import copy

# VGGT parts
from dataclasses import dataclass, field
from typing import Callable, List, Literal, Optional

import torch
from einops import rearrange
from jaxtyping import Float

from src.model.encoder.heads.gaussian_refinement_head import (
    GaussianRefinementHead,
    RefinementHeadCfg,
    GIFSplatRefinementHead,
)
from src.model.gifsplat_config import GIFSplatCfg
from src.model.encoder.heads.vggt_dpt_gs_head import VGGT_DPT_GS_Head
from src.model.encoder.heads.observation_features import ObservationFeatures
from src.model.decoder.observation_pooling import pool_observation
from src.model.encoder.vggt.utils.geometry import batchify_unproject_depth_map_to_point_map
from src.model.encoder.vggt.utils.pose_enc import pose_encoding_to_extri_intri
from torch import nn, Tensor
from torch_scatter import scatter_add, scatter_max

from ..types import Gaussians
from .common.gaussian_adapter import (
    GaussianAdapter,
    GaussianAdapterCfg,
    UnifiedGaussianAdapter,
)
from .common.gaussians import build_covariance
from .encoder import Encoder, EncoderOutput
from src.model.encoder.vggt.models.vggt import VGGT

inf = float("inf")


@dataclass
class OpacityMappingCfg:
    initial: float
    final: float
    warm_up: int


@dataclass
class GSHeadParams:
    dec_depth: int = 23
    patch_size: tuple[int, int] = (14, 14)
    enc_embed_dim: int = 2048
    dec_embed_dim: int = 2048
    feature_dim: int = 256
    depth_mode = ("exp", -inf, inf)
    conf_mode = True


@dataclass
class EncoderAnySplatCfg:
    name: Literal["anysplat"]
    anchor_feat_dim: int
    voxel_size: float
    n_offsets: int
    d_feature: int
    add_view: bool
    num_monocular_samples: int
    gaussian_adapter: GaussianAdapterCfg
    opacity_mapping: OpacityMappingCfg
    gaussians_per_pixel: int
    num_surfaces: int
    gs_params_head_type: str
    input_mean: tuple[float, float, float] = (0.5, 0.5, 0.5)
    input_std: tuple[float, float, float] = (0.5, 0.5, 0.5)
    pose_free: bool = True
    pred_pose: bool = True
    gt_pose_to_pts: bool = False
    gs_prune: bool = False
    opacity_threshold: float = 0.001
    gs_keep_ratio: float = 1.0
    pred_head_type: Literal["depth", "point"] = "point"
    freeze_backbone: bool = False
    freeze_module: Literal[
        "all",
        "global",
        "frame",
        "patch_embed",
        "patch_embed+frame",
        "patch_embed+global",
        "global+frame",
        "None",
    ] = "None"
    distill: bool = False
    render_conf: bool = False
    opacity_conf: bool = False
    conf_threshold: float = 0.1
    intermediate_layer_idx: Optional[List[int]] = None
    voxelize: bool = False
    gaussian_downsample: int = 1
    refinement: RefinementHeadCfg = field(default_factory=RefinementHeadCfg)
    gifsplat: GIFSplatCfg = field(default_factory=GIFSplatCfg)
    gaussian_head_hidden_channels: int | None = None




class EncoderAnySplat(Encoder[EncoderAnySplatCfg]):
    gaussian_adapter: GaussianAdapter

    def __init__(self, cfg: EncoderAnySplatCfg, load_pretrained: bool = True) -> None:
        super().__init__(cfg)
        if getattr(cfg, "gifsplat", GIFSplatCfg()).enabled and not cfg.refinement.enabled:
            raise ValueError("GIFSplat requires refinement.enabled=true.")
        if cfg.refinement.enabled and cfg.voxelize:
            raise ValueError("Gaussian refinement requires `voxelize: false`.")
        if cfg.refinement.enabled and cfg.gs_prune:
            raise ValueError("Gaussian refinement does not support `gs_prune: true`.")
        if type(cfg.gaussian_downsample) is not int or cfg.gaussian_downsample < 1:
            raise ValueError("gaussian_downsample must be a positive integer.")
        if cfg.gaussian_downsample != 1 and cfg.voxelize:
            raise ValueError("gaussian_downsample requires `voxelize: false`.")

        model_full = VGGT.from_pretrained("facebook/VGGT-1B") if load_pretrained else VGGT()
        # model_full = VGGT()
        self.aggregator = model_full.aggregator.to(torch.bfloat16)
        self.freeze_backbone = cfg.freeze_backbone
        self.distill = cfg.distill and not cfg.refinement.enabled
        self.pred_pose = cfg.pred_pose

        self.camera_head = model_full.camera_head
        if self.cfg.pred_head_type == "depth":
            self.depth_head = model_full.depth_head
        else:
            self.point_head = model_full.point_head

        if self.distill:
            self.distill_aggregator = copy.deepcopy(self.aggregator)
            self.distill_camera_head = copy.deepcopy(self.camera_head)
            self.distill_depth_head = copy.deepcopy(self.depth_head)
            for module in [
                self.distill_aggregator,
                self.distill_camera_head,
                self.distill_depth_head,
            ]:
                for param in module.parameters():
                    param.requires_grad = False
                    param.data = param.data.cpu()

        if self.freeze_backbone:
            # Freeze backbone components
            if self.cfg.pred_head_type == "depth":
                for module in [self.aggregator, self.camera_head, self.depth_head]:
                    for param in module.parameters():
                        param.requires_grad = False
            else:
                for module in [self.aggregator, self.camera_head, self.point_head]:
                    for param in module.parameters():
                        param.requires_grad = False
        else:
            # aggregator freeze
            freeze_module = self.cfg.freeze_module
            if freeze_module == "None":
                pass

            elif freeze_module == "all":
                for param in self.aggregator.parameters():
                    param.requires_grad = False

            else:
                module_pairs = {
                    "patch_embed+frame": ["patch_embed", "frame"],
                    "patch_embed+global": ["patch_embed", "global"],
                    "global+frame": ["global", "frame"],
                }

                if freeze_module in module_pairs:
                    for name, param in self.aggregator.named_parameters():
                        if any(m in name for m in module_pairs[freeze_module]):
                            param.requires_grad = False
                else:
                    for name, param in self.named_parameters():
                        param.requires_grad = (
                            freeze_module not in name and "distill" not in name
                        )

        self.pose_free = cfg.pose_free
        if self.pose_free:
            self.gaussian_adapter = UnifiedGaussianAdapter(cfg.gaussian_adapter)
        else:
            self.gaussian_adapter = GaussianAdapter(cfg.gaussian_adapter)

        self.raw_gs_dim = 1 + self.gaussian_adapter.d_in  # 1 for opacity
        self.voxel_size = cfg.voxel_size
        self.gs_params_head_type = cfg.gs_params_head_type
        # fake backbone for head parameters
        head_params = GSHeadParams()
        self.gaussian_param_head = VGGT_DPT_GS_Head(
            dim_in=2048,
            patch_size=head_params.patch_size,
            output_dim=self.raw_gs_dim + 1,
            activation="norm_exp",
            conf_activation="expp1",
            features=head_params.feature_dim,
            head_hidden_channels=cfg.gaussian_head_hidden_channels,
        )
        self.refinement_renderer: Optional[Callable] = None
        self.refinement_head: GaussianRefinementHead | None = None
        if cfg.refinement.enabled:
            sh_dim = 3 * self.gaussian_adapter.d_sh
            head_class = GIFSplatRefinementHead if cfg.gifsplat.enabled else GaussianRefinementHead
            self.refinement_head = head_class(
                gaussian_token_dim=3 + 3 + 4 + 1 + sh_dim,
                init_feature_dim=128,
                delta_dim=3 + 3 + 4 + 1 + sh_dim,
                cfg=cfg.refinement,
            )
            self.observation_features = ObservationFeatures(pretrained=load_pretrained)
            for name, param in self.named_parameters():
                param.requires_grad_(name.startswith("refinement_head."))
            self.train(self.training)

    def train(self, mode: bool = True):
        super().train(mode)
        if self.cfg.refinement.enabled:
            for name, module in self.named_children():
                if name != "refinement_head":
                    module.eval()
        return self

    def set_refinement_renderer(self, renderer: Callable) -> None:
        self.refinement_renderer = renderer

    def get_generative_prior(self, device):
        from src.model.difix_enhancer import load_difix_enhancer
        from .heads.generative_prior import GenerativePriorCues

        # Runtime-only frozen dependency: exclude diffusion weights from state_dict
        # and DDP/optimizer registration. Construct after rank/device assignment.
        cached = self.__dict__.get("_prior_runtime")
        if cached is None or cached[0] != device:
            provider = GenerativePriorCues(load_difix_enhancer(self.cfg.gifsplat, device),
                                           self.observation_features)
            self.__dict__["_prior_runtime"] = (device, provider)
        return self.__dict__["_prior_runtime"][1]

    def map_pdf_to_opacity(
        self,
        pdf: Float[Tensor, " *batch"],
        global_step: int,
    ) -> Float[Tensor, " *batch"]:
        # https://www.desmos.com/calculator/opvwti3ba9

        # Figure out the exponent.
        cfg = self.cfg.opacity_mapping
        x = cfg.initial + min(global_step / cfg.warm_up, 1) * (cfg.final - cfg.initial)
        exponent = 2**x

        # Map the probability density to an opacity.
        return 0.5 * (1 - (1 - pdf) ** exponent + pdf ** (1 / exponent))



    def pad_tensor_list(self, tensor_list, pad_shape, value=0.0):
        padded = []
        for t in tensor_list:
            pad_len = pad_shape[0] - t.shape[0]
            if pad_len > 0:
                padding = torch.full(
                    (pad_len, *t.shape[1:]), value, device=t.device, dtype=t.dtype
                )
                t = torch.cat([t, padding], dim=0)
            padded.append(t)
        return torch.stack(padded)

    def voxelizaton_with_fusion(self, img_feat, pts3d, voxel_size, conf=None):
        # img_feat: B*V, C, H, W
        # pts3d: B*V, 3, H, W
        V, C, H, W = img_feat.shape
        pts3d_flatten = pts3d.permute(0, 2, 3, 1).flatten(0, 2)

        voxel_indices = (pts3d_flatten / voxel_size).round().int()  # [B*V*N, 3]
        unique_voxels, inverse_indices, counts = torch.unique(
            voxel_indices, dim=0, return_inverse=True, return_counts=True
        )

        # Flatten confidence scores and features
        conf_flat = conf.flatten()  # [B*V*N]
        anchor_feats_flat = img_feat.permute(0, 2, 3, 1).flatten(0, 2)  # [B*V*N, ...]

        # Compute softmax weights per voxel
        conf_voxel_max, _ = scatter_max(conf_flat, inverse_indices, dim=0)
        conf_exp = torch.exp(conf_flat - conf_voxel_max[inverse_indices])
        voxel_weights = scatter_add(
            conf_exp, inverse_indices, dim=0
        )  # [num_unique_voxels]
        weights = (conf_exp / (voxel_weights[inverse_indices] + 1e-6)).unsqueeze(
            -1
        )  # [B*V*N, 1]

        # Compute weighted average of positions and features
        weighted_pts = pts3d_flatten * weights
        weighted_feats = anchor_feats_flat.squeeze(1) * weights

        # Aggregate per voxel
        voxel_pts = scatter_add(
            weighted_pts, inverse_indices, dim=0
        )  # [num_unique_voxels, 3]
        voxel_feats = scatter_add(
            weighted_feats, inverse_indices, dim=0
        )  # [num_unique_voxels, feat_dim]

        return voxel_pts, voxel_feats

    def build_pred_context_pose(
        self,
        extrinsic: torch.Tensor,
        intrinsic: torch.Tensor,
        height: int,
        width: int,
    ) -> dict[str, torch.Tensor]:
        b, v = extrinsic.shape[:2]
        extrinsic_padding = (
            torch.tensor([0, 0, 0, 1], device=extrinsic.device, dtype=extrinsic.dtype)
            .view(1, 1, 1, 4)
            .repeat(b, v, 1, 1)
        )
        normalized_intrinsic = intrinsic.clone()
        normalized_intrinsic = torch.stack(
            [
                normalized_intrinsic[:, :, 0] / width,
                normalized_intrinsic[:, :, 1] / height,
                normalized_intrinsic[:, :, 2],
            ],
            dim=2,
        )
        return {
            "extrinsic": torch.cat([extrinsic, extrinsic_padding], dim=2).inverse(),
            "intrinsic": normalized_intrinsic,
        }

    def get_num_refine_steps(self) -> int:
        return self.cfg.refinement.num_refine

    def gaussians_to_tokens(
        self,
        gaussians: Gaussians,
        scene_scale: torch.Tensor,
    ) -> torch.Tensor:
        scale = scene_scale.view(-1, 1, 1).clamp_min(1e-8)
        shs = gaussians.harmonics.flatten(2)
        return torch.cat(
            (
                gaussians.means / scale,
                (gaussians.scales / scale).clamp_min(torch.finfo(torch.float32).tiny).log(),
                gaussians.rotations,
                torch.logit(gaussians.opacities.clamp(1e-6, 1 - 1e-6)).unsqueeze(-1),
                shs,
            ),
            dim=-1,
        )

    def detach_gaussians(self, gaussians: Gaussians) -> Gaussians:
        return Gaussians(
            means=gaussians.means.detach(),
            covariances=gaussians.covariances.detach(),
            harmonics=gaussians.harmonics.detach(),
            opacities=gaussians.opacities.detach(),
            scales=gaussians.scales.detach(),
            rotations=gaussians.rotations.detach(),
        )

    def apply_refinement_deltas(
        self,
        latent: torch.Tensor,
        initial_gaussians: Gaussians,
        delta_gaussians: torch.Tensor,
        scene_scale: torch.Tensor,
        gaussian_valid_mask: torch.Tensor,
    ) -> tuple[Gaussians, torch.Tensor, dict[str, torch.Tensor]]:
        sh_dim = 3 * self.gaussian_adapter.d_sh
        valid_mask = gaussian_valid_mask.unsqueeze(-1)
        delta = delta_gaussians.float().masked_fill(~valid_mask, 0)
        latent = latent.float() + delta
        means, log_scales, raw_rotations, opacity_logits, shs = latent.split((3, 3, 4, 1, sh_dim), dim=-1)
        scale = scene_scale.view(-1, 1, 1).clamp_min(1e-8)
        # Decode relative to the fixed initializer to avoid round-trip drift
        # (world -> normalized -> world) when the predicted update is zero.
        means = initial_gaussians.means + (means - initial_gaussians.means / scale) * scale
        initial_log_scales = (initial_gaussians.scales / scale).clamp_min(torch.finfo(torch.float32).tiny).log()
        scales = initial_gaussians.scales * (log_scales - initial_log_scales).exp()
        norm = raw_rotations.norm(dim=-1, keepdim=True)
        rotations = torch.where(norm > 1e-8, raw_rotations / norm.clamp_min(1e-8), initial_gaussians.rotations)
        opacities = opacity_logits.sigmoid().squeeze(-1)
        means = torch.where(valid_mask, means, initial_gaussians.means)
        scales = torch.where(valid_mask, scales, initial_gaussians.scales)
        rotations = torch.where(valid_mask, rotations, initial_gaussians.rotations)
        shs = torch.where(valid_mask, shs, initial_gaussians.harmonics.flatten(2))
        opacities = torch.where(gaussian_valid_mask, opacities, initial_gaussians.opacities)

        covariances = build_covariance(scales, rotations)
        gaussians = Gaussians(
            means=means.float(),
            covariances=covariances.float(),
            harmonics=shs.view_as(initial_gaussians.harmonics).float(),
            opacities=opacities.float(),
            scales=scales.float(),
            rotations=rotations.float(),
        )
        metrics = {
            "delta_mean_norm": (delta[..., :3] * scale).norm(dim=-1)[gaussian_valid_mask].mean(),
            "delta_scale_norm": delta[..., 3:6].norm(dim=-1)[gaussian_valid_mask].mean(),
            "delta_opacity_norm": delta[..., 10].abs()[gaussian_valid_mask].mean(),
        }
        return gaussians, latent, metrics

    def run_refinement(
        self,
        image: torch.Tensor,
        gaussians: Gaussians,
        gaussian_valid_mask: torch.Tensor,
        initial_features: torch.Tensor,
        scene_scale: torch.Tensor,
        pred_context_pose: dict[str, torch.Tensor],
        image_shape: tuple[int, int],
        generative_prior=None,
        novel_pose=None,
        prior_generator=None,
    ) -> tuple[Gaussians, list, list[Gaussians], dict]:
        use_prior = generative_prior is not None
        if use_prior != (novel_pose is not None):
            raise ValueError("Provide both generative_prior and novel_pose, or neither.")
        is_gifsplat = getattr(self.refinement_head, "architecture", None) == "gifsplat"
        if use_prior != is_gifsplat:
            raise ValueError("Generative priors require a GIFSplat head and vice versa.")
        if not self.cfg.refinement.enabled:
            return gaussians, [], [], {"num_refine": 0}
        if self.refinement_head is None or self.refinement_renderer is None:
            raise RuntimeError("Refinement is enabled but the refinement modules are not fully initialized.")

        num_refine = self.get_num_refine_steps()
        if num_refine == 0:
            return gaussians, [], [], {"num_refine": 0}

        b, v, _, h, w = image.shape
        near = torch.full((b, v), 0.01, device=image.device, dtype=image.dtype)
        far = torch.full((b, v), 100.0, device=image.device, dtype=image.dtype)

        # Input features are cached; neither rendering evidence nor the frozen
        # image encoder builds an autograd graph. Target renders carry the loss.
        with torch.no_grad(), torch.autocast(image.device.type, enabled=False):
            observed = [[self.observation_features(image[i, j:j+1]) for j in range(v)] for i in range(b)]

        def render(current, collect_evidence):
            sums = image.new_zeros(b, gaussians.means.shape[1], 256, dtype=torch.float32)
            masses = image.new_zeros(b, gaussians.means.shape[1], dtype=torch.float32)

            def collect(i, j, color, meta):
                predicted = self.observation_features(color)
                residual = tuple(a - p for a, p in zip(observed[i][j], predicted))
                numerator, denominator = pool_observation(meta, residual, gaussians.means.shape[1])
                sums[i].add_(numerator)
                masses[i].add_(denominator)

            with torch.no_grad(), torch.autocast(image.device.type, enabled=False):
                rendered = self.refinement_renderer(
                    self.detach_gaussians(current), pred_context_pose["extrinsic"],
                    pred_context_pose["intrinsic"], near, far, image_shape, "depth",
                    evidence_callback=collect if collect_evidence else None,
                )
            return rendered, sums / masses.clamp_min(1e-8)[..., None]

        current_gaussians = gaussians
        current_render, evidence = render(current_gaussians, True)
        initial_render = current_render
        state = self.refinement_head.initialize_state(initial_features.detach())
        state = torch.where(
            gaussian_valid_mask.unsqueeze(-1), state, torch.zeros_like(state)
        )

        refinement_outputs = []
        refinement_gaussians = []
        delta_mean_norms = []
        delta_scale_norms = []
        delta_opacity_norms = []
        prior_norms = []
        # Keep log-scales, opacity logits and raw quaternions across iterations;
        # render-time activation must not overwrite the recurrent latent state.
        latent = self.gaussians_to_tokens(gaussians, scene_scale)

        for step in range(num_refine):
            if self.cfg.refinement.detach_prev_iter:
                latent = latent.detach()
            head_kwargs = {}
            if use_prior:
                prior, _ = generative_prior.render_cues(
                    self.refinement_renderer, self.detach_gaussians(current_gaussians),
                    image, pred_context_pose, novel_pose, scene_scale, image_shape,
                    generator=prior_generator,
                )
                prior = prior.detach().masked_fill(~gaussian_valid_mask[..., None], 0)
                prior_norms.append(prior.norm(dim=-1)[gaussian_valid_mask].mean())
                head_kwargs["prior"] = prior
            next_state, delta_gaussians = self.refinement_head(
                latent,
                state,
                evidence,
                gaussian_valid_mask,
                **head_kwargs,
            )
            state = torch.where(
                gaussian_valid_mask.unsqueeze(-1), next_state, state
            )
            current_gaussians, latent, metrics = self.apply_refinement_deltas(
                latent,
                gaussians,
                delta_gaussians,
                scene_scale,
                gaussian_valid_mask,
            )
            current_render, evidence = render(current_gaussians, step + 1 < num_refine)
            refinement_gaussians.append(current_gaussians)
            refinement_outputs.append(current_render)
            delta_mean_norms.append(metrics["delta_mean_norm"])
            delta_scale_norms.append(metrics["delta_scale_norm"])
            delta_opacity_norms.append(metrics["delta_opacity_norm"])

        return current_gaussians, refinement_outputs, refinement_gaussians, {
            "num_refine": num_refine,
            "initial_render": initial_render,
            "delta_mean_norm": delta_mean_norms,
            "delta_scale_norm": delta_scale_norms,
            "delta_opacity_norm": delta_opacity_norms,
            **({"prior_norm": prior_norms} if use_prior else {}),
        }

    def forward(
        self,
        image: torch.Tensor,
        global_step: int = 0,
        visualization_dump: Optional[dict] = None,
        *,
        generative_prior=None,
        novel_pose=None,
        prior_generator=None,
        unprojection_cameras=None,
    ) -> EncoderOutput:
        if unprojection_cameras is not None and self.cfg.refinement.enabled:
            raise ValueError("unprojection_cameras is for initializer training, not refinement.")
        auto_prior = getattr(self.cfg, "gifsplat", GIFSplatCfg()).enabled
        if auto_prior and generative_prior is None:
            generative_prior = self.get_generative_prior(image.device)
        if auto_prior and prior_generator is None:
            prior_generator = torch.Generator(device=image.device).manual_seed(self.cfg.gifsplat.seed)
        if not auto_prior and (generative_prior is None) != (novel_pose is None):
            raise ValueError("Provide both generative_prior and novel_pose, or neither.")
        if generative_prior is not None and not self.cfg.refinement.enabled:
            raise ValueError("Generative prior requires refinement.enabled=true.")
        with torch.set_grad_enabled(torch.is_grad_enabled() and not self.cfg.refinement.enabled):
            output, features, valid = self._forward_initial(
                image, global_step, visualization_dump, unprojection_cameras=unprojection_cameras)
        if self.cfg.refinement.enabled:
            if auto_prior and novel_pose is None:
                from .heads.generative_prior import sample_prior_cameras
                novel_pose = sample_prior_cameras(output.pred_context_pose, output.infos["scene_scale"],
                                                  self.cfg.gifsplat.num_views)
            output.gaussians, output.refinement_outputs, output.refinement_gaussians, metrics = self.run_refinement(
                image, output.gaussians, valid, features, output.infos["scene_scale"],
                output.pred_context_pose, image.shape[-2:],
                generative_prior=generative_prior, novel_pose=novel_pose,
                prior_generator=prior_generator,
            )
            output.refinement_initial_output = metrics.pop("initial_render")
            output.refinement_metrics = metrics
            output.infos["refinement_num_steps"] = metrics["num_refine"]
        return output

    def _forward_initial(self, image, global_step=0, visualization_dump=None, unprojection_cameras=None):
        device = image.device
        b, v, _, h, w = image.shape
        downsample = self.cfg.gaussian_downsample
        if h % downsample or w % downsample:
            raise ValueError("Image dimensions must be divisible by gaussian_downsample.")
        if unprojection_cameras is not None and self.cfg.pred_head_type != "depth":
            raise ValueError("unprojection_cameras requires pred_head_type: depth.")
        distill_infos = {}
        if self.distill:
            distill_image = image.clone().detach()
            for module in [
                self.distill_aggregator,
                self.distill_camera_head,
                self.distill_depth_head,
            ]:
                for param in module.parameters():
                    param.data = param.data.to(device, non_blocking=True)

            with torch.no_grad():
                # Process with bfloat16 precision
                with torch.amp.autocast("cuda", enabled=True, dtype=torch.bfloat16):
                    distill_aggregated_tokens_list, distill_patch_start_idx = (
                        self.distill_aggregator(
                            distill_image.to(torch.bfloat16),
                            intermediate_layer_idx=self.cfg.intermediate_layer_idx,
                        )
                    )

                # Process with default precision
                with torch.amp.autocast("cuda", enabled=False):
                    # Get camera pose information
                    distill_pred_pose_enc_list = self.distill_camera_head(
                        distill_aggregated_tokens_list
                    )
                    last_distill_pred_pose_enc = distill_pred_pose_enc_list[-1]
                    distill_extrinsic, distill_intrinsic = pose_encoding_to_extri_intri(
                        last_distill_pred_pose_enc, image.shape[-2:]
                    )

                    # Get depth information
                    distill_depth_map, distill_depth_conf = self.distill_depth_head(
                        distill_aggregated_tokens_list,
                        images=distill_image,
                        patch_start_idx=distill_patch_start_idx,
                    )

                    # Convert depth to 3D points
                    distill_pts_all = batchify_unproject_depth_map_to_point_map(
                        distill_depth_map, distill_extrinsic, distill_intrinsic
                    )
                # Store results
                distill_infos["pred_pose_enc_list"] = distill_pred_pose_enc_list
                distill_infos["pts_all"] = distill_pts_all
                distill_infos["depth_map"] = distill_depth_map

                conf_threshold = torch.quantile(
                    distill_depth_conf.flatten(2, 3), 0.3, dim=-1, keepdim=True
                )  # Get threshold for each view
                conf_mask = distill_depth_conf > conf_threshold.unsqueeze(-1)
                distill_infos["conf_mask"] = conf_mask

                for module in [
                    self.distill_aggregator,
                    self.distill_camera_head,
                    self.distill_depth_head,
                ]:
                    for param in module.parameters():
                        param.data = param.data.cpu()
                # Clean up to save memory
                del distill_aggregated_tokens_list, distill_patch_start_idx
                del distill_pred_pose_enc_list, last_distill_pred_pose_enc
                del distill_extrinsic, distill_intrinsic
                del distill_depth_map, distill_depth_conf
                torch.cuda.empty_cache()

        from simple_timer import tic, toc, report
        tic("agg")
        with torch.amp.autocast("cuda", enabled=True, dtype=torch.bfloat16):
            aggregated_tokens_list, patch_start_idx = self.aggregator(
                image.to(torch.bfloat16),
                intermediate_layer_idx=self.cfg.intermediate_layer_idx,
            )
        toc("agg")


        with torch.amp.autocast("cuda", enabled=False):
            tic("h_c")
            pred_pose_enc_list = self.camera_head(aggregated_tokens_list)
            last_pred_pose_enc = pred_pose_enc_list[-1]
            extrinsic, intrinsic = pose_encoding_to_extri_intri(
                last_pred_pose_enc, image.shape[-2:]
            )  # only for debug
            toc("h_c")
            if self.cfg.pred_head_type == "point":
                tic("h_d")
                pts_all, pts_conf = self.point_head(
                    aggregated_tokens_list,
                    images=image,
                    patch_start_idx=patch_start_idx,
                )
                depth_map = pts_all[..., -1:].contiguous()
                depth_conf = pts_conf
                toc("h_d")
            elif self.cfg.pred_head_type == "depth":
                tic("h_p")
                depth_map, depth_conf = self.depth_head(
                    aggregated_tokens_list,
                    images=image,
                    patch_start_idx=patch_start_idx,
                )
                # Training-only hook: (predicted w2c [B,V,3,4], pixel K) -> cameras
                # placing Gaussian centers. pred_context_pose stays predicted.
                unproject_extrinsic, unproject_intrinsic = (
                    unprojection_cameras(extrinsic, intrinsic)
                    if unprojection_cameras is not None else (extrinsic, intrinsic)
                )
                pts_all = batchify_unproject_depth_map_to_point_map(
                    depth_map, unproject_extrinsic, unproject_intrinsic
                )
                toc("h_p")
            else:
                raise ValueError(f"Invalid pred_head_type: {self.cfg.pred_head_type}")

            if self.cfg.render_conf:
                conf_valid = torch.quantile(
                    depth_conf.flatten(0, 1), self.cfg.conf_threshold
                )
                conf_valid_mask = depth_conf > conf_valid
            else:
                conf_valid_mask = torch.ones_like(depth_conf, dtype=torch.bool)

        # dpt style gs_head input format
        tic("h_g")
        out = self.gaussian_param_head(
            aggregated_tokens_list,
            pts_all.flatten(0, 1).permute(0, 3, 1, 2),
            image,
            patch_start_idx=patch_start_idx,
            image_size=(h, w),
            return_features=self.cfg.refinement.enabled,
        )
        initial_feature_map = None
        if self.cfg.refinement.enabled:
            out, initial_feature_map = out
        toc("h_g")


        del aggregated_tokens_list, patch_start_idx
        torch.cuda.empty_cache()

        pts_flat = pts_all.flatten(2, 3)
        scene_scale = pts_flat.norm(dim=-1).mean(dim=(1, 2)).clamp_min(1e-8)
        initial_features = None
        anchor_feats, conf = out[:, :, : self.raw_gs_dim], out[:, :, self.raw_gs_dim]

        neural_feats_list, neural_pts_list = [], []
        gaussian_valid_mask_list = []
        tic("voxel")
        if self.cfg.voxelize:
            print("VOXELIZE ENABLED", self.voxel_size)
            self.voxel_size = 0.002
            print("VOXELIZE CHANGED", self.voxel_size)
            for b_i in range(b):
                neural_pts, neural_feats = self.voxelizaton_with_fusion(
                    anchor_feats[b_i],
                    pts_all[b_i].permute(0, 3, 1, 2).contiguous(),
                    self.voxel_size,
                    conf=conf[b_i],
                )
                neural_feats_list.append(neural_feats)
                neural_pts_list.append(neural_pts)
                gaussian_valid_mask_list.append(
                    torch.ones(neural_pts.shape[0], device=device, dtype=torch.bool)
                )
        else:
            # Sampling unprojected rays is equivalent to unprojecting sampled
            # depth with intrinsics / d. Images and depth diagnostics stay full size.
            sampled_mask = conf_valid_mask[:, :, ::downsample, ::downsample]
            sampled_pts = pts_all[:, :, ::downsample, ::downsample]
            sampled_feats = anchor_feats[:, :, :, ::downsample, ::downsample]
            if initial_feature_map is not None:
                sampled_init = initial_feature_map[:, :, :, ::downsample, ::downsample]
                selected = [sampled_init[i].permute(0, 2, 3, 1)[sampled_mask[i]] for i in range(b)]
                initial_features = self.pad_tensor_list(selected, (max(len(x) for x in selected),), value=0.0)
            for b_i in range(b):
                selected_mask = sampled_mask[b_i]
                neural_feats_list.append(
                    sampled_feats[b_i].permute(0, 2, 3, 1)[selected_mask]
                )
                neural_pts_list.append(sampled_pts[b_i][selected_mask])
                gaussian_valid_mask_list.append(
                    torch.ones(neural_pts_list[-1].shape[0], device=device, dtype=torch.bool)
                )
        toc("voxel")
        report()
        max_voxels = max(f.shape[0] for f in neural_feats_list)
        neural_feats = self.pad_tensor_list(
            neural_feats_list, (max_voxels,), value=-1e10
        )

        neural_pts = self.pad_tensor_list(
            neural_pts_list, (max_voxels,), -1e4
        )  # -1 == invalid voxel
        gaussian_valid_mask = self.pad_tensor_list(
            gaussian_valid_mask_list, (max_voxels,), value=False
        )

        depths = neural_pts[..., -1].unsqueeze(-1)
        densities = neural_feats[..., 0].sigmoid()

        assert len(densities.shape) == 2, "the shape of densities should be (B, N)"
        assert neural_pts.shape[1] > 1, "the number of voxels should be greater than 1"

        opacity = self.map_pdf_to_opacity(densities, global_step).squeeze(-1)
        if self.cfg.opacity_conf:
            shift = torch.quantile(depth_conf, self.cfg.conf_threshold)
            opacity_conf = torch.sigmoid(depth_conf[:, :, ::downsample, ::downsample] - shift)
            masks = conf_valid_mask[:, :, ::downsample, ::downsample]
            opacity = opacity * self.pad_tensor_list(
                [opacity_conf[i][masks[i]] for i in range(b)], (max_voxels,), value=0.0
            )

        # GS Prune, but only works when bs = 1
        # if want to support bs > 1, need to random prune gaussians based on the rank of opacity like LongLRM
        # Note: we not prune gaussians here, but we will try it in the future
        if self.cfg.gs_prune and b == 1:
            opacity_threshold = self.cfg.opacity_threshold
            gaussian_usage = opacity > opacity_threshold  # (B, N)

            print(
                f"based on opacity threshold {opacity_threshold}, pruned {gaussian_usage.shape[1] - neural_pts.shape[1]} gaussians out of {gaussian_usage.shape[1]}"
            )

            if (gaussian_usage.sum() / gaussian_usage.numel()) > self.cfg.gs_keep_ratio:
                # rank by opacity
                num_keep = int(gaussian_usage.shape[1] * self.cfg.gs_keep_ratio)
                idx_sort = opacity.argsort(dim=1, descending=True)
                keep_idx = idx_sort[:, :num_keep]
                gaussian_usage = torch.zeros_like(gaussian_usage, dtype=torch.bool)
                gaussian_usage.scatter_(1, keep_idx, True)

            neural_pts = neural_pts[gaussian_usage].view(b, -1, 3).contiguous()
            depths = depths[gaussian_usage].view(b, -1, 1).contiguous()
            neural_feats = (
                neural_feats[gaussian_usage].view(b, -1, self.raw_gs_dim).contiguous()
            )
            opacity = opacity[gaussian_usage].view(b, -1).contiguous()
            gaussian_valid_mask = torch.ones(
                opacity.shape,
                device=opacity.device,
                dtype=torch.bool,
            )

            print(
                f"finally pruned {gaussian_usage.shape[1] - neural_pts.shape[1]} gaussians out of {gaussian_usage.shape[1]}"
            )

        gaussians = self.gaussian_adapter.forward(
            neural_pts,
            depths,
            opacity,
            neural_feats[..., 1:].squeeze(2),
        )

        if visualization_dump is not None:
            visualization_dump["depth"] = rearrange(
                pts_all[..., -1].flatten(2, 3).unsqueeze(-1).unsqueeze(-1),
                "b v (h w) srf s -> b v h w srf s",
                h=h,
                w=w,
            )

        pred_context_pose = self.build_pred_context_pose(extrinsic, intrinsic, h, w)
        infos = {}
        infos["scene_scale"] = scene_scale
        infos["voxelize_ratio"] = densities.shape[1] / (h * w * v)
        infos["refinement_enabled"] = self.cfg.refinement.enabled
        infos["refinement_num_steps"] = 0

        print(
            f"scene scale: {scene_scale.mean():.3f}, pixel-wise num: {h*w*v}, after voxelize: {neural_pts.shape[1]}, voxelize ratio: {infos['voxelize_ratio']:.3f}"
        )
        print(
            f"Gaussians attributes: \n"
            f"opacities: mean: {gaussians.opacities.mean()}, min: {gaussians.opacities.min()}, max: {gaussians.opacities.max()} \n"
            f"scales: mean: {gaussians.scales.mean()}, min: {gaussians.scales.min()}, max: {gaussians.scales.max()}"
        )

        print("B:", b, "V:", v, "H:", h, "W:", w)

        return EncoderOutput(
            gaussians=gaussians,
            pred_pose_enc_list=pred_pose_enc_list,
            pred_context_pose=pred_context_pose,
            depth_dict=dict(depth=depth_map, conf_valid_mask=conf_valid_mask),
            infos=infos,
            distill_infos=distill_infos,
        ), initial_features, gaussian_valid_mask

