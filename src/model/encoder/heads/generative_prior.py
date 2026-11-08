# Copyright (c) 2026 The GIFSplat Authors.
# SPDX-License-Identifier: CC-BY-NC-4.0
"""Frozen GIFSplat prior cues; enabled only through explicit runtime arguments."""

import math

import torch
from torch import nn

from ...decoder.observation_pooling import pool_observation


@torch.no_grad()
def sample_prior_cameras(context_pose, scene_scale, num_views=2):
    """Deterministic nearest-pair SO(3) midpoints; no targets or frame order.

    Select distinct unordered pairs by center distance/scene scale + angle/pi.
    Exact distance ties use input indices. Repeated poses are allowed; with two
    contexts there is one pair, regardless of the requested upper budget.
    """
    from scipy.spatial.transform import Rotation

    poses = context_pose["extrinsic"].detach().float()
    intrinsics = context_pose["intrinsic"].detach().float()
    if poses.ndim != 4 or poses.shape[-2:] != (4, 4) or poses.shape[1] < 2:
        raise ValueError("GIFSplat needs at least two context cameras [B,V,4,4].")
    if intrinsics.shape != (*poses.shape[:2], 3, 3) or not torch.isfinite(intrinsics).all():
        raise ValueError("Expected finite normalized context intrinsics [B,V,3,3].")
    if type(num_views) is not int or num_views < 1:
        raise ValueError("num_views must be a positive integer.")
    # Reuse the pose/scale validation, including finite transforms.
    closest_reference_indices(poses, poses[:, :1], scene_scale)
    with torch.autocast(poses.device.type, enabled=False):
        left, right = torch.triu_indices(poses.shape[1], poses.shape[1], 1, device=poses.device)
        distance = (poses[:, left, :3, 3] - poses[:, right, :3, 3]).norm(dim=-1)
        trace = (poses[:, left, :3, :3] * poses[:, right, :3, :3]).sum((-1, -2))
        score = distance / scene_scale.float()[:, None] + ((trace - 1) / 2).clamp(-1, 1).acos() / math.pi
        selected = score.argsort(dim=1, stable=True)[:, :min(num_views, len(left))]
        batch = torch.arange(len(poses), device=poses.device)[:, None]
        a, b = left[selected], right[selected]
        first, second = poses[batch, a], poses[batch, b]
        result = first.clone()
        result[..., :3, 3] = (first[..., :3, 3] + second[..., :3, 3]) / 2
        r0 = Rotation.from_matrix(first[..., :3, :3].reshape(-1, 3, 3).cpu().numpy())
        r1 = Rotation.from_matrix(second[..., :3, :3].reshape(-1, 3, 3).cpu().numpy())
        midpoint = (r0 * Rotation.from_rotvec((r0.inv() * r1).as_rotvec() * .5)).as_matrix()
        result[..., :3, :3] = torch.as_tensor(midpoint, device=poses.device, dtype=torch.float32).reshape_as(result[..., :3, :3])
        return {"extrinsic": result, "intrinsic": (intrinsics[batch, a] + intrinsics[batch, b]) / 2}


@torch.no_grad()
def closest_reference_indices(context_c2w, novel_c2w, scene_scale):
    """Select context references by normalized translation + rotation angle/pi.

    Inputs are [B,V,4,4], [B,K,4,4], [B]; output is [B,K]. The paper does
    not specify a pose-distance metric; this dimensionless score is our choice.
    """
    if (context_c2w.ndim != 4 or novel_c2w.ndim != 4
            or context_c2w.shape[-2:] != (4, 4) or novel_c2w.shape[-2:] != (4, 4)
            or context_c2w.shape[0] != novel_c2w.shape[0]
            or context_c2w.shape[1] == 0 or novel_c2w.shape[1] == 0
            or scene_scale.shape != context_c2w.shape[:1]):
        raise ValueError("Expected nonempty [B,V,4,4], [B,K,4,4] cameras and [B] scene scale.")
    if (not torch.isfinite(context_c2w).all() or not torch.isfinite(novel_c2w).all()
            or not torch.isfinite(scene_scale).all() or (scene_scale <= 0).any()):
        raise ValueError("Camera poses must be finite and scene scale positive.")
    with torch.autocast(context_c2w.device.type, enabled=False):
        context, novel = context_c2w.float(), novel_c2w.float()
        distance = (novel[:, :, None, :3, 3] - context[:, None, :, :3, 3]).norm(dim=-1)
        trace = (novel[:, :, None, :3, :3] * context[:, None, :, :3, :3]).sum((-1, -2))
        angle = ((trace - 1) / 2).clamp(-1, 1).acos()
        return (distance / scene_scale.float()[:, None, None] + angle / math.pi).argmin(-1)


class GenerativePriorCues(nn.Module):
    """Eq. (4): lift enhanced-minus-rendered features using original render weights.

    Process one scene and a bounded caller-selected view set at a time. No GT
    target images are accepted, and no generated image becomes a loss target.
    """

    def __init__(self, enhancer, feature_encoder):
        super().__init__()
        self.enhancer = enhancer
        self.feature_encoder = feature_encoder
        self.requires_grad_(False)
        self.eval()

    def train(self, mode=True):
        return super().train(False)

    @torch.no_grad()
    def pool_view(self, rendered, reference, meta, num_gaussians, generator=None):
        if (meta["height"], meta["width"]) != rendered.shape[-2:]:
            raise ValueError("Raster metadata must correspond to the unresized original render.")
        with torch.autocast(rendered.device.type, enabled=False):
            rendered = rendered.detach().float()
            enhanced = self.enhancer(rendered, reference, generator=generator)
            original_features = self.feature_encoder(rendered)
            enhanced_features = self.feature_encoder(enhanced)
            residuals = tuple(e - r for e, r in zip(enhanced_features, original_features, strict=True))
            return pool_observation(meta, residuals, num_gaussians)

    @torch.no_grad()
    def render_cues(self, renderer, gaussians, context_images, context_pose,
                    novel_pose, scene_scale, image_shape, generator=None):
        """Stream novel views of CURRENT Gaussians; references are context-only.

        novel_pose uses predicted-world c2w and normalized intrinsics, like the
        renderer. Its K-view budget/trajectory is explicit, never inferred from GT.
        """
        cameras, intrinsics = novel_pose["extrinsic"], novel_pose["intrinsic"]
        b, n = gaussians.means.shape[:2]
        if (cameras.ndim != 4 or cameras.shape[0] != b
                or intrinsics.shape != (*cameras.shape[:2], 3, 3)
                or not torch.isfinite(intrinsics).all()
                or context_images.shape[:2] != context_pose["extrinsic"].shape[:2]
                or tuple(context_images.shape[-2:]) != tuple(image_shape)):
            raise ValueError("Prior cameras/images must share the batch and unresized image frame.")
        references = closest_reference_indices(context_pose["extrinsic"], cameras, scene_scale)
        sums = gaussians.means.new_zeros(b, n, 256, dtype=torch.float32)
        masses = gaussians.means.new_zeros(b, n, dtype=torch.float32)
        seen = set()

        def collect(i, j, color, meta):
            if (i, j) in seen:
                raise ValueError("Prior renderer emitted a duplicate view.")
            seen.add((i, j))
            reference = context_images[i, references[i, j]].unsqueeze(0)
            numerator, denominator = self.pool_view(color, reference, meta, n, generator)
            sums[i].add_(numerator)
            masses[i].add_(denominator)

        with torch.autocast(context_images.device.type, enabled=False):
            renderer(gaussians, cameras.float(), intrinsics.float(),
                     masses.new_full(cameras.shape[:2], .01),
                     masses.new_full(cameras.shape[:2], 100.), image_shape, "depth",
                     evidence_callback=collect)
        if len(seen) != b * cameras.shape[1]:
            raise RuntimeError("Prior renderer must supply raster evidence for every novel view.")
        return sums / masses.clamp_min(1e-8)[..., None], masses

    @torch.no_grad()
    def forward(self, rendered_views, reference_views, raster_metadata, num_gaussians,
                generator=None):
        if (rendered_views.ndim != 4 or rendered_views.shape[1] != 3
                or rendered_views.shape != reference_views.shape
                or len(rendered_views) == 0 or len(raster_metadata) != len(rendered_views)
                or num_gaussians <= 0):
            raise ValueError("Expected nonempty paired [V,3,H,W] images, one raster metadata per view, and N>0.")
        sums, masses = None, None
        with torch.autocast(rendered_views.device.type, enabled=False):
            for view, meta in enumerate(raster_metadata):
                rendered = rendered_views[view:view + 1].detach().float()
                numerator, denominator = self.pool_view(
                    rendered, reference_views[view:view + 1], meta, num_gaussians, generator)
                if sums is None:
                    sums, masses = numerator, denominator
                else:
                    sums.add_(numerator)
                    masses.add_(denominator)
        # Normalize once over all views, not as an unweighted mean of view cues.
        return sums / masses.clamp_min(1e-8)[:, None], masses
