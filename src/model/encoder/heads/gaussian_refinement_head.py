# Copyright (c) 2026 The GIFSplat Authors.
# SPDX-License-Identifier: CC-BY-NC-4.0
from dataclasses import dataclass, field

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint


@dataclass
class RefinementHeadCfg:
    enabled: bool = False
    num_refine: int = 3
    state_channels: int = 256
    num_blocks: int = 6
    num_heads: int = 4
    window_size: int = 128
    grid_resolution: int = 384
    detach_prev_iter: bool = True
    zero_init_update_head: bool = True
    loss_weights: list[float] = field(default_factory=lambda: [0.4, 0.3, 0.3])

    def __post_init__(self):
        if self.num_refine < 1 or self.num_blocks < 1 or self.window_size < 1:
            raise ValueError("Refinement steps, blocks and window size must be positive.")
        if self.num_heads < 1 or self.state_channels % self.num_heads:
            raise ValueError("state_channels must be divisible by num_heads.")
        if type(self.grid_resolution) is not int or not 2 <= self.grid_resolution <= 2**21:
            raise ValueError("grid_resolution must be between 2 and 2**21.")
        if len(self.loss_weights) != self.num_refine or any(w <= 0 for w in self.loss_weights):
            raise ValueError("Provide one positive loss weight per refinement step.")


@torch.no_grad()
def quantize_coordinates(coords: torch.Tensor, resolution: int) -> torch.Tensor:
    """Aspect-preserving bounding cube; quantization never merges Gaussians."""
    coords = coords.detach().float()
    shifted = coords - coords.amin(dim=0)
    return (shifted / shifted.amax().clamp_min(1e-8) * (resolution - 1)).long()


@torch.no_grad()
def serialization_code(grid: torch.Tensor, order: str, depth: int) -> torch.Tensor:
    """Pointcept-compatible Z/Hilbert codes (Skilling's invert/exchange algorithm)."""
    if order not in ("z", "z-trans", "hilbert", "hilbert-trans"):
        raise ValueError(f"Unknown serialization order: {order}")
    coords = grid[:, [1, 0, 2]].clone() if order.endswith("-trans") else grid.clone()
    if order.startswith("hilbert"):
        for bit in range(depth - 1, 0, -1):
            lower = (1 << bit) - 1
            for axis in range(3):
                active = (coords[:, axis] & (1 << bit)) != 0
                swap = torch.where(active, 0, (coords[:, 0] ^ coords[:, axis]) & lower)
                coords[:, 0] ^= torch.where(active, lower, 0)
                coords[:, axis] ^= swap
                coords[:, 0] ^= swap
    code = torch.zeros(len(grid), dtype=torch.long, device=grid.device)
    for bit in range(depth):
        for axis in range(3):
            code |= ((coords[:, axis] >> bit) & 1) << (3 * bit + 2 - axis)
    if order.startswith("hilbert"):
        shift = 1
        while shift < 3 * depth:
            code ^= code >> shift
            shift *= 2
    return code


class SparseCPE(nn.Module):
    def __init__(self, channels):
        super().__init__()
        import spconv.pytorch as spconv
        self.conv = spconv.SubMConv3d(channels, channels, kernel_size=3, bias=True)
        self.linear = nn.Linear(channels, channels)
        self.norm = nn.LayerNorm(channels)

    def forward(self, features, indices, inverse, counts, resolution):
        import spconv.pytorch as spconv
        # spconv does not support BF16. Keep sparse convolution and its gradients
        # in FP32 even under Lightning's BF16 mixed precision.
        with torch.autocast(features.device.type, enabled=False):
            pooled = features.new_zeros(len(indices), features.shape[-1], dtype=torch.float32)
            pooled.index_add_(0, inverse, features.float())
            pooled = pooled / counts[:, None]
            sparse = spconv.SparseConvTensor(pooled, indices, [resolution] * 3, 1)
            convolved = self.conv(sparse).features[inverse]
        return self.norm(self.linear(convolved))


class WindowBlock(nn.Module):
    def __init__(self, channels: int, heads: int, window_size: int):
        super().__init__()
        self.heads = heads
        self.window_size = window_size
        self.cpe = SparseCPE(channels)
        self.norm1 = nn.LayerNorm(channels)
        self.qkv = nn.Linear(channels, 3 * channels)
        self.proj = nn.Linear(channels, channels)
        self.norm2 = nn.LayerNorm(channels)
        self.ffn = nn.Sequential(nn.Linear(channels, 4 * channels), nn.GELU(),
                                 nn.Linear(4 * channels, channels))

    def forward(self, features, order, indices, inverse, counts, resolution):
        n, c = features.shape
        k = min(n, self.window_size)
        x = (features + self.cpe(features, indices, inverse, counts, resolution))[order]
        padded = (k - n % k) % k
        normalized = self.norm1(x)
        # PTv3 fills the final window from the preceding window, not zero tokens.
        if padded:
            normalized = torch.cat((normalized, normalized[n-k:n-k+padded]))
        windows = normalized.reshape(-1, k, c)
        outputs = []
        for start in range(0, len(windows), 256):
            part = windows[start:start + 256]
            q, key, value = self.qkv(part).reshape(-1, k, 3, self.heads, c // self.heads).permute(2, 0, 3, 1, 4)
            attn = F.scaled_dot_product_attention(q, key, value)
            outputs.append(attn.transpose(1, 2).reshape(-1, k, c))
        x = x + self.proj(torch.cat(outputs).reshape(-1, c)[:n])
        x = x + self.ffn(self.norm2(x))
        return x[order.argsort()]


class GaussianRefinementHead(nn.Module):
    architecture = "ifsplat"

    def __init__(self, gaussian_token_dim: int, init_feature_dim: int,
                 delta_dim: int, cfg: RefinementHeadCfg) -> None:
        super().__init__()
        self.cfg = cfg
        c = cfg.state_channels
        self.init_state = nn.Sequential(nn.LayerNorm(init_feature_dim), nn.Linear(init_feature_dim, c))
        self.input_proj = nn.Linear(gaussian_token_dim + c + 256, c)
        self.blocks = nn.ModuleList([WindowBlock(c, cfg.num_heads, cfg.window_size)
                                    for _ in range(cfg.num_blocks)])
        self.update_head = nn.Sequential(nn.LayerNorm(c), nn.Linear(c, c), nn.GELU(), nn.Linear(c, delta_dim))
        if cfg.zero_init_update_head:
            nn.init.zeros_(self.update_head[-1].weight)
            nn.init.zeros_(self.update_head[-1].bias)

    def initialize_state(self, init_features):
        return self.init_state(init_features)

    def fuse_inputs(self, gaussian_tokens, state, evidence, prior=None):
        if prior is not None:
            raise ValueError("Generative prior requires a GIFSplat refinement head.")
        return self.input_proj(torch.cat((gaussian_tokens, state, evidence), dim=-1))

    def forward(self, gaussian_tokens, state, evidence, valid_mask, prior=None):
        fused = self.fuse_inputs(gaussian_tokens, state, evidence, prior)
        next_states = []
        orders = ("z", "z-trans", "hilbert", "hilbert-trans")
        # Scenes are never padded into a shared attention window.
        for batch_idx, valid in enumerate(valid_mask):
            x = fused[batch_idx, valid]
            if x.shape[0] == 0:
                raise ValueError("Refinement requires at least one valid Gaussian per scene.")
            coords = gaussian_tokens[batch_idx, valid, :3]
            grid = quantize_coordinates(coords, self.cfg.grid_resolution)
            cells, inverse, counts = grid.unique(dim=0, return_inverse=True, return_counts=True)
            indices = F.pad(cells, (1, 0)).int()
            depth = (self.cfg.grid_resolution - 1).bit_length()
            serialized = [serialization_code(grid, order, depth).argsort(stable=True) for order in orders]
            for i, block in enumerate(self.blocks):
                args = (serialized[i % len(orders)], indices, inverse, counts, self.cfg.grid_resolution)
                if self.training and torch.is_grad_enabled():
                    x = checkpoint(block, x, *args, use_reentrant=False)
                else:
                    x = block(x, *args)
            updated = state[batch_idx, valid] + x
            full = updated.new_zeros(state[batch_idx].shape)
            full[valid] = updated
            next_states.append(full)
        next_state = torch.stack(next_states)
        delta = self.update_head(next_state).masked_fill(~valid_mask[..., None], 0)
        return next_state, delta


class GIFSplatRefinementHead(GaussianRefinementHead):
    """Observation + prior fusion, with unchanged spatial/recurrent blocks."""

    architecture = "gifsplat"

    def __init__(self, gaussian_token_dim, init_feature_dim, delta_dim, cfg):
        super().__init__(gaussian_token_dim, init_feature_dim, delta_dim, cfg)
        # Equivalent to concatenating 256 prior channels into input_proj.
        # Zero prior columns preserve the pretrained IFSplat function initially.
        self.prior_proj = nn.Linear(256, cfg.state_channels, bias=False)
        nn.init.zeros_(self.prior_proj.weight)


    def fuse_inputs(self, gaussian_tokens, state, evidence, prior=None):
        if prior is None or prior.shape != evidence.shape or prior.shape[-1] != 256:
            raise ValueError("GIFSplat requires matching [B,N,256] observation and prior cues.")
        return super().fuse_inputs(gaussian_tokens, state, evidence) + self.prior_proj(prior.detach())
