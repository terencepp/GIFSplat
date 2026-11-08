# Copyright (c) 2026 The GIFSplat Authors.
# SPDX-License-Identifier: CC-BY-NC-4.0
"""Forward-only feature lifting using gsplat's depth-sorted tile intersections."""

import torch
from torch.nn import functional as F


@torch.no_grad()
def pool_observation(meta, feature_pyramid, num_gaussians):
    """Return weighted sums and masses for one view (normalize across views).

    Enumerate only splat/tile intersections. Temporary weights are bounded by
    16 tiles x 32 splats x tile_size**2 pixels, never N x H x W.
    Match classic gsplat's pixel centers, alpha culling and exclusive early stop.
    """
    means = meta["means2d"].reshape(-1, 2).float()
    conics = meta["conics"].reshape(-1, 3).float()
    opacity = meta["opacities"].reshape(-1).float()
    ids = meta["flatten_ids"].long()
    offsets = meta["isect_offsets"].reshape(-1).long()
    ends = torch.cat((offsets[1:], offsets.new_tensor([len(ids)])))
    channels = sum(f.shape[1] for f in feature_pyramid)
    numerator = means.new_zeros(num_gaussians, channels)
    denominator = means.new_zeros(num_gaussians)
    if len(ids) == 0:
        return numerator, denominator
    tile = meta["tile_size"]
    height, width = meta["height"], meta["width"]
    tile_width = (width + tile - 1) // tile
    py, px = torch.meshgrid(torch.arange(tile, device=means.device),
                            torch.arange(tile, device=means.device), indexing="ij")
    local = torch.stack((px.flatten(), py.flatten()), dim=-1)
    for start in range(0, len(offsets), 16):
        tiles = torch.arange(start, min(start + 16, len(offsets)), device=means.device)
        origins = torch.stack((tiles % tile_width, tiles // tile_width), dim=-1) * tile
        pixels = origins[:, None] + local[None] + .5
        inside = (pixels[..., 0] < width) & (pixels[..., 1] < height)
        grid = pixels / pixels.new_tensor([width, height]) * 2 - 1
        features = torch.cat([
            F.grid_sample(f.float(), grid[None], mode="bilinear", padding_mode="border",
                          align_corners=False)[0].permute(1, 2, 0)
            for f in feature_pyramid
        ], dim=-1)
        trans = means.new_ones(len(tiles), tile * tile)
        counts = ends[tiles] - offsets[tiles]
        for rank in range(0, int(counts.max()), 32):
            ranks = torch.arange(rank, rank + 32, device=means.device)
            valid = ranks[None] < counts[:, None]
            slots = offsets[tiles, None] + ranks[None]
            gs = ids[slots.clamp_max(len(ids) - 1)]
            d = pixels[:, None] - means[gs, None]
            a, b, c = conics[gs].unbind(-1)
            sigma = .5 * (a[..., None] * d[..., 0].square() + c[..., None] * d[..., 1].square()) + b[..., None] * d[..., 0] * d[..., 1]
            alpha = (opacity[gs, None] * (-sigma).exp()).clamp_max(.999)
            keep = valid[..., None] & inside[:, None] & (sigma >= 0) & (alpha >= 1 / 255)
            alpha = torch.where(keep, alpha, 0)
            after = (1 - alpha).cumprod(dim=1) * trans[:, None]
            before = torch.cat((trans[:, None], after[:, :-1]), dim=1)
            weights = torch.where(after > 1e-4, before * alpha, 0)
            trans = after[:, -1]
            numerator.index_add_(0, gs.flatten(), torch.bmm(weights, features).flatten(0, 1))
            denominator.index_add_(0, gs.flatten(), weights.sum(-1).flatten())
    return numerator, denominator


