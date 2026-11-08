# Copyright (c) 2026 The GIFSplat Authors.
# SPDX-License-Identifier: CC-BY-NC-4.0
"""DiFiX tensor adapter without AnySplat encoder/renderer import dependencies."""

import torch
from torch import nn


def load_difix_enhancer(cfg, device):
    import importlib.util
    from pathlib import Path

    cfg.validate_paths()
    pipeline, weights = Path(cfg.pipeline_path).expanduser(), Path(cfg.weights_path).expanduser()
    spec = importlib.util.spec_from_file_location("gifsplat_local_difix", pipeline)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    pipe = module.DifixPipeline.from_pretrained(
        str(weights.resolve()), local_files_only=True, torch_dtype=torch.float32,
    ).to(device)
    return FrozenDifixEnhancer(pipe)


class FrozenDifixEnhancer(nn.Module):
    """Wrap an already loaded DiFiX reference pipeline; one pair per call.

    Loading weights/device placement stays with the caller. Importing this
    module never downloads weights or imports diffusers into the trainer.
    """

    def __init__(self, pipeline, timestep=200):
        super().__init__()
        self.pipeline = pipeline
        self.timestep = timestep
        self.frozen_modules = nn.ModuleList(
            module for module in pipeline.components.values() if isinstance(module, nn.Module)
        )
        self.requires_grad_(False)
        self.eval()

    def train(self, mode=True):
        return super().train(False)

    @torch.no_grad()
    def forward(self, rendered, reference, generator=None):
        if (rendered.ndim != 4 or rendered.shape[:2] != (1, 3)
                or rendered.shape != reference.shape):
            raise ValueError("DiFiX requires one aligned [1,3,H,W] render/reference pair per call.")
        height, width = rendered.shape[-2:]
        if height % 8 or width % 8 or min(height, width) == 0:
            raise ValueError("DiFiX image dimensions must be positive multiples of 8; no implicit resize.")
        if not torch.isfinite(rendered).all() or not torch.isfinite(reference).all():
            raise ValueError("DiFiX inputs must be finite.")
        enhanced = self.pipeline(
            prompt="remove degradation",
            image=rendered.detach().float().clamp(0, 1),
            ref_image=reference.detach().float().clamp(0, 1),
            height=height, width=width,
            num_inference_steps=1, timesteps=[self.timestep], guidance_scale=0.0,
            output_type="pt", generator=generator,
        ).images
        if (not isinstance(enhanced, torch.Tensor) or enhanced.shape != rendered.shape
                or not torch.isfinite(enhanced).all()
                or (enhanced < 0).any() or (enhanced > 1).any()):
            raise ValueError("DiFiX must return finite [1,3,H,W] RGB in [0,1] without resizing.")
        return enhanced.detach().to(device=rendered.device, dtype=torch.float32)
