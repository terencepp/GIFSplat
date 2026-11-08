# Copyright (c) 2026 The GIFSplat Authors.
# SPDX-License-Identifier: CC-BY-NC-4.0
from dataclasses import dataclass


@dataclass
class GIFSplatCfg:
    enabled: bool = False
    policy: str = "nearest_pair_midpoint"
    num_views: int = 2
    seed: int = 0
    pipeline_path: str | None = None
    weights_path: str | None = None

    def __post_init__(self):
        if self.policy != "nearest_pair_midpoint":
            raise ValueError("Unsupported GIFSplat camera policy.")
        if type(self.num_views) is not int or self.num_views < 1:
            raise ValueError("GIFSplat num_views must be a positive integer.")

    def protocol(self):
        return self.policy, self.num_views, self.seed

    def validate_paths(self):
        from pathlib import Path

        if not self.pipeline_path or not self.weights_path:
            raise ValueError("GIFSplat requires DiFiX: set pipeline_path and weights_path.")
        if (not Path(self.pipeline_path).expanduser().is_file()
                or not (Path(self.weights_path).expanduser() / "model_index.json").is_file()):
            raise FileNotFoundError("DiFiX needs a local pipeline_difix.py and complete model snapshot directory.")
