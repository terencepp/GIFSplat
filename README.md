# GIFSplat
GIFSplat: Generative Prior-Guided Iterative Feed-Forward 3D Gaussian Splatting from Sparse Views.

Tianyu Chen, Wei Xiang, Kang Han, Yu Lu, Di Wu, Gaowen Liu, Ramana Rao Kompella

## Overview

GIFSplat starts from a feed-forward 3D Gaussian Splatting initializer
(AnySplat) and refines the Gaussians with a small number of shared-weight
residual updates. Each update reads two kinds of evidence:

- **Observation cues.** Features of the input views minus features of the
  current renderings, pooled onto each Gaussian with its rasterization weights.
- **Generative prior cues.** A frozen DiFiX enhancer is applied to renderings
  of the current Gaussians at novel views. Features of the enhanced images
  minus those of the renderings are pooled onto the Gaussians with the same
  rasterization weights.

```text
Input RGB views
  -> Initializer: Gaussians, predicted cameras, initial features
  -> Repeat T = 3 times (shared weights):
       observation cues  (observed  - rendered features)
       prior cues        (enhanced  - rendered features at novel views)
       Gaussian tokens + recurrent state + cues
         -> spatial attention -> state update -> Gaussian parameter increments
  -> Refined Gaussians
```

Both cues are recomputed from the current Gaussians at every step. Enhanced
views are used only as cues, never as supervision targets. Without the prior
branch, the model reduces to the observation-only variant (IFSplat).

## Repository contents

This release contains the model architecture: the initializer, the recurrent
refinement head, observation pooling and the generative prior branch.

| Path under `src/` | Contents |
| --- | --- |
| [model/encoder/anysplat.py](src/model/encoder/anysplat.py) | Initializer, Gaussian tokens, parameter updates and the refinement loop (`EncoderAnySplat.forward`, `run_refinement`) |
| [model/encoder/vggt/](src/model/encoder/vggt/) | VGGT network: aggregator (alternating frame/global attention), camera head, DPT depth/point heads, track head and DINOv2 layers |
| [model/encoder/heads/gaussian_refinement_head.py](src/model/encoder/heads/gaussian_refinement_head.py) | IFSplat / GIFSplat refinement heads: serialization, sparse conditional positional encoding, window attention |
| [model/encoder/heads/generative_prior.py](src/model/encoder/heads/generative_prior.py) | Prior-view cameras, reference-view selection, feature residuals and prior pooling |
| [model/encoder/heads/observation_features.py](src/model/encoder/heads/observation_features.py) | Frozen ResNet18 feature pyramid |
| [model/encoder/heads/vggt_dpt_gs_head.py](src/model/encoder/heads/vggt_dpt_gs_head.py) | Gaussian DPT head, including the features that initialize the recurrent state |
| [model/encoder/common/gaussian_adapter.py](src/model/encoder/common/gaussian_adapter.py) | Decoding of raw Gaussian parameters |
| [model/encoder/common/gaussians.py](src/model/encoder/common/gaussians.py) | Quaternion conversion and covariance construction |
| [model/decoder/observation_pooling.py](src/model/decoder/observation_pooling.py) | Pooling of image-space features onto Gaussians with rasterization weights |
| [model/decoder/decoder_splatting_cuda.py](src/model/decoder/decoder_splatting_cuda.py) | gsplat rendering and the pooling callback |
| [model/difix_enhancer.py](src/model/difix_enhancer.py) | Frozen DiFiX adapter |
| [model/gifsplat_config.py](src/model/gifsplat_config.py) | Generative prior options: prior-view policy, number of prior views, seed, DiFiX paths |
| [model/types.py](src/model/types.py) | Gaussian tensor fields |
| [model/encoder/encoder.py](src/model/encoder/encoder.py), [model/decoder/decoder.py](src/model/decoder/decoder.py) | Encoder and decoder base classes and output types |
| [geometry/projection.py](src/geometry/projection.py) | Camera transforms, projection/unprojection and ray utilities |

The DiFiX pipeline is used from its original repository and is not included here.

## Dependencies

- PyTorch, torchvision, einops, jaxtyping
- huggingface_hub (VGGT model wrapper)
- [gsplat](https://github.com/nerfstudio-project/gsplat) 1.4.0
- [torch_scatter](https://github.com/rusty1s/pytorch_scatter) 2.1.2
- [spconv](https://github.com/traveller59/spconv) 2.3.6
- [DiFiX](https://github.com/nv-tlabs/Difix3D) pipeline and weights (prior branch only)

## Model conventions

- The initializer follows AnySplat with voxelization removed. It predicts the
  cameras and the initial Gaussians (position, scale, orientation, color and
  opacity).
- The refinement head applies T = 3 shared-weight residual updates to a fixed
  set of Gaussians and uses six window-attention blocks.
- Training has two stages: Stage I fine-tunes the initializer, and Stage II
  trains the refinement head by unrolling the T refinement steps. Losses,
  schedules and other training details are given in the paper and its
  supplementary material.

## Citation

```bibtex
@inproceedings{chen2026gifsplat,
  title     = {GIFSplat: Generative Prior-Guided Iterative Feed-Forward 3D Gaussian Splatting from Sparse Views},
  author    = {Chen, Tianyu and Xiang, Wei and Han, Kang and Lu, Yu and Wu, Di and Liu, Gaowen and Kompella, Ramana Rao},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year      = {2026}
}
```

## License and acknowledgements

Code written for GIFSplat is released under
[CC BY-NC 4.0](LICENSE) (non-commercial use). Files derived from
[AnySplat](https://github.com/OpenRobotLab/AnySplat) keep its MIT license
([LICENSE-AnySplat](LICENSE-AnySplat)). The Gaussian DPT head
(`src/model/encoder/heads/vggt_dpt_gs_head.py`) is modified from the original
Naver implementation, and the original and our modifications are licensed under
CC BY-NC-SA 4.0. The VGGT code in `src/model/encoder/vggt/` is distributed
under the [VGGT License](src/model/encoder/vggt/LICENSE), and its
DINOv2-derived layers under
[Apache License 2.0](src/model/encoder/vggt/layers/LICENSE).

We thank the authors of [AnySplat](https://github.com/OpenRobotLab/AnySplat),
[VGGT](https://github.com/facebookresearch/vggt),
[DiFiX](https://github.com/nv-tlabs/Difix3D),
[gsplat](https://github.com/nerfstudio-project/gsplat),
[Point Transformer V3](https://github.com/Pointcept/PointTransformerV3) and
[SplatFormer](https://github.com/ChenYutongTHU/SplatFormer) for their code.
