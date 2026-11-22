---
license: mit
library_name: pytorch
pipeline_tag: image-segmentation
tags:
  - hyperspectral
  - semantic-segmentation
  - opsin
---

# PaleVision

Official implementation of **Paleoinspired Vision: From Exploring Colour Vision Evolution to Inspiring Camera Design**, accepted at **NeurIPS 2026**.

[arXiv](https://arxiv.org/abs/2412.19439) · [Project website](https://luyijie-research.github.io/PaleVision/) · [Code](https://github.com/luyijie-research/PaleVision) · [Weights](https://huggingface.co/smilelab/PaleVision)

PaleVision models opsin spectral responses with learnable evolutionary visual perception layers (EVPL) and trains semantic segmentation on hyperspectral imagery.

## Structure

```
configs/       experiment configs
dataloader/    dataset loaders
models/        EVPL, encoders, SegFormer head
util/          training helpers
weights/       SOTA weights (download; not stored in git)
datasets/      put datasets here (not shipped)
work_dir/      logs and checkpoints (not shipped)
```

## Install

```bash
conda env create -f environment.yml
conda activate palevision
```

`environment.yml` installs Python 3.10 and the packages in `requirements.txt`. The released training run used PyTorch 2.6.0 with CUDA 12.6. Install a CUDA build of PyTorch that matches your driver before `mmcv` if the default wheel does not.

Hardware used for the released runs: an NVIDIA GPU with enough memory for MiT-B0 through Twins-SVT-Small at the resolutions in `configs/`. CPU-only loading of the checkpoints does not need that GPU.

## Data layout (Mars example)

```
datasets/Mars/MSL/train/{img,hsi,label}
datasets/Mars/MSL/test/{img,hsi,label}
```

`hsi` files are HDF5 with a `cube` dataset of shape `[31, H, W]`.

## Train (Mars)

```bash
torchrun --nproc_per_node=1 mars_exp.py \
  --config configs/mars.yaml \
  --workdir work_dir/mars \
  --train_path datasets/Mars/MSL/train \
  --val_path datasets/Mars/MSL/test \
  --encoder_type mit_b0
```

Multi-GPU:

```bash
torchrun --nproc_per_node=2 mars_exp.py \
  --config configs/mars.yaml \
  --workdir work_dir/mars \
  --train_path datasets/Mars/MSL/train \
  --val_path datasets/Mars/MSL/test
```

Smoke check (few steps, small resolution):

```bash
torchrun --nproc_per_node=1 mars_exp.py \
  --config configs/mars.yaml \
  --workdir work_dir/mars_smoke \
  --train_path datasets/Mars/MSL/train \
  --val_path datasets/Mars/MSL/test \
  --batch_size 1 \
  --input_size 128 \
  --max_steps 3 \
  --skip_val \
  --num_workers 2
```

Other entry points: `modid_exp.py`, `mhsi_exp.py`, `lib_hsi_exp_v2.py`, `minneapple_exp.py`, `encoder_swap_exp.py`, `minneapple_det_count_exp.py`.

Encoder swap on MODID or MHSI:

```bash
torchrun --nproc_per_node=1 encoder_swap_exp.py \
  --dataset modid \
  --encoder twins_svt_small \
  --opsin_mode free \
  --config configs/modid.yaml \
  --workdir work_dir/modid_twins_free
```

## SOTA weights

Released weights keep only module parameters. Optimizer state, epoch counters, and training hyperparameters are not included.

| File | Setting | Reported score |
| --- | --- | --- |
| `weights/modid_twins_svt_small_free.pth` | MODID, Twins-SVT-Small, FreeConv | mIoU 0.6006 |
| `weights/mhsi_pvt_v2_b2_free.pth` | MHSI, PVT-v2-B2, FreeConv | mIoU 0.5727 |
| `weights/lib_hsi.pth` | LIB-HSI, MiT-B3, 3-opsin EVPL with a 2-to-3 projection | mIoU 0.3407 |
| `weights/minneapple.pth` | MinnApple segmentation, opsin evolve | mIoU 0.4845 |
| `weights/minneapple_detection.pth` | MinnApple detection, fixed opsin | AP@50 0.3105 |
| `weights/minneapple_counting.pth` | MinnApple counting, trainable opsin | MAE 2.446 |
| `weights/mars_spectral.pth` | Mars spectral EVPL, epoch 1000 | spectral model |

Segmentation checkpoints are dictionaries with `backbone`, `head`, and `evpl`. Detection and counting checkpoints are a single `state_dict`. `mars_spectral.pth` is the Mars `EVPL` `state_dict`.

```python
import torch
ckpt = torch.load("weights/modid_twins_svt_small_free.pth", map_location="cpu")
backbone.load_state_dict(ckpt["backbone"])
head.load_state_dict(ckpt["head"])
evpl.load_state_dict(ckpt["evpl"])
```

The weight files sit next to this code under `weights/` in the release folder. Git ignores `*.pth` because the largest file is above GitHub's file size limit. Download the same files from [smilelab/PaleVision](https://huggingface.co/smilelab/PaleVision). Code for this release is at [luyijie-research/PaleVision](https://github.com/luyijie-research/PaleVision).

## Model card

PaleVision is a research model that maps a hyperspectral image through an opsin layer into a segmentation or counting network. The released files are the best finished checkpoints for each reported task. They contain module parameters only.

**Intended use.** Reproduce the paper's segmentation, detection, counting, and Mars spectral-filter experiments, and inspect the learned opsin peak wavelengths.

**Limitations.** Checkpoints are tied to the datasets and class counts used in training. They are not a general medical or navigation system. The Mars file is the finished epoch-1000 spectral EVPL; that run did not report a segmentation mIoU. LIB-HSI was assembled from the separately saved best backbone, head, and EVPL of the best finished run.

**Training data.** MODID, MHSI, LIB-HSI, MinnApple, and the Mars MSL hyperspectral set. Place them under `datasets/` using the layout in this README and the loaders in `dataloader/`. The datasets themselves are not redistributed.

**Evaluation.** Scores for the released weights are in the table above. Training hyperparameters live in `configs/`. Entry points default to random seed 42 (`--seed 42`).

**Inference.** Load a segmentation checkpoint with the snippet above, or run the matching entry point (`modid_exp.py`, `mhsi_exp.py`, `lib_hsi_exp_v2.py`, `minneapple_exp.py`, `encoder_swap_exp.py`, `minneapple_det_count_exp.py`, `mars_exp.py`). Dataset paths are arguments or `datasets/<name>`, not machine-specific directories.

## Notes

- Checkpoints written under `work_dir/` during training are ignored by git.

## Citation

If you use PaleVision in your research, please cite our NeurIPS 2026 paper:

```bibtex
@inproceedings{lu2026paleoinspired,
  title     = {Paleoinspired Vision: From Exploring Colour Vision Evolution to Inspiring Camera Design},
  author    = {Lu, Yijie and Zong, Zhimin and Zhang, Junjie and Su, Shenghan and Zhao, Zeyun and Cui, Ziteng and Pu, Yan and Lu, Jing and Kojima, Daisuke and Gu, Lin and Fang, Ruogu},
  booktitle = {Advances in Neural Information Processing Systems},
  year      = {2026},
  eprint    = {2412.19439},
  archivePrefix = {arXiv},
  url       = {https://arxiv.org/abs/2412.19439}
}
```
