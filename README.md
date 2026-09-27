# HQ-DM: Single-Hadamard Low-Bit Diffusion Models

[![arXiv](https://img.shields.io/badge/arXiv-2512.05746-b31b1b.svg)](https://arxiv.org/abs/2512.05746)
[![ECCV 2026 Poster](https://img.shields.io/badge/ECCV%202026-Poster-0054a6.svg)](https://arxiv.org/abs/2512.05746)

This repository contains the HQ-DM quantization-aware training code and an
integer inference implementation for quantized `Linear` and `Conv2d` layers.

![HQ-DM framework](assets/hq-dm-framework.png)

## Getting Started

Run all commands from the repository root.

### Step 1: Setup

```bash
git clone --recurse-submodules https://github.com/msz12345/HQ-DM.git
cd HQ-DM

python -m venv .venv
source .venv/bin/activate
```

If the repository was cloned without `--recurse-submodules`, run:

```bash
git submodule update --init --recursive
```

Install a CUDA-enabled PyTorch and torchvision build for the target machine,
then install the remaining dependencies:

```bash
pip install -r requirements.txt
```

### Step 2: Download Pretrained Models

```bash
mkdir -p models/ldm/cin256-v2
wget -O models/ldm/cin256-v2/model.ckpt \
  https://ommer-lab.com/files/latent-diffusion/nitro/cin/model.ckpt
```

Place the HQ-DM checkpoint at the repository root. For example, the W4A4,
20-step configuration expects:

```text
quantw4a4_20steps_SingleH32.pth
```

### Step 3: Collect Calibration Data

Skip this step if calibration data or a trained HQ-DM checkpoint is already
available.

```bash
python -m tools.collect_calibration
```

### Step 4: Train HQ-DM

Keep the weight/activation bit widths, DDIM steps, calibration data, input
checkpoint, and output checkpoint name consistent in the training and sampling
scripts.

```bash
mkdir -p experiments_log
python -m tools.train
```

### Step 5: Sample with HQ-DM

```bash
python -m tools.sample
```

### Step 6 (Optional): Efficient Inference with CUTLASS

SM80, SM90, and SM120 use [NVIDIA CUTLASS](https://github.com/NVIDIA/cutlass).
SM70 uses the integer fallback and does not require CUTLASS.

```bash
pip install ninja

# CUTLASS >= 3.5.1 for SM80/SM90; CUTLASS >= 4.2.0 for SM120.
export HQDM_CUTLASS_PATH=/absolute/path/to/cutlass

python -m tools.validate_checkpoint

python -m tools.benchmark \
  --mode int_amp \
  --steps 20 \
  --batch-size 1 \
  --class-label 333 \
  --warmup 0 \
  --repeats 1 \
  --channels-last \
  --cuda-graphs \
  --output-dir outputs/efficient
```

The CUDA extension is compiled and loaded on the first run. The integer module
replacement covers both `Linear` and `Conv2d` layers.

## Citation

```bibtex
@inproceedings{mao2026hqdm,
  title     = {HQ-DM: Single Hadamard Transformation-Based Quantization-Aware Training for Low-Bit Diffusion Models},
  author    = {Mao, Shizhuo and Zou, Hongtao and Xie, Qihu and Chen, Song and Kang, Yi},
  booktitle = {European Conference on Computer Vision (ECCV)},
  year      = {2026}
}
```

## Acknowledgements

Our codebase builds heavily on
[latent-diffusion](https://github.com/CompVis/latent-diffusion) and
[EfficientDM](https://github.com/ThisisBillhe/EfficientDM). Thanks for
open-sourcing these projects!

We also thank
[Taming Transformers](https://github.com/CompVis/taming-transformers),
[OpenAI CLIP](https://github.com/openai/CLIP), and
[NVIDIA CUTLASS](https://github.com/NVIDIA/cutlass).
