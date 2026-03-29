# Generative Modeling via Drifting — JAX & PyTorch Release

<p align="center">
  <a href="http://arxiv.org/abs/2602.04770"><img src="https://img.shields.io/badge/arXiv-2602.04770-b31b1b.svg" alt="arXiv" /></a>
  <a href="https://colab.research.google.com/github/lambertae/drifting/blob/main/notebooks/inference_demo.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab" /></a>
  <a href="https://huggingface.co/Goodeat/drifting"><img src="https://img.shields.io/badge/HuggingFace-Models-yellow.svg" alt="HuggingFace" /></a>
</p>

<p align="center">
  <img src="assets/teaser_main.png" width="90%" alt="Drifting Models overview" />
</p>

Official codebase for the ImageNet experiments of [*Generative Modeling via Drifting*](http://arxiv.org/abs/2602.04770) (Deng*, Li*, Li, Du & He, 2026).
We provide training, inference, and pretrained weights for one-step image generation on ImageNet 256×256.

This repository contains **two independent implementations** of the Drift framework:

| Directory | Backend | Status |
|-----------|---------|--------|
| [`jax/`](jax/) | JAX + Flax (original) | Pretrained weights available |
| [`torch/`](torch/) | PyTorch | Re-implementation |

Root-level directories (`assets/`, `configs/`, `notebooks/`) are shared between both implementations.

---

## Method Overview

### What is Drifting?

**Drifting** is a new paradigm for generative modeling that produces high-quality images in a **single forward pass** (1 NFE — Number of Function Evaluations). Unlike diffusion models that require tens to thousands of iterative denoising steps, a Drifting generator maps random noise directly to photorealistic images in one shot, achieving state-of-the-art FID 1.54 on ImageNet 256×256.

The core idea is conceptually simple: instead of learning to reverse a noising process, the generator is trained so that its output distribution **drifts** toward the real data distribution. During training, each generated sample is simultaneously **attracted** toward real images of the same class (positive anchors) and **repelled** from unrelated images (negative anchors). These attraction–repulsion forces, computed across multiple scales, gradually sculpt the generator's output distribution until it matches the data.

### Key Technical Components

#### 1. Drift Loss — Attraction & Repulsion in Feature Space

The drift loss is the heart of the method. It operates entirely in **feature space** (not pixel space) to capture high-level semantics.

**How it works:**

1. A batch of images is generated from random noise via the generator.
2. Both generated and real images are passed through a frozen MAE (Masked Autoencoder) feature extractor to obtain multi-scale feature representations.
3. For each generated sample, **positive anchors** (real images from the same class) and **negative anchors** (unconditional/cross-class images) are drawn from the memory bank.
4. Pairwise distances are computed between generated features and anchor features, yielding **soft affinities** via a softmax over distances:

$$\text{affinity}(i, j) = \sqrt{\text{softmax}_{\text{row}}\!\left(-\frac{d_{ij}}{R}\right) \cdot \text{softmax}_{\text{col}}\!\left(-\frac{d_{ij}}{R}\right)}$$

5. These affinities define directional forces: positive anchors **pull** the generated sample closer, while negative anchors **push** it away. The forces are combined into a **goal position** for each generated sample.
6. The loss is simply the MSE between the current generated features and the goal (with stop-gradient on the goal, so only the generator is updated).

**Multi-scale design:** The loss is computed at multiple temperature scales $R \in \{0.02, 0.05, 0.2\}$, enabling the generator to learn both fine-grained details (small $R$) and global structure (large $R$) simultaneously.

#### 2. Memory Bank — Efficient Anchor Storage

A **class-wise circular buffer** stores real image features for each of the 1000 ImageNet classes. During each training step:

- **Push:** A batch of real images is added to the memory bank (both class-specific positive bank and a shared negative bank).
- **Sample:** For each training label, positive anchors are sampled from the same class, and negative anchors are sampled unconditionally.

This design decouples the batch size from the number of anchor comparisons, allowing rich positive/negative supervision (e.g., 64 positive + 16 negative anchors per sample) without enormous batch sizes.

#### 3. Generator Architecture — Adapted DiT (Diffusion Transformer)

The generator is a **Diffusion Transformer (DiT)** repurposed for one-step generation:

| Component | Details |
|-----------|---------|
| **Backbone** | Transformer with adaptive Layer Norm (adaLN-Zero) |
| **Variants** | DiT-B (768 hidden, 12 layers) / DiT-L (1024 hidden, 24 layers) |
| **Input** | Random noise (Gaussian), either in pixel space or VAE latent space |
| **Conditioning** | Class label embedding via learnable tokens (16 class tokens concatenated to the sequence) |
| **Attention** | Multi-head self-attention with QK-norm, RoPE, and optional FP32 precision |
| **MLP** | SwiGLU activation with 4× expansion ratio |
| **Normalization** | RMSNorm (instead of standard LayerNorm) |

At inference time, the generator takes in pure Gaussian noise and a class label, and outputs an image in a single pass. Classifier-Free Guidance (CFG) is supported by running two forward passes (conditional + unconditional) and interpolating:

$$x_{\text{out}} = x_{\text{unc}} + s \cdot (x_{\text{cond}} - x_{\text{unc}})$$

#### 4. MAE Feature Extractor — Semantic Supervision Signal

A **ResNet-based Masked Autoencoder (MAE)** is pre-trained via self-supervised learning on ImageNet to serve as the feature extractor for the drift loss:

- **Architecture:** ResNet-50 backbone with configurable channel width (256 for ablation, 640 for SOTA).
- **Training:** Masked image modeling — reconstructs 50% randomly masked patches from the remaining visible patches.
- **Usage during generator training:** The MAE is **frozen**; only its multi-scale feature maps (layers 1–4 + global tokens) are extracted. This provides 5 complementary views of each image, from low-level textures (layer 1) to high-level semantics (layer 4 / global tokens).

The MAE can operate in **latent space** (on 4-channel VAE-encoded representations) or **pixel space** (on raw RGB images).

#### 5. Training Pipeline

Training proceeds in two stages:

**Stage 1 — MAE Pretraining (optional):** Train the ResNet-based MAE on ImageNet via masked image reconstruction. Pre-trained weights are provided on HuggingFace, so this step is optional.

**Stage 2 — Generator Training:**

```
For each training step:
  1. Push a batch of real images into the memory bank (positive + negative)
  2. Sample positive & negative anchors from the memory bank
  3. Sample random noise and a CFG scale from [cfg_min, cfg_max]
  4. Generate images via the DiT generator (one forward pass)
  5. Extract frozen MAE features from generated + anchor images
  6. Compute multi-scale drift loss across all feature levels
  7. Backpropagate and update generator (AdamW + gradient clipping)
  8. Update EMA (Exponential Moving Average) parameters
  9. Periodically evaluate FID on the EMA model at multiple CFG scales
```

**Key training hyperparameters (SOTA configuration):**

| Parameter | Value | Description |
|-----------|-------|-------------|
| `total_steps` | 200,000 | Total training iterations |
| `batch_size` | 2,048 (global) | Distributed across 128 TPU v6e |
| `pos_per_sample` | 64 | Positive anchors per generated sample |
| `neg_per_sample` | 32 | Negative anchors per generated sample |
| `gen_per_label` | 64 | Samples generated per class label per step |
| `R_list` | [0.02, 0.05, 0.2] | Multi-scale temperature values |
| `cfg_min / cfg_max` | 1.0 / 4.0 | CFG scale sampling range during training |
| `ema_decay` | 0.999 | EMA coefficient |
| `learning_rate` | 4e-4 | AdamW learning rate (constant after warmup) |

### How is Drifting Different from Diffusion?

| Aspect | Diffusion / Flow Matching | Drifting |
|--------|--------------------------|----------|
| **Generation steps** | 10–1000 iterative steps | **1 step** (single forward pass) |
| **Training signal** | Predict noise / velocity field | Attraction–repulsion forces from real samples |
| **Loss space** | Pixel or noise space | **Feature space** (via frozen MAE) |
| **Noise schedule** | Required (timestep-dependent) | **None** — no timestep concept |
| **Memory bank** | Not needed | Class-wise anchor storage |
| **Architecture** | U-Net or DiT | DiT (adapted) |
| **ImageNet-256 FID** | ~1.5–2.0 (multi-step) | **1.54** (single step) |

## Generated Samples

Uncurated conditional ImageNet 256×256 samples (1 NFE, CFG scale 1.0, FID 1.54):

<p align="center">
  <img src="assets/class_095_jacamar.jpg" width="24%" alt="Jacamar" />
  <img src="assets/class_022_bald_eagle.jpg" width="24%" alt="Bald Eagle" />
  <img src="assets/class_088_macaw.jpg" width="24%" alt="Macaw" />
  <img src="assets/class_108_sea_anemone.jpg" width="24%" alt="Sea Anemone" />
</p>
<p align="center">
  <img src="assets/class_386_African_elephant.jpg" width="24%" alt="African Elephant" />
  <img src="assets/class_296_ice_bear.jpg" width="24%" alt="Ice Bear" />
  <img src="assets/class_483_castle.jpg" width="24%" alt="Castle" />
  <img src="assets/class_698_palace.jpg" width="24%" alt="Palace" />
</p>
<p align="center">
  <img src="assets/class_970_alp.jpg" width="24%" alt="Alp" />
  <img src="assets/class_975_lakeside.jpg" width="24%" alt="Lakeside" />
  <img src="assets/class_973_coral_reef.jpg" width="24%" alt="Coral Reef" />
  <img src="assets/class_812_space_shuttle.jpg" width="24%" alt="Space Shuttle" />
</p>

## Training Dynamics

The generated distribution **q** evolves toward the data distribution **p** during training.
Try the interactive toy demo to see the algorithm in action:

[![Open Toy Demo In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/lambertae/lambertae.github.io/blob/main/projects/drifting/notebooks/drifting_model_demo.ipynb)

<table align="center">
  <tr>
    <th align="center">Middle Init</th>
    <th align="center">Far-Away Init</th>
    <th align="center">Collapsed Init</th>
  </tr>
  <tr>
    <td align="center"><img src="assets/toy_case1.gif" width="100%" alt="Middle init" /></td>
    <td align="center"><img src="assets/toy_case2.gif" width="100%" alt="Far-away init" /></td>
    <td align="center"><img src="assets/toy_case3.gif" width="100%" alt="Collapsed init" /></td>
  </tr>
</table>

---

## Table of Contents

- [Method Overview](#method-overview)
  - [What is Drifting?](#what-is-drifting)
  - [Key Technical Components](#key-technical-components)
  - [How is Drifting Different from Diffusion?](#how-is-drifting-different-from-diffusion)
- [Repository Structure](#repository-structure)
- [Quick Start (Inference)](#quick-start-inference)
- [Pretrained Models](#pretrained-models)
- [Environment Setup](#environment-setup)
- [FID Evaluation](#fid-evaluation)
- [Training](#training)
- [Checkpoints and Logs](#checkpoints-and-logs)
- [Citation](#citation)

## Repository Structure

```
.
├── assets/              # Images and GIFs used in README
├── configs/             # Shared YAML configs for gen/mae training
│   ├── gen/
│   └── mae/
├── notebooks/           # Colab inference demo
├── jax/                 # JAX + Flax implementation (original)
│   ├── dataset/
│   ├── models/
│   ├── utils/
│   ├── drift_loss.py
│   ├── memory_bank.py
│   ├── train.py
│   ├── train_mae.py
│   ├── inference.py
│   ├── main.py
│   └── requirements.txt
└── torch/               # PyTorch re-implementation
    ├── dataset/
    ├── models/
    ├── utils/
    ├── drift_loss.py
    ├── memory_bank.py
    ├── train.py
    ├── train_mae.py
    ├── inference.py
    ├── main.py
    └── requirements.txt
```

## Quick Start (Inference)

The self-contained Colab notebook lets you generate samples interactively — no local setup required:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/lambertae/drifting/blob/main/notebooks/inference_demo.ipynb)

Default notebook configuration:

- `init_from = hf://latent_L_sota`
- `class_ids = 95,22,88,108,386,296,483,698`

Class indices follow the ImageNet-1k label order.

## Pretrained Models

### Generators

| Model    | Space  | Feature Encoder  | Encoder HF ID           | Generator HF ID      | CFG | FID (repo / paper) | IS (repo / paper) |
| -------- | ------ | ---------------- | ----------------------- | -------------------- | --- | ------------------ | ----------------- |
| Drift-L  | latent | MAE-640 (latent) | `hf://mae_latent_640`   | `hf://latent_L_sota` | 1.0 | 1.53 / 1.54        | 260.1 / 258.9     |
| Drift-B  | latent | MAE-640 (latent) | `hf://mae_latent_640`   | `hf://latent_B_sota` | 1.1 | 1.74 / 1.75        | 263.4 / 263.2     |
| Drift-L  | pixel  | MAE-640 (pixel)  | `hf://mae_pixel_640`    | `hf://pixel_L_sota`  | 1.0 | 1.62 / 1.61        | 308.6 / 307.5     |
| Drift-B  | pixel  | MAE-640 (pixel)  | `hf://mae_pixel_640`    | `hf://pixel_B_sota`  | 1.0 | 1.73 / 1.76        | 300.1 / 299.7     |
| Ablation | latent | MAE-256 (latent) | `hf://mae_latent_256`   | `hf://ablation`      | 2.0 | 8.49 / 8.46        | 144.0 / —         |

### Feature Extractors

| Model              | Space  | HF ID                 |
| ------------------ | ------ | --------------------- |
| MAE-640 (latent)   | latent | `hf://mae_latent_640` |
| MAE-640 (pixel)    | pixel  | `hf://mae_pixel_640`  |
| MAE-256 (ablation) | latent | `hf://mae_latent_256` |

All artifacts are hosted on HuggingFace at [`Goodeat/drifting`](https://huggingface.co/Goodeat/drifting) and are downloaded automatically.

## Environment Setup

### JAX Implementation

```bash
conda create -n drifting-jax python=3.10 -y
conda activate drifting-jax
pip install -r jax/requirements.txt
export JAX_PLATFORMS=tpu,cpu
```

### PyTorch Implementation

```bash
conda create -n drifting-torch python=3.10 -y
conda activate drifting-torch
pip install -r torch/requirements.txt
```

### Install Dependencies (Legacy — JAX)

```bash
conda create -n drifting-release python=3.10 -y
conda activate drifting-release
pip install -r jax/requirements.txt
export JAX_PLATFORMS=tpu,cpu
```

For local TPU runs, keep `JAX_PLATFORMS=tpu,cpu` in the shell before running
latent-cache building, training, or evaluation. This keeps TPU as the default
backend while still exposing a CPU backend for Flax VAE / checkpoint restore
paths that expect it.

### Download ImageNet

Download the [ImageNet](https://image-net.org/download) dataset and extract it to your desired location. The dataset should have the following structure:

```
imagenet/
├── train/
│   ├── n01440764/
│   ├── n01443537/
│   └── ...
└── val/
    ├── n01440764/
    ├── n01443537/
    └── ...
```

### Path Configuration

Before running training or evaluation, open `jax/utils/env.py` (or `torch/utils/env.py` for PyTorch) and set these constants for your machine:

- `IMAGENET_PATH`: root of the ImageNet directory (expects `train/` and `val/` subdirectories).
- `IMAGENET_CACHE_PATH`: root of the latent cache directory (only needed for latent-generator training).
- `IMAGENET_FID_NPZ`: path to the ImageNet-256 FID reference stats `.npz`.
- `IMAGENET_PR_NPZ`: path to the ImageNet precision/recall reference stats `.npz`.
- `HF_ROOT`: local cache directory for downloaded HuggingFace artifacts.
- `HF_REPO_ID`: HuggingFace repo ID for the release checkpoints (keep as `Goodeat/drifting`).

FID/PR reference stats can be downloaded from [Google Drive](https://drive.google.com/drive/folders/1Tr_6PXF2WMYkSlCbbkP_0FRhEjAXx5gb) (migrated from MeanFlow).

### Build Latent Cache

Only needed for latent-space generators. Run from the `jax/` directory:

```bash
cd jax
python -m dataset.latent \
  --data-path /path/to/imagenet \
  --target-path /path/to/latent_cache \
  --local-batch-size 128 \
  --num-workers 8 \
  --pin-memory
```

This encodes ImageNet images through the VAE and writes `.pt` files to `/path/to/latent_cache/{train,val}/`. After building the cache, update `IMAGENET_CACHE_PATH` in `utils/env.py`.

## FID Evaluation

### JAX

Reproduce paper FID numbers on ImageNet-256 (50k samples, CFG=1.0):

```bash
cd jax
# Latent model
python inference.py --init-from "hf://latent_L_sota" --cfg-scale 1.0 \
  --num-samples 50000 --eval-batch-size 256 --json-out results_latent.json

# Pixel model
python inference.py --init-from "hf://pixel_L_sota" --cfg-scale 1.0 \
  --num-samples 50000 --eval-batch-size 256 --json-out results_pixel.json
```

### PyTorch

```bash
cd torch
python inference.py --init-from /path/to/params_ema --cfg-scale 1.0 \
  --num-samples 50000 --eval-batch-size 256 --json-out results.json
```

To stream metrics and preview images to W&B, add `--use-wandb --wandb-entity YOUR_ENTITY_HERE --wandb-project YOUR_PROJECT_HERE` to either command.

Expected FID numbers match the [Pretrained Models](#pretrained-models) table above. Output JSON contains `fid`, `isc_mean`, `isc_std`, `precision`, `recall`. Precision/recall are only computed when `num_samples >= 50000`.

**Requirements:**

- TPU v4-8 (otherwise reduce `--eval-batch-size` to avoid OOM during VAE decoding)
- ImageNet-256 path configured in `utils/env.py`. Images are generated using the class labels from the ImageNet validation set.
- Precomputed FID/PR reference stats configured in `utils/env.py`

## Training

### JAX — Generator Training

```bash
cd jax
python main.py --gen --config ../configs/gen/latent_ablation.yaml --workdir runs/gen_latent_ablation
python main.py --gen --config ../configs/gen/latent_sota_B.yaml   --workdir runs/gen_latent_sota_B
python main.py --gen --config ../configs/gen/latent_sota_L.yaml   --workdir runs/gen_latent_sota_L
python main.py --gen --config ../configs/gen/pixel_sota_B.yaml    --workdir runs/gen_pixel_sota_B
python main.py --gen --config ../configs/gen/pixel_sota_L.yaml    --workdir runs/gen_pixel_sota_L
```

### PyTorch — Generator Training

```bash
cd torch
python main.py --gen --config ../configs/gen/latent_ablation.yaml --workdir runs/gen_latent_ablation
```

### JAX — Generator Training (Details)

```bash
cd jax
python main.py --gen --config ../configs/gen/latent_ablation.yaml --workdir runs/gen_latent_ablation
python main.py --gen --config ../configs/gen/latent_sota_B.yaml   --workdir runs/gen_latent_sota_B
python main.py --gen --config ../configs/gen/latent_sota_L.yaml   --workdir runs/gen_latent_sota_L
python main.py --gen --config ../configs/gen/pixel_sota_B.yaml    --workdir runs/gen_pixel_sota_B
python main.py --gen --config ../configs/gen/pixel_sota_L.yaml    --workdir runs/gen_pixel_sota_L
```

MAE pretrained weights are downloaded automatically from HuggingFace via the `feature.mae_path` config field. No need to train MAE unless experimenting with custom feature extractors.

FID is evaluated during training at intervals set by `train.eval_per_step`.

**Ablation run intermediate FID (EMA model, best CFG):**

| Steps | CFG | FID   |
| ----- | --- | ----- |
| 5k    | 3.5 | 35.20 |
| 10k   | 2.5 | 13.33 |
| 15k   | 2.0 | 10.70 |
| 20k   | 2.0 | 9.47  |
| 25k   | 2.0 | 8.84  |
| 30k   | 2.0 | 8.34  |

We used 64 TPU v6e for the ablation run and 128 TPU v6e for the SOTA runs. Each host maintains its own memory bank (16 hosts for ablation, 32 for SOTA). When using fewer hosts (e.g., DDP on one H100 node = 8 hosts), increase `push_per_step` to keep the memory bank update rate sufficient.

### JAX — MAE Pretraining (Optional)

Pretrained MAE weights are already available at `hf://mae_latent_640`, `hf://mae_latent_256`, and `hf://mae_pixel_640`. Training code is provided for users who want to train their own:

```bash
cd jax
python main.py --config ../configs/mae/latent_ablation_256.yaml --workdir runs/mae_latent_ablation_256
python main.py --config ../configs/mae/latent_640.yaml          --workdir runs/mae_latent_640
python main.py --config ../configs/mae/pixel_640.yaml           --workdir runs/mae_pixel_640
```

### Using a Local MAE Checkpoint as Feature Extractor

1. Train an MAE (see above).
2. Point the generator config at the MAE workdir:

```yaml
feature:
  mae_path: /abs/path/to/runs/mae_latent_640
  use_mae: true
  use_convnext: false
  use_post_x: false
```

3. Run generator training.

## Checkpoints and Logs

### JAX

Each `--workdir <dir>` produces:

```
<dir>/
├── checkpoints/                        # Orbax checkpoints (full training state)
├── params_ema/                         # EMA-only artifact
│   ├── ema_params.msgpack
│   └── metadata.json
└── log/
    ├── metrics.jsonl                   # Metrics (when use_wandb: false)
    └── images/*.jpg                    # Sample preview grids
```

Local artifacts in `params_ema/` can be loaded directly for inference (JAX):

```bash
cd jax
python inference.py --init-from /path/to/workdir --cfg-scale 1.0 \
  --num-samples 50000 --eval-batch-size 256
```

### PyTorch

```
<dir>/
├── checkpoints/                        # torch.save() checkpoints
├── params_ema/                         # EMA params artifact
│   ├── ema_params.pt
│   └── metadata.json
└── log/
    ├── metrics.jsonl
    └── images/*.jpg
```

## Citation

```bibtex
@article{deng2026generative,
  title={Generative Modeling via Drifting},
  author={Deng, Mingyang and Li, He and Li, Tianhong and Du, Yilun and He, Kaiming},
  journal={arXiv preprint arXiv:2602.04770},
  year={2026}
}
```

## Acknowledgments

We thank Hanhong Zhao for sanity checking this repository.
