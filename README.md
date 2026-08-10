# EasyColdDiffusion

Kickstarted project scaffold for cold diffusion research, focused only on model and training script foundations.

## What is included

All implementation lives under `src/`:

```text
src/
  models/
    __init__.py
    unet.py
    diffusion_vit.py
    sparse_video_transformer.py
    utils/
      __init__.py
      blocks.py
  training/
    train.py
```

Additionally:

- `install.sh` for dependency installation via `pip`.

## Models

### `src/models/unet.py`
- `ColdDiffusionUNet`: U-Net-style denoising model for image-based cold diffusion experiments.
- Includes timestep conditioning via sinusoidal embeddings.

### `src/models/diffusion_vit.py`
- `DiffusionViT`: Vision Transformer denoiser for image diffusion.
- Uses patch embeddings, transformer blocks, and timestep conditioning.

### `src/models/sparse_video_transformer.py`
- `SparseDiffusionVideoTransformer`: video diffusion transformer scaffold.
- Uses a sparse attention backend through `fla` (`MultiScaleRetention`) when available, with a PyTorch attention fallback.

### `src/models/utils/`
- Shared model utilities:
  - `SinusoidalTimeEmbedding`
  - `MLP`

## Training

### `src/training/train.py`
- Contains a basic training scaffold:
  - `TrainConfig`
  - `train_one_epoch`
  - `main`
- Data loading is intentionally not implemented yet:
  - `build_dataloader(...)` is left as `TODO` and raises `NotImplementedError`.

## Install

```bash
./install.sh
```

This installs:
- `torch`
- `torchvision`
- `einops`
- `tqdm`
- `pyyaml`
- `flash-linear-attention`

## Notes

- No dataset, preprocessing pipeline, or cold-diffusion degradation operators are implemented yet.
- This repository state is intentionally a scaffold so data and experiment details can be added later.
