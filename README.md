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
- `DiffusionViT`: DiT-style image denoiser.
- Uses per-block adaptive LayerNorm conditioning (`adaLN`) with residual gating.
- Includes final adaptive modulation before reconstruction and zero-init output projection for stable startup.

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

## Notes

- No dataset, preprocessing pipeline, or cold-diffusion degradation operators are implemented yet.
- This repository state is intentionally a scaffold so data and experiment details can be added later.

## TODO

- [ ] Prepare data and augmentation and so on.
