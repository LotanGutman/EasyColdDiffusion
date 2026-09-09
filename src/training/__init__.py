"""Training package exports for EasyColdDiffusion."""

from .train_unet import train_cold_diffusion as train_unet
from .train_dit import train_dit

__all__ = ["train_unet", "train_dit"]
