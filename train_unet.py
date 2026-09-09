"""Root entry point for training the U-Net Cold Diffusion restoration model."""

import sys
from pathlib import Path

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.training.train_unet import parse_args, train_cold_diffusion

if __name__ == "__main__":
    train_cold_diffusion(parse_args())
