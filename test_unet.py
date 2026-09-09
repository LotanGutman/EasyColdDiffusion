"""Root entry point for testing and evaluating the U-Net Cold Diffusion restoration model."""

import sys
from pathlib import Path

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.infer.infer_unet import parse_args, test_unet

if __name__ == "__main__":
    test_unet(parse_args())
