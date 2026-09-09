"""Root entry point for training the DiT Cold Diffusion model."""

import argparse
import sys
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.training.train_dit import DiTTrainConfig, train_dit

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train DiT Cold Diffusion Model")
    parser.add_argument("--clean_dir", type=str, default=str(PROJECT_ROOT / "data" / "clean"))
    parser.add_argument("--degraded_dir", type=str, default=str(PROJECT_ROOT / "data" / "degraded"))
    parser.add_argument("--image_size", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=0.5 * 1e-4)
    parser.add_argument("--timesteps", type=int, default=1000)
    parser.add_argument("--checkpoint_dir", type=str, default=str(PROJECT_ROOT / "checkpoints"))
    args = parser.parse_args()

    cfg = DiTTrainConfig(
        data_dir_clean=args.clean_dir,
        data_dir_degraded=args.degraded_dir,
        image_size=args.image_size,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.lr,
        timesteps=args.timesteps,
        checkpoint_dir=args.checkpoint_dir,
    )
    train_dit(cfg)
