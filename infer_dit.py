"""Root entry point for running inference with the DiT Cold Diffusion model."""

import argparse
import sys
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.infer.infer_dit import InferConfig, run_infer_dit

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Infer DiT Cold Diffusion Restoration")
    parser.add_argument("--checkpoint", type=str, default=str(PROJECT_ROOT / "checkpoints" / "dit_cold_diffusion_epoch_6.pth"))
    parser.add_argument("--input_dir", type=str, default=str(PROJECT_ROOT / "data" / "degraded"))
    parser.add_argument("--output_dir", type=str, default=str(PROJECT_ROOT / "results" / "dit_restored"))
    parser.add_argument("--image_size", type=int, default=128)
    parser.add_argument("--timesteps", type=int, default=50)
    args = parser.parse_args()

    cfg = InferConfig(
        checkpoint_path=args.checkpoint,
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        image_size=args.image_size,
        timesteps=args.timesteps,
    )
    run_infer_dit(cfg)
