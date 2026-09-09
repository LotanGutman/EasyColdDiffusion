"""
EasyColdDiffusion - Unified CLI & Interactive GUI Entry Point

Usage:
    python main.py                  # Launches the Interactive GUI (Default)
    python main.py gui              # Launches the Interactive GUI
    python main.py test-unet        # Run U-Net evaluation & restoration
    python main.py train-unet       # Train U-Net model from scratch
    python main.py infer-dit        # Run DiT tiled inference
    python main.py train-dit        # Train DiT model from scratch
    python main.py create-dataset   # Crop and prepare 128x128 dataset
    python main.py download-coco    # Download raw COCO training images
"""

import sys
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def main():
    if len(sys.argv) == 1:
        # Default action: launch interactive GUI
        from src.gui import launch_gui
        launch_gui()
        return

    subcommand = sys.argv[1].lower().strip()

    if subcommand in ["gui", "app"]:
        from src.gui import launch_gui
        launch_gui()

    elif subcommand in ["test-unet", "test_unet", "test"]:
        from src.infer.infer_unet import parse_args, test_unet
        # Remove subcommand from sys.argv so sub-parser works seamlessly
        sys.argv.pop(1)
        test_unet(parse_args())

    elif subcommand in ["train-unet", "train_unet"]:
        from src.training.train_unet import parse_args, train_cold_diffusion
        sys.argv.pop(1)
        train_cold_diffusion(parse_args())

    elif subcommand in ["infer-dit", "infer_dit"]:
        from src.infer.infer_dit import main as dit_infer_main
        sys.argv.pop(1)
        dit_infer_main()

    elif subcommand in ["train-dit", "train_dit"]:
        from src.training.train_dit import main as dit_train_main
        sys.argv.pop(1)
        dit_train_main()

    elif subcommand in ["create-dataset", "create_dataset"]:
        from src.datasets.create_dataset import main as create_dataset_main
        sys.argv.pop(1)
        create_dataset_main()

    elif subcommand in ["download-coco", "download_coco"]:
        from src.datasets.download_coco import main as download_coco_main
        sys.argv.pop(1)
        download_coco_main()

    elif subcommand in ["train-estimator", "train_estimator"]:
        from src.training.train_estimator import parse_args, train_estimator
        sys.argv.pop(1)
        train_estimator(parse_args())

    elif subcommand in ["-h", "--help", "help"]:
        print(__doc__)

    else:
        print(f"Unknown command: '{subcommand}'\n")
        print(__doc__)
        sys.exit(1)


if __name__ == "__main__":
    main()
