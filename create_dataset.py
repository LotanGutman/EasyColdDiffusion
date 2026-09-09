"""Root entry point to process raw images into 128x128 clean training dataset."""

import argparse
import os
import sys
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.datasets.create_dataset import create_clean_dataset

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process raw images into 128x128 clean training dataset.")
    parser.add_argument(
        "--raw_dir",
        type=str,
        default=os.path.join(str(PROJECT_ROOT), "data", "coco_raw_train2017"),
        help="Path to raw image folder",
    )
    parser.add_argument(
        "--clean_dir",
        type=str,
        default=os.path.join(str(PROJECT_ROOT), "data", "coco_128_clean"),
        help="Path to save processed 128x128 clean images",
    )
    parser.add_argument(
        "--max_images",
        type=int,
        default=30000,
        help="Maximum number of images to process",
    )
    args = parser.parse_args()

    create_clean_dataset(args.raw_dir, args.clean_dir, max_images=args.max_images)
