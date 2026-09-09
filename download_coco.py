"""Root entry point to download a subset of COCO 2017 dataset."""

import argparse
import os
import sys
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.datasets.download_coco import download_coco_dataset

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download and extract subset of COCO 2017 train images.")
    parser.add_argument(
        "--output_dir",
        type=str,
        default=os.path.join(str(PROJECT_ROOT), "data", "coco_raw_train2017"),
        help="Target directory to save extracted images",
    )
    parser.add_argument(
        "--max_images",
        type=int,
        default=30000,
        help="Number of images to extract",
    )
    args = parser.parse_args()

    download_coco_dataset(target_dir=args.output_dir, max_images=args.max_images)
