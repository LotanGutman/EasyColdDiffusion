"""Prepares clean, center-cropped 128x128 images from the raw dataset."""

import argparse
import os
import random
import cv2
import numpy as np
from PIL import Image

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def create_clean_dataset(coco_raw_dir: str, clean_dir: str, max_images: int = 30000) -> None:
    """
    Prepares clean, center-cropped 128x128 images from the raw dataset.
    Recursive degradation is applied on-the-fly inside the DataLoader.
    """
    os.makedirs(clean_dir, exist_ok=True)

    if not os.path.exists(coco_raw_dir):
        print(f"Error: Raw directory not found at '{coco_raw_dir}'")
        return

    all_files = [
        f for f in os.listdir(coco_raw_dir)
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
    ]
    random.shuffle(all_files)
    files_to_process = all_files[:max_images]

    processed_count = 0
    print(f"Starting to process {len(files_to_process)} clean images...")

    for filename in files_to_process:
        img_path = os.path.join(coco_raw_dir, filename)

        try:
            with open(img_path, "rb") as f:
                pil_img = Image.open(f).convert("RGB")
                w, h = pil_img.size

                min_dim = min(w, h)
                left = (w - min_dim) / 2
                top = (h - min_dim) / 2
                right = (w + min_dim) / 2
                bottom = (h + min_dim) / 2

                pil_img = pil_img.crop((left, top, right, bottom))
                pil_img = pil_img.resize((128, 128), Image.Resampling.LANCZOS)

                img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

            cv2.imwrite(os.path.join(clean_dir, filename), img)

            processed_count += 1
            if processed_count % 2000 == 0:
                print(f"Processed {processed_count}/{len(files_to_process)} images...")

        except Exception:
            continue

    print(f"Clean dataset generation complete! Saved {processed_count} images to '{clean_dir}'.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process raw images into 128x128 clean training dataset.")
    parser.add_argument(
        "--raw_dir",
        type=str,
        default=os.path.join(PROJECT_ROOT, "data", "coco_raw_train2017"),
        help="Path to raw image folder",
    )
    parser.add_argument(
        "--clean_dir",
        type=str,
        default=os.path.join(PROJECT_ROOT, "data", "coco_128_clean"),
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
