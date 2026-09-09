"""Utility script to download and extract a subset of COCO 2017 train images."""

import argparse
import os
import urllib.request
import zipfile

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def download_progress(count: int, block_size: int, total_size: int) -> None:
    if total_size > 0:
        percent = int(count * block_size * 100 / total_size)
        percent = min(percent, 100)
        print(f"\rDownloading COCO train2017.zip (~18GB)... {percent}%", end="")


def download_coco_dataset(target_dir: str = None, max_images: int = 30000) -> None:
    if target_dir is None:
        target_dir = os.path.join(PROJECT_ROOT, "data", "coco_raw_train2017")

    os.makedirs(target_dir, exist_ok=True)

    img_url = "http://images.cocodataset.org/zips/train2017.zip"
    img_zip = os.path.join(PROJECT_ROOT, "train2017.zip")

    print(f"Target directory: {target_dir}")
    print("Starting download. Please be patient, download speed depends on your network connection...")

    urllib.request.urlretrieve(img_url, img_zip, reporthook=download_progress)
    print("\nDownload complete! Starting extraction...")

    extracted_count = 0

    with zipfile.ZipFile(img_zip, "r") as zip_ref:
        jpg_files = [f for f in zip_ref.namelist() if f.lower().endswith((".jpg", ".jpeg"))]

        for file_in_zip in jpg_files:
            if extracted_count >= max_images:
                break

            file_name = os.path.basename(file_in_zip)
            if not file_name:
                continue

            target_path = os.path.join(target_dir, file_name)

            with zip_ref.open(file_in_zip) as source, open(target_path, "wb") as target:
                target.write(source.read())

            extracted_count += 1

            if extracted_count % 5000 == 0:
                print(f"Extracted {extracted_count} / {max_images} images...")

    print("Cleaning up the downloaded zip file to save disk space...")
    if os.path.exists(img_zip):
        os.remove(img_zip)

    print(f"Success! {extracted_count} images are ready in '{target_dir}'.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download and extract subset of COCO 2017 train images.")
    parser.add_argument(
        "--output_dir",
        type=str,
        default=os.path.join(PROJECT_ROOT, "data", "coco_raw_train2017"),
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
