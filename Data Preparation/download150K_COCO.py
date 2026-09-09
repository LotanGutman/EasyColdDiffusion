import os
import urllib.request
import zipfile


def download_progress(count, block_size, total_size):
    # Calculate and display download progress percentage
    percent = int(count * block_size * 100 / total_size)
    print(f"\rDownloading COCO train2017.zip (~18GB)... {percent}%", end="")


def download_coco_120k_images():
    # Target directory based on your project structure
    target_dir = "C:/python projects/sockets/FINAL DL/larger tiny test/coco_raw_train2017"
    os.makedirs(target_dir, exist_ok=True)

    # Official URL for the COCO 2017 train images
    img_url = "http://images.cocodataset.org/zips/train2017.zip"
    img_zip = "train2017.zip"
    max_images = 150000

    print("Starting download. Please be patient, this depends on your internet speed.")

    # Download the 18GB zip file with a progress hook
    urllib.request.urlretrieve(img_url, img_zip, reporthook=download_progress)
    print("\nDownload complete! Starting extraction...")

    extracted_count = 0

    # Extract exactly 50,000 images dynamically
    with zipfile.ZipFile(img_zip, 'r') as zip_ref:
        # Filter out anything that is not a JPEG image
        jpg_files = [f for f in zip_ref.namelist() if f.lower().endswith('.jpg')]

        for file_in_zip in jpg_files:
            if extracted_count >= max_images:
                break

            # Extract the filename without the internal zip folder structure
            file_name = os.path.basename(file_in_zip)
            if not file_name:
                continue

            target_path = os.path.join(target_dir, file_name)

            # Read the image directly from the zip and write it to the target directory
            with zip_ref.open(file_in_zip) as source, open(target_path, "wb") as target:
                target.write(source.read())

            extracted_count += 1

            # Print an update every 5,000 images
            if extracted_count % 5000 == 0:
                print(f"Extracted {extracted_count} / {max_images} images...")

    print("Cleaning up the 18GB zip file to save disk space...")
    os.remove(img_zip)

    print(f"Success! {extracted_count} images are ready in '{target_dir}'.")


if __name__ == "__main__":
    download_coco_120k_images()