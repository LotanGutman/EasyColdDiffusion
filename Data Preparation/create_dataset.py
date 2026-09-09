import os
import cv2
import numpy as np
import random
from PIL import Image


def create_clean_dataset(coco_raw_dir, clean_dir, max_images=30000):
    """
    מכין רק תמונות מקוריות נקיות. הלכלוך הרקורסיבי יתבצע בזמן אמת בתוך ה-DataLoader.
    """
    os.makedirs(clean_dir, exist_ok=True)

    # שליפת כל הקבצים וערבוב אקראי
    all_files = [f for f in os.listdir(coco_raw_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    random.shuffle(all_files)
    files_to_process = all_files[:max_images]

    processed_count = 0

    print(f"Starting to process {len(files_to_process)} clean images...")

    for filename in files_to_process:
        img_path = os.path.join(coco_raw_dir, filename)

        try:
            # שימוש ב-PIL לחיתוך ריבועי ושינוי גודל מדויק
            with open(img_path, 'rb') as f:
                pil_img = Image.open(f).convert('RGB')
                w, h = pil_img.size

                # מציאת הריבוע המרכזי
                min_dim = min(w, h)
                left = (w - min_dim) / 2
                top = (h - min_dim) / 2
                right = (w + min_dim) / 2
                bottom = (h + min_dim) / 2

                pil_img = pil_img.crop((left, top, right, bottom))
                pil_img = pil_img.resize((128, 128), Image.Resampling.LANCZOS)

                # המרה חזרה ל-OpenCV/Numpy לשמירה
                img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

            cv2.imwrite(os.path.join(clean_dir, filename), img)

            processed_count += 1
            if processed_count % 2000 == 0:
                print(f"Processed {processed_count}/{max_images} images...")

        except Exception as e:
            # דילוג על קבצים פגומים
            continue

    print("Clean dataset generation complete!")


if __name__ == "__main__":
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

    # נתיבי התיקיות
    COCO_RAW = os.path.join(BASE_DIR, "coco_raw_train2017")
    CLEAN_OUTPUT = os.path.join(BASE_DIR, "coco_128_clean")

    create_clean_dataset(COCO_RAW, CLEAN_OUTPUT, max_images=30000)