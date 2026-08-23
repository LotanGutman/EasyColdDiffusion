import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "0"

from pathlib import Path
import cv2
import numpy as np
from tqdm import tqdm
from deep_learning_project import operations

# 1. עדכון הנתיבים
input_dir = Path(r"C:\Users\User\Downloads\ffhq_128_70k_images")  #Path(r"C:\Users\User\PyCharmMiscProject\deep_learning_project\test_pictures_origin")
output_dir = Path(r"C:\Users\User\PyCharmMiscProject\deep_learning_project\output")  #Path(r"C:\Users\User\PyCharmMiscProject\deep_learning_project\test_pictures_deg")
output_dir.mkdir(parents=True, exist_ok=True)

# 2. איסוף מיוין של קבצים (מבטיח סדר עקבי ושומר על שמות המקור)
valid_extensions = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}
image_paths = sorted([
    f for f in input_dir.rglob("*") if f.suffix in valid_extensions and f.is_file()
])

print(f"נמצאו {len(image_paths)} תמונות. מתחיל בעיבוד...")

for image_path in tqdm(image_paths, desc="Generating Degraded Dataset"):
    image = cv2.imread(str(image_path))
    if image is None:
        print(f"Warning: Could not read {image_path.name}")
        continue

    # הקטנה מראש ל-128x128
    image = cv2.resize(image, (128, 128), interpolation=cv2.INTER_LINEAR)


    # פרמטרים מותאמים
    sepia_parameter = float(np.clip(np.random.normal(loc=0.4, scale=0.1), 0.0, 1.0))
    film_grain_parameter = float(np.clip(np.random.normal(loc=6.0, scale=2.0), 1.0, 15.0))
    blur_parameter = float(np.clip(np.random.normal(loc=0.0008, scale=0.0003), 0.0001, 0.002))
    vignette_parameter = float(np.clip(np.random.normal(loc=0.7, scale=0.12), 0.1, 1.0))
    desaturation_parameter = float(np.clip(np.random.normal(loc=0.7, scale=0.12), 0.1, 1.0))
    white_abrasion_parameter = float(np.clip(np.random.normal(loc=0.55, scale=0.12), 0.2, 0.9))
    stain_type_parameter = np.random.choice(["coffee", "tea", "grease"])

    # רשימת המניפולציות הקבועה המלאה שלך
    operation_list = [
        ("add_random_heavy_tear"),
        ("add_dust_and_flecks"),
        ("blur", blur_parameter + 0.003),
        ("sepia", sepia_parameter),
        ("gamma"),
        ("film grain", film_grain_parameter),
        ("vignette", vignette_parameter),
        ("desaturation", desaturation_parameter),
        ("cracks"),
        ("sepia",sepia_parameter),
        ("contrast"),
        ("film grain", film_grain_parameter),
        ("blur", blur_parameter),
        ("add_spilled_stain", {"stain_type": stain_type_parameter, "intensity": 0.65}),
        ("add_white_mold_abrasion")
    ]

    processed_image = operations.apply_manipulations(image, operation_list)

    # שמירה באותו שם קובץ מדויק בתיקיית הפלט
    save_path = output_dir / image_path.name
    cv2.imwrite(str(save_path), processed_image)

print("Process finished! All images saved to output folder.")
