import cv2
from pathlib import Path
from deep_learning_project import operations

input_dir = Path("train")  # שם התיקייה המכילה את התמונות
output_dir = Path("output")

# יצירת תיקיית הפלט במידה והיא עדיין לא קיימת
output_dir.mkdir(parents=True, exist_ok=True)


#רשימת המניפולציות - אפשר לשחק עם זה
operation_list = [
    ("add_random_heavy_tear"),
    ("add_dust_and_flecks"),
    ("blur", 0.003),
    ("sepia"),
    ("gamma"),
    ("film grain", 12),
    ("vignette", 0.7),
    ("desaturation", 0.7),
    ("cracks"),
    ("add_dust_and_flecks"),
    ("contrast"),
    ("film grain", 23),
    ("blur",0.007),
    ("desaturation", 1.3)
]

# 3. איסוף כל קבצי ה-jpg/png (כולל אותיות גדולות וחיפוש עמוק בתתי-תיקיות)
extensions = ["*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"]
image_paths = []
for ext in extensions:
    image_paths.extend(input_dir.rglob(ext))

# מעבר על כל התמונות שנמצאו (set מונע כפילויות)
for image_path in set(image_paths):

    # קריאת התמונה מהדיסק (str נדרש עבור cv2.imread)
    image = cv2.imread(str(image_path))

    # בדיקה שהתמונה נטענה בהצלחה (ולא קובץ פגום)
    if image is None:
        print(f"Warning: Could not read {image_path.name}")
        continue

    processed_image = operations.apply_manipulations(image, operation_list)

    # הגדרת נתיב השמירה בתיקיית הפלט
    save_path = output_dir / image_path.name

    # שמירת התמונה המעובדת
    cv2.imwrite(str(save_path), processed_image)

print("Process finished! All images saved to the output folder.")