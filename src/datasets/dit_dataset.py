"""
Dataset loader for DiT Cold Diffusion training.
Handles image loading, resizing, and on-the-fly photo aging degradation.
"""

from pathlib import Path
from typing import Tuple
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

# יבוא מודל היישון - ודא שקובץ olding.py נגיש ב-PYTHONPATH או בתיקייה הראשית
try:
    from olding import PhotoAgingModel
except ImportError:
    from ...olding import PhotoAgingModel


class DiTColdDiffusionDataset(Dataset):
    """
    Dataset class for DiT Cold Diffusion with on-the-fly PhotoAgingModel degradation.
    
    Returns:
        x_t: Image degraded up to step t (Tensor: 3 x H x W)
        x_prev: Image degraded up to step t-1 (Target for model prediction) (Tensor: 3 x H x W)
        y_deg: Fully degraded image at max_steps (Conditioning) (Tensor: 3 x H x W)
        t_val: Timestep scalar tensor (long)
    """

    def __init__(self, clean_dir: str, image_size: int = 128, max_steps: int = 50):
        super().__init__()
        self.clean_dir = Path(clean_dir)
        self.image_size = image_size
        self.max_steps = max_steps

        # אתחול מודל היישון המהיר O(1)
        self.aging_model = PhotoAgingModel(image_size=image_size, max_t=max_steps)

        valid_exts = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}
        self.clean_paths = sorted([
            p for p in self.clean_dir.glob("*") 
            if p.suffix in valid_exts and p.is_file()
        ])

        if len(self.clean_paths) == 0:
            raise FileNotFoundError(f"No valid clean images found in directory: {clean_dir}")

        # טרנספורמציה ונרמול לטווח [-1, 1] כפי שהמודל מצפה
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

    def __len__(self) -> int:
        return len(self.clean_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        clean_path = self.clean_paths[idx]

        # 1. טעינת התמונה ושינוי גודל
        clean_img = Image.open(clean_path).convert("RGB")
        clean_img = clean_img.resize((self.image_size, self.image_size), Image.Resampling.LANCZOS)
        clean_np = np.array(clean_img).astype(np.float32) / 255.0

        # 2. בחירה אקראית של step t מתוך max_steps
        t_val = np.random.randint(1, self.max_steps + 1)
        prev_t = max(0, t_val - 1)

        # 3. הפעלת מודל היישון
        self.aging_model.fit(clean_np)

        # 4. רינדור הדרגתי: x_t, x_prev, והתמונה המושחתת ביותר y_deg
        x_t_np = self.aging_model.render(t=t_val)
        x_prev_np = self.aging_model.render(t=prev_t)
        y_deg_np = self.aging_model.render(t=self.max_steps)

        # 5. המרה ל-PIL והחלת נרמול
        x_t_img = Image.fromarray((np.clip(x_t_np, 0, 1) * 255).astype(np.uint8))
        x_prev_img = Image.fromarray((np.clip(x_prev_np, 0, 1) * 255).astype(np.uint8))
        y_deg_img = Image.fromarray((np.clip(y_deg_np, 0, 1) * 255).astype(np.uint8))

        return (
            self.transform(x_t_img),
            self.transform(x_prev_img),
            self.transform(y_deg_img),
            torch.tensor(t_val, dtype=torch.long)
        )


# Alias לשמירה על תאימות לאחור
ColdDiffusionDataset = DiTColdDiffusionDataset
