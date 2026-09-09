"""Dataset class for DiT-based Cold Diffusion with pre-degraded or paired image directories."""

from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms


class ColdDiffusionDataset(Dataset):
    def __init__(self, clean_dir: str, degraded_dir: str, image_size: int = 128):
        self.clean_dir = Path(clean_dir)
        self.degraded_dir = Path(degraded_dir)

        valid_exts = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}
        self.clean_paths = sorted([
            p for p in self.clean_dir.glob("*") if p.suffix in valid_exts and p.is_file()
        ])

        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

    def __len__(self) -> int:
        return len(self.clean_paths)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        clean_path = self.clean_paths[idx]
        degraded_path = self.degraded_dir / clean_path.name

        if not degraded_path.exists():
            raise FileNotFoundError(f"Degraded pair file not found: {degraded_path}")

        clean_img = Image.open(clean_path).convert("RGB")
        degraded_img = Image.open(degraded_path).convert("RGB")

        return self.transform(clean_img), self.transform(degraded_img)
