"""Dataset class for U-Net Cold Diffusion with recursive on-the-fly degradation."""

import hashlib
import os
import random
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms


class RecursiveColdDiffusionDataset(Dataset):
    def __init__(self, clean_dir: str, paper_paths: list[str], scratch_paths: list[str], max_t: int = 100):
        """
        clean_dir: Path to directory of clean images (128x128)
        paper_paths: List of paths to paper texture images
        scratch_paths: List of paths to scratch texture images
        max_t: Maximum number of recursive degradation steps (T=100)
        """
        self.clean_dir = clean_dir
        self.image_files = (
            [f for f in os.listdir(clean_dir) if f.lower().endswith((".png", ".jpg", ".jpeg"))]
            if os.path.exists(clean_dir)
            else []
        )
        self.max_t = max_t

        # Preload textures into memory to avoid repeated disk reads during training
        self.papers = [self._load_texture(p) for p in paper_paths if os.path.exists(p)]
        self.scratches = [self._load_texture(p) for p in scratch_paths if os.path.exists(p)]

        self.transform = transforms.Compose([
            transforms.ToTensor()
        ])

    def _load_texture(self, path: str) -> np.ndarray:
        """Loads and prepares a texture image, resized to 128x128 and normalized to [0, 1]."""
        img_array = np.fromfile(path, np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        img = cv2.resize(img, (128, 128))
        return img.astype(np.float32) / 255.0

    def _get_deterministic_seed(self, filename: str) -> int:
        """
        Converts the image filename into a deterministic integer seed.
        Ensures a specific image always samples the same degradation profile/textures.
        """
        return int(hashlib.md5(filename.encode("utf-8")).hexdigest(), 16) % (10 ** 8)

    def single_degradation_step(self, img_float: np.ndarray, paper_tex: np.ndarray, scratch_tex: np.ndarray, profile: int) -> np.ndarray:
        """
        Micro-step degradation optimized for T=100.
        Applies sepia aging, paper texture overlay, and scratch texture with morphological dilation.
        """
        # 1. Sepia fade - RGB matrix for realistic aging/yellowing
        sepia_matrix = np.array([
            [0.393, 0.769, 0.189],
            [0.349, 0.686, 0.168],
            [0.272, 0.534, 0.131],
        ])
        sepia_img = np.clip(cv2.transform(img_float, sepia_matrix), 0, 1.0)
        blended = (1.0 - 0.008) * img_float + 0.008 * sepia_img

        # 2. Paper texture (0.8%)
        if profile in [0, 2] and self.papers:
            multiply_blend = blended * paper_tex
            blended = (1.0 - 0.008) * blended + 0.008 * multiply_blend

        # 3. Scratch texture (1.0%) - Smooth extraction & Dilation
        if profile in [1, 2] and self.scratches:
            scratch_gray = np.mean(scratch_tex, axis=-1, keepdims=True)
            black_point = 0.20
            scratch_clean = np.clip((scratch_gray - black_point) / (1.0 - black_point), 0.0, 1.0)

            # --- Morphological Dilation ---
            kernel = np.ones((2, 2), np.float32)
            scratch_clean = cv2.dilate(scratch_clean, kernel, iterations=1)
            scratch_clean = np.expand_dims(scratch_clean, axis=-1)

            screen_blend = 1.0 - (1.0 - blended) * (1.0 - scratch_clean)
            blended = (1.0 - 0.01) * blended + 0.01 * screen_blend

        return np.clip(blended, 0.0, 1.0)

    def __len__(self) -> int:
        return len(self.image_files)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        filename = self.image_files[idx]
        img_path = os.path.join(self.clean_dir, filename)

        # 1. Load clean ground truth image x_0
        img_array = np.fromfile(img_path, np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        x_0 = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

        # 2. Use deterministic seed for fixed texture characteristics per image
        rng = random.Random(self._get_deterministic_seed(filename))
        profile = rng.choice([0, 1, 2])  # 0: paper only, 1: scratch only, 2: both
        paper_tex = rng.choice(self.papers) if self.papers else np.ones_like(x_0)
        scratch_tex = rng.choice(self.scratches) if self.scratches else np.ones_like(x_0)

        # 3. Sample degradation step t randomly so the model encounters varied degradation levels
        t = random.randint(1, self.max_t)

        # 4. Apply recursive chaining degradation for t steps
        x_t = x_0.copy()
        for _ in range(t):
            x_t = self.single_degradation_step(x_t, paper_tex, scratch_tex, profile)

        # 5. Convert to tensors
        x_0_tensor = self.transform(x_0)
        x_t_tensor = self.transform(x_t)

        # 6. Normalized timestep tensor
        t_tensor = torch.tensor([t / self.max_t], dtype=torch.float32)

        return x_t_tensor, x_0_tensor, t_tensor
