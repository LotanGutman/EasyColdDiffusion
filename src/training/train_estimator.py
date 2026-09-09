"""
Training script for Step & Degradation Estimator.

Trains the lightweight StepAndDegradationEstimator network on synthetic
progressive degradation pairs (x_t, t, K_target).
"""

from __future__ import annotations

import argparse

import glob
import os
import random
import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

from src.models.step_estimator import StepAndDegradationEstimator

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TEXTURES_DIR = os.path.join(PROJECT_ROOT, "assets", "textures")
TEST_IMAGES_DIR = os.path.join(PROJECT_ROOT, "assets", "test_images")
DEFAULT_CLEAN_DIR = os.path.join(PROJECT_ROOT, "data", "coco_128_clean")
DEFAULT_CHECKPOINT_DIR = os.path.join(PROJECT_ROOT, "checkpoints")


class DegradationEstimationDataset(Dataset):
    def __init__(self, clean_dir: str, textures_dir: str, samples_per_epoch: int = 5000, max_t: int = 100):
        self.max_t = max_t
        self.samples_per_epoch = samples_per_epoch

        # Clean images
        self.clean_files = []
        if os.path.exists(clean_dir):
            self.clean_files = [
                os.path.join(clean_dir, f)
                for f in os.listdir(clean_dir)
                if f.lower().endswith((".png", ".jpg", ".jpeg"))
            ]

        # Fallback to test images if clean_dir empty
        if not self.clean_files and os.path.exists(TEST_IMAGES_DIR):
            self.clean_files = [
                os.path.join(TEST_IMAGES_DIR, f)
                for f in os.listdir(TEST_IMAGES_DIR)
                if f.lower().endswith((".png", ".jpg", ".jpeg"))
            ]

        if not self.clean_files:
            raise RuntimeError(f"No images found in {clean_dir} or {TEST_IMAGES_DIR}!")

        # Pre-load textures
        paper_files = sorted(glob.glob(os.path.join(textures_dir, "paper*.*")))
        scratch_files = sorted(glob.glob(os.path.join(textures_dir, "scratch*.*")))

        self.papers = [self._load_tex(p) for p in paper_files if os.path.exists(p)]
        self.scratches = [self._load_tex(p) for p in scratch_files if os.path.exists(p)]

        self.transform = transforms.ToTensor()

    def _load_tex(self, path: str) -> np.ndarray:
        arr = np.fromfile(path, np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        img = cv2.resize(img, (128, 128))
        return img.astype(np.float32) / 255.0

    def _degrade_step(self, img_float: np.ndarray, paper: np.ndarray | None, scratch: np.ndarray | None) -> np.ndarray:
        # Sepia
        sepia_matrix = np.array([
            [0.393, 0.769, 0.189],
            [0.349, 0.686, 0.168],
            [0.272, 0.534, 0.131],
        ])
        sepia_img = np.clip(cv2.transform(img_float, sepia_matrix), 0, 1.0)
        blended = (1.0 - 0.008) * img_float + 0.008 * sepia_img

        # Paper
        if paper is not None:
            blended = (1.0 - 0.008) * blended + 0.008 * (blended * paper)

        # Scratch + Dilation
        if scratch is not None:
            scratch_gray = np.mean(scratch, axis=-1, keepdims=True)
            scratch_clean = np.clip((scratch_gray - 0.20) / 0.80, 0.0, 1.0)
            kernel = np.ones((2, 2), np.float32)
            scratch_clean = cv2.dilate(scratch_clean, kernel, iterations=1)
            scratch_clean = np.expand_dims(scratch_clean, axis=-1)

            screen = 1.0 - (1.0 - blended) * (1.0 - scratch_clean)
            blended = (1.0 - 0.01) * blended + 0.01 * screen

        return np.clip(blended, 0.0, 1.0)

    def __len__(self) -> int:
        return self.samples_per_epoch

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        img_path = random.choice(self.clean_files)
        arr = np.fromfile(img_path, np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        h, w = img.shape[:2]
        min_dim = min(h, w)
        img = img[h // 2 - min_dim // 2: h // 2 + min_dim // 2, w // 2 - min_dim // 2: w // 2 + min_dim // 2]
        img = cv2.resize(img, (128, 128))
        x_0 = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

        # Sample degradation step t in [0, max_t]
        t = random.randint(0, self.max_t)
        paper = random.choice(self.papers) if self.papers and random.random() > 0.15 else None
        scratch = random.choice(self.scratches) if self.scratches and random.random() > 0.15 else None

        x_t = x_0.copy()
        for _ in range(t):
            x_t = self._degrade_step(x_t, paper, scratch)

        # Target step schedule: 1 step for t=0, up to 10-12 steps for t=100
        k_target = max(1.0, 1.0 + 9.0 * (t / float(self.max_t)))

        x_tensor = self.transform(x_t)
        t_target = torch.tensor([float(t)], dtype=torch.float32)
        k_tensor = torch.tensor([k_target], dtype=torch.float32)

        return x_tensor, t_target, k_tensor


def train_estimator(args: argparse.Namespace):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training Step & Degradation Estimator on: {device}")

    os.makedirs(args.output_dir, exist_ok=True)
    save_path = os.path.join(args.output_dir, "step_estimator.pth")

    dataset = DegradationEstimationDataset(
        clean_dir=args.clean_dir,
        textures_dir=args.textures_dir,
        samples_per_epoch=args.samples_per_epoch,
        max_t=args.max_t,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True if "cuda" in device.type else False,
    )

    model = StepAndDegradationEstimator(max_t=float(args.max_t), max_steps=args.max_steps).to(device)
    optimizer = AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    loss_fn = nn.SmoothL1Loss()

    scaler = torch.amp.GradScaler("cuda", enabled=("cuda" in device.type))

    print(f"Starting training ({args.epochs} epochs, {len(dataset)} samples/epoch)...")

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0

        pbar = tqdm(dataloader, desc=f"Epoch {epoch}/{args.epochs}")
        for x_batch, t_target, k_target in pbar:
            x_batch = x_batch.to(device, non_blocking=True)
            t_target = t_target.to(device, non_blocking=True)
            k_target = k_target.to(device, non_blocking=True)

            optimizer.zero_grad()

            with torch.amp.autocast("cuda", enabled=("cuda" in device.type)):
                t_pred, k_pred = model(x_batch)
                loss_t = loss_fn(t_pred, t_target)
                loss_k = loss_fn(k_pred, k_target) * 5.0
                loss = loss_t + loss_k

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()
            pbar.set_postfix({"Loss": f"{loss.item():.4f}", "Loss_t": f"{loss_t.item():.2f}"})

        scheduler.step()
        epoch_loss = running_loss / len(dataloader)
        print(f"Epoch {epoch} Complete - Average Loss: {epoch_loss:.4f}")

    torch.save(model.state_dict(), save_path)
    print(f"Training complete! Saved estimator checkpoint to: {save_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Step & Degradation Estimator")
    parser.add_argument("--clean_dir", type=str, default=DEFAULT_CLEAN_DIR, help="Clean images directory")
    parser.add_argument("--textures_dir", type=str, default=TEXTURES_DIR, help="Textures directory")
    parser.add_argument("--output_dir", type=str, default=DEFAULT_CHECKPOINT_DIR, help="Checkpoint save directory")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--samples_per_epoch", type=int, default=5000, help="Samples per epoch")
    parser.add_argument("--learning_rate", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--max_t", type=int, default=100, help="Maximum degradation steps")
    parser.add_argument("--max_steps", type=int, default=20, help="Maximum sampling steps")
    parser.add_argument("--num_workers", type=int, default=2, help="DataLoader workers")
    return parser.parse_args()


if __name__ == "__main__":
    train_estimator(parse_args())
