"""Training pipeline for DiT (Diffusion Transformer) Cold Diffusion model."""

import argparse
from dataclasses import dataclass
import os
from pathlib import Path
import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.datasets.dit_dataset import ColdDiffusionDataset
from src.models.diffusion_vit import DiffusionViT

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


@dataclass
class DiTTrainConfig:
    data_dir_clean: str = str(PROJECT_ROOT / "data" / "clean")
    data_dir_degraded: str = str(PROJECT_ROOT / "data" / "degraded")
    image_size: int = 128
    batch_size: int = 64
    num_workers: int = 2
    learning_rate: float = 0.5 * 1e-4
    epochs: int = 30
    timesteps: int = 1000
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint_dir: str = str(PROJECT_ROOT / "checkpoints")


def build_dataloader(config: DiTTrainConfig) -> DataLoader:
    dataset = ColdDiffusionDataset(
        clean_dir=config.data_dir_clean,
        degraded_dir=config.data_dir_degraded,
        image_size=config.image_size,
    )
    return DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=True if torch.cuda.is_available() else False,
        drop_last=True,
    )


def sample_timesteps(batch_size: int, max_timesteps: int, device: str) -> torch.Tensor:
    return torch.randint(low=0, high=max_timesteps, size=(batch_size,), device=device)


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    config: DiTTrainConfig,
    epoch: int,
) -> float:
    model.train()
    running_loss = 0.0

    pbar = tqdm(dataloader, desc=f"Epoch {epoch}/{config.epochs}")
    for clean_images, degraded_images in pbar:
        clean_images = clean_images.to(config.device, non_blocking=True)
        degraded_images = degraded_images.to(config.device, non_blocking=True)
        b = clean_images.shape[0]

        timesteps = sample_timesteps(b, config.timesteps, config.device)
        t_factor = timesteps.view(b, 1, 1, 1).float() / config.timesteps
        x_t = (1.0 - t_factor) * clean_images + t_factor * degraded_images
        model_input = torch.cat([degraded_images, x_t], dim=1)

        optimizer.zero_grad(set_to_none=True)

        use_amp = config.device == "cuda"
        with torch.amp.autocast("cuda", enabled=use_amp):
            predictions = model(model_input, timesteps)
            loss = torch.nn.functional.mse_loss(predictions, clean_images)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item()
        pbar.set_postfix({"Loss": f"{loss.item():.4f}"})

    return running_loss / max(len(dataloader), 1)


def train_dit(config: DiTTrainConfig) -> None:
    print(f"Using device: {config.device}")

    if config.device == "cuda":
        torch.set_float32_matmul_precision("high")

    os.makedirs(config.checkpoint_dir, exist_ok=True)

    model = DiffusionViT(
        image_size=config.image_size,
        patch_size=8,
        in_channels=6,
        out_channels=3,
        dim=512,
        depth=8,
        num_heads=8,
    ).to(config.device)

    optimizer = AdamW(model.parameters(), lr=config.learning_rate, weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs, eta_min=1e-6)
    scaler = torch.amp.GradScaler("cuda", enabled=(config.device == "cuda"))

    dataloader = build_dataloader(config)
    print(f"Loaded {len(dataloader.dataset)} image pairs. Starting training...")

    for epoch in range(1, config.epochs + 1):
        loss = train_one_epoch(model, dataloader, optimizer, scaler, config, epoch)
        scheduler.step()
        print(f"Epoch {epoch}/{config.epochs} Complete - Loss: {loss:.6f}")

        if epoch % 5 == 0 or epoch == config.epochs:
            ckpt_path = os.path.join(config.checkpoint_dir, f"dit_cold_diffusion_epoch_{epoch}.pth")
            torch.save(model.state_dict(), ckpt_path)
            print(f"Saved checkpoint: {ckpt_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train DiT Cold Diffusion Model")
    parser.add_argument("--clean_dir", type=str, default=str(PROJECT_ROOT / "data" / "clean"))
    parser.add_argument("--degraded_dir", type=str, default=str(PROJECT_ROOT / "data" / "degraded"))
    parser.add_argument("--image_size", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=0.5 * 1e-4)
    parser.add_argument("--timesteps", type=int, default=1000)
    parser.add_argument("--checkpoint_dir", type=str, default=str(PROJECT_ROOT / "checkpoints"))
    args = parser.parse_args()

    cfg = DiTTrainConfig(
        data_dir_clean=args.clean_dir,
        data_dir_degraded=args.degraded_dir,
        image_size=args.image_size,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.lr,
        timesteps=args.timesteps,
        checkpoint_dir=args.checkpoint_dir,
    )
    train_dit(cfg)
