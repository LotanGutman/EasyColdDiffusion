"""Training pipeline for U-Net Cold Diffusion restoration model."""

import argparse
import glob
import os
import re
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from diffusers.optimization import get_cosine_schedule_with_warmup

from src.datasets.unet_dataset import RecursiveColdDiffusionDataset
from src.models.unet import UNet

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def train_cold_diffusion(args: argparse.Namespace) -> None:
    # ==========================================
    # Configurations & Hyperparameters
    # ==========================================
    clean_dir = args.clean_dir
    if not os.path.exists(clean_dir):
        alt_dir = os.path.join(PROJECT_ROOT, "coco_128_clean")
        if os.path.exists(alt_dir):
            clean_dir = alt_dir

    textures_dir = args.textures_dir
    paper_textures = sorted(
        glob.glob(os.path.join(textures_dir, "paper*.[jJ][pP][gG]"))
        + glob.glob(os.path.join(textures_dir, "paper*.[pP][nN][gG]"))
    )
    scratch_textures = sorted(
        glob.glob(os.path.join(textures_dir, "scratch*.[jJ][pP][gG]"))
        + glob.glob(os.path.join(textures_dir, "scratch*.[pP][nN][gG]"))
    )

    batch_size = args.batch_size
    total_epochs = args.epochs
    learning_rate = args.lr
    max_t = args.max_t
    checkpoint_dir = args.checkpoint_dir
    os.makedirs(checkpoint_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")
    print(f"Clean dataset directory: {clean_dir}")
    print(f"Textures found: {len(paper_textures)} paper textures, {len(scratch_textures)} scratch textures")

    # ==========================================
    # Dataset & DataLoader
    # ==========================================
    dataset = RecursiveColdDiffusionDataset(
        clean_dir=clean_dir,
        paper_paths=paper_textures,
        scratch_paths=scratch_textures,
        max_t=max_t,
    )

    if len(dataset) == 0:
        print(f"Error: No images found in '{clean_dir}'. Please prepare dataset first.")
        return

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True if torch.cuda.is_available() else False,
    )

    # ==========================================
    # Model & Optimizers
    # ==========================================
    model = UNet().to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion_img = nn.L1Loss()

    # ==========================================
    # Resume Logic
    # ==========================================
    start_epoch = 1

    if args.resume and args.resume_checkpoint:
        if os.path.exists(args.resume_checkpoint):
            print(f"Loading weights from {args.resume_checkpoint}...")
            model.load_state_dict(
                torch.load(args.resume_checkpoint, map_location=device), strict=False
            )
            match = re.search(r"epoch_(\d+)", args.resume_checkpoint)
            if match:
                start_epoch = int(match.group(1)) + 1
                print(f"Resuming training from epoch {start_epoch} to {total_epochs}")
        else:
            print(f"Warning: Checkpoint '{args.resume_checkpoint}' not found. Starting from scratch.")

    # ==========================================
    # Cosine Learning Rate Scheduler setup
    # ==========================================
    total_training_steps = len(dataloader) * total_epochs
    warmup_steps = int(total_training_steps * 0.1)
    past_steps = (start_epoch - 1) * len(dataloader)

    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_training_steps,
        last_epoch=past_steps - 1 if past_steps > 0 else -1,
    )

    # ==========================================
    # Training Loop
    # ==========================================
    for epoch in range(start_epoch, total_epochs + 1):
        model.train()
        epoch_loss_img = 0.0

        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch}/{total_epochs}")

        for batch_idx, (x_t, x_0, t_true) in enumerate(progress_bar):
            x_t = x_t.to(device)
            x_0 = x_0.to(device)
            t_true = t_true.to(device)

            t_embedding_input = (t_true.squeeze() * max_t).float()

            optimizer.zero_grad()

            pred_x_0 = model(x_t, t_embedding_input)
            total_loss = criterion_img(pred_x_0, x_0)

            total_loss.backward()
            optimizer.step()
            scheduler.step()

            epoch_loss_img += total_loss.item()
            current_lr = scheduler.get_last_lr()[0]

            progress_bar.set_postfix({
                "Img_L1": f"{total_loss.item():.4f}",
                "LR": f"{current_lr:.6f}",
            })

        avg_loss_img = epoch_loss_img / len(dataloader)
        print(f"End of Epoch {epoch} | Avg Img Loss: {avg_loss_img:.4f}")

        checkpoint_path = os.path.join(checkpoint_dir, f"cold_diff_unet_epoch_{epoch}.pth")
        torch.save(model.state_dict(), checkpoint_path)
        print(f"Checkpoint saved: {checkpoint_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Cold Diffusion U-Net Image Restoration Model")
    parser.add_argument(
        "--clean_dir",
        type=str,
        default=os.path.join(PROJECT_ROOT, "data", "coco_128_clean"),
        help="Path to clean 128x128 images directory",
    )
    parser.add_argument(
        "--textures_dir",
        type=str,
        default=os.path.join(PROJECT_ROOT, "assets", "textures"),
        help="Path to textures directory",
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default=os.path.join(PROJECT_ROOT, "checkpoints"),
        help="Directory to save checkpoints",
    )
    parser.add_argument("--batch_size", type=int, default=16, help="Training batch size")
    parser.add_argument("--epochs", type=int, default=40, help="Total training epochs")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--max_t", type=int, default=100, help="Maximum degradation steps")
    parser.add_argument("--num_workers", type=int, default=4, help="DataLoader worker processes")
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint")
    parser.add_argument("--resume_checkpoint", type=str, default="", help="Path to checkpoint to resume from")
    return parser.parse_args()


if __name__ == "__main__":
    train_cold_diffusion(parse_args())
