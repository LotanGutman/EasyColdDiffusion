import os
import re
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from diffusers.optimization import get_cosine_schedule_with_warmup

# Import our custom dataset and model
from dataset import RecursiveColdDiffusionDataset
from model import UNet

def train_cold_diffusion():
    # ==========================================
    # Configurations & Hyperparameters
    # ==========================================
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    CLEAN_DIR = os.path.join(BASE_DIR, "coco_128_clean")

    PAPER_TEXTURES = [os.path.join(BASE_DIR, "textures", f"paper{i}.jpg") for i in range(1, 7)]
    SCRATCH_TEXTURES = [os.path.join(BASE_DIR, "textures", f"scratch{i}.jpg") for i in range(1, 7)]

    BATCH_SIZE = 16
    TOTAL_EPOCHS = 40
    LEARNING_RATE = 1e-4
    MAX_T = 100

    # Resume configurations
    RESUME = False
    RESUME_CHECKPOINT = os.path.join(BASE_DIR, "checkpoints", "cold_diff_model_epoch_10.pth")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")

    # ==========================================
    # Dataset & DataLoader
    # ==========================================
    dataset = RecursiveColdDiffusionDataset(
        clean_dir=CLEAN_DIR,
        paper_paths=PAPER_TEXTURES,
        scratch_paths=SCRATCH_TEXTURES,
        max_t=MAX_T
    )

    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)

    # ==========================================
    # Model & Optimizers
    # ==========================================
    model = UNet().to(device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Only Image L1 Loss remains
    criterion_img = nn.L1Loss()

    # ==========================================
    # Resume Logic
    # ==========================================
    start_epoch = 1

    if RESUME:
        if os.path.exists(RESUME_CHECKPOINT):
            print(f"Loading weights from {RESUME_CHECKPOINT}...")
            # strict=False allows loading older checkpoints that had the t_predictor weights
            model.load_state_dict(torch.load(RESUME_CHECKPOINT, map_location=device), strict=False)

            match = re.search(r'epoch_(\d+)', RESUME_CHECKPOINT)
            if match:
                start_epoch = int(match.group(1)) + 1
                print(f"Resuming training from epoch {start_epoch} to {TOTAL_EPOCHS}")
        else:
            print(f"Warning: Checkpoint {RESUME_CHECKPOINT} not found. Starting from scratch.")

    # ==========================================
    # Cosine Learning Rate Scheduler setup
    # ==========================================
    total_training_steps = len(dataloader) * TOTAL_EPOCHS
    warmup_steps = int(total_training_steps * 0.1)
    past_steps = (start_epoch - 1) * len(dataloader)

    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_training_steps,
        last_epoch=past_steps - 1 if past_steps > 0 else -1
    )

    # ==========================================
    # Training Loop
    # ==========================================
    os.makedirs("checkpoints", exist_ok=True)

    for epoch in range(start_epoch, TOTAL_EPOCHS + 1):
        model.train()
        epoch_loss_img = 0.0

        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch}/{TOTAL_EPOCHS}")

        for batch_idx, (x_t, x_0, t_true) in enumerate(progress_bar):
            x_t = x_t.to(device)
            x_0 = x_0.to(device)
            t_true = t_true.to(device)

            t_embedding_input = (t_true.squeeze() * MAX_T).float()

            optimizer.zero_grad()

            # The UNet now solely outputs the restored image
            pred_x_0 = model(x_t, t_embedding_input)

            # Total loss is exactly the image loss
            total_loss = criterion_img(pred_x_0, x_0)

            total_loss.backward()
            optimizer.step()
            scheduler.step()

            epoch_loss_img += total_loss.item()
            current_lr = scheduler.get_last_lr()[0]

            progress_bar.set_postfix({
                "Img_L1": f"{total_loss.item():.4f}",
                "LR": f"{current_lr:.6f}"
            })

        avg_loss_img = epoch_loss_img / len(dataloader)
        print(f"End of Epoch {epoch} | Avg Img Loss: {avg_loss_img:.4f}")

        checkpoint_path = os.path.join("checkpoints", f"cold_diff_model_epoch_{epoch}.pth")
        torch.save(model.state_dict(), checkpoint_path)
        print(f"Checkpoint saved: {checkpoint_path}")

if __name__ == "__main__":
    train_cold_diffusion()