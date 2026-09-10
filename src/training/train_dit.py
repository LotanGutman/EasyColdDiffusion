from dataclasses import dataclass
from pathlib import Path
import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
import numpy as np
import cv2

from src.models.diffusion_vit import DiffusionViT
from olding import PhotoAgingModel


@dataclass
class TrainConfig:
    data_dir_clean: str = r"C:\Users\User\Downloads\ffhq_128_70k_images"
    image_size: int = 128
    batch_size: int = 16
    num_workers: int = 4
    learning_rate: float = 0.5 * 1e-4
    epochs: int = 30
    timesteps: int = 50  # <-- עודכן ל-50 צעדי יישון
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


class ColdDiffusionDataset(Dataset):
    def __init__(self, clean_dir: str, image_size: int = 128, max_steps: int = 50):
        self.clean_dir = Path(clean_dir)
        self.image_size = image_size
        self.max_steps = max_steps

        # אתחול מודל היישון עם 50 צעדים מקסימליים
        self.aging_model = PhotoAgingModel(image_size=image_size, max_t=max_steps)

        valid_exts = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}
        self.clean_paths = sorted([
            p for p in self.clean_dir.glob("*") if p.suffix in valid_exts and p.is_file()
        ])

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

    def __len__(self):
        return len(self.clean_paths)

    def __getitem__(self, idx):
        clean_path = self.clean_paths[idx]

        # טעינת התמונה בפורמט RGB ונרמול לטווח [0.0, 1.0] כ-float32
        clean_img = Image.open(clean_path).convert("RGB")
        clean_img = clean_img.resize((self.image_size, self.image_size), Image.Resampling.LANCZOS)
        clean_np = np.array(clean_img).astype(np.float32) / 255.0

        # בחירה אקראית של זמן דיפוזיה t מתוך 50
        t_val = np.random.randint(1, self.max_steps + 1)
        prev_t = max(0, t_val - 1)

        # הפעלת fit פעם אחת עבור התמונה הנוכחית
        self.aging_model.fit(clean_np)

        # רינדור מהיר O(1) בעזרת העברת הפרמטר t בלבד
        x_t_np = self.aging_model.render(t=t_val)
        x_prev_np = self.aging_model.render(t=prev_t)
        y_deg_np = self.aging_model.render(t=self.max_steps)

        # המרה בחזרה ל-PIL Image לצורך מעבר דרך ה-transforms
        x_t_img = Image.fromarray((np.clip(x_t_np, 0, 1) * 255).astype(np.uint8))
        x_prev_img = Image.fromarray((np.clip(x_prev_np, 0, 1) * 255).astype(np.uint8))
        y_deg_img = Image.fromarray((np.clip(y_deg_np, 0, 1) * 255).astype(np.uint8))

        return (
            self.transform(x_t_img),
            self.transform(x_prev_img),
            self.transform(y_deg_img),
            torch.tensor(t_val, dtype=torch.long)
        )


def save_aging_preview(config: TrainConfig, num_samples: int = 3, step_interval: int = 10):
    print("Generating aging progression preview samples (50 steps)...")
    clean_dir = Path(config.data_dir_clean)
    valid_exts = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}
    clean_paths = sorted([p for p in clean_dir.glob("*") if p.suffix in valid_exts and p.is_file()])

    if not clean_paths:
        print("No clean images found for preview generation.")
        return

    preview_dir = Path("aging_previews")
    preview_dir.mkdir(parents=True, exist_ok=True)

    aging_model = PhotoAgingModel(image_size=config.image_size, max_t=config.timesteps)

    # בחירת צעדי הזמן להצגה בקפיצות של 10 (0, 10, 20, 30, 40, 50)
    step_indices = list(range(0, config.timesteps + 1, step_interval))
    if config.timesteps not in step_indices:
        step_indices.append(config.timesteps)

    actual_samples = min(num_samples, len(clean_paths))
    for i in range(actual_samples):
        img_path = clean_paths[i]
        clean_img = Image.open(img_path).convert("RGB")
        clean_img = clean_img.resize((config.image_size, config.image_size), Image.Resampling.LANCZOS)
        clean_np = np.array(clean_img).astype(np.float32) / 255.0

        aging_model.fit(clean_np, seed=42 + i)

        rendered_tiles = []
        for t in step_indices:
            t_np = aging_model.render(t=t)
            t_img = Image.fromarray((np.clip(t_np, 0, 1) * 255).astype(np.uint8))
            rendered_tiles.append(t_img)

        w, h = config.image_size, config.image_size
        combined_img = Image.new("RGB", (w * len(rendered_tiles), h))
        for idx, tile in enumerate(rendered_tiles):
            combined_img.paste(tile, (idx * w, 0))

        save_path = preview_dir / f"aging_progression_sample_{i+1}.png"
        combined_img.save(save_path)
        print(f"Saved aging progression preview to: {save_path}")
    print(f"Previews saved successfully in '{preview_dir.resolve()}' folder.\n")


def build_dataloader(config: TrainConfig) -> DataLoader:
    dataset = ColdDiffusionDataset(
        clean_dir=config.data_dir_clean,
        image_size=config.image_size,
        max_steps=config.timesteps
    )
    return DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=True,
        drop_last=True
    )


def train_one_epoch(
        model: nn.Module,
        dataloader: DataLoader,
        optimizer: torch.optim.Optimizer,
        scaler: torch.amp.GradScaler,
        config: TrainConfig,
        epoch: int,
) -> float:
    model.train()
    running_loss = 0.0

    pbar = tqdm(dataloader, desc=f"Epoch {epoch}/{config.epochs}")
    for x_t, x_prev, degraded_images, timesteps in pbar:
        x_t = x_t.to(config.device, non_blocking=True)
        x_prev = x_prev.to(config.device, non_blocking=True)
        degraded_images = degraded_images.to(config.device, non_blocking=True)
        timesteps = timesteps.to(config.device, non_blocking=True)

        model_input = torch.cat([degraded_images, x_t], dim=1)

        optimizer.zero_grad(set_to_none=True)

        use_amp = config.device == "cuda"
        with torch.amp.autocast('cuda', enabled=use_amp):
            predictions = model(model_input, timesteps)
            loss = torch.nn.functional.mse_loss(predictions, x_prev)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item()
        pbar.set_postfix({"Loss": f"{loss.item():.4f}"})

    return running_loss / max(len(dataloader), 1)


def main() -> None:
    config = TrainConfig()
    print(f"Using device: {config.device}")

    if config.device == "cuda":
        torch.set_float32_matmul_precision('high')

    # יצירת תצוגה מקדימה של 50 צעדים (בקפיצות של 10) לפני האימון
    save_aging_preview(config, num_samples=3, step_interval=10)

    model = DiffusionViT(
        image_size=config.image_size,
        patch_size=8,
        in_channels=6,
        out_channels=3,
        dim=512,
        depth=8,
        num_heads=8
    ).to(config.device)

    optimizer = AdamW(model.parameters(), lr=config.learning_rate, weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs, eta_min=1e-6)
    scaler = torch.amp.GradScaler('cuda', enabled=(config.device == "cuda"))

    dataloader = build_dataloader(config)
    print(f"Loaded {len(dataloader.dataset)} clean images. Starting staggered probabilistic chaining training (50 steps)...")

    for epoch in range(1, config.epochs + 1):
        loss = train_one_epoch(model, dataloader, optimizer, scaler, config, epoch)
        scheduler.step()
        print(f"Epoch {epoch}/{config.epochs} Complete - Loss: {loss:.6f}")

        if epoch % 3 == 0 or epoch == config.epochs:
            torch.save(model.state_dict(), f"dit_cold_diffusion_epoch_{epoch}.pth")


if __name__ == "__main__":
    main()