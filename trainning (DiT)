from dataclasses import dataclass
from pathlib import Path
import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
from tqdm import tqdm

from src.models.diffusion_vit import DiffusionViT


@dataclass
class TrainConfig:
    data_dir_clean: str = r"C:\Users\User\Downloads\ffhq_128_70k_images"
    data_dir_degraded: str = r"C:\Users\User\PyCharmMiscProject\deep_learning_project\output"
    image_size: int = 128
    batch_size: int = 64
    num_workers: int = 2
    learning_rate: float = 0.5*1e-4
    epochs: int = 30
    timesteps: int = 1000
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


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

    def __len__(self):
        return len(self.clean_paths)

    def __getitem__(self, idx):
        clean_path = self.clean_paths[idx]
        degraded_path = self.degraded_dir / clean_path.name

        if not degraded_path.exists():
            raise FileNotFoundError(f"הקובץ המשובש התואם לא נמצא: {degraded_path}")

        clean_img = Image.open(clean_path).convert("RGB")
        degraded_img = Image.open(degraded_path).convert("RGB")

        return self.transform(clean_img), self.transform(degraded_img)


def build_dataloader(config: TrainConfig) -> DataLoader:
    dataset = ColdDiffusionDataset(
        clean_dir=config.data_dir_clean,
        degraded_dir=config.data_dir_degraded,
        image_size=config.image_size
    )
    return DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=True,
        drop_last=True
    )


def sample_timesteps(batch_size: int, max_timesteps: int, device: str) -> torch.Tensor:
    return torch.randint(low=0, high=max_timesteps, size=(batch_size,), device=device)


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
    for clean_images, degraded_images in pbar:
        clean_images = clean_images.to(config.device, non_blocking=True)
        degraded_images = degraded_images.to(config.device, non_blocking=True)
        b = clean_images.shape[0]

        # 1. דגימת דרגת t אקראית עבור כל דוגמה באצווה
        timesteps = sample_timesteps(b, config.timesteps, config.device)

        # 2. נרמול t לטווח [0, 1] לצורך יצירת הדרגרדציה הרציפה
        t_factor = timesteps.view(b, 1, 1, 1).float() / config.timesteps

        # 3. יצירת התמונה המשובשת בזמן t: x_t = (1-t)*x_0 + t*y_deg
        x_t = (1.0 - t_factor) * clean_images + t_factor * degraded_images

        # 4. הכנת ה-Input למודל: שרשור התמונה המשובשת הסופית + התמונה ברמה t
        model_input = torch.cat([degraded_images, x_t], dim=1)

        optimizer.zero_grad(set_to_none=True)

        use_amp = config.device == "cuda"
        with torch.amp.autocast('cuda', enabled=use_amp):
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


def main() -> None:
    config = TrainConfig()
    print(f"Using device: {config.device}")

    if config.device == "cuda":
        torch.set_float32_matmul_precision('high')

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
    print(f"Loaded {len(dataloader.dataset)} image pairs. Starting training...")

    print("\n--- בדיקת התאמת קבצים ב-Dataset ---")
    dataset_ref = dataloader.dataset
    for i in range(min(5, len(dataset_ref))):
        c_p = dataset_ref.clean_paths[i]
        d_p = dataset_ref.degraded_dir / c_p.name
        print(f"זוג {i + 1}:")
        print(f"  Clean:    {c_p}")
        print(f"  Degraded: {d_p}")
    print("-" * 40 + "\n")

    for epoch in range(1, config.epochs + 1):
        loss = train_one_epoch(model, dataloader, optimizer, scaler, config, epoch)
        scheduler.step()
        print(f"Epoch {epoch}/{config.epochs} Complete - Loss: {loss:.6f}")

        if epoch % 5 == 0 or epoch == config.epochs:
            torch.save(model.state_dict(), f"dit_cold_diffusion_epoch_{epoch}.pth")


if __name__ == "__main__":
    main()
