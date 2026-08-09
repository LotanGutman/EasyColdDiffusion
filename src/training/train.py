"""Training entrypoint for EasyColdDiffusion."""

from dataclasses import dataclass

import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader

from src.models import ColdDiffusionUNet


@dataclass
class TrainConfig:
    image_size: int = 64
    batch_size: int = 8
    learning_rate: float = 1e-4
    epochs: int = 10
    timesteps: int = 1000
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


def build_dataloader(config: TrainConfig) -> DataLoader:
    """TODO: implement dataset + dataloader for training data."""
    raise NotImplementedError("TODO: data loading will be implemented later")


def sample_timesteps(batch_size: int, max_timesteps: int, device: str) -> torch.Tensor:
    return torch.randint(low=0, high=max_timesteps, size=(batch_size,), device=device)


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    config: TrainConfig,
) -> float:
    model.train()
    running_loss = 0.0

    for clean_images in dataloader:
        clean_images = clean_images.to(config.device)
        timesteps = sample_timesteps(clean_images.shape[0], config.timesteps, config.device)

        # Placeholder cold-diffusion objective until degradation/reconstruction pipeline is added.
        predictions = model(clean_images, timesteps)
        loss = torch.nn.functional.mse_loss(predictions, clean_images)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    return running_loss / max(len(dataloader), 1)


def main() -> None:
    config = TrainConfig()
    model = ColdDiffusionUNet(in_channels=3, base_channels=64).to(config.device)
    optimizer = AdamW(model.parameters(), lr=config.learning_rate)

    dataloader = build_dataloader(config)

    for epoch in range(1, config.epochs + 1):
        loss = train_one_epoch(model, dataloader, optimizer, config)
        print(f"epoch={epoch} loss={loss:.6f}")


if __name__ == "__main__":
    main()
