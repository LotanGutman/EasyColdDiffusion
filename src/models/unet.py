"""U-Net style denoising network for image cold diffusion experiments."""

import torch
from torch import nn

from .utils import SinusoidalTimeEmbedding


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(8, out_channels),
            nn.SiLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(8, out_channels),
            nn.SiLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class DownBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv = ConvBlock(in_channels, out_channels)
        self.down = nn.Conv2d(out_channels, out_channels, kernel_size=4, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.conv(x)
        return features, self.down(features)


class UpBlock(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int) -> None:
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1)
        self.conv = ConvBlock(out_channels + skip_channels, out_channels)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class ColdDiffusionUNet(nn.Module):
    """A baseline U-Net architecture inspired by image diffusion implementations."""

    def __init__(self, in_channels: int = 3, base_channels: int = 64, time_dim: int = 256) -> None:
        super().__init__()
        self.time_embed = nn.Sequential(
            SinusoidalTimeEmbedding(time_dim),
            nn.Linear(time_dim, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )

        self.input_proj = nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1)
        self.down1 = DownBlock(base_channels, base_channels * 2)
        self.down2 = DownBlock(base_channels * 2, base_channels * 4)
        self.mid = ConvBlock(base_channels * 4, base_channels * 4)
        self.up2 = UpBlock(base_channels * 4, base_channels * 4, base_channels * 2)
        self.up1 = UpBlock(base_channels * 2, base_channels * 2, base_channels)
        self.output_proj = nn.Conv2d(base_channels, in_channels, kernel_size=1)

        self.time_to_channels = nn.ModuleList(
            [
                nn.Linear(time_dim, base_channels * 2),
                nn.Linear(time_dim, base_channels * 4),
                nn.Linear(time_dim, base_channels * 4),
            ]
        )

    def _add_time(self, x: torch.Tensor, time_embedding: torch.Tensor, projector: nn.Linear) -> torch.Tensor:
        t = projector(time_embedding).unsqueeze(-1).unsqueeze(-1)
        return x + t

    def forward(self, x: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        time_embedding = self.time_embed(timesteps)

        x = self.input_proj(x)
        skip1, x = self.down1(x)
        x = self._add_time(x, time_embedding, self.time_to_channels[0])

        skip2, x = self.down2(x)
        x = self._add_time(x, time_embedding, self.time_to_channels[1])

        x = self.mid(x)
        x = self._add_time(x, time_embedding, self.time_to_channels[2])

        x = self.up2(x, skip2)
        x = self.up1(x, skip1)
        return self.output_proj(x)
