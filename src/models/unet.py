"""U-Net architecture for Cold Diffusion Image Restoration."""

import math
import torch
import torch.nn as nn
from .utils import SinusoidalTimeEmbedding


class SinusoidalPositionEmbeddings(nn.Module):
    """Converts a single integer timestep into a high-dimensional vector representation."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, time: torch.Tensor) -> torch.Tensor:
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class ConvBlock(nn.Module):
    """A standard double convolution block that injects the time embedding."""

    def __init__(self, in_ch: int, out_ch: int, time_emb_dim: int):
        super().__init__()
        self.time_mlp = nn.Linear(time_emb_dim, out_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)
        self.norm1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1)
        self.norm2 = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        h = self.relu(self.norm1(self.conv1(x)))
        time_emb = self.relu(self.time_mlp(t))
        time_emb = time_emb[(...,) + (None,) * 2]
        h = h + time_emb
        h = self.relu(self.norm2(self.conv2(h)))
        return h


class UNet(nn.Module):
    """
    Main U-Net architecture for Cold Diffusion Image Restoration.
    Trained for photograph degradation restoration (sepia, paper texture, scratches).
    """

    def __init__(
        self,
        image_channels: int = 3,
        down_channels: tuple[int, ...] = (64, 128, 256, 512),
        up_channels: tuple[int, ...] = (512, 256, 128, 64),
        out_dim: int = 3,
        time_emb_dim: int = 64,
    ):
        super().__init__()
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.ReLU(),
        )

        self.conv0 = nn.Conv2d(image_channels, down_channels[0], kernel_size=3, padding=1)

        self.downs = nn.ModuleList([
            ConvBlock(down_channels[0], down_channels[1], time_emb_dim),
            ConvBlock(down_channels[1], down_channels[2], time_emb_dim),
            ConvBlock(down_channels[2], down_channels[3], time_emb_dim),
        ])
        self.pool = nn.MaxPool2d(2)

        self.bottleneck = ConvBlock(down_channels[3], down_channels[3], time_emb_dim)

        self.ups = nn.ModuleList([
            ConvBlock(1024, up_channels[1], time_emb_dim),
            ConvBlock(512, up_channels[2], time_emb_dim),
            ConvBlock(256, up_channels[3], time_emb_dim),
        ])

        self.upconvs = nn.ModuleList([
            nn.ConvTranspose2d(down_channels[3], down_channels[3], kernel_size=2, stride=2),
            nn.ConvTranspose2d(up_channels[1], up_channels[1], kernel_size=2, stride=2),
            nn.ConvTranspose2d(up_channels[2], up_channels[2], kernel_size=2, stride=2),
        ])

        self.final_conv = nn.Conv2d(up_channels[-1], out_dim, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
        t = self.time_mlp(timestep)
        x = self.conv0(x)

        skip_connections = []
        for down in self.downs:
            x = down(x, t)
            skip_connections.append(x)
            x = self.pool(x)

        x = self.bottleneck(x, t)

        skip_connections = skip_connections[::-1]
        for i in range(len(self.ups)):
            x = self.upconvs[i](x)
            skip = skip_connections[i]
            x = torch.cat((x, skip), dim=1)
            x = self.ups[i](x, t)

        x = self.final_conv(x)
        return self.sigmoid(x)


# -------------------------------------------------------------
# Scaffold ColdDiffusionUNet for modular compatibility
# -------------------------------------------------------------
class _ScaffoldConvBlock(nn.Module):
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


class _ScaffoldDownBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv = _ScaffoldConvBlock(in_channels, out_channels)
        self.down = nn.Conv2d(out_channels, out_channels, kernel_size=4, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.conv(x)
        return features, self.down(features)


class _ScaffoldUpBlock(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int) -> None:
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1)
        self.conv = _ScaffoldConvBlock(out_channels + skip_channels, out_channels)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class ColdDiffusionUNet(nn.Module):
    """Baseline scaffold U-Net architecture with sinusoidal time embeddings."""

    def __init__(self, in_channels: int = 3, base_channels: int = 64, time_dim: int = 256) -> None:
        super().__init__()
        self.time_embed = nn.Sequential(
            SinusoidalTimeEmbedding(time_dim),
            nn.Linear(time_dim, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )

        self.input_proj = nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1)
        self.down1 = _ScaffoldDownBlock(base_channels, base_channels * 2)
        self.down2 = _ScaffoldDownBlock(base_channels * 2, base_channels * 4)
        self.mid = _ScaffoldConvBlock(base_channels * 4, base_channels * 4)
        self.up2 = _ScaffoldUpBlock(base_channels * 4, base_channels * 4, base_channels * 2)
        self.up1 = _ScaffoldUpBlock(base_channels * 2, base_channels * 2, base_channels)
        self.output_proj = nn.Conv2d(base_channels, in_channels, kernel_size=1)

        self.time_to_channels = nn.ModuleList([
            nn.Linear(time_dim, base_channels * 2),
            nn.Linear(time_dim, base_channels * 4),
            nn.Linear(time_dim, base_channels * 4),
        ])

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
