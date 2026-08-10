"""Reusable neural network blocks for cold diffusion models."""

import math

import torch
from torch import nn


class SinusoidalTimeEmbedding(nn.Module):
    """Standard sinusoidal timestep embedding used in diffusion models."""

    def __init__(self, embedding_dim: int) -> None:
        super().__init__()
        if embedding_dim % 2 != 0:
            raise ValueError("embedding_dim must be even for sinusoidal embeddings")
        self.embedding_dim = embedding_dim

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        half_dim = self.embedding_dim // 2
        exponent = -math.log(10_000) / (half_dim - 1)
        frequencies = torch.exp(
            torch.arange(half_dim, device=timesteps.device, dtype=torch.float32) * exponent
        )
        args = timesteps.float().unsqueeze(-1) * frequencies.unsqueeze(0)
        return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)


class MLP(nn.Module):
    """Simple MLP block used for projections and transformer feed-forward layers."""

    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
