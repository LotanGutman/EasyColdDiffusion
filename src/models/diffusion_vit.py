"""Vision Transformer denoiser for image cold diffusion experiments."""

import torch
from torch import nn

from .utils import MLP, SinusoidalTimeEmbedding


class TransformerBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.0, dropout: float = 0.0) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, int(dim * mlp_ratio), dim, dropout=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x)
        h, _ = self.attn(h, h, h, need_weights=False)
        x = x + h
        x = x + self.mlp(self.norm2(x))
        return x


class DiffusionViT(nn.Module):
    """A compact ViT denoiser with timestep conditioning."""

    def __init__(
        self,
        image_size: int = 64,
        patch_size: int = 8,
        in_channels: int = 3,
        dim: int = 512,
        depth: int = 8,
        num_heads: int = 8,
    ) -> None:
        super().__init__()
        if image_size % patch_size != 0:
            raise ValueError("image_size must be divisible by patch_size")

        self.patch_size = patch_size
        self.grid_size = image_size // patch_size
        self.num_patches = self.grid_size**2
        self.patch_dim = in_channels * patch_size * patch_size

        self.patch_embed = nn.Linear(self.patch_dim, dim)
        self.pos_embed = nn.Parameter(torch.randn(1, self.num_patches, dim) * 0.02)

        self.time_embed = nn.Sequential(
            SinusoidalTimeEmbedding(dim),
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
        )

        self.blocks = nn.ModuleList([TransformerBlock(dim, num_heads) for _ in range(depth)])
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, self.patch_dim)

    def _to_patches(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        p = self.patch_size
        x = x.view(b, c, h // p, p, w // p, p)
        x = x.permute(0, 2, 4, 3, 5, 1).contiguous()
        return x.view(b, -1, self.patch_dim)

    def _from_patches(self, patches: torch.Tensor) -> torch.Tensor:
        b = patches.shape[0]
        p = self.patch_size
        g = self.grid_size
        c = self.patch_dim // (p * p)
        x = patches.view(b, g, g, p, p, c)
        x = x.permute(0, 5, 1, 3, 2, 4).contiguous()
        return x.view(b, c, g * p, g * p)

    def forward(self, x: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        patches = self._to_patches(x)
        tokens = self.patch_embed(patches) + self.pos_embed
        time_cond = self.time_embed(timesteps).unsqueeze(1)
        tokens = tokens + time_cond

        for block in self.blocks:
            tokens = block(tokens)

        tokens = self.norm(tokens)
        patches = self.head(tokens)
        return self._from_patches(patches)
