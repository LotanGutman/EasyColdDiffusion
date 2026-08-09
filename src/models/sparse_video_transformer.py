"""Sparse video diffusion transformer with optional Flash Linear Attention backend."""

import torch
from torch import nn

from .utils import MLP, SinusoidalTimeEmbedding

try:
    from fla.layers import MultiScaleRetention
except ImportError:  # pragma: no cover - fallback when FLA is unavailable
    MultiScaleRetention = None


class SparseVideoBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        if MultiScaleRetention is not None:
            self.attn = MultiScaleRetention(dim=dim, num_heads=num_heads)
            self.uses_fla = True
        else:
            self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
            self.uses_fla = False

        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, dim * 4, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x)
        if self.uses_fla:
            h = self.attn(h)
        else:
            h, _ = self.attn(h, h, h, need_weights=False)
        x = x + h
        x = x + self.mlp(self.norm2(x))
        return x


class SparseDiffusionVideoTransformer(nn.Module):
    """Video denoiser for cold diffusion with sparse transformer-style token mixing."""

    def __init__(
        self,
        frame_size: int = 64,
        patch_size: int = 8,
        frames: int = 8,
        in_channels: int = 3,
        dim: int = 512,
        depth: int = 8,
        num_heads: int = 8,
    ) -> None:
        super().__init__()
        if frame_size % patch_size != 0:
            raise ValueError("frame_size must be divisible by patch_size")

        self.patch_size = patch_size
        self.frames = frames
        self.grid = frame_size // patch_size
        self.patch_dim = in_channels * patch_size * patch_size
        self.num_tokens = frames * self.grid * self.grid

        self.patch_embed = nn.Linear(self.patch_dim, dim)
        self.pos_embed = nn.Parameter(torch.randn(1, self.num_tokens, dim) * 0.02)
        self.time_embed = nn.Sequential(
            SinusoidalTimeEmbedding(dim),
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
        )

        self.blocks = nn.ModuleList([SparseVideoBlock(dim, num_heads) for _ in range(depth)])
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, self.patch_dim)

    def _to_tokens(self, x: torch.Tensor) -> torch.Tensor:
        b, t, c, h, w = x.shape
        p = self.patch_size
        x = x.view(b, t, c, h // p, p, w // p, p)
        x = x.permute(0, 1, 3, 5, 4, 6, 2).contiguous()
        return x.view(b, -1, self.patch_dim)

    def _from_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        b = tokens.shape[0]
        p = self.patch_size
        g = self.grid
        t = self.frames
        c = self.patch_dim // (p * p)
        x = tokens.view(b, t, g, g, p, p, c)
        x = x.permute(0, 1, 6, 2, 4, 3, 5).contiguous()
        return x.view(b, t, c, g * p, g * p)

    def forward(self, x: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        tokens = self._to_tokens(x)
        tokens = self.patch_embed(tokens) + self.pos_embed
        tokens = tokens + self.time_embed(timesteps).unsqueeze(1)

        for block in self.blocks:
            tokens = block(tokens)

        tokens = self.norm(tokens)
        patches = self.head(tokens)
        return self._from_tokens(patches)
