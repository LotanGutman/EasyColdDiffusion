"""Vision Transformer denoiser for image cold diffusion experiments."""

import torch
from torch import nn

from .utils import MLP, SinusoidalTimeEmbedding


class DiTBlock(nn.Module):
    """A DiT block with adaptive LayerNorm modulation and residual gating."""

    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.0, dropout: float = 0.0) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.mlp = MLP(dim, int(dim * mlp_ratio), dim, dropout=dropout)
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(dim, 6 * dim, bias=True))
        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.constant_(self.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.adaLN_modulation[-1].bias, 0)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        shifts_scales_gates = self.adaLN_modulation(cond).chunk(6, dim=-1)
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = [
            params.unsqueeze(1) for params in shifts_scales_gates
        ]

        h = self.norm1(x) * (1 + scale_msa) + shift_msa
        h, _ = self.attn(h, h, h, need_weights=False)
        x = x + gate_msa * h

        h = self.norm2(x) * (1 + scale_mlp) + shift_mlp
        x = x + gate_mlp * self.mlp(h)
        return x


class DiffusionViT(nn.Module):
    """A DiT-style denoiser with per-block adaptive timestep conditioning."""

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
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        self.time_embed = nn.Sequential(
            SinusoidalTimeEmbedding(dim),
            nn.Linear(dim, dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
        )

        self.blocks = nn.ModuleList([DiTBlock(dim, num_heads) for _ in range(depth)])
        self.final_norm = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.final_adaLN = nn.Sequential(nn.SiLU(), nn.Linear(dim, 2 * dim, bias=True))
        self.head = nn.Linear(dim, self.patch_dim, bias=True)
        self._init_final_layer()

    def _init_final_layer(self) -> None:
        nn.init.constant_(self.final_adaLN[-1].weight, 0)
        nn.init.constant_(self.final_adaLN[-1].bias, 0)
        nn.init.constant_(self.head.weight, 0)
        nn.init.constant_(self.head.bias, 0)

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
        time_cond = self.time_embed(timesteps)

        for block in self.blocks:
            tokens = block(tokens, time_cond)

        shift, scale = [params.unsqueeze(1) for params in self.final_adaLN(time_cond).chunk(2, dim=-1)]
        tokens = self.final_norm(tokens) * (1 + scale) + shift
        patches = self.head(tokens)
        return self._from_patches(patches)
