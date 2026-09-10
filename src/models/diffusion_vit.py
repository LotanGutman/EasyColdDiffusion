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


class ConvStem(nn.Module):
    """A lightweight convolutional stem with residual connection to inject local inductive bias before patch embedding."""

    def __init__(self, in_channels: int, hidden_dim: int = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, kernel_size=3, stride=1, padding=1),
            nn.SiLU(),
            nn.Conv2d(hidden_dim, in_channels, kernel_size=3, stride=1, padding=1),
            nn.SiLU()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


class MiniUNetRefiner(nn.Module):
    """A lightweight U-Net refinement head to smooth out grid artifacts and repair cracks."""

@@ -105,7 +121,7 @@


class DiffusionViT(nn.Module):
    """A DiT-style denoiser with per-block adaptive timestep conditioning and Mini-UNet refinement head."""
    """A DiT-style denoiser with ConvStem, per-block adaptive timestep conditioning, and Mini-UNet refinement head."""

    def __init__(
            self,
@@ -127,6 +143,9 @@
        self.in_channels = in_channels
        self.out_channels = out_channels

        # --- CONVOLUTIONAL STEM ---
        self.conv_stem = ConvStem(in_channels=in_channels, hidden_dim=64)

        self.in_patch_dim = in_channels * patch_size * patch_size
        self.out_patch_dim = out_channels * patch_size * patch_size

@@ -182,6 +201,9 @@
        return x.view(b, c, g * p, g * p)

    def forward(self, x: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        # 1. Apply Convolutional Stem for local feature extraction & artifact smoothing
        x = self.conv_stem(x)

        patches = self._to_patches(x)
        tokens = self.patch_embed(patches) + self.pos_embed
        time_cond = self.time_embed(timesteps)
@@ -192,11 +214,11 @@
        shift, scale = [params.unsqueeze(1) for params in self.final_adaLN(time_cond).chunk(2, dim=-1)]
        tokens = self.final_norm(tokens) * (1 + scale) + shift

        # 1. Project to discrete patches and reconstruct image
        # 2. Project to discrete patches and reconstruct image
        patches = self.head(tokens)
        img_discrete = self._from_patches(patches)

        # 2. Refine using Mini U-Net with a residual connection
        # 3. Refine using Mini U-Net with a residual connection
        refined_img = img_discrete + self.unet_head(img_discrete)

        return refined_img
