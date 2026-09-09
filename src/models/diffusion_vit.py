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

    def __init__(self, channels: int = 3, hidden_dim: int = 32) -> None:
        super().__init__()
        # Encoder (Contracting path)
        self.enc1 = nn.Sequential(
            nn.Conv2d(channels, hidden_dim, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.SiLU()
        )
        self.pool1 = nn.MaxPool2d(2)  # 128 -> 64

        self.enc2 = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim * 2, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv2d(hidden_dim * 2, hidden_dim * 2, kernel_size=3, padding=1),
            nn.SiLU()
        )
        self.pool2 = nn.MaxPool2d(2)  # 64 -> 32

        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv2d(hidden_dim * 2, hidden_dim * 4, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv2d(hidden_dim * 4, hidden_dim * 2, kernel_size=3, padding=1),
            nn.SiLU()
        )

        # Decoder (Expanding path) with Skip Connections
        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.dec2 = nn.Sequential(
            nn.Conv2d(hidden_dim * 4, hidden_dim * 2, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv2d(hidden_dim * 2, hidden_dim, kernel_size=3, padding=1),
            nn.SiLU()
        )

        self.up1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.dec1 = nn.Sequential(
            nn.Conv2d(hidden_dim * 2, hidden_dim, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv2d(hidden_dim, channels, kernel_size=3, padding=1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder
        e1 = self.enc1(x)
        p1 = self.pool1(e1)

        e2 = self.enc2(p1)
        p2 = self.pool2(e2)

        # Bottleneck
        b = self.bottleneck(p2)

        # Decoder with Skip Connections
        u2 = self.up2(b)
        cat2 = torch.cat([u2, e2], dim=1)
        d2 = self.dec2(cat2)

        u1 = self.up1(d2)
        cat1 = torch.cat([u1, e1], dim=1)
        out = self.dec1(cat1)

        return out


class DiffusionViT(nn.Module):
    """A DiT-style denoiser with ConvStem, per-block adaptive timestep conditioning, and Mini-UNet refinement head."""

    def __init__(
            self,
            image_size: int = 128,
            patch_size: int = 8,
            in_channels: int = 6,
            out_channels: int = 3,
            dim: int = 512,
            depth: int = 8,
            num_heads: int = 8,
    ) -> None:
        super().__init__()
        if image_size % patch_size != 0:
            raise ValueError("image_size must be divisible by patch_size")

        self.patch_size = patch_size
        self.grid_size = image_size // patch_size
        self.num_patches = self.grid_size ** 2
        self.in_channels = in_channels
        self.out_channels = out_channels

        # --- CONVOLUTIONAL STEM ---
        self.conv_stem = ConvStem(in_channels=in_channels, hidden_dim=64)

        self.in_patch_dim = in_channels * patch_size * patch_size
        self.out_patch_dim = out_channels * patch_size * patch_size

        self.patch_embed = nn.Linear(self.in_patch_dim, dim)
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

        # Linear head producing the discrete patches
        self.head = nn.Linear(dim, self.out_patch_dim, bias=True)

        # --- MINI U-NET REFINEMENT HEAD ---
        self.unet_head = MiniUNetRefiner(channels=self.out_channels, hidden_dim=32)

        self._init_final_layer()

    def _init_final_layer(self) -> None:
        nn.init.constant_(self.final_adaLN[-1].weight, 0)
        nn.init.constant_(self.final_adaLN[-1].bias, 0)

        nn.init.constant_(self.head.weight, 0)
        nn.init.constant_(self.head.bias, 0)

        # Zero-initialization for the final U-Net layer to start as identity mapping
        final_conv = self.unet_head.dec1[-1]
        nn.init.constant_(final_conv.weight, 0)
        nn.init.constant_(final_conv.bias, 0)

    def _to_patches(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        p = self.patch_size
        x = x.view(b, c, h // p, p, w // p, p)
        x = x.permute(0, 2, 4, 3, 5, 1).contiguous()
        return x.view(b, -1, self.in_patch_dim)

    def _from_patches(self, patches: torch.Tensor) -> torch.Tensor:
        b = patches.shape[0]
        p = self.patch_size
        g = self.grid_size
        c = self.out_channels
        x = patches.view(b, g, g, p, p, c)
        x = x.permute(0, 5, 1, 3, 2, 4).contiguous()
        return x.view(b, c, g * p, g * p)

    def forward(self, x: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        # 1. Apply Convolutional Stem for local feature extraction & artifact smoothing
        x = self.conv_stem(x)

        patches = self._to_patches(x)
        tokens = self.patch_embed(patches) + self.pos_embed
        time_cond = self.time_embed(timesteps)

        for block in self.blocks:
            tokens = block(tokens, time_cond)

        shift, scale = [params.unsqueeze(1) for params in self.final_adaLN(time_cond).chunk(2, dim=-1)]
        tokens = self.final_norm(tokens) * (1 + scale) + shift

        # 2. Project to discrete patches and reconstruct image
        patches = self.head(tokens)
        img_discrete = self._from_patches(patches)

        # 3. Refine using Mini U-Net with a residual connection
        refined_img = img_discrete + self.unet_head(img_discrete)

        return refined_img
