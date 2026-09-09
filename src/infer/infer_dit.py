"""Inference module for DiT Cold Diffusion restoration model using patch tiling."""

import argparse
from dataclasses import dataclass
from pathlib import Path
import torch
from torchvision import transforms
from PIL import Image
from tqdm import tqdm

from src.models.diffusion_vit import DiffusionViT

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


@dataclass
class InferConfig:
    checkpoint_path: str = str(PROJECT_ROOT / "checkpoints" / "dit_cold_diffusion_epoch_6.pth")
    input_dir: str = str(PROJECT_ROOT / "data" / "degraded")
    output_dir: str = str(PROJECT_ROOT / "results" / "dit_restored")
    image_size: int = 128
    timesteps: int = 50
    sampling_steps: int = 50
    overlap_ratio: float = 0.5
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


def restore_single_image(model: torch.nn.Module, degraded_tensor: torch.Tensor, config: InferConfig) -> torch.Tensor:
    x_t = degraded_tensor.clone()
    step_size = 1
    time_steps = list(range(config.timesteps, 0, -step_size))
    use_amp = config.device == "cuda"

    with torch.no_grad():
        for t_val in time_steps:
            t_tensor = torch.tensor([t_val], device=config.device)
            model_input = torch.cat([degraded_tensor, x_t], dim=1)

            with torch.amp.autocast("cuda", enabled=use_amp):
                pred_next_state = model(model_input, t_tensor)

            x_t = pred_next_state.clamp(-1.0, 1.0)

    return x_t


def restore_large_image_tiled(
    model: torch.nn.Module,
    degraded_tensor: torch.Tensor,
    config: InferConfig,
) -> torch.Tensor:
    """Performs tiled inference with Hann window blending to eliminate patch boundary artifacts."""
    _, C, H, W = degraded_tensor.shape
    tile_size = config.image_size

    if H == tile_size and W == tile_size:
        return restore_single_image(model, degraded_tensor, config)

    stride = int(tile_size * (1 - config.overlap_ratio))

    hann_1d = torch.hann_window(tile_size, periodic=False, device=config.device)
    window_2d = torch.outer(hann_1d, hann_1d).unsqueeze(0).unsqueeze(0)

    output_acc = torch.zeros((1, C, H, W), device=config.device)
    weight_acc = torch.zeros((1, 1, H, W), device=config.device)

    h_steps = list(range(0, H - tile_size + 1, stride))
    if h_steps and h_steps[-1] + tile_size < H:
        h_steps.append(H - tile_size)
    if not h_steps:
        h_steps = [0]

    w_steps = list(range(0, W - tile_size + 1, stride))
    if w_steps and w_steps[-1] + tile_size < W:
        w_steps.append(W - tile_size)
    if not w_steps:
        w_steps = [0]

    for y in h_steps:
        for x in w_steps:
            tile = degraded_tensor[:, :, y : y + tile_size, x : x + tile_size]
            actual_h, actual_w = tile.shape[2], tile.shape[3]

            if actual_h < tile_size or actual_w < tile_size:
                tile = torch.nn.functional.pad(
                    tile, (0, tile_size - actual_w, 0, tile_size - actual_h), mode="reflect"
                )

            restored_tile = restore_single_image(model, tile, config)
            restored_tile = restored_tile[:, :, :actual_h, :actual_w]
            tile_window = window_2d[:, :, :actual_h, :actual_w]

            output_acc[:, :, y : y + actual_h, x : x + actual_w] += restored_tile * tile_window
            weight_acc[:, :, y : y + actual_h, x : x + actual_w] += tile_window

    restored_full = output_acc / torch.clamp(weight_acc, min=1e-8)
    return restored_full


def run_infer_dit(config: InferConfig) -> None:
    print(f"Using device: {config.device}")

    input_path = Path(config.input_dir)
    output_path = Path(config.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    valid_exts = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}
    image_paths = sorted([
        p for p in input_path.rglob("*") if p.suffix in valid_exts and p.is_file()
    ])

    if not image_paths:
        print(f"No images found in {config.input_dir}")
        return

    print(f"Found {len(image_paths)} images to restore.")

    model = DiffusionViT(
        image_size=config.image_size,
        patch_size=8,
        in_channels=6,
        out_channels=3,
        dim=512,
        depth=8,
        num_heads=8,
    ).to(config.device)

    if Path(config.checkpoint_path).exists():
        checkpoint = torch.load(config.checkpoint_path, map_location=config.device, weights_only=True)
        model.load_state_dict(checkpoint)
        print(f"Loaded checkpoint from: {config.checkpoint_path}")
    else:
        print(f"Warning: Checkpoint not found at {config.checkpoint_path}")

    model.eval()

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    to_pil = transforms.ToPILImage()

    for img_path in tqdm(image_paths, desc="Restoring Images"):
        try:
            degraded_img = Image.open(img_path).convert("RGB")
            degraded_tensor = transform(degraded_img).unsqueeze(0).to(config.device)

            restored_tensor = restore_large_image_tiled(model, degraded_tensor, config)
            restored_tensor = (restored_tensor.squeeze(0).cpu() * 0.5 + 0.5).clamp(0.0, 1.0)
            restored_image = to_pil(restored_tensor)

            save_path = output_path / img_path.name
            restored_image.save(save_path)
        except Exception as e:
            print(f"\nError processing {img_path.name}: {e}")

    print(f"\nAll restored images saved successfully to {config.output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Infer DiT Cold Diffusion Restoration")
    parser.add_argument("--checkpoint", type=str, default=str(PROJECT_ROOT / "checkpoints" / "dit_cold_diffusion_epoch_6.pth"))
    parser.add_argument("--input_dir", type=str, default=str(PROJECT_ROOT / "data" / "degraded"))
    parser.add_argument("--output_dir", type=str, default=str(PROJECT_ROOT / "results" / "dit_restored"))
    parser.add_argument("--image_size", type=int, default=128)
    parser.add_argument("--timesteps", type=int, default=50)
    args = parser.parse_args()

    cfg = InferConfig(
        checkpoint_path=args.checkpoint,
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        image_size=args.image_size,
        timesteps=args.timesteps,
    )
    run_infer_dit(cfg)
