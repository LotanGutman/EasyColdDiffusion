from dataclasses import dataclass
from pathlib import Path
import torch
from torchvision import transforms
from PIL import Image
from tqdm import tqdm

from src.models.diffusion_vit import DiffusionViT


@dataclass
class InferConfig:
    checkpoint_path: str = "dit_cold_diffusion_epoch_10.pth"
    input_dir: str = r"C:\Users\User\PyCharmMiscProject\deep_learning_project\test_pictures_deg"
    output_dir: str = r"C:\Users\User\PyCharmMiscProject\deep_learning_project\test_pictures_recreate"
    image_size: int = 128
    timesteps: int = 1000
    sampling_steps: int = 50
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


def restore_single_image(model: torch.nn.Module, degraded_tensor: torch.Tensor, config: InferConfig) -> torch.Tensor:
    # מתחילים מ-x_T שהוא התמונה המשובשת לחלוטין
    x_t = degraded_tensor.clone()
    step_size = config.timesteps // config.sampling_steps
    time_steps = list(range(0, config.timesteps, step_size))[::-1]

    use_amp = config.device == "cuda"

    with torch.no_grad():
        for t_val in time_steps:
            t_tensor = torch.tensor([t_val], device=config.device)

            # 1. התאמת סדר השרשור בדיוק כמו ב-train.py: [degraded, x_t]
            model_input = torch.cat([degraded_tensor, x_t], dim=1)

            with torch.amp.autocast('cuda', enabled=use_amp):
                pred_clean = model(model_input, t_tensor)

            # קטימת הניבוי לטווח המורגל [-1, 1]
            pred_clean = pred_clean.clamp(-1.0, 1.0)

            if t_val > 0:
                t_next = max(0, t_val - step_size)

                # המרה לנורמליזציה [0, 1]
                t_norm = t_val / config.timesteps
                t_next_norm = t_next / config.timesteps

                # Cold Diffusion Sampling Algorithm:
                # x_{t_next} = x_t - D(x_0, t) + D(x_0, t_next)
                # D(x_0, t) = (1 - t) * x_0 + t * y_deg
                # מעבר אלגברי מפשט את השינוי ל: x_t + (t_next - t) * (y_deg - x_0)
                dt = t_next_norm - t_norm
                x_t = x_t + dt * (degraded_tensor - pred_clean)
                x_t = x_t.clamp(-1.0, 1.0)
            else:
                x_t = pred_clean

    return x_t


def main():
    config = InferConfig()
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
        num_heads=8
    ).to(config.device)

    checkpoint = torch.load(config.checkpoint_path, map_location=config.device, weights_only=True)
    model.load_state_dict(checkpoint)
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((config.image_size, config.image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    to_pil = transforms.ToPILImage()

    for img_path in tqdm(image_paths, desc="Restoring Images"):
        try:
            degraded_img = Image.open(img_path).convert("RGB")
            degraded_tensor = transform(degraded_img).unsqueeze(0).to(config.device)

            restored_tensor = restore_single_image(model, degraded_tensor, config)

            # ביטול הנורמליזציה מ-[-1, 1] ל-[0, 1]
            restored_tensor = (restored_tensor.squeeze(0).cpu() * 0.5 + 0.5).clamp(0.0, 1.0)
            restored_image = to_pil(restored_tensor)

            save_path = output_path / img_path.name
            restored_image.save(save_path)
        except Exception as e:
            print(f"\nError processing {img_path.name}: {e}")

    print(f"\nAll restored images saved successfully to {config.output_dir}")


if __name__ == "__main__":
    main()
