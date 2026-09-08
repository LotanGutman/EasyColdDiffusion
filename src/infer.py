from dataclasses import dataclass
from pathlib import Path
import torch
from torchvision import transforms
from PIL import Image
from tqdm import tqdm

from src.models.diffusion_vit import DiffusionViT


@dataclass
class InferConfig:
    checkpoint_path: str = "dit_cold_diffusion_epoch_6.pth"
    input_dir: str = r"C:\Users\User\PyCharmMiscProject\deep_learning_project\degraded_output_images" #r"C:\Users\User\PyCharmMiscProject\deep_learning_project\test_pictures_deg"
    output_dir: str = r"C:\Users\User\PyCharmMiscProject\deep_learning_project\recreated_output_images"
    image_size: int = 128
    timesteps: int = 50
    sampling_steps: int = 50
    overlap_ratio: float = 0.5  # יחס החפיפה בין טאצ'ים (50% חפיפה)
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

            with torch.amp.autocast('cuda', enabled=use_amp):
                pred_next_state = model(model_input, t_tensor)

            x_t = pred_next_state.clamp(-1.0, 1.0)

    return x_t


def restore_large_image_tiled(model: torch.nn.Module, degraded_tensor: torch.Tensor,
                              config: InferConfig) -> torch.Tensor:
    """
    מבצע הסקה על תמונה (או חלק ממנה) תוך שימוש בחלונות חופפים ומיזוג מרחבי חלק (Hann Window)
    כדי למנוע תפרים ופגמים גיאומטריים בין ה-patches.
    """
    _, C, H, W = degraded_tensor.shape
    tile_size = config.image_size

    # אם התמונה בגודל המדויק של האימון, נריץ אותה ישירות בלי חפיפה מיותרת
    if H == tile_size and W == tile_size:
        return restore_single_image(model, degraded_tensor, config)

    stride = int(tile_size * (1 - config.overlap_ratio))

    # יצירת חלון משקולות דו-ממדי (Hann Window) להחלקה והקטנת משקל השוליים של כל טאץ'
    hann_1d = torch.hann_window(tile_size, periodic=False, device=config.device)
    window_2d = torch.outer(hann_1d, hann_1d)
    window_2d = window_2d.unsqueeze(0).unsqueeze(0)  # צורה: (1, 1, tile_size, tile_size)

    output_acc = torch.zeros((1, C, H, W), device=config.device)
    weight_acc = torch.zeros((1, 1, H, W), device=config.device)

    # בניית רשימת הצעדים (קואורדינטות התחלתיות של כל טאץ')
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
            # חילוץ הטאץ' הנוכחי
            tile = degraded_tensor[:, :, y:y + tile_size, x:x + tile_size]
            actual_h, actual_w = tile.shape[2], tile.shape[3]

            # במקרה שהטאץ' בשוליים קטן מהגודל המלא, נבצע מילוי (Padding) זמני
            if actual_h < tile_size or actual_w < tile_size:
                tile = torch.nn.functional.pad(tile, (0, tile_size - actual_w, 0, tile_size - actual_h), mode='reflect')

            # הרצת תהליך השחזור על הטאץ'
            restored_tile = restore_single_image(model, tile, config)

            # חזרה לגודל המקורי של הטאץ' אם בוצע Padding
            restored_tile = restored_tile[:, :, :actual_h, :actual_w]
            tile_window = window_2d[:, :, :actual_h, :actual_w]

            # צבירת הפיקסלים המשוקללים לתוך התמונה הגדולה
            output_acc[:, :, y:y + actual_h, x:x + actual_w] += restored_tile * tile_window
            weight_acc[:, :, y:y + actual_h, x:x + actual_w] += tile_window

    # נרמול התוצאה הסופית בסכום המשקולות כדי לשמור על בהירות אחידה
    restored_full = output_acc / torch.clamp(weight_acc, min=1e-8)
    return restored_full


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

    # טרנספורמציה לשמירה על טווח ערכים נכון בלי כפיית Resize גלובלי
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    to_pil = transforms.ToPILImage()

    for img_path in tqdm(image_paths, desc="Restoring Images"):
        try:
            degraded_img = Image.open(img_path).convert("RGB")
            degraded_tensor = transform(degraded_img).unsqueeze(0).to(config.device)

            # הרצת שחזור מבוסס Tiling וחפיפה
            restored_tensor = restore_large_image_tiled(model, degraded_tensor, config)

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
