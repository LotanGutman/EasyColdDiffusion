"""Inference and evaluation script for U-Net Cold Diffusion restoration model."""

import argparse
import glob
import os
import random
import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision import transforms

from src.models.unet import UNet

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def single_degradation_step(img_float: np.ndarray, paper_tex: np.ndarray, scratch_tex: np.ndarray) -> np.ndarray:
    """
    Applies a single micro-step of degradation.
    Optimized for MAX_T = 100 with Morphological Dilation.
    """
    # 1. Sepia fade
    sepia_matrix = np.array([
        [0.393, 0.769, 0.189],
        [0.349, 0.686, 0.168],
        [0.272, 0.534, 0.131],
    ])
    sepia_img = np.clip(cv2.transform(img_float, sepia_matrix), 0, 1.0)
    blended = (1.0 - 0.008) * img_float + 0.008 * sepia_img

    # 2. Paper texture
    if paper_tex is not None:
        multiply_blend = blended * paper_tex
        blended = (1.0 - 0.008) * blended + 0.008 * multiply_blend

    # 3. Scratch texture
    if scratch_tex is not None:
        scratch_gray = np.mean(scratch_tex, axis=-1, keepdims=True)
        black_point = 0.20
        scratch_clean = np.clip((scratch_gray - black_point) / (1.0 - black_point), 0.0, 1.0)

        kernel = np.ones((2, 2), np.float32)
        scratch_clean = cv2.dilate(scratch_clean, kernel, iterations=1)
        scratch_clean = np.expand_dims(scratch_clean, axis=-1)

        screen_blend = 1.0 - (1.0 - blended) * (1.0 - scratch_clean)
        blended = (1.0 - 0.01) * blended + 0.01 * screen_blend

    return np.clip(blended, 0.0, 1.0)


def load_texture(path: str) -> np.ndarray:
    """Loads and prepares a texture image supporting Unicode paths."""
    img_array = np.fromfile(path, np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    img = cv2.resize(img, (128, 128))
    return img.astype(np.float32) / 255.0


def test_unet(args: argparse.Namespace) -> None:
    if not args.show:
        matplotlib.use("Agg")

    test_dir = args.test_dir
    checkpoint_path = args.checkpoint
    textures_dir = args.textures_dir
    output_dir = args.output_dir
    max_t = args.max_t
    num_samples = args.num_samples

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    paper_files = sorted(
        glob.glob(os.path.join(textures_dir, "paper*.[jJ][pP][gG]"))
        + glob.glob(os.path.join(textures_dir, "paper*.[pP][nN][gG]"))
    )
    scratch_files = sorted(
        glob.glob(os.path.join(textures_dir, "scratch*.[jJ][pP][gG]"))
        + glob.glob(os.path.join(textures_dir, "scratch*.[pP][nN][gG]"))
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Testing on device: {device}")

    valid_papers = [load_texture(p) for p in paper_files if os.path.exists(p)]
    valid_scratches = [load_texture(p) for p in scratch_files if os.path.exists(p)]

    # ==========================================
    # Load Trained Model
    # ==========================================
    model = UNet().to(device)
    if os.path.exists(checkpoint_path):
        model.load_state_dict(torch.load(checkpoint_path, map_location=device), strict=False)
        model.eval()
        print(f"Successfully loaded model from {checkpoint_path}")
    else:
        print(f"Error: Model checkpoint not found at {checkpoint_path}")
        return

    transform = transforms.ToTensor()

    # ==========================================
    # Process Test Images
    # ==========================================
    if not os.path.exists(test_dir):
        print(f"Error: Test directory '{test_dir}' not found.")
        return

    test_files = [f for f in os.listdir(test_dir) if f.lower().endswith((".png", ".jpg", ".jpeg"))]
    if not test_files:
        print(f"No images found in '{test_dir}'!")
        return

    if num_samples is not None and num_samples < len(test_files):
        test_files = random.sample(test_files, num_samples)

    for idx, filename in enumerate(test_files):
        img_path = os.path.join(test_dir, filename)
        img_array = np.fromfile(img_path, np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

        h, w = img.shape[:2]
        min_dim = min(h, w)
        img = img[
            h // 2 - min_dim // 2 : h // 2 + min_dim // 2,
            w // 2 - min_dim // 2 : w // 2 + min_dim // 2,
        ]
        img = cv2.resize(img, (128, 128))

        x_0 = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

        paper_tex = random.choice(valid_papers) if valid_papers else None
        scratch_tex = random.choice(valid_scratches) if valid_scratches else None

        t_true = random.randint(5, max_t)

        x_t = x_0.copy()
        for _ in range(t_true):
            x_t = single_degradation_step(x_t, paper_tex, scratch_tex)

        x_t_tensor = transform(x_t).unsqueeze(0).to(device)
        t_input = torch.tensor([t_true], dtype=torch.float32).to(device)

        # ==========================================
        # Model Inference
        # ==========================================
        with torch.no_grad():
            pred_x_0 = model(x_t_tensor, t_input)

        restored_img = pred_x_0.squeeze().cpu().permute(1, 2, 0).numpy()

        # ==========================================
        # Visualization & Saving
        # ==========================================
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        fig.suptitle(f"Testing on {filename}", fontsize=16)

        axes[0].imshow(x_0)
        axes[0].set_title("Original (Clean)")
        axes[0].axis("off")

        axes[1].imshow(x_t)
        axes[1].set_title(f"Degraded\nTrue Steps (t): {t_true}")
        axes[1].axis("off")

        axes[2].imshow(restored_img)
        axes[2].set_title("Restored by Model")
        axes[2].axis("off")

        plt.tight_layout()

        if output_dir:
            save_path = os.path.join(output_dir, f"result_{os.path.splitext(filename)[0]}.png")
            plt.savefig(save_path, dpi=150)
            print(f"Saved result: {save_path}")

        if args.show:
            plt.show()
        else:
            plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Test Cold Diffusion U-Net Restoration Model")
    parser.add_argument(
        "--test_dir",
        type=str,
        default=os.path.join(PROJECT_ROOT, "assets", "test_images"),
        help="Directory with test images",
    )
    parser.add_argument(
        "--textures_dir",
        type=str,
        default=os.path.join(PROJECT_ROOT, "assets", "textures"),
        help="Directory with textures",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=os.path.join(PROJECT_ROOT, "trained_model.pth"),
        help="Path to trained model checkpoint",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=os.path.join(PROJECT_ROOT, "results"),
        help="Directory to save restoration comparison images",
    )
    parser.add_argument(
        "--max_t",
        type=int,
        default=100,
        help="Maximum degradation steps",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=5,
        help="Number of test images to evaluate",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display matplotlib interactive plots",
    )
    return parser.parse_args()


if __name__ == "__main__":
    test_unet(parse_args())
