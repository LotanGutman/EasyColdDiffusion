import os
import cv2
import numpy as np
import torch
import random
import matplotlib.pyplot as plt
from torchvision import transforms
import matplotlib
matplotlib.use('TkAgg')

# Import the model architecture
from model import UNet

def single_degradation_step(img_float, paper_tex, scratch_tex):
    """
    Applies a single micro-step of degradation.
    Optimized for MAX_T = 100 with Morphological Dilation.
    """
    # 1. Sepia fade
    sepia_matrix = np.array([
        [0.393, 0.769, 0.189],
        [0.349, 0.686, 0.168],
        [0.272, 0.534, 0.131]
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

def load_texture(path):
    """
    Loads and prepares a texture image.
    """
    img_array = np.fromfile(path, np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    img = cv2.resize(img, (128, 128))
    return img.astype(np.float32) / 255.0

def test_model():
    # ==========================================
    # Configurations
    # ==========================================
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    TEST_DIR = os.path.join(BASE_DIR, "test_images")

    # UPDATE THIS PATH to your saved checkpoint!
    CHECKPOINT_PATH = os.path.join(BASE_DIR, "checkpoints", "cold_diff_model_epoch_40.pth")

    PAPER_TEXTURES = [os.path.join(BASE_DIR, "textures", f"paper{i}.jpg") for i in range(1, 7)]
    SCRATCH_TEXTURES = [os.path.join(BASE_DIR, "textures", f"scratch{i}.jpg") for i in range(1, 7)]

    MAX_T = 100

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Testing on device: {device}")

    valid_papers = [load_texture(p) for p in PAPER_TEXTURES if os.path.exists(p)]
    valid_scratches = [load_texture(p) for p in SCRATCH_TEXTURES if os.path.exists(p)]

    # ==========================================
    # Load the Trained Model
    # ==========================================
    model = UNet().to(device)
    if os.path.exists(CHECKPOINT_PATH):
        # strict=False safely ignores missing/extra weights from previous versions
        model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device), strict=False)
        model.eval()
        print(f"Successfully loaded model from {CHECKPOINT_PATH}")
    else:
        print(f"Error: Model checkpoint not found at {CHECKPOINT_PATH}")
        return

    transform = transforms.ToTensor()

    # ==========================================
    # Process Test Images
    # ==========================================
    test_files = [f for f in os.listdir(TEST_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    if not test_files:
        print("No images found in test_images folder!")
        return

    for filename in test_files:
        img_path = os.path.join(TEST_DIR, filename)
        img_array = np.fromfile(img_path, np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

        h, w = img.shape[:2]
        min_dim = min(h, w)
        img = img[h // 2 - min_dim // 2: h // 2 + min_dim // 2, w // 2 - min_dim // 2: w // 2 + min_dim // 2]
        img = cv2.resize(img, (128, 128))

        x_0 = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

        paper_tex = random.choice(valid_papers) if valid_papers else None
        scratch_tex = random.choice(valid_scratches) if valid_scratches else None

        t_true = random.randint(5, MAX_T)

        x_t = x_0.copy()
        for _ in range(t_true):
            x_t = single_degradation_step(x_t, paper_tex, scratch_tex)

        x_t_tensor = transform(x_t).unsqueeze(0).to(device)
        t_input = torch.tensor([t_true], dtype=torch.float32).to(device)

        # ==========================================
        # Model Inference
        # ==========================================
        with torch.no_grad():
            # Model returns ONLY the restored image
            pred_x_0 = model(x_t_tensor, t_input)

        restored_img = pred_x_0.squeeze().cpu().permute(1, 2, 0).numpy()

        # ==========================================
        # Visualization
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
        plt.show()

if __name__ == "__main__":
    test_model()