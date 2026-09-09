"""Interactive Tkinter GUI tool for visual Cold Diffusion photo degradation."""

import glob
import os
import random
import tkinter as tk
from tkinter import filedialog
import cv2
import numpy as np
from PIL import Image, ImageTk

# ==========================================
# CONFIGURATION
# ==========================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEXTURES_DIR = os.path.join(BASE_DIR, "assets", "textures")
TEST_IMAGES_DIR = os.path.join(BASE_DIR, "assets", "test_images")

PAPER_TEXTURES = sorted(
    glob.glob(os.path.join(TEXTURES_DIR, "paper*.[jJ][pP][gG]"))
    + glob.glob(os.path.join(TEXTURES_DIR, "paper*.[pP][nN][gG]"))
)
SCRATCH_TEXTURES = sorted(
    glob.glob(os.path.join(TEXTURES_DIR, "scratch*.[jJ][pP][gG]"))
    + glob.glob(os.path.join(TEXTURES_DIR, "scratch*.[pP][nN][gG]"))
)

MAX_T = 100


class InteractiveAgingTool:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Cold Diffusion - Interactive Degradation Tool")
        self.root.geometry("800x800")
        self.root.configure(bg="#2b2b2b")

        self.orig_img = None
        self.paper_texture = None
        self.scratch_texture = None

        # Load texture paths
        self.valid_papers = [p for p in PAPER_TEXTURES if os.path.exists(p)]
        self.valid_scratches = [p for p in SCRATCH_TEXTURES if os.path.exists(p)]

        # UI Variables
        self.use_paper = tk.BooleanVar(value=True)
        self.use_scratches = tk.BooleanVar(value=True)

        self.setup_ui()

    def setup_ui(self) -> None:
        control_frame = tk.Frame(self.root, bg="#2b2b2b")
        control_frame.pack(pady=20)

        btn_load = tk.Button(
            control_frame,
            text="Load Image",
            command=self.load_image,
            bg="#4CAF50",
            fg="white",
            font=("Arial", 12, "bold"),
            padx=10,
        )
        btn_load.grid(row=0, column=0, rowspan=2, padx=20)

        chk_paper = tk.Checkbutton(
            control_frame,
            text="Apply Paper Texture",
            variable=self.use_paper,
            command=self.update_view,
            bg="#2b2b2b",
            fg="white",
            selectcolor="#444444",
            font=("Arial", 10),
        )
        chk_paper.grid(row=0, column=1, sticky="w", padx=10)

        chk_scratches = tk.Checkbutton(
            control_frame,
            text="Apply Scratches",
            variable=self.use_scratches,
            command=self.update_view,
            bg="#2b2b2b",
            fg="white",
            selectcolor="#444444",
            font=("Arial", 10),
        )
        chk_scratches.grid(row=1, column=1, sticky="w", padx=10)

        self.slider = tk.Scale(
            self.root,
            from_=0,
            to=MAX_T,
            orient=tk.HORIZONTAL,
            command=lambda val: self.update_view(),
            length=600,
            label="Degradation Step (t)",
            bg="#2b2b2b",
            fg="white",
            highlightthickness=0,
            font=("Arial", 12),
        )
        self.slider.pack(pady=10)

        self.lbl_img = tk.Label(
            self.root,
            text="Load an image to start...",
            bg="gray",
            fg="white",
            font=("Arial", 14),
        )
        self.lbl_img.pack(pady=20, expand=True)

    def load_texture(self, path: str) -> np.ndarray:
        """Loads and prepares a texture image supporting Unicode/multilingual paths."""
        img_array = np.fromfile(path, np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        img = cv2.resize(img, (128, 128))
        return img.astype(np.float32) / 255.0

    def load_image(self) -> None:
        """Opens file dialog, loads image, and prepares random textures for it."""
        initial_dir = TEST_IMAGES_DIR if os.path.exists(TEST_IMAGES_DIR) else BASE_DIR
        path = filedialog.askopenfilename(
            initialdir=initial_dir,
            filetypes=[("Image Files", "*.png;*.jpg;*.jpeg")]
        )
        if not path:
            return

        img_array = np.fromfile(path, np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

        h, w = img.shape[:2]
        min_dim = min(h, w)
        img = img[
            h // 2 - min_dim // 2 : h // 2 + min_dim // 2,
            w // 2 - min_dim // 2 : w // 2 + min_dim // 2,
        ]
        self.orig_img = cv2.resize(img, (128, 128))

        if self.valid_papers and self.valid_scratches:
            self.paper_texture = self.load_texture(random.choice(self.valid_papers))
            self.scratch_texture = self.load_texture(random.choice(self.valid_scratches))
        else:
            print("Error: Textures missing from 'assets/textures' folder!")
            return

        self.slider.set(0)
        self.update_view()

    def single_degradation_step(self, img_float: np.ndarray) -> np.ndarray:
        """
        Applies a single micro-step of degradation.
        Includes morphological dilation so cracks physically widen over time.
        """
        # 1. Sepia fade
        sepia_matrix = np.array([
            [0.393, 0.769, 0.189],
            [0.349, 0.686, 0.168],
            [0.272, 0.534, 0.131],
        ])
        sepia_img = np.clip(cv2.transform(img_float, sepia_matrix), 0, 1.0)
        blended = (1.0 - 0.008) * img_float + 0.008 * sepia_img

        # 2. Paper texture (0.8%)
        if self.use_paper.get() and self.paper_texture is not None:
            multiply_blend = blended * self.paper_texture
            blended = (1.0 - 0.008) * blended + 0.008 * multiply_blend

        # 3. Scratch texture (1.0%) with spatial widening (Dilation)
        if self.use_scratches.get() and self.scratch_texture is not None:
            scratch_gray = np.mean(self.scratch_texture, axis=-1, keepdims=True)
            black_point = 0.20
            scratch_clean = np.clip(
                (scratch_gray - black_point) / (1.0 - black_point), 0.0, 1.0
            )

            # --- Morphological Dilation ---
            kernel = np.ones((2, 2), np.float32)
            scratch_clean = cv2.dilate(scratch_clean, kernel, iterations=1)
            scratch_clean = np.expand_dims(scratch_clean, axis=-1)

            screen_blend = 1.0 - (1.0 - blended) * (1.0 - scratch_clean)
            blended = (1.0 - 0.01) * blended + 0.01 * screen_blend

        return np.clip(blended, 0.0, 1.0)

    def update_view(self, *args) -> None:
        """Applies the recursive loop based on slider value and checkbox states."""
        if self.orig_img is None:
            return

        t = self.slider.get()

        if t == 0:
            self.show_img(self.orig_img)
            return

        x_t = cv2.cvtColor(self.orig_img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

        for _ in range(t):
            x_t = self.single_degradation_step(x_t)

        final_img_bgr = cv2.cvtColor((x_t * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
        self.show_img(final_img_bgr)

    def show_img(self, img_bgr: np.ndarray) -> None:
        """Displays the image in the GUI."""
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img_rgb).resize((512, 512), Image.Resampling.LANCZOS)
        tk_img = ImageTk.PhotoImage(pil_img)
        self.lbl_img.config(image=tk_img, text="")
        self.lbl_img.image = tk_img


if __name__ == "__main__":
    root = tk.Tk()
    app = InteractiveAgingTool(root)
    root.mainloop()
