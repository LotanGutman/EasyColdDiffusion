"""Interactive Tkinter GUI for EasyColdDiffusion Image Aging and Restoration."""

from __future__ import annotations

import glob

import io
import os
import random
import threading
import cv2
import numpy as np
from PIL import Image, ImageTk, ImageGrab
import torch
import torchvision.transforms as transforms
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

# Optional drag-and-drop support
try:
    from tkinterdnd2 import DND_FILES, TkinterDnD
    HAS_DND = True
except ImportError:
    HAS_DND = False

from src.models.unet import UNet
from src.models.step_estimator import StepAndDegradationEstimator, estimate_damage_heuristic
from src.infer.convergence import CauchyConvergenceMonitor


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEFAULT_CHECKPOINT = os.path.join(PROJECT_ROOT, "trained_model.pth")
TEXTURES_DIR = os.path.join(PROJECT_ROOT, "assets", "textures")
TEST_IMAGES_DIR = os.path.join(PROJECT_ROOT, "assets", "test_images")

MODEL_SIZE = (128, 128)
DISPLAY_SIZE = (260, 260)


class ColdDiffusionGUI:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("EasyColdDiffusion - Interactive Aging & Restoration Lab")
        self.root.geometry("1180x820")
        self.root.minsize(1050, 750)
        self.root.configure(bg="#1e1e24")

        # PyTorch Setup
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"

        # State
        self.checkpoint_path = DEFAULT_CHECKPOINT
        self.model = None
        self.transform = transforms.ToTensor()

        self.loaded_image_pil = None
        self.degraded_image_pil = None
        self.restored_image_pil = None

        self.paper_texture = None
        self.scratch_texture = None
        self.auto_restore_timer = None

        # Convergence & Step Estimator Setup
        self.estimator_path = os.path.join(PROJECT_ROOT, "checkpoints", "step_estimator.pth")
        self.estimator = None
        self.convergence_monitor = CauchyConvergenceMonitor(tolerance=0.008, patience=1, min_steps=2)

        # GPU Degradation Operator Setup for Cold Diffusion Algorithm 2
        self.sepia_matrix = torch.tensor([
            [0.393, 0.769, 0.189],
            [0.349, 0.686, 0.168],
            [0.272, 0.534, 0.131],
        ], dtype=torch.float32, device=self.device)
        self.gpu_paper_tensor = None
        self.gpu_scratch_tensor = None



        # Load Textures
        self.paper_files = sorted(
            glob.glob(os.path.join(TEXTURES_DIR, "paper*.[jJ][pP][gG]"))
            + glob.glob(os.path.join(TEXTURES_DIR, "paper*.[pP][nN][gG]"))
        )
        self.scratch_files = sorted(
            glob.glob(os.path.join(TEXTURES_DIR, "scratch*.[jJ][pP][gG]"))
            + glob.glob(os.path.join(TEXTURES_DIR, "scratch*.[pP][nN][gG]"))
        )
        self.valid_papers = [self._load_texture_array(p) for p in self.paper_files if os.path.exists(p)]
        self.valid_scratches = [self._load_texture_array(p) for p in self.scratch_files if os.path.exists(p)]
        if self.valid_papers:
            self.paper_texture = self.valid_papers[0]
        if self.valid_scratches:
            self.scratch_texture = self.valid_scratches[0]
        self._update_gpu_textures()


        # Sample images
        self.sample_images = sorted(glob.glob(os.path.join(TEST_IMAGES_DIR, "*.*")))

        # Build UI
        self.setup_styles()
        self.build_ui()

        # Keyboard shortcuts
        self.root.bind("<Control-v>", lambda e: self.paste_from_clipboard())
        self.root.bind("<Control-V>", lambda e: self.paste_from_clipboard())

        # Auto-load model & estimator
        self.load_model(self.checkpoint_path)
        self.load_estimator()


        # Load a default sample image if available
        if self.sample_images:
            self.load_image_from_path(self.sample_images[0])

    def _load_texture_array(self, path: str) -> np.ndarray:
        img_array = np.fromfile(path, np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        img = cv2.resize(img, MODEL_SIZE)
        return img.astype(np.float32) / 255.0

    def setup_styles(self):
        self.style = ttk.Style()
        try:
            self.style.theme_use("clam")
        except Exception:
            pass
        self.style.configure("TFrame", background="#1e1e24")
        self.style.configure("TLabelframe", background="#282830", foreground="#ffffff")
        self.style.configure("TLabelframe.Label", background="#282830", foreground="#61afef", font=("Segoe UI", 10, "bold"))
        self.style.configure("TLabel", background="#1e1e24", foreground="#e0e0e0", font=("Segoe UI", 9))

    def build_ui(self):
        # 1. Header Bar
        header = tk.Frame(self.root, bg="#1e1e24", pady=6, padx=15)
        header.pack(fill=tk.X)

        title = tk.Label(
            header, text="❄️ EasyColdDiffusion Studio",
            font=("Segoe UI", 16, "bold"), fg="#61afef", bg="#1e1e24"
        )
        title.pack(side=tk.LEFT)

        self.device_badge = tk.Label(
            header, text=f"Device: {self.device_name}",
            font=("Segoe UI", 9, "bold"),
            fg="#98c379" if "cuda" in self.device.type else "#e5c07b",
            bg="#2c313a", padx=10, pady=3, relief=tk.RIDGE
        )
        self.device_badge.pack(side=tk.RIGHT)

        # 2. Checkpoint Bar
        ckpt_bar = tk.Frame(self.root, bg="#282830", padx=15, pady=6)
        ckpt_bar.pack(fill=tk.X, padx=15, pady=(0, 8))

        tk.Label(ckpt_bar, text="Model Checkpoint:", font=("Segoe UI", 9, "bold"), fg="#abb2bf", bg="#282830").pack(side=tk.LEFT, padx=(0, 5))
        self.ckpt_entry = tk.Entry(ckpt_bar, font=("Segoe UI", 9), bg="#1e1e24", fg="#abb2bf", relief=tk.FLAT)
        self.ckpt_entry.insert(0, self.checkpoint_path)
        self.ckpt_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)

        tk.Button(
            ckpt_bar, text="Browse...", command=self.browse_checkpoint,
            bg="#3e4451", fg="#ffffff", font=("Segoe UI", 8), relief=tk.GROOVE, padx=8
        ).pack(side=tk.LEFT, padx=3)

        tk.Button(
            ckpt_bar, text="Reload", command=lambda: self.load_model(self.ckpt_entry.get().strip()),
            bg="#4b5263", fg="#ffffff", font=("Segoe UI", 8), relief=tk.GROOVE, padx=8
        ).pack(side=tk.LEFT)

        # 3. Control Panel Frame
        panel = tk.LabelFrame(
            self.root, text=" Controls & Operations ", bg="#282830", fg="#61afef",
            font=("Segoe UI", 10, "bold"), padx=12, pady=8
        )
        panel.pack(fill=tk.X, padx=15, pady=(0, 8))

        # Row 1: Image Inputs
        row1 = tk.Frame(panel, bg="#282830")
        row1.pack(fill=tk.X, pady=(0, 6))

        tk.Button(
            row1, text="📂 Browse Image...", command=self.browse_image,
            bg="#61afef", fg="#1e1e24", font=("Segoe UI", 9, "bold"), padx=10, relief=tk.RAISED
        ).pack(side=tk.LEFT, padx=(0, 5))

        tk.Button(
            row1, text="📋 Paste (Ctrl+V)", command=self.paste_from_clipboard,
            bg="#98c379", fg="#1e1e24", font=("Segoe UI", 9, "bold"), padx=10, relief=tk.RAISED
        ).pack(side=tk.LEFT, padx=(0, 5))

        tk.Button(
            row1, text="🎲 Random Sample", command=self.load_random_sample,
            bg="#c678dd", fg="#ffffff", font=("Segoe UI", 9), padx=8, relief=tk.GROOVE
        ).pack(side=tk.LEFT, padx=(0, 8))

        self.path_entry = tk.Entry(row1, font=("Segoe UI", 9), bg="#1e1e24", fg="#abb2bf", relief=tk.SUNKEN)
        self.path_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=3)
        self.path_entry.bind("<Return>", lambda e: self.load_image_from_path(self.path_entry.get().strip()))

        tk.Button(
            row1, text="Load", command=lambda: self.load_image_from_path(self.path_entry.get().strip()),
            bg="#4b5263", fg="#ffffff", font=("Segoe UI", 9), padx=8, relief=tk.GROOVE
        ).pack(side=tk.LEFT, padx=(3, 0))

        # Row 2: Aging Degradation & Restoration
        row2 = tk.Frame(panel, bg="#282830")
        row2.pack(fill=tk.X, pady=(4, 0))

        # Degradation Controls (Left Box)
        deg_box = tk.Frame(row2, bg="#333842", padx=8, pady=6, relief=tk.GROOVE, bd=1)
        deg_box.pack(side=tk.LEFT, fill=tk.BOTH, padx=(0, 8))

        tk.Label(deg_box, text="Aging Degradation:", font=("Segoe UI", 8, "bold"), fg="#e5c07b", bg="#333842").pack(anchor=tk.W)

        opts_frame = tk.Frame(deg_box, bg="#333842")
        opts_frame.pack(fill=tk.X, pady=2)

        self.use_paper_var = tk.BooleanVar(value=True)
        self.use_scratch_var = tk.BooleanVar(value=True)

        tk.Checkbutton(
            opts_frame, text="Paper", variable=self.use_paper_var, command=self._on_toggle_aging,
            bg="#333842", fg="#e0e0e0", selectcolor="#1e1e24", activebackground="#333842", font=("Segoe UI", 8)
        ).pack(side=tk.LEFT, padx=2)

        tk.Checkbutton(
            opts_frame, text="Scratches", variable=self.use_scratch_var, command=self._on_toggle_aging,
            bg="#333842", fg="#e0e0e0", selectcolor="#1e1e24", activebackground="#333842", font=("Segoe UI", 8)
        ).pack(side=tk.LEFT, padx=2)

        tk.Button(
            opts_frame, text="🎲 Textures", command=self.randomize_textures,
            bg="#5c6370", fg="#ffffff", font=("Segoe UI", 7), relief=tk.GROOVE, padx=4
        ).pack(side=tk.LEFT, padx=4)

        tk.Button(
            opts_frame, text="➡️ Direct", command=self.use_as_direct_input,
            bg="#4b5263", fg="#ffffff", font=("Segoe UI", 7), relief=tk.GROOVE, padx=4
        ).pack(side=tk.LEFT, padx=(0, 2))

        tk.Button(
            opts_frame, text="⚡ Auto-Detect", command=lambda: self.estimate_and_apply_current(update_t=True),
            bg="#2c3e50", fg="#61afef", font=("Segoe UI", 7, "bold"), relief=tk.GROOVE, padx=4
        ).pack(side=tk.LEFT)

        # Slider Box (Middle - Aging Timestep t)
        slider_box = tk.Frame(row2, bg="#333842", padx=8, pady=6, relief=tk.GROOVE, bd=1)
        slider_box.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 8))

        slider_hdr = tk.Frame(slider_box, bg="#333842")
        slider_hdr.pack(fill=tk.X)
        tk.Label(slider_hdr, text="Degradation / Timestep (t):", font=("Segoe UI", 8, "bold"), fg="#61afef", bg="#333842").pack(side=tk.LEFT)
        self.lbl_t_val = tk.Label(slider_hdr, text="50", font=("Segoe UI", 8, "bold"), fg="#98c379", bg="#333842")
        self.lbl_t_val.pack(side=tk.RIGHT)

        self.t_slider = tk.Scale(
            slider_box, from_=0, to=100, orient=tk.HORIZONTAL, bg="#282830", fg="#abb2bf",
            highlightthickness=0, troughcolor="#1e1e24", activebackground="#61afef",
            command=self._on_slider_move
        )
        self.t_slider.set(50)
        self.t_slider.pack(fill=tk.X, pady=(2, 0))
        self.t_slider.bind("<ButtonRelease-1>", lambda e: self.schedule_auto_restore(delay_ms=50))

        # Restoration Box (Right - Denoise Steps Slider)
        rest_box = tk.Frame(row2, bg="#333842", padx=8, pady=6, relief=tk.GROOVE, bd=1)
        rest_box.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        steps_hdr = tk.Frame(rest_box, bg="#333842")
        steps_hdr.pack(fill=tk.X)
        tk.Label(steps_hdr, text="Denoise Steps:", font=("Segoe UI", 8, "bold"), fg="#e5c07b", bg="#333842").pack(side=tk.LEFT)
        self.lbl_steps_val = tk.Label(steps_hdr, text="1", font=("Segoe UI", 8, "bold"), fg="#98c379", bg="#333842")
        self.lbl_steps_val.pack(side=tk.RIGHT)

        self.auto_estimate_var = tk.BooleanVar(value=True)
        self.chk_auto = tk.Checkbutton(
            steps_hdr, text="⚡ Live Auto-Estimate", variable=self.auto_estimate_var,
            command=self._on_toggle_auto_estimate, bg="#333842", fg="#61afef",
            selectcolor="#1e1e24", activebackground="#333842", font=("Segoe UI", 8, "bold")
        )
        self.chk_auto.pack(side=tk.RIGHT, padx=(0, 6))

        self.steps_slider = tk.Scale(
            rest_box, from_=1, to=20, orient=tk.HORIZONTAL, bg="#282830", fg="#abb2bf",
            highlightthickness=0, troughcolor="#1e1e24", activebackground="#98c379",
            command=self._on_steps_slider_move
        )
        self.steps_slider.set(1)
        self.steps_slider.pack(fill=tk.X, pady=(2, 0))
        self.steps_slider.bind("<ButtonRelease-1>", lambda e: self.schedule_auto_restore(delay_ms=50))


        # 4. Display Panels (3 Side-by-Side)
        disp_frame = tk.Frame(self.root, bg="#1e1e24")
        disp_frame.pack(fill=tk.BOTH, expand=True, padx=15, pady=5)

        self.panel_orig, self.canvas_orig = self.create_image_card(disp_frame, "1. Original (Clean)", "No image loaded")
        self.panel_orig.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))

        self.panel_deg, self.canvas_deg = self.create_image_card(disp_frame, "2. Degraded (Model Input)", "Ready for aging")
        self.panel_deg.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=3)

        self.panel_rest, self.canvas_rest = self.create_image_card(disp_frame, "3. Restored Output", "Awaiting Denoising...")
        self.panel_rest.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(5, 0))

        # 5. Bottom Action Bar
        action_bar = tk.Frame(self.root, bg="#282830", padx=15, pady=6)
        action_bar.pack(fill=tk.X, padx=15, pady=(4, 4))

        tk.Button(
            action_bar, text="💾 Save Restored Image...", command=self.save_restored_image,
            bg="#61afef", fg="#1e1e24", font=("Segoe UI", 9, "bold"), padx=10, relief=tk.RAISED
        ).pack(side=tk.LEFT, padx=(0, 6))

        tk.Button(
            action_bar, text="📄 Copy Result to Clipboard", command=self.copy_restored_to_clipboard,
            bg="#c678dd", fg="#ffffff", font=("Segoe UI", 9, "bold"), padx=10, relief=tk.RAISED
        ).pack(side=tk.LEFT, padx=(0, 6))

        tk.Button(
            action_bar, text="🧹 Clear All", command=self.clear_all,
            bg="#e06c75", fg="#ffffff", font=("Segoe UI", 9), padx=8, relief=tk.GROOVE
        ).pack(side=tk.RIGHT)

        # 6. Status Bar
        self.status_bar = tk.Label(
            self.root, text="Ready. Drag & drop images, paste (Ctrl+V), or browse.",
            font=("Segoe UI", 9), fg="#98c379", bg="#18181c", anchor=tk.W, padx=15, pady=4
        )
        self.status_bar.pack(fill=tk.X, side=tk.BOTTOM)

        # Setup Drag & drop
        self.setup_dnd()

    def create_image_card(self, parent, title, placeholder):
        frame = tk.LabelFrame(
            parent, text=f" {title} ", bg="#282830", fg="#abb2bf",
            font=("Segoe UI", 9, "bold"), padx=6, pady=6
        )
        canvas = tk.Label(
            frame, text=placeholder, bg="#18181c", fg="#5c6370",
            font=("Segoe UI", 9, "italic"), relief=tk.SUNKEN, bd=1
        )
        canvas.pack(fill=tk.BOTH, expand=True)
        return frame, canvas

    def setup_dnd(self):
        if HAS_DND:
            try:
                self.root.drop_target_register(DND_FILES)
                self.root.dnd_bind('<<Drop>>', self.on_drop)
            except Exception:
                pass

    def on_drop(self, event):
        path = event.data.strip('{}').strip('"').strip("'")
        if os.path.exists(path):
            self.load_image_from_path(path)

    def set_status(self, message: str, error: bool = False):
        color = "#e06c75" if error else "#98c379"
        self.status_bar.config(text=message, fg=color)

    # ==========================================
    # MODEL MANAGEMENT
    # ==========================================
    def browse_checkpoint(self):
        path = filedialog.askopenfilename(
            title="Select Model Checkpoint",
            filetypes=[("PyTorch Model", "*.pth;*.pt;*.bin"), ("All Files", "*.*")]
        )
        if path:
            self.ckpt_entry.delete(0, tk.END)
            self.ckpt_entry.insert(0, path)
            self.load_model(path)

    def load_model(self, path: str):
        if not os.path.exists(path):
            self.set_status(f"Checkpoint not found: {path}", error=True)
            return

        self.set_status(f"Loading weights from {os.path.basename(path)}...")

        def _loader():
            try:
                model = UNet().to(self.device)
                ckpt = torch.load(path, map_location=self.device)
                if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
                    model.load_state_dict(ckpt["model_state_dict"], strict=False)
                elif isinstance(ckpt, dict):
                    model.load_state_dict(ckpt, strict=False)

                model.eval()
                self.model = model
                self.checkpoint_path = path
                self.root.after(0, lambda: self.set_status(f"Model loaded successfully on {self.device_name}!"))
            except Exception as e:
                self.root.after(0, lambda: self.set_status(f"Failed to load model: {e}", error=True))

        threading.Thread(target=_loader, daemon=True).start()

    def load_estimator(self):
        """Loads the trained Step & Degradation Estimator if available."""
        if not os.path.exists(self.estimator_path):
            self.estimator = None
            return

        try:
            estimator = StepAndDegradationEstimator().to(self.device)
            ckpt = torch.load(self.estimator_path, map_location=self.device)
            if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
                estimator.load_state_dict(ckpt["model_state_dict"])
            elif isinstance(ckpt, dict):
                estimator.load_state_dict(ckpt)
            estimator.eval()
            self.estimator = estimator
        except Exception:
            self.estimator = None


    # ==========================================
    # IMAGE INPUT & SELECTION
    # ==========================================
    def browse_image(self):
        path = filedialog.askopenfilename(
            title="Select Image",
            initialdir=TEST_IMAGES_DIR if os.path.exists(TEST_IMAGES_DIR) else PROJECT_ROOT,
            filetypes=[("Image Files", "*.png;*.jpg;*.jpeg;*.bmp;*.webp"), ("All Files", "*.*")]
        )
        if path:
            self.load_image_from_path(path)

    def load_random_sample(self):
        if not self.sample_images:
            self.set_status("No sample images found in assets/test_images!", error=True)
            return
        path = random.choice(self.sample_images)
        self.load_image_from_path(path)

    def load_image_from_path(self, path: str):
        if not path or not os.path.exists(path):
            self.set_status(f"File not found: {path}", error=True)
            return
        try:
            img_array = np.fromfile(path, np.uint8)
            img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            h, w = img.shape[:2]
            min_dim = min(h, w)
            img = img[h // 2 - min_dim // 2: h // 2 + min_dim // 2, w // 2 - min_dim // 2: w // 2 + min_dim // 2]
            img = cv2.resize(img, MODEL_SIZE)
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(img_rgb)

            self.path_entry.delete(0, tk.END)
            self.path_entry.insert(0, path)

            self.process_new_image(pil_img, source=os.path.basename(path))
        except Exception as e:
            self.set_status(f"Failed to load image: {e}", error=True)

    def paste_from_clipboard(self):
        try:
            clip_data = ImageGrab.grabclipboard()
            if isinstance(clip_data, Image.Image):
                self.process_new_image(clip_data.convert("RGB"), source="Clipboard Image")
                return
            elif isinstance(clip_data, list):
                for p in clip_data:
                    if isinstance(p, str) and os.path.exists(p):
                        self.load_image_from_path(p)
                        return

            text = self.root.clipboard_get().strip().strip('"').strip("'")
            if os.path.exists(text) and os.path.isfile(text):
                self.load_image_from_path(text)
                return

            self.set_status("Clipboard does not contain an image or valid file path.", error=True)
        except Exception as e:
            self.set_status(f"Clipboard read error: {e}", error=True)

    def process_new_image(self, pil_img: Image.Image, source: str = ""):
        self.loaded_image_pil = pil_img.resize(MODEL_SIZE, Image.Resampling.LANCZOS)
        self.randomize_textures()

        self.display_on_canvas(self.loaded_image_pil, self.canvas_orig)
        self.canvas_rest.config(image="", text="Awaiting Denoising...")
        self.canvas_rest.image = None
        self.restored_image_pil = None

        # Re-apply current slider degradation
        self.apply_aging_slider()
        self.set_status(f"Loaded: {source} (128x128)")

    def schedule_auto_restore(self, delay_ms: int = 350):
        """Debounced automatic restoration trigger."""
        if self.auto_restore_timer is not None:
            try:
                self.root.after_cancel(self.auto_restore_timer)
            except Exception:
                pass
            self.auto_restore_timer = None
        self.auto_restore_timer = self.root.after(delay_ms, self.restore_image_threaded)

    def _update_gpu_textures(self):
        """Prepares GPU tensors for Cold Diffusion Algorithm 2 degradation operator."""
        if self.paper_texture is not None:
            p_tensor = torch.from_numpy(self.paper_texture.astype(np.float32)).permute(2, 0, 1).unsqueeze(0).to(self.device)
            self.gpu_paper_tensor = p_tensor
        else:
            self.gpu_paper_tensor = None

        if self.scratch_texture is not None:
            scratch_gray = np.mean(self.scratch_texture, axis=-1).astype(np.float32)
            black_point = 0.20
            scratch_clean = np.clip((scratch_gray - black_point) / (1.0 - black_point), 0.0, 1.0)
            kernel = np.ones((2, 2), np.float32)
            scratch_clean = cv2.dilate(scratch_clean, kernel, iterations=1)
            # Shape: (128, 128) -> (1, 1, 128, 128)
            s_tensor = torch.from_numpy(scratch_clean.astype(np.float32)).unsqueeze(0).unsqueeze(0).to(self.device)
            self.gpu_scratch_tensor = s_tensor
        else:
            self.gpu_scratch_tensor = None


    def degrade_tensor(self, x_0: torch.Tensor, num_steps: int) -> torch.Tensor:
        """Applies num_steps of Cold Diffusion degradation entirely on GPU."""
        if num_steps <= 0:
            return x_0

        use_paper = self.use_paper_var.get() and (self.gpu_paper_tensor is not None)
        use_scratch = self.use_scratch_var.get() and (self.gpu_scratch_tensor is not None)

        x_t = x_0
        for _ in range(num_steps):
            # 1. Sepia fade
            sepia = torch.einsum("bchw,oc->bohw", x_t, self.sepia_matrix).clamp(0.0, 1.0)
            x_t = 0.992 * x_t + 0.008 * sepia

            # 2. Paper texture
            if use_paper:
                x_t = 0.992 * x_t + 0.008 * (x_t * self.gpu_paper_tensor)

            # 3. Scratch texture
            if use_scratch:
                screen = 1.0 - (1.0 - x_t) * (1.0 - self.gpu_scratch_tensor)
                x_t = 0.99 * x_t + 0.01 * screen

        return x_t.clamp(0.0, 1.0)

    def randomize_textures(self):
        if self.valid_papers:
            self.paper_texture = random.choice(self.valid_papers)
        if self.valid_scratches:
            self.scratch_texture = random.choice(self.valid_scratches)
        self._update_gpu_textures()
        self.apply_aging_slider()
        self.schedule_auto_restore(delay_ms=100)


    def use_as_direct_input(self):
        if self.loaded_image_pil is None:
            self.set_status("Load an image first!", error=True)
            return
        self.degraded_image_pil = self.loaded_image_pil.copy()
        self.display_on_canvas(self.degraded_image_pil, self.canvas_deg)
        if self.auto_estimate_var.get():
            self.estimate_and_apply_current(self.degraded_image_pil, update_t=True)
        else:
            self.set_status("Input passed directly to Model Input (No artificial aging).")
        self.schedule_auto_restore(delay_ms=100)

    def estimate_and_apply_current(self, target_pil=None, update_t: bool = True):
        """Analyze current image and update t and/or step sliders using Estimator/Heuristic."""
        if target_pil is None:
            target_pil = self.degraded_image_pil or self.loaded_image_pil
        if target_pil is None:
            self.set_status("Load an image first!", error=True)
            return

        try:
            x_tensor = self.transform(target_pil).unsqueeze(0).to(self.device)
            if self.estimator is not None:
                est_t, est_k = self.estimator.predict_image(x_tensor)
                src_name = "Neural Estimator"
            else:
                est_t, est_k = estimate_damage_heuristic(x_tensor)
                src_name = "Heuristic Estimator"

            if update_t:
                self.t_slider.set(int(round(est_t)))
                self.lbl_t_val.config(text=str(int(round(est_t))))

            self.steps_slider.set(est_k)
            self.lbl_steps_val.config(text=str(est_k))
            self.set_status(f"⚡ {src_name}: Estimated t={est_t:.0f}, Optimal Steps={est_k}")
            self.schedule_auto_restore(delay_ms=100)
        except Exception as e:
            self.set_status(f"Estimation error: {e}", error=True)

    def _on_toggle_auto_estimate(self):
        if self.auto_estimate_var.get():
            self.estimate_and_apply_current(update_t=False)

    # ==========================================
    # PROGRESSIVE COMPOUND DEGRADATION
    # ==========================================
    def _on_toggle_aging(self):
        self.apply_aging_slider()
        self.schedule_auto_restore(delay_ms=100)

    def _on_slider_move(self, val):
        self.lbl_t_val.config(text=str(val))
        self.apply_aging_slider()
        if self.auto_estimate_var.get():
            # Dynamically compute optimal steps based on degradation severity
            t_int = int(val)
            est_k = max(1, min(20, int(round(1.0 + (t_int / 100.0) * 12.0))))
            self.steps_slider.set(est_k)
            self.lbl_steps_val.config(text=str(est_k))
        self.schedule_auto_restore(delay_ms=350)

    def _on_steps_slider_move(self, val):
        self.lbl_steps_val.config(text=str(val))
        self.schedule_auto_restore(delay_ms=350)

    def apply_aging_slider(self):
        if self.loaded_image_pil is None:
            return

        t = int(self.t_slider.get())
        if t == 0:
            self.degraded_image_pil = self.loaded_image_pil.copy()
            self.display_on_canvas(self.degraded_image_pil, self.canvas_deg)
            return

        x_0 = np.array(self.loaded_image_pil).astype(np.float32) / 255.0
        x_t = x_0.copy()

        use_paper = self.use_paper_var.get()
        use_scratch = self.use_scratch_var.get()

        for _ in range(t):
            # 1. Sepia fade
            sepia_matrix = np.array([
                [0.393, 0.769, 0.189],
                [0.349, 0.686, 0.168],
                [0.272, 0.534, 0.131],
            ])
            sepia_img = np.clip(cv2.transform(x_t, sepia_matrix), 0, 1.0)
            x_t = (1.0 - 0.008) * x_t + 0.008 * sepia_img

            # 2. Paper Texture (Multiply)
            if use_paper and self.paper_texture is not None:
                x_t = (1.0 - 0.008) * x_t + 0.008 * (x_t * self.paper_texture)

            # 3. Scratch Texture (Screen + Dilation)
            if use_scratch and self.scratch_texture is not None:
                scratch_gray = np.mean(self.scratch_texture, axis=-1, keepdims=True)
                black_point = 0.20
                scratch_clean = np.clip((scratch_gray - black_point) / (1.0 - black_point), 0.0, 1.0)
                kernel = np.ones((2, 2), np.float32)
                scratch_clean = cv2.dilate(scratch_clean, kernel, iterations=1)
                scratch_clean = np.expand_dims(scratch_clean, axis=-1)

                screen_blend = 1.0 - (1.0 - x_t) * (1.0 - scratch_clean)
                x_t = (1.0 - 0.01) * x_t + 0.01 * screen_blend

        x_t_uint8 = np.clip(x_t * 255.0, 0, 255).astype(np.uint8)
        self.degraded_image_pil = Image.fromarray(x_t_uint8)
        self.display_on_canvas(self.degraded_image_pil, self.canvas_deg)

    # ==========================================
    # COLD DIFFUSION RESTORATION
    # ==========================================
    def restore_image_threaded(self):
        if self.model is None:
            self.set_status("Model is not loaded! Check checkpoint path.", error=True)
            return

        input_pil = self.degraded_image_pil or self.loaded_image_pil
        if input_pil is None:
            self.set_status("No image to restore! Load an image first.", error=True)
            return

        t_val = max(1.0, float(self.t_slider.get()))
        num_steps = int(self.steps_slider.get())
        self.set_status(f"Restoring image ({num_steps} steps, t={int(t_val)})...")

        def _infer():
            try:
                x_tensor = self.transform(input_pil).unsqueeze(0).to(self.device)
                self.convergence_monitor.reset()
                status_note = ""

                with torch.no_grad():
                    if num_steps <= 1 or t_val <= 1:
                        # 1-pass direct reconstruction
                        t_input = torch.tensor([float(t_val)], dtype=torch.float32, device=self.device)
                        pred_x_0 = self.model(x_tensor, t_input).clamp(0.0, 1.0)
                        status_note = "1-pass direct reconstruction."
                    else:
                        # Full Cold Diffusion Algorithm 2 with re-degradation residual subtraction:
                        # x_{s_{i+1}} = x_{s_i} - D(x_0, s_i) + D(x_0, s_{i+1})
                        raw_schedule = [int(round(t_val * (num_steps - k) / num_steps)) for k in range(num_steps + 1)]
                        timesteps = []
                        for ts in raw_schedule:
                            if not timesteps or ts < timesteps[-1]:
                                timesteps.append(ts)
                        if timesteps[-1] != 0:
                            timesteps.append(0)

                        curr_x = x_tensor
                        pred_x_0 = None
                        steps_completed = 0
                        stopped_early = False
                        stop_reason = ""
                        num_iterations = len(timesteps) - 1

                        for i in range(num_iterations):
                            steps_completed += 1
                            t_curr = timesteps[i]
                            t_next = timesteps[i + 1]

                            # 1. Predict clean image x_0 from current state at timestep t_curr
                            t_input = torch.tensor([float(t_curr)], dtype=torch.float32, device=self.device)
                            pred_x_0 = self.model(curr_x, t_input).clamp(0.0, 1.0)

                            # 2. Check convergence on predicted clean image
                            should_stop, delta, monitor_msg = self.convergence_monitor.check(pred_x_0, i, num_iterations)

                            if t_next == 0 or should_stop:
                                if should_stop:
                                    stopped_early = True
                                    stop_reason = monitor_msg
                                break

                            # 3. Cold Diffusion Algorithm 2 update (Eq. 4 in Bansal et al.):
                            d_next = self.degrade_tensor(pred_x_0, t_next)
                            d_curr = self.degrade_tensor(d_next, t_curr - t_next)

                            curr_x = (curr_x - d_curr + d_next).clamp(0.0, 1.0)

                        if stopped_early:
                            status_note = f"Early stop: {stop_reason}"
                        else:
                            status_note = f"Completed all {steps_completed} Cold Diffusion Alg 2 steps."


                restored_np = pred_x_0.squeeze(0).cpu().permute(1, 2, 0).numpy()
                restored_np = np.clip(restored_np * 255.0, 0, 255).astype(np.uint8)
                self.restored_image_pil = Image.fromarray(restored_np)

                self.root.after(0, lambda: self._on_restore_done(status_note))
            except Exception as e:
                self.root.after(0, lambda: self.set_status(f"Restoration error: {e}", error=True))

        threading.Thread(target=_infer, daemon=True).start()

    def _on_restore_done(self, status_note: str = ""):
        self.display_on_canvas(self.restored_image_pil, self.canvas_rest)
        msg = f"Restored! {status_note}" if status_note else "Restoration complete!"
        self.set_status(msg)


    # ==========================================
    # CANVAS & EXPORT UTILITIES
    # ==========================================
    def display_on_canvas(self, pil_img: Image.Image, label_widget: tk.Label):
        disp_img = pil_img.resize(DISPLAY_SIZE, Image.Resampling.LANCZOS)
        tk_img = ImageTk.PhotoImage(disp_img)
        label_widget.config(image=tk_img, text="")
        label_widget.image = tk_img

    def save_restored_image(self):
        if self.restored_image_pil is None:
            self.set_status("No restored image to save!", error=True)
            return
        save_path = filedialog.asksaveasfilename(
            title="Save Restored Image",
            defaultextension=".png",
            filetypes=[("PNG Image", "*.png"), ("JPEG Image", "*.jpg;*.jpeg"), ("All Files", "*.*")]
        )
        if save_path:
            try:
                self.restored_image_pil.save(save_path)
                self.set_status(f"Saved to: {save_path}")
            except Exception as e:
                self.set_status(f"Save error: {e}", error=True)

    def copy_restored_to_clipboard(self):
        if self.restored_image_pil is None:
            self.set_status("No restored image to copy!", error=True)
            return
        try:
            output = io.BytesIO()
            self.restored_image_pil.convert("RGB").save(output, "BMP")
            data = output.getvalue()[14:]
            output.close()

            import win32clipboard
            win32clipboard.OpenClipboard()
            win32clipboard.EmptyClipboard()
            win32clipboard.SetClipboardData(win32clipboard.CF_DIB, data)
            win32clipboard.CloseClipboard()
            self.set_status("Copied restored image to system clipboard!")
        except Exception:
            self.set_status("Clipboard copy fallback: Use 'Save Restored Image' to export.", error=False)

    def clear_all(self):
        self.loaded_image_pil = None
        self.degraded_image_pil = None
        self.restored_image_pil = None

        self.canvas_orig.config(image="", text="No image loaded")
        self.canvas_orig.image = None
        self.canvas_deg.config(image="", text="Model Input")
        self.canvas_deg.image = None
        self.canvas_rest.config(image="", text="Restored Output")
        self.canvas_rest.image = None

        self.path_entry.delete(0, tk.END)
        self.set_status("All panels cleared.")


def launch_gui():
    if HAS_DND:
        root = TkinterDnD.Tk()
    else:
        root = tk.Tk()
    app = ColdDiffusionGUI(root)
    root.mainloop()


if __name__ == "__main__":
    launch_gui()
