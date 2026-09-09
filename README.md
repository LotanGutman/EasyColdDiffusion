# EasyColdDiffusion

A unified deep learning framework implementing **Cold Diffusion** for photographic aging degradation and image restoration. The project supports both **U-Net** and **DiT (Diffusion Transformer)** architectures.

---

## 📁 Repository Structure

```text
EasyColdDiffusion/
├── assets/
│   ├── textures/             # Paper and scratch degradation textures
│   └── test_images/          # Sample images for testing & restoration
├── checkpoints/              # Directory for model checkpoint weights
├── data/                     # Dataset storage (raw and processed 128x128)
├── results/                  # Inference and restoration output images
├── src/
│   ├── datasets/             # Data loading and preprocessing
│   │   ├── unet_dataset.py   # Recursive on-the-fly degradation dataset (U-Net)
│   │   ├── dit_dataset.py    # Paired dataset loader (DiT)
│   │   ├── download_coco.py  # COCO dataset downloader
│   │   └── create_dataset.py # 128x128 center crop & resize pipeline
│   ├── models/               # Model architectures
│   │   ├── unet.py           # Time-conditioned U-Net restoration model
│   │   ├── diffusion_vit.py  # Diffusion Vision Transformer (DiT)
│   │   └── utils/            # Shared embedding & layer blocks
│   ├── training/             # Training routines
│   │   ├── train_unet.py     # U-Net Cold Diffusion trainer (Cosine Warmup)
│   │   └── train_dit.py      # DiT trainer
│   ├── infer/                # Inference & evaluation
│   │   ├── infer_unet.py     # U-Net evaluation & multi-step visualizer
│   │   └── infer_dit.py      # DiT tiled reconstruction with Hann window blending
│   ├── operations.py         # Degradation operators (sepia, cracks, stains, tears)
│   └── operations_main.py    # Offline dataset degradation generator
├── aging_gui.py              # Interactive Tkinter GUI for continuous aging preview
├── train_unet.py             # Entry point: Train U-Net model
├── train_dit.py              # Entry point: Train DiT model
├── test_unet.py              # Entry point: Test & restore images with U-Net
├── infer_dit.py              # Entry point: Restore images with DiT
├── download_coco.py          # Entry point: Download raw COCO images
├── create_dataset.py         # Entry point: Crop and prepare clean dataset
├── trained_model.pth         # Pretrained U-Net weights
├── requirements.txt          # Python dependencies
└── install.sh                # Setup script
```

---

## 🛠️ Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/LotanGutman/EasyColdDiffusion.git
   cd EasyColdDiffusion
   ```

2. **Create and activate a virtual environment:**
   ```bash
   # Windows:
   python -m venv venv
   venv\Scripts\activate

   # Linux / macOS:
   python -m venv venv
   source venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

---

## 🚀 Usage Guide

### 1. Interactive Aging Degradation GUI
Launch the visual degradation tool to experiment with real-time continuous degradation (sepia tone, paper grain, scratch dilation):
```bash
python aging_gui.py
```

---

### 2. U-Net Restoration Model

#### Test Pretrained U-Net Model:
Run restoration on sample test images and save comparison figures to `results/`:
```bash
python test_unet.py --checkpoint trained_model.pth --num_samples 5 --output_dir results
```
To view interactive matplotlib windows directly:
```bash
python test_unet.py --show
```

#### Train U-Net from Scratch:
```bash
# Step 1: Download COCO raw images
python download_coco.py --output_dir data/coco_raw_train2017 --max_images 30000

# Step 2: Prepare 128x128 clean training set
python create_dataset.py --raw_dir data/coco_raw_train2017 --clean_dir data/coco_128_clean

# Step 3: Run training
python train_unet.py --clean_dir data/coco_128_clean --epochs 40 --batch_size 16
```

---

### 3. DiT (Diffusion Transformer) Model

#### Train DiT Model:
```bash
python train_dit.py --clean_dir data/clean --degraded_dir data/degraded --epochs 30 --batch_size 64
```

#### Run DiT Inference with Tiling & Hann Window Blending:
```bash
python infer_dit.py --checkpoint checkpoints/dit_cold_diffusion_epoch_6.pth --input_dir data/degraded --output_dir results/dit_restored
```

---

## 🔬 Technical Methodology
- **Cold Diffusion:** Employs deterministic degradation operators $D(x, t)$ rather than Gaussian noise.
- **Recursive Degradation:** Micro-step compounding sepia matrix transforms, paper texture multi-layering, and morphological scratch dilation.
- **U-Net Architecture:** Multi-scale encoder-decoder network with sinusoidal time embeddings and skip connections for high-fidelity reconstruction.
- **DiT Architecture:** Vision Transformer operating over image patches with adaptive LayerNorm (`adaLN`) modulation and Hann window overlap blending for large images.
