# EasyColdDiffusion

A unified deep learning framework implementing **Cold Diffusion** for photographic aging degradation and image restoration. The project supports both **U-Net** and **DiT (Diffusion Transformer)** architectures.

---

## 📁 Repository Structure

```text
EasyColdDiffusion/
├── assets/                  # Degradation textures & sample test images
│   ├── textures/            # Paper and scratch textures
│   └── test_images/         # Sample images for testing & restoration
├── checkpoints/             # Directory for model checkpoint weights
├── data/                    # Dataset storage (raw and processed 128x128)
├── results/                 # Inference and restoration output images
├── src/
│   ├── datasets/            # Data loading and preprocessing
│   │   ├── create_dataset.py# 128x128 center crop & resize pipeline
│   │   ├── dit_dataset.py   # Paired dataset loader (DiT)
│   │   ├── download_coco.py # COCO dataset downloader
│   │   └── unet_dataset.py  # On-the-fly degradation dataset (U-Net)
│   ├── gui/                 # Interactive Tkinter restoration studio
│   │   └── app.py           # Drag-and-drop, paste & live restoration GUI
│   ├── infer/               # Specialized inference modules
│   │   ├── __init__.py
│   │   ├── convergence.py   # Cauchy closed-loop convergence monitor
│   │   ├── infer_dit.py     # DiT inference & reconstruction pipeline
│   │   └── infer_unet.py    # U-Net evaluation & multi-step visualizer
│   ├── models/              # Model architectures
│   │   ├── utils/           # Shared embedding & layer blocks
│   │   │   └── __init__.py
│   │   ├── diffusion_vit.py # Diffusion Vision Transformer (DiT)
│   │   ├── step_estimator.py# Lightweight damage & step budget estimator
│   │   └── unet.py          # Time-conditioned U-Net restoration model
│   ├── training/            # Training routines
│   │   ├── __init__.py
│   │   ├── train_dit.py     # DiT trainer
│   │   ├── train_estimator.py # Step & degradation estimator trainer
│   │   └── train_unet.py    # U-Net Cold Diffusion trainer
│   ├── __init__.py
│   ├── infer.py             # Top-level inference module
│   ├── operations.py        # Degradation operators (sepia, cracks, stains)
│   └── operations_main.py   # Offline dataset degradation generator
├── .gitignore               # Git ignore rules
├── DiT_trained_model.pth    # Pretrained DiT model weights
├── LICENSE                  # Project license
├── README.md                # Project documentation
├── install.sh               # Setup script
├── main.py                  # Unified CLI & GUI entry point
├── requirements.txt         # Python dependencies
└── trained_model.pth        # Pretrained U-Net weights
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

### 1. Interactive Aging & Restoration Studio (GUI)
Launch the visual degradation and real-time restoration tool (supports drag & drop, clipboard paste `Ctrl+V`, texture selection, and instant U-Net restoration):
```bash
python main.py
# or: python main.py gui
```

---

### 2. U-Net Restoration Model

#### Test Pretrained U-Net Model:
Run restoration on sample test images and save comparison figures to `results/`:
```bash
python main.py test-unet --checkpoint trained_model.pth --num_samples 5 --output_dir results
```
To view interactive matplotlib windows directly:
```bash
python main.py test-unet --show
```

#### Train U-Net from Scratch:
```bash
# Step 1: Download COCO raw images
python main.py download-coco --output_dir data/coco_raw_train2017 --max_images 30000

# Step 2: Prepare 128x128 clean training set
python main.py create-dataset --raw_dir data/coco_raw_train2017 --clean_dir data/coco_128_clean

# Step 3: Run training
python main.py train-unet --clean_dir data/coco_128_clean --epochs 40 --batch_size 16
```

---

### 3. DiT (Diffusion Transformer) Model

#### Train DiT Model:
```bash
# Step 1: Download COCO raw images

# Step 2: Prepare 128x128 clean training set

# Step 3: Run training
```

#### Run DiT Inference with Tiling & Hann Window Blending:
```bash
# 1. open infer.DiT.py
# 2. update the relevant paths of the weights file "DiT_trained_model.pth"
# 3. update the file where the 128X128 pictutres are saved
# 4. update the directory where you want to save the restored images
# 5. run the infer_DiT.py
```

---

### 4. Step & Convergence Estimator (Architecture-Agnostic)

An intelligent hybrid system combining **Instance-Level Neural Prediction** (predicting damage severity $t \in [0, 100]$ and step budget $K \in [1, 20]$) and a **Closed-Loop Cauchy Convergence Monitor** ($\Delta_k = \frac{\|x^{(k)} - x^{(k-1)}\|_2}{\|x^{(k-1)}\|_2 + \epsilon} < \tau$) for early termination. Compatible with both U-Net and DiT backbones.

#### Train the Step Estimator:
```bash
python main.py train-estimator --epochs 10 --batch_size 64
```
The resulting checkpoint is saved to `checkpoints/step_estimator.pth` and automatically loaded by the interactive GUI.

---

## 🔬 Technical Methodology
- **Cold Diffusion:** Employs deterministic degradation operators $D(x, t)$ rather than Gaussian noise.
- **Recursive Degradation:** Micro-step compounding sepia matrix transforms, paper texture multi-layering, and morphological scratch dilation.
- **U-Net Architecture:** Multi-scale encoder-decoder network with sinusoidal time embeddings and skip connections for high-fidelity reconstruction.
- **DiT Architecture:** Vision Transformer operating over image patches with adaptive LayerNorm (`adaLN`) modulation and Hann window overlap blending for large images.
- **Adaptive Convergence Estimation:**
  - *Neural Estimator (`StepAndDegradationEstimator`):* Lightweight 4-stage convolutional encoder + dual-head regression (~180k parameters, <2ms GPU latency) predicting degradation timestep $t^*$ and recommended multi-pass budget $K^*$.
  - *Cauchy Convergence Monitor (`CauchyConvergenceMonitor`):* Tracks successive reconstruction residual norms on-the-fly and detects plateau/inflection to halt sampling early, reducing compute overhead while preventing over-smoothing.


