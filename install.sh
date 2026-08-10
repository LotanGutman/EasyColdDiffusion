#!/usr/bin/env bash
set -euo pipefail

python -m pip install --upgrade pip
python -m pip install torch torchvision einops tqdm pyyaml flash-linear-attention
