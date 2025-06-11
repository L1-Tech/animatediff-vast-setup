#!/bin/bash
set -e

echo "🚀 Setting up AnimateDiff environment..."

# === System dependencies ===
apt update && apt install -y wget curl ffmpeg libgl1 unzip python3 python3-venv python3-pip

# === Python virtual environment ===
cd ~
python3 -m venv AnimateDiff-venv
source AnimateDiff-venv/bin/activate

# === Upgrade pip and install packages ===
pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install diffusers compel accelerate omegaconf safetensors tqdm imageio imageio-ffmpeg einops transformers

# === Create directory structure ===
mkdir -p ~/AnimateDiff/outputs

# === Download inference.py ===
wget https://raw.githubusercontent.com/L1-Tech/animatediff-vast-setup/main/scripts/inference.py -O ~/AnimateDiff/inference.py

echo "✅ Setup complete."
echo "To run:"
echo "source ~/AnimateDiff-venv/bin/activate && python3 ~/AnimateDiff/inference.py"
