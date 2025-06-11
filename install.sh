#!/bin/bash
set -e

# === Dependencies ===
apt update && apt install -y wget curl ffmpeg libgl1 unzip python3 python3-venv python3-pip

# === Python Env ===
cd ~
python3 -m venv AnimateDiff-venv
source AnimateDiff-venv/bin/activate

pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install imageio imageio-ffmpeg einops transformers accelerate omegaconf safetensors tqdm

# === Download AnimateDiff Core ===
mkdir -p AnimateDiff/scripts
wget https://raw.githubusercontent.com/L1-Tech/animatediff-vast-setup/main/inference.py -O AnimateDiff/scripts/inference.py

# === Download Models ===
mkdir -p AnimateDiff/models/StableDiffusion
wget https://huggingface.co/runwayml/stable-diffusion-v1-5/resolve/main/v1-5-pruned-emaonly.ckpt -O AnimateDiff/models/StableDiffusion/model.ckpt

mkdir -p AnimateDiff/models/Motion_Module
wget https://huggingface.co/guoyww/animatediff-motion-model/resolve/main/mm_sd_v15_v2.ckpt -O AnimateDiff/models/Motion_Module/mm_sd_v15_v2.ckpt

# === Run Test ===
python AnimateDiff/scripts/inference.py \
  --sd_model_path AnimateDiff/models/StableDiffusion/model.ckpt \
  --motion_module_path AnimateDiff/models/Motion_Module/mm_sd_v15_v2.ckpt \
  --prompt "cinematic drone shot of the Douro River in Porto, Portugal, sunset" \
  --seed 42 --steps 25 --frames 16 --fps 8 \
  --output_dir AnimateDiff/outputs/test_run
