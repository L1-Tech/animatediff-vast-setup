#!/bin/bash
set -e

# === Basic System Update and Dependencies ===
apt update && apt install -y git wget curl python3 python3-venv python3-pip ffmpeg libgl1 unzip

# === Clone AnimateDiff ===
cd ~
git clone https://github.com/continue-revolution/AnimateDiff
cd AnimateDiff

# === Python Environment Setup ===
python3 -m venv venv
source venv/bin/activate

pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt

# === Download Stable Diffusion Model ===
mkdir -p models/StableDiffusion
cd models/StableDiffusion
wget https://huggingface.co/runwayml/stable-diffusion-v1-5/resolve/main/v1-5-pruned-emaonly.ckpt -O model.ckpt
cd ../../

# === Download AnimateDiff Motion Module ===
mkdir -p models/Motion_Module
cd models/Motion_Module
wget https://huggingface.co/guoyww/animatediff-motion-model/resolve/main/mm_sd_v15_v2.ckpt
cd ../../

# === Test Prompt Run ===
python3 scripts/inference.py \
  --sd_model_path models/StableDiffusion/model.ckpt \
  --motion_module_path models/Motion_Module/mm_sd_v15_v2.ckpt \
  --prompt "a cinematic drone shot of Lisbon’s Alfama district at sunset" \
  --seed 42 \
  --steps 25 \
  --frames 16 \
  --fps 8 \
  --output_dir outputs/test_run
