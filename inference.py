import torch
from diffusers import StableDiffusionPipeline, DDIMScheduler
from compel import Compel
from PIL import Image
import os

# ==== Model Paths ====
base_model_path = "runwayml/stable-diffusion-v1-5"
motion_module_path = "AnimateDiff/models/Motion_Module/mm_sd_v15_v2.ckpt"

# ==== Set up pipeline ====
pipe = StableDiffusionPipeline.from_pretrained(
    base_model_path,
    torch_dtype=torch.float16,
    safety_checker=None
).to("cuda")

# Load motion module weights (LoRA-style injection)
motion_weights = torch.load(motion_module_path, map_location="cuda")
pipe.unet.load_state_dict(motion_weights, strict=False)

# Set scheduler
pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)

# ==== Prompt ====
prompt = "A cinematic drone shot of Lisbon at golden hour, 4k"
compel_proc = Compel(tokenizer=pipe.tokenizer, text_encoder=pipe.text_encoder)
conditioning = compel_proc(prompt)

# ==== Generate animation frames ====
frames = pipe(prompt_embeds=conditioning, num_inference_steps=25, guidance_scale=7.5, num_images_per_prompt=16).images

# ==== Save frames ====
output_dir = "AnimateDiff/outputs"
os.makedirs(output_dir, exist_ok=True)

for i, frame in enumerate(frames):
    frame.save(f"{output_dir}/frame_{i:03d}.png")

print("✅ Generation complete. Frames saved in AnimateDiff/outputs")
