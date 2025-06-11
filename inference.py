import torch
from diffusers import AnimateDiffPipeline, DDIMScheduler
from compel import Compel
from PIL import Image
import os

model_id = "cerspense/zeroscope_v2_XL"

pipe = AnimateDiffPipeline.from_pretrained(
    model_id,
    torch_dtype=torch.float16
).to("cuda")

pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)

prompt = "a cinematic drone shot of Lisbon at golden hour"
compel_proc = Compel(tokenizer=pipe.tokenizer, text_encoder=pipe.text_encoder)
conditioning = compel_proc(prompt)

frames = pipe(prompt_embeds=conditioning, num_inference_steps=25).frames[0]

output_dir = "./outputs"
os.makedirs(output_dir, exist_ok=True)

for i, frame in enumerate(frames):
    frame.save(f"{output_dir}/frame_{i:03d}.png")

print("✅ Frames saved to ./outputs/")
