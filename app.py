%%writefile app.py
import streamlit as st
import torch
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
from PIL import Image
import os, json, time
from datetime import datetime

MODEL_ID = "runwayml/stable-diffusion-v1-5"
OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"

@st.cache_resource
def load_pipeline():
    pipe = StableDiffusionPipeline.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32
    )
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    return pipe.to(device)

st.title("Text → Image Generator (Stable Diffusion)")
st.caption(f"Running on: {device}")

pipe = load_pipeline()

with st.sidebar:
    st.header("Settings")
    style = st.selectbox("Style", ["None", "Photorealistic", "Artistic", "Anime", "Van Gogh"])
    neg = st.text_input("Negative Prompt", "lowres, text, watermark, blurry")
    steps = st.slider("Sampling Steps", 10, 50, 25)
    cfg = st.slider("CFG Scale", 1.0, 15.0, 7.0)
    count = st.slider("Images", 1, 4, 1)

prompt = st.text_area("Enter your prompt", "a futuristic city at sunset")

if st.button("Generate"):
    final_prompt = prompt
    if style != "None":
        final_prompt = f"{style}, {prompt}, highly detailed, 4k rendering"

    st.info("Generating images...")

    results = pipe(
        [final_prompt] * count,
        negative_prompt=[neg] * count,
        num_inference_steps=steps,
        guidance_scale=cfg
    ).images

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    for i, img in enumerate(results):
        path = f"{OUTPUT_DIR}/img_{ts}_{i}.png"
        img.save(path)
        st.image(path)
        st.download_button("Download", open(path, "rb").read(), file_name=f"img_{ts}_{i}.png")

