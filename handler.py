import runpod
from runpod import RunPodLogger
import torch
import base64
import io
import os
from PIL import Image
from fashn_vton import TryOnPipeline

# Initialize the RunPod logger
log = RunPodLogger()

# 1. Model Initialization with Logging
log.info("--- Starting Worker Initialization ---")
WEIGHTS_PATH = "./weights"

if not os.path.exists(WEIGHTS_PATH):
    log.error(f"CRITICAL: Weights directory not found at {WEIGHTS_PATH}!")
else:
    log.info(f"Weights directory found. Contents: {os.listdir(WEIGHTS_PATH)}")

try:
    log.info("Loading FASHN VTON Pipeline into VRAM...")
    # Loading outside the handler to prevent reload on every request
    pipeline = TryOnPipeline(weights_dir=WEIGHTS_PATH)
    log.info("✅ Pipeline loaded successfully.")
except Exception as e:
    log.error(f"❌ Failed to initialize pipeline: {str(e)}")
    raise e

def decode_base64_to_image(base64_str):
    if "," in base64_str:
        base64_str = base64_str.split(",")[1]
    image_data = base64.b64decode(base64_str)
    return Image.open(io.BytesIO(image_data)).convert("RGB")

def encode_image_to_base64(image):
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode('utf-8')

def handler(job):
    job_input = job["input"]
    log.info(f"Received new job ID: {job['id']}")
    
    try:
        log.info("Decoding input images...")
        person_img = decode_base64_to_image(job_input["person_image"])
        garment_img = decode_base64_to_image(job_input["garment_image"])
        category = job_input.get("category", "tops")

        log.info(f"Running inference for category: {category}...")
        result = pipeline(
            person_image=person_img,
            garment_image=garment_img,
            category=category,
            num_timesteps=job_input.get("steps", 30),
            guidance_scale=job_input.get("guidance", 1.5)
        )

        log.info("Inference complete. Encoding output...")
        output_base64 = encode_image_to_base64(result.images[0])
        
        log.info("✅ Job successful.")
        return {"image": output_base64}

    except Exception as e:
        log.error(f"❌ Error during job execution: {str(e)}")
        return {"error": str(e)}

# Start the worker
runpod.serverless.start({"handler": handler})