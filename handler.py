import runpod
import torch
import base64
import io
from PIL import Image
from fashn_vton import TryOnPipeline

# Initialize the pipeline globally so it stays in VRAM
# This path matches the 'weights' folder we download in the Dockerfile
pipeline = TryOnPipeline(weights_dir="./weights")

def decode_base64_to_image(base64_str):
    """Converts base64 string to a PIL Image."""
    if "," in base64_str:
        base64_str = base64_str.split(",")[1]
    image_data = base64.b64decode(base64_str)
    return Image.open(io.BytesIO(image_data)).convert("RGB")

def encode_image_to_base64(image):
    """Converts PIL Image to base64 string for API response."""
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode('utf-8')

def handler(job):
    """
    The main handler function for RunPod Serverless.
    Input format: { "person_image": "base64...", "garment_image": "base64...", "category": "tops" }
    """
    job_input = job["input"]
    
    try:
        # 1. Decode inputs
        person_img = decode_base64_to_image(job_input["person_image"])
        garment_img = decode_base64_to_image(job_input["garment_image"])
        category = job_input.get("category", "tops") # tops, bottoms, or one-pieces

        # 2. Run Inference
        # Optimized for speed with 30 timesteps and bfloat16 (auto-detected on GPU)
        result = pipeline(
            person_image=person_img,
            garment_image=garment_img,
            category=category,
            num_timesteps=job_input.get("steps", 30),
            guidance_scale=job_input.get("guidance", 1.5),
            seed=job_input.get("seed", 42)
        )

        # 3. Return the result
        output_base64 = encode_image_to_base64(result.images[0])
        return {"image": output_base64}

    except Exception as e:
        return {"error": str(e)}

# Start the serverless worker
runpod.serverless.start({"handler": handler})