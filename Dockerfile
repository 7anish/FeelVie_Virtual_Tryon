# Use CUDA 12.1 for modern GPU support
FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04

# Prevent interactive prompts
ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /app

# Install Python and essential system tools
RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-dev \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
# 'fashn-vton' is the main model library, 'runpod' is the serverless SDK
RUN pip3 install --no-cache-dir \
    runpod \
    fashn-vton \
    huggingface_hub \
    onnxruntime-gpu

# Download the model weights during the build so they are baked in
# This prevents 2-3 minute delays (cold starts) when your API is called
RUN python3 -c "from huggingface_hub import snapshot_download; \
    snapshot_download(repo_id='fashn-ai/fashn-vton-1.5', local_dir='./weights')"

# Copy your handler script into the container
COPY handler.py /app/handler.py

# Start the worker
CMD ["python3", "-u", "/app/handler.py"]