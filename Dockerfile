# Use a development-ready CUDA image
FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1

WORKDIR /app

# 1. Install system dependencies first
RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-dev \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# 2. Upgrade core build tools (This prevents 90% of Exit Code 1 errors)
RUN pip3 install --no-cache-dir --upgrade pip setuptools wheel

# 3. Install onnxruntime-gpu ALONE first
# We use a specific version to ensure stability with CUDA 12.1
RUN pip3 install --no-cache-dir onnxruntime-gpu==1.17.1

# 4. Install the rest of the packages
RUN pip3 install --no-cache-dir \
    runpod \
    fashn-vton \
    huggingface_hub

# 5. Download weights
RUN python3 -c "from huggingface_hub import snapshot_download; \
    snapshot_download(repo_id='fashn-ai/fashn-vton-1.5', local_dir='./weights')"

COPY handler.py /app/handler.py

CMD ["python3", "-u", "/app/handler.py"]