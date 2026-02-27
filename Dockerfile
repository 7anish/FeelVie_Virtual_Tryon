# Use a development-ready CUDA image
FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# 1. Install system dependencies
RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-dev \
    git \
    libgl1-mesa-glx \
    libglib2.0-0 \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# 2. Upgrade core build tools
RUN pip3 install --upgrade pip setuptools wheel

# 3. Install heavy dependencies from official sources
RUN pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu121
RUN pip3 install onnxruntime-gpu==1.17.1
RUN pip3 install runpod huggingface_hub

# 4. Install the package directly from GitHub
# This bypasses PyPI issues and ensures you get the latest version
RUN pip3 install git+https://github.com/fashn-AI/fashn-vton-1.5.git

# 5. Bake the weights into the image
RUN python3 -c "from huggingface_hub import snapshot_download; \
    snapshot_download(repo_id='fashn-ai/fashn-vton-1.5', local_dir='./weights')"

COPY handler.py /app/handler.py

CMD ["python3", "-u", "/app/handler.py"]