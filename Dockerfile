# Use a runtime image that includes both CUDA 12.1 and cuDNN 8
FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH}

WORKDIR /app

# 1. Install system dependencies + C++ libraries for ONNX
RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-dev \
    git \
    build-essential \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    ffmpeg \
    wget \
    && rm -rf /var/lib/apt/lists/*

# 2. Upgrade pip and HARD PIN NumPy 1.x
RUN pip3 install --upgrade pip setuptools wheel
RUN pip3 install "numpy<2.0.0"

# 3. Install PyTorch (Matches CUDA 12.1)
RUN pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# 4. Install ONNX Runtime GPU (Version 1.17.1 is the last stable one for cuDNN 8)
# We uninstall any existing onnxruntime first to prevent conflicts
RUN pip3 uninstall -y onnxruntime onnxruntime-gpu && \
    pip3 install onnxruntime-gpu==1.17.1

# 5. Install the VTON project
RUN pip3 install runpod huggingface_hub
RUN pip3 install git+https://github.com/fashn-AI/fashn-vton-1.5.git

# 6. Pre-download weights
RUN python3 -c "from huggingface_hub import snapshot_download; \
    snapshot_download(repo_id='fashn-ai/fashn-vton-1.5', local_dir='./weights')"

COPY handler.py /app/handler.py
RUN chmod +x /app/handler.py

# Launch with unbuffered python
CMD ["python3", "-u", "/app/handler.py"]