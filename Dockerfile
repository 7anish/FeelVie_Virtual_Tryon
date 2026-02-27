FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1

WORKDIR /app

# 1. Install system dependencies
RUN apt-get update && apt-get install -y \
    python3-pip python3-dev git build-essential \
    libgl1-mesa-glx libglib2.0-0 libsm6 libxext6 libxrender-dev ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# 2. Upgrade pip and force NumPy 1.x (This is the fix)
RUN pip3 install --upgrade pip setuptools wheel
RUN pip3 install "numpy<2.0"

# 3. Install PyTorch and ONNX
RUN pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu121
RUN pip3 install onnxruntime-gpu==1.17.1 runpod huggingface_hub

# 4. Install VTON
RUN pip3 install git+https://github.com/fashn-AI/fashn-vton-1.5.git

# 5. Weights and Handler
RUN python3 -c "from huggingface_hub import snapshot_download; \
    snapshot_download(repo_id='fashn-ai/fashn-vton-1.5', local_dir='./weights')"

COPY handler.py /app/handler.py
RUN chmod +x /app/handler.py

CMD ["python3", "-u", "/app/handler.py"]