# 1. Switch to the official NVIDIA PyTorch base (highly stable)
FROM nvcr.io/nvidia/pytorch:23.10-py3

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1

WORKDIR /app

# 2. Install essential system vision libraries
RUN apt-get update && apt-get install -y \
    git libgl1-mesa-glx libglib2.0-0 ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# 3. Clean and fix NumPy/ONNX (The core fix)
RUN pip3 install --upgrade pip
RUN pip3 uninstall -y numpy onnxruntime onnxruntime-gpu
RUN pip3 install "numpy<2.0.0"
RUN pip3 install onnxruntime-gpu==1.17.1

# 4. Install Fashn-VTON and RunPod
RUN pip3 install runpod huggingface_hub
RUN pip3 install git+https://github.com/fashn-AI/fashn-vton-1.5.git

# 5. Bake weights 
RUN python3 -c "from huggingface_hub import snapshot_download; \
    snapshot_download(repo_id='fashn-ai/fashn-vton-1.5', local_dir='./weights')"

COPY handler.py /app/handler.py
RUN chmod +x /app/handler.py

CMD ["python3", "-u", "/app/handler.py"]