# XAI-IDS Benchmark — Reproducibility container
# Reference: Python 3.11, PyTorch 2.10.0+cu128 for NVIDIA RTX 5070 (Compute 12.0)

FROM nvidia/cuda:12.8.0-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    TZ=Etc/UTC

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.11 \
        python3.11-dev \
        python3-pip \
        git \
        wget \
        ca-certificates \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

RUN ln -sf /usr/bin/python3.11 /usr/bin/python && \
    ln -sf /usr/bin/python3.11 /usr/bin/python3

WORKDIR /app

# Install PyTorch with CUDA 12.8 first (large download; cache layer)
RUN pip install --upgrade pip && \
    pip install torch==2.10.0 torchvision==0.25.0 torchaudio==2.10.0 \
        --index-url https://download.pytorch.org/whl/cu128

# Copy and install remaining requirements
COPY requirements.txt /app/requirements.txt
RUN pip install -r /app/requirements.txt

# Copy project
COPY . /app

# Default seeds & config
ENV PYTHONPATH=/app
ENV PYTHONHASHSEED=42

# Verification step (build will fail fast if CUDA is misconfigured)
RUN python -c "import torch; print('Torch:', torch.__version__, 'CUDA:', torch.cuda.is_available())"

ENTRYPOINT ["python", "main.py"]
CMD ["--mode", "multiclass", "--stage", "all"]
