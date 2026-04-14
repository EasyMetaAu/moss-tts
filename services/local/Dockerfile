# Python 3.12 + CUDA 12.1 for MOSS-TTS-Local-Transformer compatibility
FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y \
    python3.12 python3.12-venv python3.12-dev python3-pip \
    libsndfile1 ffmpeg git curl \
    && rm -rf /var/lib/apt/lists/*

# Install CUDA runtime (for PyTorch GPU)
RUN curl -fsSL https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb -o /tmp/cuda-keyring.deb \
    && dpkg -i /tmp/cuda-keyring.deb \
    && apt-get update \
    && apt-get install -y libcublas-12-1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Use python3.12 as default
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.12 1
RUN python3.12 -m pip install --upgrade pip setuptools wheel

COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

COPY server.py .

ENV MODEL_PATH=/models/MOSS-TTS-Local-Transformer
ENV VOICES_DIR=/app/voices
ENV HF_HOME=/hf_cache
ENV PORT=6007

CMD ["python3", "server.py"]
