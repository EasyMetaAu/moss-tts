# MOSS-TTS Local-GGUF: llama.cpp first-class pipeline (Qwen3 backbone + ONNX audio tokenizer)
# Target VRAM: ~5-6GB on GPU1 (vs ~12.7GB for bf16 local profile)
#
# Multi-stage build:
#   Stage 1 (builder): clone OpenMOSS/llama.cpp moss-tts-firstclass + cmake CUDA build llama-moss-tts
#   Stage 2 (runtime): copy binary + tools/tts python helpers + FastAPI server; weights mounted via volume

# ─────────────────────────────────────────────
# Stage 1: builder
# ─────────────────────────────────────────────
FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04 AS builder

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    cmake ninja-build gcc g++ git ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build
ARG LLAMA_CPP_BRANCH=moss-tts-firstclass
ARG CUDA_ARCHS="86"
# RTX 3090 = sm_86, RTX 4090 = sm_89 ; set to "86;89" for multi
RUN git clone --depth 1 --branch ${LLAMA_CPP_BRANCH} \
        https://github.com/OpenMOSS/llama.cpp.git llama.cpp

ENV LIBRARY_PATH=/usr/local/cuda/lib64/stubs:${LIBRARY_PATH}
RUN cd llama.cpp \
    && cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON \
             -DCMAKE_CUDA_ARCHITECTURES="${CUDA_ARCHS}" \
             -DCMAKE_EXE_LINKER_FLAGS="-L/usr/local/cuda/lib64/stubs" \
             -DCMAKE_SHARED_LINKER_FLAGS="-L/usr/local/cuda/lib64/stubs" \
    && cmake --build build --target llama-moss-tts -j$(nproc)

# ─────────────────────────────────────────────
# Stage 2: runtime
# ─────────────────────────────────────────────
FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-venv python3-pip libsndfile1 ca-certificates curl \
    && rm -rf /var/lib/apt/lists/*

RUN update-alternatives --install /usr/bin/python  python  /usr/bin/python3 1

WORKDIR /app

# Copy the llama-moss-tts binary + tools/tts python helpers + libs
COPY --from=builder /build/llama.cpp/build/bin/llama-moss-tts  /app/bin/llama-moss-tts
COPY --from=builder /build/llama.cpp/tools/tts/                /app/tools/tts/

# ggml shared libs (if any)
RUN mkdir -p /app/lib
COPY --from=builder /build/llama.cpp/build/bin/                /app/lib_raw/
RUN find /app/lib_raw -name '*.so*' -exec cp -a {} /app/lib/ \; || true \
    && rm -rf /app/lib_raw
ENV LD_LIBRARY_PATH=/app/lib:${LD_LIBRARY_PATH}

# Python deps for hybrid e2e wrapper + FastAPI server
COPY requirements.txt /app/requirements.txt
RUN pip3 install --no-cache-dir --break-system-packages -r /app/requirements.txt

# Server code
COPY server.py /app/server.py

# Defaults (override via env/volume mount)
ENV MODEL_GGUF=/models/MOSS-TTS-GGUF/first_class/MOSS_TTS_FIRST_CLASS_Q4_K_M.gguf \
    TOKENIZER_DIR=/models/MOSS-TTS-GGUF/tokenizer \
    ONNX_ENCODER=/models/MOSS-Audio-Tokenizer-ONNX/encoder.onnx \
    ONNX_DECODER=/models/MOSS-Audio-Tokenizer-ONNX/decoder.onnx \
    LLAMA_BIN=/app/bin/llama-moss-tts \
    VOICES_DIR=/app/voices \
    MAX_SEGMENT_CHARS=500 \
    MAX_NEW_TOKENS=4096 \
    N_GPU_LAYERS=-1 \
    PORT=6009

EXPOSE 6009

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD curl -fsS http://localhost:${PORT}/health || exit 1

CMD ["python3", "/app/server.py"]
