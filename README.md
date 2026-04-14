# moss-tts

Multi-model MOSS-TTS deployment with Docker Compose — three model sizes, one unified API.

| Profile | Model | Parameters | Sample Rate | VRAM | Best For |
|---------|-------|-----------|-------------|------|----------|
| `nano` | MOSS-TTS-Nano | ~0.1B | 48kHz stereo | ~2GB | Low-latency, real-time |
| `local` | MOSS-TTS-Local-Transformer | 1.7B (3B w/ audio tokenizer) | 24kHz mono | ~13GB (bf16) | High-quality article TTS, max fidelity |
| `local-gguf` | MOSS-TTS-Delay (Q4_K_M) | 1.7B (~3B effective) | 24kHz mono | ~5-6GB | High-quality article TTS, memory-constrained GPUs (e.g. 8GB) |
| `realtime` | MOSS-TTS-Realtime | 1.7B | 24kHz mono | ~13GB | Multi-turn dialogue |

All services expose a unified `/api/generate` API compatible with the official MOSS-TTS-Nano interface.

## Setup

### 1. Prerequisites

- Docker + NVIDIA Container Toolkit
- CUDA 12.1+
- MOSS-TTS-Nano Docker image: `moss-tts-nano:latest`
  ```bash
  # Pull from DockerHub (check OpenMOSS-Team page for the exact image)
  docker pull mossmoss/moss-tts-nano:latest
  docker tag mossmoss/moss-tts-nano:latest moss-tts-nano:latest
  ```

### 2. Configure environment

```bash
cp .env.example .env
# Edit .env:
#   HF_HOME=/path/to/huggingface/cache
#   VOICES_DIR=/path/to/voices
#   NANO_PORT=6006   LOCAL_PORT=6007   REALTIME_PORT=6008
```

### 3. Set up voices

```bash
# Copy Chatterbox voices + MOSS demo voices, generate demo.jsonl
CHATTERBOX_VOICES_DIR=/home/lukin/code/chatterbox-api/voices \
    bash scripts/setup-voices.sh
```

### 4. Download model weights (for local/realtime profiles)

```bash
# Requires Python + huggingface_hub
HF_ENDPOINT=https://hf-mirror.com python3 -c "
from huggingface_hub import snapshot_download
snapshot_download('OpenMOSS-Team/MOSS-TTS-Local-Transformer')
snapshot_download('OpenMOSS-Team/MOSS-TTS-Realtime')
"
```

### 5. Start services

```bash
# Nano only (uses pre-built image, fast start)
docker compose --profile nano up -d

# Local-Transformer (build + run)
docker compose --profile local up -d --build

# Realtime
docker compose --profile realtime up -d --build

# Multiple at once
docker compose --profile nano --profile local up -d
```

### 6. `local-gguf` profile: memory-optimized variant

The `local-gguf` profile runs MOSS-TTS-Delay via [OpenMOSS/llama.cpp first-class pipeline](https://github.com/OpenMOSS/llama.cpp/blob/moss-tts-firstclass/docs/moss-tts-firstclass-e2e.md) (Qwen3 backbone as GGUF + ONNX audio tokenizer). Same `/api/generate` contract as `local`, but ~60% smaller VRAM footprint.

**Weights needed** (place under `MODEL_WEIGHTS_DIR`):

```
MODEL_WEIGHTS_DIR/
├── MOSS-TTS-GGUF/
│   ├── first_class/MOSS_TTS_FIRST_CLASS_Q4_K_M.gguf   # ~5.2GB
│   └── tokenizer/tokenizer.json + merges.txt + vocab.json
└── MOSS-Audio-Tokenizer-ONNX/
    ├── encoder.onnx  (+ encoder.data)
    └── decoder.onnx
```

Download via:

```bash
# Note: MOSS-TTS-GGUF is a gated repo — request access at
# https://huggingface.co/OpenMOSS-Team/MOSS-TTS-GGUF
HF_ENDPOINT=https://hf-mirror.com \
  hf download OpenMOSS-Team/MOSS-TTS-GGUF \
    --include 'first_class/MOSS_TTS_FIRST_CLASS_Q4_K_M.gguf' 'tokenizer/*' \
    --local-dir /home/lukin/models/MOSS-TTS-GGUF

HF_ENDPOINT=https://hf-mirror.com \
  hf download OpenMOSS-Team/MOSS-Audio-Tokenizer-ONNX \
    --local-dir /home/lukin/models/MOSS-Audio-Tokenizer-ONNX
```

**Build & run:**

```bash
# First build is slow (~30-60 min on 1x 3090): compiles llama.cpp CUDA
docker compose --profile local-gguf up -d --build

# Smoke test
curl -X POST http://localhost:6009/api/generate \
  -F "text=你好，世界" \
  --output out.wav
```

**Tradeoffs vs `local`:**

- ✅ VRAM: ~5-6GB vs ~13GB (fits on 8GB GPUs / shared GPUs)
- ✅ No pytorch in runtime image (smaller image)
- ⚠️ First-token latency: ~5-10s (subprocess cold-load per request). Acceptable for batch article TTS; not suited for interactive use.
- ⚠️ Q4_K_M quantization; subjective quality delta vs bf16 reported by OpenMOSS as minimal for Chinese/English.

## API

All services expose the same API at their respective ports:

### POST `/api/generate`

```bash
# Using a voice from the voices/ library
curl -X POST http://localhost:6006/api/generate \
  -F "text=欢迎收听今日新闻" \
  -F "demo_id=demo-1" \
  --output output.wav

# Using a custom reference audio
curl -X POST http://localhost:6006/api/generate \
  -F "text=Hello world" \
  -F "prompt_audio=@reference.wav" \
  --output output.wav
```

**Parameters:**
| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `text` | string | required | Text to synthesize (supports long texts — auto-segmented at `MAX_SEGMENT_CHARS`) |
| `demo_id` | string | null | Voice from `voices/demo.jsonl` (e.g. `demo-1`) |
| `prompt_audio` | file | null | Reference audio file for voice cloning |
| `enable_text_normalization` | bool | true | Apply text normalization |

### GET `/health`

Returns `{"status": "ok", "model": "...", "loaded": true}`.

## Text Segmentation

Long texts (articles, 10+ minutes) are automatically split at sentence boundaries (。！？.!?) into segments of `MAX_SEGMENT_CHARS` characters (default: 500). Each segment is synthesized independently and concatenated. Voice consistency is maintained by using the first segment's output as a reference prompt for subsequent segments.

## Voice Library

- **Chatterbox voices** (English): 45 WAV files — `Adam`, `Sarah`, `Emily`, etc.
- **MOSS demo voices** (Chinese/English/Japanese): 13 renamed WAV files — `moss_zh_news1`, `moss_en_clear`, etc.

Run `python3 scripts/gen-demo-jsonl.py` to regenerate `voices/demo.jsonl` after adding new voices.

## MicroServices Integration

This service is integrated into the MicroServices `/tts/speech` endpoint as the `moss` engine:

```bash
# Article TTS with MOSS
curl -X POST http://microservices/tts/speech \
  -H 'Content-Type: application/json' \
  -d '{"text": "...article...", "engine": "moss", "language": "zh-cn"}'
```

Set `MOSS_SERVICE_URL` environment variable in MicroServices to point to this service.
