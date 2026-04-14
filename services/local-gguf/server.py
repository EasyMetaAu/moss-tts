"""
MOSS-TTS Local-GGUF API Server
Compatible with MOSS-TTS-Nano /api/generate interface.

Path: llama.cpp first-class pipeline (Qwen3 backbone GGUF + ONNX audio tokenizer).
VRAM target: ~5-6GB vs ~12.7GB for the transformers bf16 variant.

Note v1: each request spawns `tools/tts/moss-tts-firstclass-e2e.py` as a subprocess.
The llama-moss-tts binary cold-loads the GGUF per invocation, so first-token latency
is ~5-10s. Throughput is adequate for batch article TTS; future work will move to
a persistent inference loop via llama-cpp-python once the MOSS-specific kernels
are upstreamed.
"""
import io
import json
import logging
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Optional

import numpy as np
import soundfile as sf
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse, Response
import uvicorn

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
)
logger = logging.getLogger("moss-local-gguf")

app = FastAPI(title="MOSS-TTS Local-GGUF API")

# ───── Config (override via env) ──────────────────────────────────────────
APP_ROOT          = Path("/app")
LLAMA_BIN         = Path(os.getenv("LLAMA_BIN", APP_ROOT / "bin/llama-moss-tts"))
E2E_SCRIPT        = APP_ROOT / "tools/tts/moss-tts-firstclass-e2e.py"
MODEL_GGUF        = Path(os.getenv("MODEL_GGUF", "/models/MOSS-TTS-GGUF/first_class/MOSS_TTS_FIRST_CLASS_Q4_K_M.gguf"))
TOKENIZER_DIR     = Path(os.getenv("TOKENIZER_DIR", "/models/MOSS-TTS-GGUF/tokenizer"))
ONNX_ENCODER      = Path(os.getenv("ONNX_ENCODER", "/models/MOSS-Audio-Tokenizer-ONNX/encoder.onnx"))
ONNX_DECODER      = Path(os.getenv("ONNX_DECODER", "/models/MOSS-Audio-Tokenizer-ONNX/decoder.onnx"))
VOICES_DIR        = Path(os.getenv("VOICES_DIR", APP_ROOT / "voices"))
MAX_SEGMENT_CHARS = int(os.getenv("MAX_SEGMENT_CHARS", "500"))
MAX_NEW_TOKENS    = int(os.getenv("MAX_NEW_TOKENS", "4096"))
N_GPU_LAYERS      = int(os.getenv("N_GPU_LAYERS", "-1"))
TEXT_TEMPERATURE  = float(os.getenv("TEXT_TEMPERATURE", "1.5"))
AUDIO_TEMPERATURE = float(os.getenv("AUDIO_TEMPERATURE", "1.7"))
SAMPLE_RATE       = int(os.getenv("SAMPLE_RATE", "24000"))  # MOSS-TTS-Delay output is 24kHz mono
LANGUAGE_DEFAULT  = os.getenv("LANGUAGE_DEFAULT", "zh")
# Run ONNX audio encoder/decoder on CPU (default=1: avoids cuDNN runtime requirement).
# Set to 0 only if cuDNN 9.x is available and onnxruntime-gpu >= 1.20 is installed.
AUDIO_DECODER_CPU = os.getenv("AUDIO_DECODER_CPU", "1") not in ("0", "false", "False")

# ───── Preflight ──────────────────────────────────────────────────────────
MISSING: list[str] = []
if os.getenv("MOSS_PREFLIGHT", "1") not in ("0", "false", "False"):
    for name, p in [
        ("LLAMA_BIN", LLAMA_BIN),
        ("E2E_SCRIPT", E2E_SCRIPT),
        ("MODEL_GGUF", MODEL_GGUF),
        ("TOKENIZER_DIR/tokenizer.json", TOKENIZER_DIR / "tokenizer.json"),
        ("ONNX_ENCODER", ONNX_ENCODER),
        ("ONNX_DECODER", ONNX_DECODER),
    ]:
        if not p.exists():
            MISSING.append(f"{name}={p}")
            logger.warning("preflight: missing %s -> %s", name, p)
        else:
            logger.info("preflight: %s -> %s (ok)", name, p)

READY = len(MISSING) == 0
if not READY:
    logger.error("preflight FAILED; /api/generate will 503 until weights are mounted")


# ───── Helpers ─────────────────────────────────────────────────────────────
def detect_language(text: str) -> str:
    """zh if >=1 Han char, else en."""
    return "zh" if re.search(r"[\u4e00-\u9fff]", text) else "en"


def split_text(text: str, max_chars: int = MAX_SEGMENT_CHARS) -> list[str]:
    if len(text) <= max_chars:
        return [text]
    segments, buf = [], ""
    for sentence in re.split(r"(?<=[。！？.!?])\s*", text):
        if not sentence.strip():
            continue
        if len(buf) + len(sentence) <= max_chars:
            buf += sentence
        else:
            if buf:
                segments.append(buf.strip())
                buf = ""
            if len(sentence) > max_chars:
                for chunk in re.split(r"(?<=[，,、；;])\s*", sentence):
                    if not chunk.strip():
                        continue
                    if len(buf) + len(chunk) <= max_chars:
                        buf += chunk
                    else:
                        if buf:
                            segments.append(buf.strip())
                        buf = chunk
            else:
                buf = sentence
    if buf.strip():
        segments.append(buf.strip())
    return segments or [text]


def find_voice_file(voice_name: str) -> Optional[Path]:
    for ext in (".wav", ".flac"):
        p = VOICES_DIR / f"{voice_name}{ext}"
        if p.exists():
            return p
    return None


def resolve_voice(demo_id: Optional[str]) -> Optional[Path]:
    if not demo_id:
        return None
    # Prefer voices.jsonl (new layout), fall back to demo.jsonl
    for manifest_name in ("voices.jsonl", "demo.jsonl"):
        manifest = VOICES_DIR / manifest_name
        if not manifest.exists():
            continue
        for line in manifest.read_text().splitlines():
            if not line.strip():
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            if entry.get("demo_id") == demo_id or entry.get("id") == demo_id:
                voice_name = entry.get("voice_name") or entry.get("name")
                if voice_name:
                    return find_voice_file(voice_name)
    return None


def run_e2e(text: str, reference_audio: Optional[Path], language: str, out_wav: Path) -> None:
    """Invoke the official hybrid e2e script once; produces a wav file."""
    cmd = [
        sys.executable,
        str(E2E_SCRIPT),
        "--model-gguf",   str(MODEL_GGUF),
        "--tokenizer-dir", str(TOKENIZER_DIR),
        "--onnx-encoder",  str(ONNX_ENCODER),
        "--onnx-decoder",  str(ONNX_DECODER),
        "--llama-bin",     str(LLAMA_BIN),
        "--output-wav",    str(out_wav),
        "--text",          text,
        "--language",      language,
        "--max-new-tokens", str(MAX_NEW_TOKENS),
        "--text-temperature", str(TEXT_TEMPERATURE),
        "--audio-temperature", str(AUDIO_TEMPERATURE),
        "--n-gpu-layers",  str(N_GPU_LAYERS),
    ]
    if AUDIO_DECODER_CPU:
        cmd.append("--audio-decoder-cpu")
    if reference_audio:
        cmd.extend(["--reference-audio", str(reference_audio)])

    logger.info("invoke llama-moss-tts: out=%s text_len=%d ref=%s", out_wav.name, len(text), reference_audio)
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        logger.error("llama-moss-tts failed rc=%d\nstdout:\n%s\nstderr:\n%s",
                     proc.returncode, proc.stdout[-2000:], proc.stderr[-2000:])
        raise RuntimeError(f"llama-moss-tts rc={proc.returncode}; {proc.stderr[-400:]}")
    if not out_wav.is_file():
        raise RuntimeError(f"llama-moss-tts produced no wav at {out_wav}")


def generate_speech(text: str, reference_audio: Optional[Path], language: Optional[str]) -> bytes:
    lang = language or detect_language(text)
    segments = split_text(text)
    logger.info("generate: segments=%d lang=%s", len(segments), lang)

    parts: list[np.ndarray] = []
    current_ref = reference_audio
    with tempfile.TemporaryDirectory(prefix="moss-gguf-") as tmpdir:
        tmp = Path(tmpdir)
        for i, seg in enumerate(segments):
            out_wav = tmp / f"seg_{i:03d}.wav"
            run_e2e(seg, current_ref, lang, out_wav)
            wav, sr = sf.read(out_wav, dtype="float32", always_2d=False)
            if wav.ndim > 1:
                wav = wav.mean(axis=1)
            parts.append(wav)
            # Voice consistency: reuse first segment as reference for subsequent ones
            if reference_audio is None and i == 0 and len(segments) > 1:
                current_ref = out_wav  # keep tmpdir alive until all segments done

    full = np.concatenate(parts) if len(parts) > 1 else parts[0]
    buf = io.BytesIO()
    sf.write(buf, full, SAMPLE_RATE, format="WAV", subtype="PCM_16")
    return buf.getvalue()


# ───── Routes ─────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    payload = {
        "status":       "ok" if READY else "degraded",
        "ready":        READY,
        "missing":      MISSING,
        "model_gguf":   str(MODEL_GGUF),
        "tokenizer":    str(TOKENIZER_DIR),
        "onnx_encoder": str(ONNX_ENCODER),
        "onnx_decoder": str(ONNX_DECODER),
        "llama_bin":    str(LLAMA_BIN),
        "sample_rate":  SAMPLE_RATE,
        "n_gpu_layers": N_GPU_LAYERS,
    }
    if not READY:
        return JSONResponse(status_code=503, content=payload)
    return payload


@app.post("/api/generate")
async def api_generate(
    text: str = Form(...),
    demo_id: Optional[str] = Form(None),
    enable_text_normalization: bool = Form(True),  # accepted but not enforced in v1
    language: Optional[str] = Form(None),
    prompt_audio: Optional[UploadFile] = File(None),
):
    if not text.strip():
        raise HTTPException(400, "text is required")
    if not READY:
        raise HTTPException(503, f"service not ready; missing: {MISSING}")

    ref_path: Optional[Path] = None
    tmp_ref: Optional[tempfile._TemporaryFileWrapper] = None  # type: ignore
    try:
        if prompt_audio is not None:
            tmp_ref = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            tmp_ref.write(await prompt_audio.read())
            tmp_ref.close()
            ref_path = Path(tmp_ref.name)
        elif demo_id:
            ref_path = resolve_voice(demo_id)
            if ref_path is None:
                logger.warning("demo_id %s not found in %s", demo_id, VOICES_DIR)

        wav_bytes = generate_speech(text, reference_audio=ref_path, language=language)
        return Response(content=wav_bytes, media_type="audio/wav")
    except Exception as e:
        logger.error("generate error: %s", e, exc_info=True)
        raise HTTPException(500, str(e))
    finally:
        if tmp_ref is not None:
            try:
                os.unlink(tmp_ref.name)
            except OSError:
                pass


if __name__ == "__main__":
    port = int(os.getenv("PORT", "6009"))
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")
