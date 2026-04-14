"""
MOSS-TTS-Realtime (1.7B) API Server
Compatible with MOSS-TTS-Nano /api/generate interface
Low-latency streaming inference with multi-turn context
"""
import os, re, io, logging, json
from pathlib import Path
from typing import Optional

import numpy as np
import soundfile as sf
import torch
from fastapi import FastAPI, Form, UploadFile, File, HTTPException
from fastapi.responses import Response, StreamingResponse
import uvicorn

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("moss-realtime")

app = FastAPI(title="MOSS-TTS-Realtime API")

MODEL_ID   = os.getenv("MODEL_PATH", "/home/lukin/models/MOSS-TTS-Realtime")
HF_HOME    = os.getenv("HF_HOME", "/hf_cache")
VOICES_DIR = Path(os.getenv("VOICES_DIR", "/app/voices"))
MAX_SEGMENT_CHARS = int(os.getenv("MAX_SEGMENT_CHARS", "500"))
SAMPLE_RATE = 24000

os.environ["HF_HOME"] = HF_HOME

device = "cuda" if torch.cuda.is_available() else "cpu"
dtype  = torch.bfloat16 if device == "cuda" else torch.float32

if device == "cuda":
    torch.backends.cuda.enable_cudnn_sdp(False)
    torch.backends.cuda.enable_flash_sdp(True)
    torch.backends.cuda.enable_mem_efficient_sdp(True)
    torch.backends.cuda.enable_math_sdp(True)

logger.info(f"Loading realtime model from {MODEL_ID} on {device} ...")
try:
    # Compatibility patches for transformers 5.0.0
    from transformers import processing_utils
    if not hasattr(processing_utils, 'MODALITY_TO_BASE_CLASS_MAPPING'):
        processing_utils.MODALITY_TO_BASE_CLASS_MAPPING = {}

    from transformers import AutoProcessor, AutoModel
    processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
    if hasattr(processor, 'audio_tokenizer'):
        processor.audio_tokenizer = processor.audio_tokenizer.to(device)
    SAMPLE_RATE = getattr(getattr(processor, 'model_config', None), 'sampling_rate', 24000)

    model = AutoModel.from_pretrained(
        MODEL_ID,
        trust_remote_code=True,
        attn_implementation="sdpa" if device == "cuda" else "eager",
        torch_dtype=dtype,
    ).to(device)
    model.eval()

    # Patch num_hidden_layers for DynamicCache compatibility
    cfg = model.config
    if not hasattr(cfg, 'num_hidden_layers'):
        lm_cfg = getattr(cfg, 'language_config', None)
        if lm_cfg and hasattr(lm_cfg, 'num_hidden_layers'):
            cfg.__class__.num_hidden_layers = property(lambda self: self.language_config.num_hidden_layers)
        elif hasattr(cfg, 'local_num_layers'):
            cfg.__class__.num_hidden_layers = property(lambda self: self.local_num_layers)

    logger.info(f"Realtime model loaded. SAMPLE_RATE={SAMPLE_RATE}")
    MODEL_LOADED = True
except Exception as e:
    logger.error(f"Failed to load model: {e}", exc_info=True)
    processor = model = None
    MODEL_LOADED = False


def split_text(text: str, max_chars: int = MAX_SEGMENT_CHARS) -> list[str]:
    if len(text) <= max_chars:
        return [text]
    segments, buf = [], ""
    for sentence in re.split(r'(?<=[。！？.!?])\s*', text):
        if not sentence.strip():
            continue
        if len(buf) + len(sentence) <= max_chars:
            buf += sentence
        else:
            if buf:
                segments.append(buf.strip())
            buf = sentence
    if buf.strip():
        segments.append(buf.strip())
    return segments or [text]


def load_voice_audio(voice_name: str) -> Optional[np.ndarray]:
    for ext in [".wav", ".flac"]:
        p = VOICES_DIR / f"{voice_name}{ext}"
        if p.exists():
            audio, sr = sf.read(str(p))
            if audio.ndim > 1:
                audio = audio.mean(axis=1)
            return audio
    return None


@torch.inference_mode()
def generate_segment(text: str, prompt_audio: Optional[np.ndarray] = None) -> np.ndarray:
    if model is None:
        raise RuntimeError("Model not loaded")
    inputs = processor(text=text, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    if prompt_audio is not None:
        ai = processor(audio=prompt_audio, sampling_rate=SAMPLE_RATE, return_tensors="pt")
        inputs.update({k: v.to(model.device) for k, v in ai.items()})
    out = model.generate(**inputs, max_new_tokens=2048)
    if hasattr(processor, "decode_audio"):
        return processor.decode_audio(out)
    return out[0].float().cpu().numpy()


def generate_speech(text: str, voice_name: Optional[str] = None,
                    prompt_audio_bytes: Optional[bytes] = None) -> bytes:
    prompt_audio = None
    if prompt_audio_bytes:
        with io.BytesIO(prompt_audio_bytes) as bio:
            prompt_audio, _ = sf.read(bio)
    elif voice_name:
        prompt_audio = load_voice_audio(voice_name)

    segments = split_text(text)
    logger.info(f"Realtime: {len(segments)} segment(s), total chars={len(text)}")
    parts = []
    for i, seg in enumerate(segments):
        wav = generate_segment(seg, prompt_audio)
        parts.append(wav)
        if prompt_audio is None and i == 0:
            prompt_audio = wav[:SAMPLE_RATE * 3] if len(wav) > SAMPLE_RATE * 3 else wav

    full = np.concatenate(parts) if len(parts) > 1 else parts[0]
    buf = io.BytesIO()
    sf.write(buf, full, SAMPLE_RATE, format="WAV", subtype="PCM_16")
    return buf.getvalue()


@app.get("/health")
def health():
    return {"status": "ok", "model": MODEL_ID, "loaded": model is not None}


@app.post("/api/generate")
async def api_generate(
    text: str = Form(...),
    demo_id: Optional[str] = Form(None),
    enable_text_normalization: bool = Form(True),
    prompt_audio: Optional[UploadFile] = File(None),
):
    if not text.strip():
        raise HTTPException(400, "text is required")

    voice_name: Optional[str] = None
    if demo_id:
        jsonl_path = VOICES_DIR / "demo.jsonl"
        if jsonl_path.exists():
            for line in jsonl_path.read_text().splitlines():
                entry = json.loads(line)
                if entry.get("demo_id") == demo_id:
                    voice_name = Path(entry.get("audio_path", "")).stem
                    break

    prompt_bytes = await prompt_audio.read() if prompt_audio else None

    try:
        wav_bytes = generate_speech(text, voice_name=voice_name, prompt_audio_bytes=prompt_bytes)
        return Response(content=wav_bytes, media_type="audio/wav")
    except Exception as e:
        logger.error(f"Generation error: {e}")
        raise HTTPException(500, str(e))


if __name__ == "__main__":
    port = int(os.getenv("PORT", "6008"))
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")
