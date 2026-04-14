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

MODEL_ID   = "OpenMOSS-Team/MOSS-TTS-Realtime"
HF_HOME    = os.getenv("HF_HOME", "/hf_cache")
VOICES_DIR = Path(os.getenv("VOICES_DIR", "/app/voices"))
MAX_SEGMENT_CHARS = int(os.getenv("MAX_SEGMENT_CHARS", "500"))
SAMPLE_RATE = 24000

os.environ["HF_HOME"] = HF_HOME

logger.info(f"Loading realtime model {MODEL_ID} ...")
try:
    from transformers import AutoProcessor, AutoModelForCausalLM
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
    )
    model.eval()
    logger.info("Realtime model loaded.")
except Exception as e:
    logger.error(f"Failed to load model: {e}")
    processor = model = None


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
