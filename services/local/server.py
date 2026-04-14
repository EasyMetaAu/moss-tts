"""
MOSS-TTS-Local-Transformer (1.7B) API Server
Compatible with MOSS-TTS-Nano /api/generate interface

Uses the official MOSS-TTS processor.build_user_message() API with trust_remote_code=True
"""
import os, re, io, logging, json, tempfile
from pathlib import Path
from typing import Optional

import torch
import torchaudio
import soundfile as sf
import numpy as np
from fastapi import FastAPI, Form, UploadFile, File, HTTPException
from fastapi.responses import Response
import uvicorn

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("moss-local")

app = FastAPI(title="MOSS-TTS-Local-Transformer API")

# ───── Config ──────────────────────────────────────────────────────────────
MODEL_PATH  = os.getenv("MODEL_PATH", "/home/lukin/models/MOSS-TTS-Local-Transformer")
VOICES_DIR  = Path(os.getenv("VOICES_DIR", "/app/voices"))
MAX_SEGMENT_CHARS = int(os.getenv("MAX_SEGMENT_CHARS", "500"))
HF_HOME    = os.getenv("HF_HOME", "/hf_cache")
os.environ["HF_HOME"] = HF_HOME

# ───── Model loading ──────────────────────────────────────────────────────
logger.info(f"Loading model from {MODEL_PATH} ...")

device = "cuda" if torch.cuda.is_available() else "cpu"
dtype  = torch.bfloat16 if device == "cuda" else torch.float32
logger.info(f"Device: {device}, dtype: {dtype}")

# Fix cuDNN SDPA backend (broken on some setups)
if device == "cuda":
    torch.backends.cuda.enable_cudnn_sdp(False)
    torch.backends.cuda.enable_flash_sdp(True)
    torch.backends.cuda.enable_mem_efficient_sdp(True)
    torch.backends.cuda.enable_math_sdp(True)

try:
    from transformers import AutoModel, AutoProcessor
    processor = AutoProcessor.from_pretrained(MODEL_PATH, trust_remote_code=True)
    processor.audio_tokenizer = processor.audio_tokenizer.to(device)
    SAMPLE_RATE = processor.model_config.sampling_rate
    
    model = AutoModel.from_pretrained(
        MODEL_PATH,
        trust_remote_code=True,
        attn_implementation="sdpa" if device == "cuda" else "eager",
        torch_dtype=dtype,
    ).to(device)
    model.eval()
    logger.info(f"Model loaded. SAMPLE_RATE={SAMPLE_RATE}")
    MODEL_LOADED = True
except Exception as e:
    logger.error(f"Failed to load model: {e}", exc_info=True)
    processor = model = None
    SAMPLE_RATE = 24000
    MODEL_LOADED = False


# ───── Helpers ─────────────────────────────────────────────────────────────
def split_text(text: str, max_chars: int = MAX_SEGMENT_CHARS) -> list[str]:
    """Split text at sentence boundaries."""
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
            # Long single sentence: split on comma
            if len(sentence) > max_chars:
                for chunk in re.split(r'(?<=[，,、；;])\s*', sentence):
                    if chunk.strip():
                        if len(buf) + len(chunk) <= max_chars:
                            buf = chunk
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
    """Find a voice WAV file by name."""
    for ext in [".wav", ".flac"]:
        p = VOICES_DIR / f"{voice_name}{ext}"
        if p.exists():
            return p
    return None


@torch.inference_mode()
def generate_segment(text: str, ref_audio_path: Optional[str] = None) -> np.ndarray:
    """Generate audio for a single text segment using the MOSS Local-Transformer API."""
    if model is None or processor is None:
        raise RuntimeError("Model not loaded")
    
    if ref_audio_path:
        conversation = [processor.build_user_message(text=text, reference=[ref_audio_path])]
    else:
        conversation = [processor.build_user_message(text=text)]
    
    batch = processor(conversation, mode="generation")
    input_ids       = batch["input_ids"].to(device)
    attention_mask  = batch["attention_mask"].to(device)
    
    outputs = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=4096,
    )
    
    decoded = processor.decode(outputs)
    audio_tensor = decoded[0].audio_codes_list[0]  # shape: [C, T] or [T]
    
    # Convert to numpy mono
    wav = audio_tensor.float().cpu()
    if wav.dim() > 1:
        wav = wav.mean(0)
    return wav.numpy()


def generate_speech(text: str, ref_audio_path: Optional[str] = None) -> bytes:
    """Full speech generation with text segmentation."""
    segments = split_text(text)
    logger.info(f"Generating {len(segments)} segment(s) for {len(text)} chars")
    
    parts = []
    for i, seg in enumerate(segments):
        logger.info(f"  Segment {i+1}/{len(segments)}: {seg[:60]}")
        wav = generate_segment(seg, ref_audio_path=ref_audio_path)
        parts.append(wav)
        # After first segment, use its output as ref for voice consistency
        if ref_audio_path is None and i == 0 and len(segments) > 1:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                sf.write(tmp.name, wav, SAMPLE_RATE, subtype="PCM_16")
                ref_audio_path = tmp.name
    
    full_audio = np.concatenate(parts) if len(parts) > 1 else parts[0]
    buf = io.BytesIO()
    sf.write(buf, full_audio, SAMPLE_RATE, format="WAV", subtype="PCM_16")
    return buf.getvalue()


# ───── Routes ─────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {
        "status": "ok",
        "model": MODEL_PATH,
        "loaded": MODEL_LOADED,
        "device": device,
        "sample_rate": SAMPLE_RATE,
    }


@app.post("/api/generate")
async def api_generate(
    text: str = Form(...),
    demo_id: Optional[str] = Form(None),
    enable_text_normalization: bool = Form(True),
    prompt_audio: Optional[UploadFile] = File(None),
):
    """Generate speech — compatible with MOSS-TTS-Nano /api/generate interface."""
    if not text.strip():
        raise HTTPException(400, "text is required")
    if not MODEL_LOADED:
        raise HTTPException(503, "Model not loaded")
    
    # Resolve voice from demo_id
    ref_audio_path: Optional[str] = None
    if prompt_audio:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp.write(await prompt_audio.read())
            ref_audio_path = tmp.name
    elif demo_id:
        jsonl_path = VOICES_DIR / "demo.jsonl"
        voice_name = None
        if jsonl_path.exists():
            for line in jsonl_path.read_text().splitlines():
                entry = json.loads(line)
                if entry.get("demo_id") == demo_id:
                    voice_name = entry.get("voice_name")
                    break
        if voice_name:
            vf = find_voice_file(voice_name)
            if vf:
                ref_audio_path = str(vf)
    
    try:
        wav_bytes = generate_speech(text, ref_audio_path=ref_audio_path)
        return Response(content=wav_bytes, media_type="audio/wav")
    except Exception as e:
        logger.error(f"Generation error: {e}", exc_info=True)
        raise HTTPException(500, str(e))


if __name__ == "__main__":
    port = int(os.getenv("PORT", "6007"))
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")
