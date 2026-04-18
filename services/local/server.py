"""
MOSS-TTS Local API Server
Compatible with MOSS-TTS-Nano /api/generate interface.

Path: llama.cpp first-class pipeline (Qwen3 backbone GGUF + ONNX audio tokenizer).
VRAM target: ~5-6GB vs ~12.7GB for the transformers bf16 variant.

v2: Daemon mode – GGUF stays resident in GPU VRAM; requests communicate
via stdin/stdout JSON with the persistent llama-moss-tts --daemon-mode process.
Fallback to subprocess-per-request (v1) if daemon fails.
"""
import io
import json
import logging
import os
import re
import struct
import subprocess
import sys
import tempfile
import threading
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
logger = logging.getLogger("moss-local")

app = FastAPI(title="MOSS-TTS Local API")

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

# Add tools/tts to sys.path so we can import moss_tts_processor / moss_tts_onnx
_tts_tools_dir = str(APP_ROOT / "tools/tts")
if _tts_tools_dir not in sys.path:
    sys.path.insert(0, _tts_tools_dir)

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


# ───── MossDaemon: persistent llama-moss-tts process ─────────────────────
class MossDaemon:
    """Persistent llama-moss-tts daemon: loads GGUF once, serves requests via stdin/stdout."""

    def __init__(self):
        self._proc: Optional[subprocess.Popen] = None
        self._lock = threading.Lock()
        self._start()

    def _start(self):
        cmd = [
            str(LLAMA_BIN), "--daemon-mode",
            "-m", str(MODEL_GGUF),
            "-ngl", str(N_GPU_LAYERS),
            "--max-new-tokens", str(MAX_NEW_TOKENS),
            "--text-temperature", str(TEXT_TEMPERATURE),
            "--audio-temperature", str(AUDIO_TEMPERATURE),
        ]
        logger.info("starting moss daemon: %s", " ".join(cmd))
        self._proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True, bufsize=1,
        )
        # Wait for ready signal (reads one JSON line from stdout)
        ready_line = self._proc.stdout.readline().strip()
        if '"ready"' not in ready_line:
            stderr_peek = ""
            try:
                stderr_peek = self._proc.stderr.read(2000)
            except Exception:
                pass
            raise RuntimeError(f"daemon did not send ready signal; got: {ready_line!r}; stderr: {stderr_peek[:500]}")
        logger.info("moss daemon ready (pid=%d)", self._proc.pid)

    def _ensure_alive(self):
        if self._proc is None or self._proc.poll() is not None:
            logger.warning("moss daemon died (rc=%s), restarting",
                           self._proc.poll() if self._proc else "?")
            self._start()

    def generate_codes(
        self,
        ref_path: Path,
        codes_path: Path,
        text_temperature: float = TEXT_TEMPERATURE,
        audio_temperature: float = AUDIO_TEMPERATURE,
        seed: int = 0,
    ) -> dict:
        req = json.dumps({
            "ref_path": str(ref_path),
            "codes_path": str(codes_path),
            "max_new_tokens": MAX_NEW_TOKENS,
            "text_temperature": text_temperature,
            "audio_temperature": audio_temperature,
            "seed": seed,
        })
        with self._lock:
            self._ensure_alive()
            self._proc.stdin.write(req + "\n")
            self._proc.stdin.flush()
            resp_line = self._proc.stdout.readline().strip()
        if not resp_line:
            raise RuntimeError("daemon returned empty response (may have crashed)")
        return json.loads(resp_line)

    def is_alive(self) -> bool:
        return self._proc is not None and self._proc.poll() is None


# ───── Lazy singletons for daemon / tokenizer / onnx ─────────────────────
_tokenizer = None
_onnx_tok = None
_daemon: Optional[MossDaemon] = None

REF_MAGIC   = 0x4652474D  # "MGRF"
REF_VERSION = 1

def _get_tokenizer():
    global _tokenizer
    if _tokenizer is None:
        from moss_tts_processor import Tokenizer
        _tokenizer = Tokenizer(str(TOKENIZER_DIR))
    return _tokenizer

def _get_onnx_tok():
    global _onnx_tok
    if _onnx_tok is None:
        from moss_tts_onnx import OnnxAudioTokenizer
        _onnx_tok = OnnxAudioTokenizer(
            encoder_path=str(ONNX_ENCODER),
            decoder_path=str(ONNX_DECODER),
            use_gpu=not AUDIO_DECODER_CPU,
        )
    return _onnx_tok

def _get_daemon() -> MossDaemon:
    global _daemon
    if _daemon is None:
        _daemon = MossDaemon()
    return _daemon


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


# ───── Audio decode helper (codes.bin → wav) ─────────────────────────────
def _decode_codes_to_wav(codes_path: Path, n_vq: int, out_wav: Path) -> None:
    """Decode raw audio codes to wav using the moss-tts-audio-decode.py script."""
    decode_script = APP_ROOT / "tools/tts/moss-tts-audio-decode.py"
    cmd = [
        sys.executable, str(decode_script),
        "--codes-bin", str(codes_path),
        "--wav-out", str(out_wav),
        "--encoder-onnx", str(ONNX_ENCODER),
        "--decoder-onnx", str(ONNX_DECODER),
    ]
    if AUDIO_DECODER_CPU:
        cmd.append("--cpu")
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0 or not out_wav.is_file():
        raise RuntimeError(f"audio decode failed rc={proc.returncode}: {proc.stderr[-400:]}")


# ───── Daemon-based e2e generation ────────────────────────────────────────
def run_e2e_daemon(text: str, reference_audio: Optional[Path], language: str, out_wav: Path) -> None:
    """Build gen ref in-process, call persistent daemon for inference, decode in-process."""
    from moss_tts_processor import build_generation_prompt, AUDIO_PAD_CODE

    tok = _get_tokenizer()

    # 1. Process reference audio (if any)
    ref_codes = None
    if reference_audio:
        wav_data, sr = sf.read(str(reference_audio), dtype="float32")
        if wav_data.ndim > 1:
            wav_data = wav_data.mean(axis=1)
        if sr != 24000:
            raise RuntimeError(f"reference audio must be 24kHz, got {sr}")
        onnx_tok = _get_onnx_tok()
        ref_codes = np.asarray(onnx_tok.encode(wav_data), dtype=np.int64)

    # 2. Build generation prompt tokens
    input_ids = build_generation_prompt(
        tokenizer=tok, text=text,
        reference_codes=ref_codes, language=language,
    )

    # 3. Write .ref.bin
    with tempfile.NamedTemporaryFile(suffix=".ref.bin", delete=False) as ref_f:
        ref_path = Path(ref_f.name)
    try:
        prompt_frames = int(input_ids.shape[0])
        n_vq = int(input_ids.shape[1] - 1)  # first column is text token
        ref_path.write_bytes(
            struct.pack("<IIIIIII",
                REF_MAGIC, REF_VERSION,
                prompt_frames, n_vq, int(AUDIO_PAD_CODE),
                prompt_frames, 0)
            + input_ids.astype(np.int32).tobytes(order="C")
        )

        # 4. Call daemon to generate raw codes
        with tempfile.NamedTemporaryFile(suffix=".codes.bin", delete=False) as codes_f:
            codes_path = Path(codes_f.name)
        try:
            resp = _get_daemon().generate_codes(ref_path, codes_path)
            if resp.get("status") != "ok":
                raise RuntimeError(f"daemon error: {resp.get('message', resp)}")

            # 5. Decode codes → WAV via ONNX decoder script
            _decode_codes_to_wav(codes_path, n_vq, out_wav)
        finally:
            codes_path.unlink(missing_ok=True)
    finally:
        ref_path.unlink(missing_ok=True)


# ───── Legacy subprocess-per-request e2e ──────────────────────────────────
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

            # Prefer daemon mode; fallback to legacy subprocess
            try:
                run_e2e_daemon(seg, current_ref, lang, out_wav)
                logger.info("daemon ok: seg=%d/%d len=%d", i + 1, len(segments), len(seg))
            except Exception as daemon_err:
                logger.warning("daemon failed for seg %d, falling back to subprocess: %s", i, daemon_err)
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
    daemon_alive = False
    try:
        daemon_alive = _daemon is not None and _daemon.is_alive()
    except Exception:
        pass
    payload = {
        "status":       "ok" if READY else "degraded",
        "ready":        READY,
        "daemon_alive": daemon_alive,
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


@app.on_event("startup")
async def startup_event():
    """Eagerly start the daemon on server boot so model is loaded before first request."""
    if READY:
        try:
            _get_daemon()
            logger.info("daemon pre-started on server boot")
        except Exception as e:
            logger.warning("failed to pre-start daemon: %s (will retry on first request)", e)


if __name__ == "__main__":
    port = int(os.getenv("PORT", "6009"))
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")
