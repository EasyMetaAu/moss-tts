# Use the MOSS-TTS-Nano image as base (already has torch, transformers, soundfile, fastapi, uvicorn)
FROM moss-tts-nano:latest

WORKDIR /app

# Copy server code
COPY server.py /local_server.py

ENV MODEL_PATH=/models/MOSS-TTS-Local-Transformer
ENV VOICES_DIR=/app/voices
ENV PORT=6007

CMD ["python3", "/local_server.py"]
