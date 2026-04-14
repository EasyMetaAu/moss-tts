FROM moss-tts-nano:latest

WORKDIR /app

COPY server.py /realtime_server.py

ENV MODEL_PATH=/models/MOSS-TTS-Realtime
ENV VOICES_DIR=/app/voices
ENV PORT=6008

CMD ["python3", "/realtime_server.py"]
