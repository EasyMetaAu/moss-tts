#!/usr/bin/env python3
"""
Auto-scan voices/ directory and generate demo.jsonl
Usage: python3 scripts/gen-demo-jsonl.py [voices_dir] [output_jsonl]
"""
import json, sys
from pathlib import Path

VOICES_DIR = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).parent.parent / "voices"
OUTPUT     = Path(sys.argv[2]) if len(sys.argv) > 2 else VOICES_DIR / "demo.jsonl"

SAMPLE_TEXTS = {
    "zh": "欢迎使用 MOSS 语音合成系统，这是一段示例语音。",
    "en": "Welcome to MOSS text-to-speech. This is a sample audio.",
    "ja": "MOSSテキスト音声合成へようこそ。これはサンプル音声です。",
}

def infer_lang(stem: str) -> str:
    s = stem.lower()
    if s.startswith("moss_zh") or s.startswith("zh_"):
        return "zh"
    if s.startswith("moss_en") or s.startswith("en_"):
        return "en"
    if s.startswith("moss_jp") or s.startswith("jp_"):
        return "ja"
    return "en"   # default

entries = []
voice_files = sorted([
    f for f in VOICES_DIR.iterdir()
    if f.is_file() and f.suffix.lower() in {".wav", ".flac", ".mp3"}
    and f.name != "demo.jsonl"
])

for idx, vf in enumerate(voice_files, start=1):
    lang = infer_lang(vf.stem)
    entry = {
        "demo_id": f"demo-{idx}",
        "audio_path": str(vf),
        "language": lang,
        "sample_text": SAMPLE_TEXTS.get(lang, SAMPLE_TEXTS["en"]),
        "voice_name": vf.stem,
    }
    entries.append(entry)

OUTPUT.write_text("\n".join(json.dumps(e, ensure_ascii=False) for e in entries) + "\n")
print(f"Generated {len(entries)} entries → {OUTPUT}")
for e in entries:
    print(f"  {e['demo_id']:10s} {e['voice_name']}")
