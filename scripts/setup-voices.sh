#!/bin/bash
# Copy voice files from their source locations into the voices/ directory
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
VOICES_DIR="$REPO_ROOT/voices"
mkdir -p "$VOICES_DIR"

# ── Chatterbox voices ──────────────────────────────────────────────────────
CHATTERBOX_SRC="${CHATTERBOX_VOICES_DIR:-/home/lukin/code/chatterbox-api/voices}"
if [ -d "$CHATTERBOX_SRC" ]; then
    echo "Copying Chatterbox voices from $CHATTERBOX_SRC ..."
    cp "$CHATTERBOX_SRC"/*.wav "$VOICES_DIR/" 2>/dev/null || true
    echo "  ✅ Chatterbox WAV voices copied"
else
    echo "  ⚠️  Chatterbox voices dir not found: $CHATTERBOX_SRC"
fi

# ── MOSS demo voices (extract from running container) ─────────────────────
CONTAINER="${MOSS_NANO_CONTAINER:-moss-tts-nano}"
if docker ps --format '{{.Names}}' | grep -q "^${CONTAINER}$"; then
    echo "Extracting MOSS demo voices from container $CONTAINER ..."
    TMPDIR=$(mktemp -d)
    docker cp "$CONTAINER":/app/assets "$TMPDIR/"
    
    # Rename and copy MOSS voices with moss_ prefix
    declare -A RENAME=(
        ["zh_1"]="moss_zh_news1"
        ["zh_3"]="moss_zh_formal"
        ["zh_4"]="moss_zh_clear"
        ["zh_6"]="moss_zh_broadcast"
        ["zh_10"]="moss_zh_culture"
        ["zh_11"]="moss_zh_calm"
        ["en_2"]="moss_en_warm"
        ["en_3"]="moss_en_clear"
        ["en_4"]="moss_en_news"
        ["en_6"]="moss_en_energetic"
        ["en_7"]="moss_en_professional"
        ["en_8"]="moss_en_soft"
        ["jp_2"]="moss_jp_standard"
    )
    
    for src_stem in "${!RENAME[@]}"; do
        dst_name="${RENAME[$src_stem]}"
        src_file="$TMPDIR/assets/${src_stem}.wav"
        if [ -f "$src_file" ]; then
            cp "$src_file" "$VOICES_DIR/${dst_name}.wav"
            echo "  ✅ $src_stem → $dst_name.wav"
        fi
    done
    rm -rf "$TMPDIR"
else
    echo "  ⚠️  Container $CONTAINER not running — skipping MOSS voices extraction"
fi

# ── Generate demo.jsonl ──────────────────────────────────────────────────
python3 "$SCRIPT_DIR/gen-demo-jsonl.py" "$VOICES_DIR"
echo ""
echo "✅ Voice setup complete. Files in $VOICES_DIR:"
ls "$VOICES_DIR"/*.wav 2>/dev/null | wc -l | xargs -I{} echo "  {} WAV files"
