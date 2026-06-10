#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Build F5-TTS training CSV from Whisper transcriptions."""
import json
import os
import glob
import sys

trans_dir = sys.argv[1] if len(sys.argv) > 1 else "/mnt/18tb/vintage_voice_processed/transcriptions"
out_csv = sys.argv[2] if len(sys.argv) > 2 else "/mnt/18tb/vintage_voice_processed/f5_train.csv"

csv_lines = []
for jf in sorted(glob.glob(os.path.join(trans_dir, "*.json"))):
    with open(jf) as f:
        data = json.load(f)
    audio = data.get("audio_path", "")
    text = data.get("text", "").strip()
    if audio and text and os.path.exists(audio):
        text = text.replace("|", " ").replace("\n", " ").strip()
        if len(text) > 5:
            csv_lines.append(f"{audio}|{text}")

with open(out_csv, "w") as f:
    f.write("audio_file|text\n")
    f.write("\n".join(csv_lines) + "\n")

print(f"Training samples: {len(csv_lines)}")
