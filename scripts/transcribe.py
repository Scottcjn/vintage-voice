#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""
VintageVoice — Whisper Transcription Pipeline
Transcribes vintage audio recordings and creates text-audio alignment data.
"""
import argparse
import json
import os
import subprocess
from pathlib import Path


def transcribe_file(audio_path, whisper_model="large-v3-turbo-q5_0", output_dir="data/transcribed"):
    """Transcribe a single audio file using whisper.cpp"""
    stem = Path(audio_path).stem
    out_path = os.path.join(output_dir, f"{stem}.json")

    if os.path.exists(out_path):
        return out_path

    # Convert to 16kHz WAV for whisper
    wav_tmp = f"/tmp/vv_{stem}.wav"
    subprocess.run([
        "ffmpeg", "-y", "-i", audio_path,
        "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le",
        wav_tmp,
    ], capture_output=True, timeout=300)

    if not os.path.exists(wav_tmp):
        print(f"  FFmpeg failed for {audio_path}")
        return None

    # Run whisper.cpp with word-level timestamps
    result = subprocess.run([
        "whisper-cpp",
        "--model", whisper_model,
        "--output-json-full",
        "--output-file", os.path.join(output_dir, stem),
        wav_tmp,
    ], capture_output=True, text=True, timeout=600)

    os.unlink(wav_tmp)

    if os.path.exists(out_path):
        return out_path

    # Fallback: use Python whisper if whisper.cpp not available
    try:
        import whisper
        model = whisper.load_model("large-v3")
        result = model.transcribe(audio_path, word_timestamps=True)
        with open(out_path, "w") as f:
            json.dump(result, f, indent=2)
        return out_path
    except ImportError:
        print("  Neither whisper.cpp nor openai-whisper available")
        return None


def main():
    parser = argparse.ArgumentParser(description="Transcribe vintage audio with Whisper")
    parser.add_argument("--input", default="data/raw", help="Input directory with audio files")
    parser.add_argument("--output", default="data/transcribed", help="Output directory for transcriptions")
    parser.add_argument("--model", default="large-v3-turbo-q5_0", help="Whisper model name")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    audio_exts = {".mp3", ".ogg", ".wav", ".flac", ".m4a"}

    files = []
    for root, _, filenames in os.walk(args.input):
        for fn in filenames:
            if Path(fn).suffix.lower() in audio_exts:
                files.append(os.path.join(root, fn))

    print(f"Found {len(files)} audio files to transcribe")

    for i, f in enumerate(sorted(files)):
        print(f"[{i+1}/{len(files)}] {os.path.basename(f)}")
        result = transcribe_file(f, args.model, args.output)
        if result:
            print(f"  -> {result}")


if __name__ == "__main__":
    main()
