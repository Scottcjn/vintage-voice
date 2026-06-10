#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""
VintageVoice — GPU-Accelerated Whisper Transcription
Uses transformers pipeline on V100 for fast batch transcription.
Creates aligned text-audio pairs for F5-TTS fine-tuning.
"""
import argparse
import csv
import json
import os
import torch
from pathlib import Path


def load_whisper(model_name="openai/whisper-large-v3-turbo", device="cuda:0"):
    """Load Whisper model on GPU"""
    from transformers import pipeline

    pipe = pipeline(
        "automatic-speech-recognition",
        model=model_name,
        torch_dtype=torch.float16,
        device=device,
        return_timestamps="word",
    )
    return pipe


def transcribe_segment(pipe, audio_path):
    """Transcribe a single audio segment with word timestamps"""
    try:
        result = pipe(audio_path, return_timestamps="word")
        text = result["text"].strip()
        chunks = result.get("chunks", [])
        return {
            "text": text,
            "words": [
                {"word": c["text"], "start": c["timestamp"][0], "end": c["timestamp"][1]}
                for c in chunks
                if c.get("timestamp") and c["timestamp"][0] is not None
            ],
        }
    except Exception as e:
        print(f"  Transcription error: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(description="Transcribe vintage audio segments with Whisper")
    parser.add_argument("--manifest", default="/mnt/18tb/vintage_voice_processed/manifest.csv")
    parser.add_argument("--output", default="/mnt/18tb/vintage_voice_processed/transcriptions")
    parser.add_argument("--model", default="openai/whisper-large-v3-turbo")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    # Load manifest
    segments = []
    with open(args.manifest) as f:
        reader = csv.DictReader(f, delimiter="|")
        for row in reader:
            segments.append(row)

    print(f"Loaded {len(segments)} segments from manifest")
    print(f"Loading Whisper on {args.device}...")
    pipe = load_whisper(args.model, args.device)
    print("Whisper ready")

    # Transcribe all segments
    results = []
    for i, seg in enumerate(segments):
        audio_path = seg["path"]
        if not os.path.exists(audio_path):
            continue

        stem = Path(audio_path).stem
        out_json = os.path.join(args.output, f"{stem}.json")

        if os.path.exists(out_json):
            with open(out_json) as f:
                results.append(json.load(f))
            continue

        result = transcribe_segment(pipe, audio_path)
        if result and result["text"]:
            result["audio_path"] = audio_path
            result["duration"] = float(seg["duration"])

            with open(out_json, "w") as f:
                json.dump(result, f, indent=2)
            results.append(result)

        if (i + 1) % 25 == 0:
            print(f"  [{i+1}/{len(segments)}] Transcribed. Total with text: {len(results)}")

    # Write training manifest (F5-TTS format)
    train_manifest = os.path.join(args.output, "train.csv")
    with open(train_manifest, "w") as f:
        f.write("audio_path|text|duration\n")
        for r in results:
            text = r["text"].replace("|", " ").replace("\n", " ")
            f.write(f"{r['audio_path']}|{text}|{r['duration']:.2f}\n")

    print(f"\nDone! {len(results)} transcribed segments")
    print(f"Training manifest: {train_manifest}")

    # Stats
    total_dur = sum(r["duration"] for r in results)
    avg_words = sum(len(r["text"].split()) for r in results) / max(len(results), 1)
    print(f"Total transcribed audio: {total_dur/3600:.1f} hours")
    print(f"Average words per segment: {avg_words:.0f}")


if __name__ == "__main__":
    main()
