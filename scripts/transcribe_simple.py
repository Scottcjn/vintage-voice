#!/usr/bin/env python3
"""
VintageVoice — Simple Whisper Transcription (no torchcodec dependency)
Uses openai-whisper or faster-whisper directly instead of transformers pipeline.
"""
import argparse
import csv
import json
import os
import sys
from pathlib import Path


def transcribe_with_faster_whisper(manifest_path, output_dir, device="cuda:0"):
    """Use faster-whisper (CTranslate2 backend, fast, no torchcodec)"""
    from faster_whisper import WhisperModel

    gpu_id = int(device.split(":")[-1]) if "cuda" in device else 0
    model = WhisperModel("large-v3", device="cuda", compute_type="float16", device_index=gpu_id)

    segments_list = []
    with open(manifest_path) as f:
        reader = csv.DictReader(f, delimiter="|")
        for row in reader:
            segments_list.append(row)

    print(f"Transcribing {len(segments_list)} segments with faster-whisper on {device}")
    results = []

    for i, seg in enumerate(segments_list):
        audio_path = seg["path"]
        if not os.path.exists(audio_path):
            continue

        stem = Path(audio_path).stem
        out_json = os.path.join(output_dir, f"{stem}.json")
        if os.path.exists(out_json):
            with open(out_json) as f:
                results.append(json.load(f))
            continue

        try:
            segments, info = model.transcribe(audio_path, word_timestamps=True)
            text_parts = []
            for s in segments:
                text_parts.append(s.text)

            text = " ".join(text_parts).strip()
            if text:
                result = {
                    "audio_path": audio_path,
                    "text": text,
                    "duration": float(seg["duration"]),
                }
                with open(out_json, "w") as f:
                    json.dump(result, f, indent=2)
                results.append(result)
        except Exception as e:
            print(f"  Error on {audio_path}: {e}")

        if (i + 1) % 100 == 0:
            print(f"  [{i+1}/{len(segments_list)}] Transcribed {len(results)} segments")

    return results


def transcribe_with_openai_whisper(manifest_path, output_dir, device="cuda:0"):
    """Use openai-whisper (original, reliable)"""
    import whisper

    model = whisper.load_model("large-v3", device=device)

    segments_list = []
    with open(manifest_path) as f:
        reader = csv.DictReader(f, delimiter="|")
        for row in reader:
            segments_list.append(row)

    print(f"Transcribing {len(segments_list)} segments with openai-whisper on {device}")
    results = []

    for i, seg in enumerate(segments_list):
        audio_path = seg["path"]
        if not os.path.exists(audio_path):
            continue

        stem = Path(audio_path).stem
        out_json = os.path.join(output_dir, f"{stem}.json")
        if os.path.exists(out_json):
            with open(out_json) as f:
                results.append(json.load(f))
            continue

        try:
            result = model.transcribe(audio_path, word_timestamps=True)
            text = result["text"].strip()
            if text:
                entry = {
                    "audio_path": audio_path,
                    "text": text,
                    "duration": float(seg["duration"]),
                }
                with open(out_json, "w") as f:
                    json.dump(entry, f, indent=2)
                results.append(entry)
        except Exception as e:
            print(f"  Error on {audio_path}: {e}")

        if (i + 1) % 100 == 0:
            print(f"  [{i+1}/{len(segments_list)}] Transcribed {len(results)} segments")

    return results


def main():
    parser = argparse.ArgumentParser(description="Transcribe vintage audio (no torchcodec)")
    parser.add_argument("--manifest", default="/mnt/18tb/vintage_voice_processed/manifest_37k.csv")
    parser.add_argument("--output", default="/mnt/18tb/vintage_voice_processed/transcriptions_37k")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--backend", default="auto", choices=["auto", "faster-whisper", "openai-whisper"])
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    # Try faster-whisper first (faster), fall back to openai-whisper
    if args.backend == "auto":
        try:
            import faster_whisper
            args.backend = "faster-whisper"
            print("Using faster-whisper backend")
        except ImportError:
            try:
                import whisper
                args.backend = "openai-whisper"
                print("Using openai-whisper backend")
            except ImportError:
                print("ERROR: Install either faster-whisper or openai-whisper")
                print("  pip install faster-whisper")
                print("  pip install openai-whisper")
                sys.exit(1)

    if args.backend == "faster-whisper":
        results = transcribe_with_faster_whisper(args.manifest, args.output, args.device)
    else:
        results = transcribe_with_openai_whisper(args.manifest, args.output, args.device)

    # Write training manifest
    train_csv = os.path.join(args.output, "train.csv")
    with open(train_csv, "w") as f:
        f.write("audio_path|text|duration\n")
        for r in results:
            text = r["text"].replace("|", " ").replace("\n", " ")
            f.write(f"{r['audio_path']}|{text}|{r['duration']:.2f}\n")

    total_dur = sum(r["duration"] for r in results)
    print(f"\nDone! {len(results)} transcribed segments")
    print(f"Total audio: {total_dur/3600:.1f} hours")
    print(f"Training manifest: {train_csv}")


if __name__ == "__main__":
    main()
