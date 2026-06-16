#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Benchmark VintageVoice inference latency on the current host."""
import argparse
import json
import os
import platform
import sys
import time
import wave
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.generate import PRESET_REFS, generate_speech
from scripts.portable_infer import choose_device, configure_threads


DEFAULT_TEXT = "Good evening, ladies and gentlemen, this is VintageVoice running on real hardware."


def wav_duration(path):
    """Return WAV duration in seconds using only the standard library."""
    with wave.open(str(path), "rb") as wav:
        frames = wav.getnframes()
        rate = wav.getframerate()
    return frames / float(rate) if rate else 0.0


def benchmark(args):
    threads = configure_threads(args.threads)
    device = choose_device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    runs = []
    for index in range(args.runs):
        output = output_dir / f"benchmark_{index + 1}.wav"
        start = time.perf_counter()
        result = generate_speech(
            text=args.text,
            preset=args.preset,
            model_path=args.model,
            vocab_path=args.vocab,
            ref_audio=args.ref_audio,
            ref_text=args.ref_text,
            output_path=str(output),
            device=device,
            speed=args.speed,
            remove_silence=not args.keep_silence,
        )
        elapsed = time.perf_counter() - start
        if not result:
            raise RuntimeError("inference failed")

        audio_seconds = wav_duration(output)
        runs.append(
            {
                "run": index + 1,
                "wall_seconds": elapsed,
                "audio_seconds": audio_seconds,
                "real_time_factor": elapsed / audio_seconds if audio_seconds else None,
                "output": str(output),
            }
        )

    avg_wall = sum(run["wall_seconds"] for run in runs) / len(runs)
    avg_audio = sum(run["audio_seconds"] for run in runs) / len(runs)
    return {
        "host": {
            "system": platform.system(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "python": platform.python_version(),
            "cpu_count": os.cpu_count(),
        },
        "inference": {
            "device": device,
            "threads": threads,
            "preset": args.preset,
            "runs": args.runs,
            "text_chars": len(args.text),
        },
        "summary": {
            "avg_wall_seconds": avg_wall,
            "avg_audio_seconds": avg_audio,
            "avg_real_time_factor": avg_wall / avg_audio if avg_audio else None,
        },
        "runs": runs,
    }


def main():
    parser = argparse.ArgumentParser(description="Benchmark portable VintageVoice inference")
    parser.add_argument("--text", default=DEFAULT_TEXT)
    parser.add_argument("--preset", default="transatlantic", choices=list(PRESET_REFS.keys()))
    parser.add_argument("--model", required=True, help="Path to VintageVoice checkpoint")
    parser.add_argument("--vocab", required=True, help="Path to vocab.txt matching the checkpoint")
    parser.add_argument("--ref-audio", required=True, help="Reference WAV")
    parser.add_argument("--ref-text", required=True, help="Transcript of the reference WAV")
    parser.add_argument("--output-dir", default="benchmarks/out")
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--threads", type=int, default=None)
    parser.add_argument("--speed", type=float, default=0.9)
    parser.add_argument("--keep-silence", action="store_true")
    parser.add_argument("--json", default=None, help="Optional path to write benchmark JSON")
    args = parser.parse_args()

    if args.runs < 1:
        parser.error("--runs must be at least 1")

    result = benchmark(args)
    rendered = json.dumps(result, indent=2)
    print(rendered)
    if args.json:
        path = Path(args.json)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
