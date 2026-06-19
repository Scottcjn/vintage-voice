#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Benchmark VintageVoice inference on portable hardware targets."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
from time import perf_counter

try:
    from scripts.generate import generate_speech
    from scripts.portable_inference import choose_inference_plan
except ModuleNotFoundError:
    from generate import generate_speech
    from portable_inference import choose_inference_plan


DEFAULT_TEXT = "Good evening, ladies and gentlemen. This is a portable VintageVoice test."


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Benchmark VintageVoice inference")
    parser.add_argument("--text", default=DEFAULT_TEXT)
    parser.add_argument("--preset", default="transatlantic")
    parser.add_argument("--model", default=None, help="Path to model checkpoint")
    parser.add_argument("--vocab", default=None, help="Path to vocab.txt")
    parser.add_argument("--ref-audio", default=None, help="Reference WAV path")
    parser.add_argument("--ref-text", default="", help="Reference transcript")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=0)
    parser.add_argument("--output-dir", default="benchmark_out")
    parser.add_argument("--json", default=None, help="Write benchmark metadata JSON")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Resolve the portable plan without importing/running F5-TTS",
    )
    return parser


def run_benchmark(args: argparse.Namespace, generate=generate_speech) -> dict:
    if args.runs < 1:
        raise ValueError("--runs must be at least 1")
    if args.warmup < 0:
        raise ValueError("--warmup must be zero or greater")

    plan = choose_inference_plan(args.device)
    output_dir = Path(args.output_dir)
    result = {
        "plan": asdict(plan),
        "runs": [],
        "warmup": args.warmup,
        "dry_run": bool(args.dry_run),
    }

    if args.dry_run:
        return result

    output_dir.mkdir(parents=True, exist_ok=True)
    for index in range(args.warmup + args.runs):
        output_path = output_dir / f"portable_inference_{index + 1}.wav"
        started = perf_counter()
        generated = generate(
            text=args.text,
            preset=args.preset,
            model_path=args.model,
            vocab_path=args.vocab,
            ref_audio=args.ref_audio,
            ref_text=args.ref_text,
            output_path=str(output_path),
            device=plan.device,
        )
        elapsed = perf_counter() - started
        if index >= args.warmup:
            result["runs"].append(
                {
                    "seconds": round(elapsed, 4),
                    "output": str(generated or output_path),
                }
            )
    return result


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    result = run_benchmark(args)
    payload = json.dumps(result, indent=2, sort_keys=True)
    print(payload)
    if args.json:
        Path(args.json).write_text(payload + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
