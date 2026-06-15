#!/usr/bin/env python3
"""
VintageVoice — Watchdog wrapper for transcribe_cajun.py

Whisper on CUDA can wedge (stuck kernel sync: 99% CPU, 0% GPU, no output).
Observed 2026-06-12 after ~7h: frozen in detect_language for 1h+ on an 11s clip.

Strategy: run the transcriber as a child; if no new JSON lands for --stale-min
minutes, kill and restart it (transcribe_cajun.py resumes from existing JSONs).
If the SAME segment wedges twice in a row, write a stub JSON (qc-flagged) so
the resume skips it — one bad segment must never cost more than two restarts.
"""
import argparse
import csv
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def newest_json_mtime(output_dir):
    newest = 0.0
    with os.scandir(output_dir) as it:
        for e in it:
            if e.name.endswith(".json"):
                m = e.stat().st_mtime
                if m > newest:
                    newest = m
    return newest


def first_missing_stem(manifest, output_dir):
    """First manifest row with no output JSON = the segment in flight when wedged."""
    with open(manifest) as f:
        for row in csv.DictReader(f, delimiter="|"):
            stem = Path(row["path"]).stem
            if not os.path.exists(os.path.join(output_dir, f"{stem}.json")):
                return stem, row
    return None, None


def write_stub(output_dir, stem, row, reason):
    stub = {
        "audio_path": row["path"],
        "text": "",
        "duration": float(row.get("duration", 0)),
        "language": "unknown",
        "qc_flags": [reason],
        "no_speech_prob": 1.0,
        "avg_logprob": -10.0,
        "compression_ratio": 0.0,
        "source": row.get("source", ""),
    }
    with open(os.path.join(output_dir, f"{stem}.json"), "w") as f:
        json.dump(stub, f, indent=2)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--model", default="medium")
    p.add_argument("--device", default="cuda")
    p.add_argument("--stale-min", type=float, default=15.0)
    p.add_argument("--max-restarts", type=int, default=12)
    args = p.parse_args()

    os.makedirs(args.output, exist_ok=True)
    cmd = [
        sys.executable, os.path.join(SCRIPT_DIR, "transcribe_cajun.py"),
        "--manifest", args.manifest, "--output", args.output,
        "--model", args.model, "--device", args.device,
    ]
    stale_s = args.stale_min * 60
    last_wedge_stem = None
    restarts = 0

    while True:
        start = time.time()
        print(f"[watchdog] launching transcriber (restart {restarts}/{args.max_restarts})", flush=True)
        proc = subprocess.Popen(cmd)

        wedged = False
        while True:
            try:
                proc.wait(timeout=60)
                break  # child exited
            except subprocess.TimeoutExpired:
                pass
            # Stale only if BOTH the child has run long enough AND output is old —
            # pre-existing JSONs from a prior run must not trigger at startup.
            now = time.time()
            if now - start > stale_s and now - newest_json_mtime(args.output) > stale_s:
                stem, row = first_missing_stem(args.manifest, args.output)
                print(f"[watchdog] WEDGE: no output for {args.stale_min:.0f}min, "
                      f"in-flight segment: {stem}", flush=True)
                if stem and stem == last_wedge_stem:
                    print(f"[watchdog] second wedge on {stem} — writing stub, skipping it", flush=True)
                    write_stub(args.output, stem, row, "cuda_wedge_skip")
                    last_wedge_stem = None
                else:
                    last_wedge_stem = stem
                proc.send_signal(signal.SIGKILL)
                proc.wait()
                wedged = True
                break

        if not wedged:
            if proc.returncode == 0:
                print("[watchdog] transcriber finished cleanly", flush=True)
                return 0
            print(f"[watchdog] transcriber exited {proc.returncode}", flush=True)

        restarts += 1
        if restarts > args.max_restarts:
            print("[watchdog] FATAL: restart budget exhausted", flush=True)
            return 1
        time.sleep(10)  # let CUDA context fully tear down


if __name__ == "__main__":
    sys.exit(main())
