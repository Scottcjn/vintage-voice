#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Portable VintageVoice inference entry point.

This wrapper keeps the regular F5-TTS/VintageVoice inference path, but makes
the device and threading defaults friendly to non-CUDA hosts such as PowerPC,
IBM POWER, ARM SBCs, and Apple Silicon.
"""
import argparse
import os
import platform
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.generate import PRESET_REFS, generate_speech


PORTABLE_ARCHES = {
    "aarch64",
    "arm64",
    "armv7l",
    "ppc64le",
    "ppc64",
    "ppc",
    "powerpc",
}


def normalized_machine(machine=None):
    """Return a normalized architecture string for platform checks."""
    return (machine or platform.machine() or "unknown").lower()


def is_portable_target(machine=None):
    """True for the hardware families covered by the portable path."""
    arch = normalized_machine(machine)
    return arch in PORTABLE_ARCHES or arch.startswith(("arm", "ppc"))


def choose_device(requested="auto", machine=None):
    """Choose an inference device without importing torch unless needed."""
    if requested != "auto":
        return requested

    arch = normalized_machine(machine)
    system = platform.system().lower()

    try:
        import torch
    except Exception:
        return "cpu"

    if system == "darwin" and arch in {"arm64", "aarch64"}:
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    if torch.cuda.is_available() and not is_portable_target(arch):
        return "cuda:0"
    return "cpu"


def configure_threads(threads=None):
    """Set conservative CPU threading defaults for small or unusual hosts."""
    if threads is None:
        threads = max(1, min(os.cpu_count() or 1, 4))
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ.setdefault(name, str(threads))

    try:
        import torch
    except Exception:
        return threads

    try:
        torch.set_num_threads(threads)
    except Exception:
        pass
    return threads


def main():
    parser = argparse.ArgumentParser(
        description="Run VintageVoice inference on CPU/MPS/CUDA with portable defaults"
    )
    parser.add_argument("text", help="Text to speak")
    parser.add_argument("--preset", default="transatlantic", choices=list(PRESET_REFS.keys()))
    parser.add_argument("--model", required=True, help="Path to VintageVoice checkpoint")
    parser.add_argument("--vocab", required=True, help="Path to vocab.txt matching the checkpoint")
    parser.add_argument("--ref-audio", required=True, help="Reference WAV, 5-15 seconds recommended")
    parser.add_argument("--ref-text", required=True, help="Exact transcript of the reference audio")
    parser.add_argument("--output", default="portable_out.wav", help="Output WAV path")
    parser.add_argument(
        "--device",
        default="auto",
        help="auto, cpu, mps, cuda:0, or any F5-TTS/PyTorch device string",
    )
    parser.add_argument("--threads", type=int, default=None, help="CPU threads for torch/BLAS")
    parser.add_argument("--speed", type=float, default=0.9)
    parser.add_argument("--keep-silence", action="store_true")
    args = parser.parse_args()

    threads = configure_threads(args.threads)
    device = choose_device(args.device)
    print(f"Host: {platform.system()} {platform.machine()} | device={device} | threads={threads}")

    output = generate_speech(
        text=args.text,
        preset=args.preset,
        model_path=args.model,
        vocab_path=args.vocab,
        ref_audio=args.ref_audio,
        ref_text=args.ref_text,
        output_path=args.output,
        device=device,
        speed=args.speed,
        remove_silence=not args.keep_silence,
    )
    return 0 if output else 1


if __name__ == "__main__":
    sys.exit(main())
