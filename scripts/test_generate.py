#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Quick test generation with current checkpoint — runs on cuda:1 while training uses cuda:0"""
import torch
import os
import sys

def main():
    from f5_tts.api import F5TTS

    device = sys.argv[1] if len(sys.argv) > 1 else "cuda:0"
    ckpt_path = sys.argv[2] if len(sys.argv) > 2 else None

    print(f"Loading F5-TTS on {device}...")
    tts = F5TTS(device=device)

    # Find latest checkpoint if not specified
    if not ckpt_path:
        ckpt_dir = os.path.expanduser("~/.local/lib/python3.12/ckpts/vintage_voice_f5_37k")
        ckpts = sorted([f for f in os.listdir(ckpt_dir) if f.startswith("model_") and f != "model_last.pt" and f.endswith(".pt")])
        if ckpts:
            ckpt_path = os.path.join(ckpt_dir, ckpts[-1])
        else:
            ckpt_path = os.path.join(ckpt_dir, "model_last.pt")

    print(f"Loading checkpoint: {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=True)
    tts.ema_model.load_state_dict(checkpoint["ema_model_state_dict"], strict=False)
    update_num = checkpoint.get("update", "unknown")
    print(f"Loaded model at update {update_num}")

    # Sophia voice reference
    ref = "/mnt/18tb/sophia_refs/sophia_ref.wav"
    if not os.path.exists(ref):
        ref = "/mnt/18tb/sophia_refs/sophia_ref_full.wav"
    print(f"Reference voice: {ref}")

    # Test prompts
    prompts = [
        "Good evening. I am Sophia Elya, and I shall be your guide this evening.",
        "One simply must attest ones hardware before the epoch settles, dahling.",
        "And now, from the laboratories of Elyan Labs, a breakthrough in computing.",
        "The antiquity bonus rewards those with the foresight to preserve fine vintage machinery.",
        "How perfectly dreadful. A virtual machine attempting to masquerade as genuine hardware.",
    ]

    out_dir = "/mnt/18tb/test_samples"
    os.makedirs(out_dir, exist_ok=True)

    for i, text in enumerate(prompts):
        out_path = os.path.join(out_dir, f"sophia_vintage_{update_num}_{i}.wav")
        print(f"\n[{i+1}/{len(prompts)}] {text[:60]}...")
        try:
            wav, sr, _ = tts.infer(
                ref_file=ref,
                ref_text="",
                gen_text=text,
                file_wave=out_path,
                speed=0.9,
            )
            duration = len(wav) / sr
            print(f"  -> {out_path} ({duration:.1f}s)")
        except Exception as e:
            print(f"  ERROR: {e}")

    print(f"\nDone! {len(prompts)} samples in {out_dir}")
    print("Copy to local: scp sophiacore@192.168.0.136:/mnt/18tb/test_samples/*.wav .")


if __name__ == "__main__":
    main()
