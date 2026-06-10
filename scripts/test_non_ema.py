#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""
Test generation using NON-EMA checkpoint.
F5-TTS's EMA (exponential moving average) keeps weights close to pretrained,
which may be preventing accent transfer from showing up.
Using the non-EMA (raw trained) weights should show stronger accent shift.
"""
import torch
import os
import sys


def main():
    from f5_tts.api import F5TTS

    device = sys.argv[1] if len(sys.argv) > 1 else "cuda:0"
    ckpt_path = sys.argv[2] if len(sys.argv) > 2 else "/home/sophiacore/.local/lib/python3.12/ckpts/vintage_voice_f5_37k/model_last.pt"

    print(f"Loading F5-TTS on {device}...")
    tts = F5TTS(device=device)

    print(f"Loading NON-EMA checkpoint: {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=True)

    # Use model_state_dict instead of ema_model_state_dict
    # Strip "model." prefix if present (F5-TTS wraps in CFM)
    raw_sd = checkpoint["model_state_dict"]
    print(f"Keys in model_state_dict: {len(raw_sd)}")
    print(f"First 3 keys: {list(raw_sd.keys())[:3]}")

    # The model_state_dict is for the CFM wrapper containing transformer+vocoder refs
    # Try loading into ema_model (same architecture)
    try:
        # Strip "transformer." prefix for ema_model
        stripped = {}
        for k, v in raw_sd.items():
            if k.startswith("transformer."):
                stripped[k[len("transformer."):]] = v
            elif k.startswith("model."):
                stripped[k[len("model."):]] = v
            else:
                stripped[k] = v

        missing, unexpected = tts.ema_model.load_state_dict(stripped, strict=False)
        print(f"Loaded non-EMA weights. Missing: {len(missing)}, Unexpected: {len(unexpected)}")
    except Exception as e:
        print(f"Strip attempt failed: {e}")
        # Fallback: direct load
        missing, unexpected = tts.ema_model.load_state_dict(raw_sd, strict=False)
        print(f"Direct loaded. Missing: {len(missing)}, Unexpected: {len(unexpected)}")

    update_num = checkpoint.get("update", "unknown")
    print(f"Loaded model at update {update_num} (NON-EMA)")

    ref = "/mnt/18tb/sophia_refs/sophia_ref.wav"
    print(f"Reference voice: {ref}")

    prompts = [
        "Good evening. I am Sophia Elya, and I shall be your guide this evening.",
        "One simply must attest ones hardware before the epoch settles, darling.",
        "And now, from the laboratories of Elyan Labs, a breakthrough in computing.",
        "The antiquity bonus rewards those with the foresight to preserve fine vintage machinery.",
        "How perfectly dreadful. A virtual machine attempting to masquerade as genuine hardware.",
    ]

    out_dir = "/mnt/18tb/test_samples"
    os.makedirs(out_dir, exist_ok=True)

    for i, text in enumerate(prompts):
        out_path = os.path.join(out_dir, f"sophia_NON_EMA_{update_num}_{i}.wav")
        print(f"\n[{i+1}/{len(prompts)}] {text[:60]}...")
        try:
            wav, sr, _ = tts.infer(
                ref_file=ref,
                ref_text="",
                gen_text=text,
                file_wave=out_path,
                speed=0.85,
            )
            print(f"  -> {out_path} ({len(wav)/sr:.1f}s)")
        except Exception as e:
            print(f"  ERROR: {e}")

    print(f"\nDone! Non-EMA samples in {out_dir}")


if __name__ == "__main__":
    main()
