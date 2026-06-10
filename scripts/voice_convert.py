#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""
VintageVoice — Two-Stage Pipeline
Stage 1: F5-TTS generates speech with vintage reference (Hepburn/Welles)
Stage 2: Converts the voice timbre to Sophia while keeping accent+delivery

Uses F5-TTS's own architecture for conversion:
- Feed Sophia ref as reference
- Feed Stage 1 output as the "continuation" source
- F5-TTS will match Sophia's voice characteristics to the input prosody
"""
import torch
import torchaudio
import os
import sys
import numpy as np


def pitch_shift_to_target(source_path, target_ref_path, output_path):
    """
    Simple voice conversion via pitch shifting + formant preservation.
    Not as good as a real VC model, but works immediately.

    Matches the pitch contour of source to the target speaker's range.
    """
    source, sr = torchaudio.load(source_path)
    target, sr_t = torchaudio.load(target_ref_path)

    if sr != sr_t:
        target = torchaudio.functional.resample(target, sr_t, sr)

    # Estimate pitch ranges
    # This is a crude but functional approach
    import librosa

    source_np = source.squeeze().numpy()
    target_np = target.squeeze().numpy()

    # Get F0 for both
    f0_src, _, _ = librosa.pyin(source_np, fmin=60, fmax=500, sr=sr)
    f0_tgt, _, _ = librosa.pyin(target_np, fmin=60, fmax=500, sr=sr)

    f0_src_valid = f0_src[~np.isnan(f0_src)]
    f0_tgt_valid = f0_tgt[~np.isnan(f0_tgt)]

    if len(f0_src_valid) == 0 or len(f0_tgt_valid) == 0:
        print("  Could not detect pitch, copying source as-is")
        torchaudio.save(output_path, source, sr)
        return

    # Calculate semitone shift needed
    median_src = np.median(f0_src_valid)
    median_tgt = np.median(f0_tgt_valid)
    semitone_shift = 12 * np.log2(median_tgt / median_src)

    print(f"  Source F0: {median_src:.0f}Hz, Target F0: {median_tgt:.0f}Hz")
    print(f"  Shift: {semitone_shift:.1f} semitones")

    # Apply pitch shift via sox/ffmpeg
    import subprocess
    # Use rubberband for high-quality pitch shift preserving formants
    cmd = [
        "ffmpeg", "-y", "-i", source_path,
        "-af", f"rubberband=pitch={2**(semitone_shift/12):.4f}:formant=preserved",
        "-ar", str(sr),
        output_path
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        # Fallback: simple asetrate pitch shift
        cmd = [
            "ffmpeg", "-y", "-i", source_path,
            "-af", f"asetrate={int(sr * 2**(semitone_shift/12))},aresample={sr}",
            output_path
        ]
        subprocess.run(cmd, capture_output=True)

    print(f"  Output: {output_path}")


def two_stage_pipeline(
    text,
    vintage_ref,
    sophia_ref,
    output_path,
    vintage_ckpt=None,
    device="cuda:0",
):
    """Full two-stage: vintage TTS → voice convert to Sophia"""
    from f5_tts.api import F5TTS

    stage1_path = output_path.replace(".wav", "_stage1.wav")

    # Stage 1: Generate with vintage reference
    print(f"Stage 1: Generating with vintage reference...")
    tts = F5TTS(device=device)

    if vintage_ckpt:
        checkpoint = torch.load(vintage_ckpt, map_location=device, weights_only=True)
        raw_sd = checkpoint["model_state_dict"]
        stripped = {k[len("transformer."):]: v for k, v in raw_sd.items() if k.startswith("transformer.")}
        tts.ema_model.load_state_dict(stripped, strict=False)

    tts.infer(
        ref_file=vintage_ref,
        ref_text="",
        gen_text=text,
        file_wave=stage1_path,
        speed=0.85,
    )
    print(f"  Stage 1 output: {stage1_path}")

    # Stage 2: Pitch-shift to Sophia's voice range
    print(f"Stage 2: Converting to Sophia's voice...")
    pitch_shift_to_target(stage1_path, sophia_ref, output_path)

    print(f"Final output: {output_path}")
    return output_path


def main():
    vintage_ref = sys.argv[1] if len(sys.argv) > 1 else "/mnt/18tb/female_vintage_refs/hepburn_philly_clip.wav"
    sophia_ref = sys.argv[2] if len(sys.argv) > 2 else "/mnt/18tb/sophia_refs/sophia_ref.wav"
    device = sys.argv[3] if len(sys.argv) > 3 else "cuda:0"

    ckpt_dir = os.path.expanduser("~/.local/lib/python3.12/ckpts/vintage_voice_f5_37k")
    ckpts = sorted([f for f in os.listdir(ckpt_dir) if f.startswith("model_") and f != "model_last.pt" and f.endswith(".pt")])
    vintage_ckpt = os.path.join(ckpt_dir, ckpts[-1]) if ckpts else os.path.join(ckpt_dir, "model_last.pt")

    prompts = [
        "Good evening darling. I am Sophia Elya, and I shall be your guide this evening.",
        "One simply must attest ones hardware before the epoch settles.",
        "How perfectly dreadful. A virtual machine masquerading as genuine hardware.",
    ]

    out_dir = "/mnt/18tb/test_samples"
    os.makedirs(out_dir, exist_ok=True)

    for i, text in enumerate(prompts):
        out = os.path.join(out_dir, f"SOPHIA_TRANSATLANTIC_{i}.wav")
        print(f"\n{'='*60}")
        print(f"Prompt: {text}")
        two_stage_pipeline(text, vintage_ref, sophia_ref, out, vintage_ckpt, device)

    print(f"\n{'='*60}")
    print("All samples generated!")


if __name__ == "__main__":
    main()
