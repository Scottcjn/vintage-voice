#!/usr/bin/env python3
"""
VintageVoice — Speech Generation with Historical Voice Presets

Generate speech that sounds like it's from the 1888-1955 era.
Uses F5-TTS (v1 Base architecture) with vintage fine-tuned weights.

IMPORTANT — Architecture: VintageVoice was fine-tuned on top of
`F5TTS_v1_Base` (not the older `F5TTS_Base`/v0). Loading our weights
into the wrong architecture silently produces garbled output because
F5-TTS's load_checkpoint uses strict=False and drops mismatched keys.
This script pins the correct architecture explicitly.
"""
import argparse
import os
import sys


# Reference audio clips for each preset (included in the model release).
# Each is a clean 5-15 sec WAV (24 kHz mono) that captures the target
# acoustic style — transatlantic broadcast, Edison cylinder, etc.
PRESET_REFS = {
    "transatlantic": "refs/transatlantic_ref.wav",
    "newsreel":      "refs/newsreel_narrator_ref.wav",
    "fireside":      "refs/fdr_fireside_ref.wav",
    "radio_drama":   "refs/radio_drama_ref.wav",
    "edison":        "refs/edison_cylinder_ref.wav",
    "wartime":       "refs/wartime_broadcast_ref.wav",
    "announcer":     "refs/radio_announcer_ref.wav",
}


def resolve_ref(preset, override, model_dir):
    """Find the reference audio path for a preset, honoring an override and a model dir."""
    if override:
        return override
    rel = PRESET_REFS.get(preset)
    if not rel:
        return None
    if os.path.exists(rel):
        return rel
    if model_dir:
        candidate = os.path.join(model_dir, rel)
        if os.path.exists(candidate):
            return candidate
    return rel  # let caller surface a clear error if missing


def generate_speech(
    text,
    preset="transatlantic",
    model_path=None,
    vocab_path=None,
    ref_audio=None,
    ref_text="",
    output_path="output.wav",
    device="cuda:0",
    speed=0.9,
    remove_silence=True,
):
    """Generate vintage-styled speech.

    Passing an explicit `ref_text` that matches `ref_audio` is recommended;
    it skips F5-TTS's internal Whisper auto-transcribe and avoids a common
    artifact where a half-second of the reference speaker's voice leaks
    into the start of the generated output.
    """
    from f5_tts.api import F5TTS

    model_dir = os.path.dirname(model_path) if model_path else None
    ref_audio = resolve_ref(preset, ref_audio, model_dir)
    if not ref_audio or not os.path.exists(ref_audio):
        print(f"ERROR: reference audio not found: {ref_audio}", file=sys.stderr)
        print(f"  Preset: {preset}", file=sys.stderr)
        print("  Provide --ref-audio <path/to/your.wav> or drop the preset file into refs/",
              file=sys.stderr)
        return None

    if not ref_text:
        print("", file=sys.stderr)
        print("⚠️  VintageVoice: ref_text is empty — will auto-transcribe via Whisper.",
              file=sys.stderr)
        print("   This occasionally leaks ~0.5s of the reference speaker's voice into",
              file=sys.stderr)
        print("   the start of generated audio. To avoid it, pass --ref-text with the",
              file=sys.stderr)
        print("   exact transcript of your reference clip.", file=sys.stderr)
        print("", file=sys.stderr)

    print("VintageVoice Generation")
    print(f"  Preset:    {preset}")
    print(f"  Text:      {text[:80]}{'...' if len(text) > 80 else ''}")
    print(f"  Reference: {ref_audio}")
    print(f"  Output:    {output_path}")
    print(f"  Device:    {device}")

    tts = F5TTS(
        model="F5TTS_v1_Base",
        ckpt_file=model_path or "",
        vocab_file=vocab_path or "",
        device=device,
        use_ema=True,
    )
    if model_path:
        print(f"  Vintage weights loaded: {model_path}")

    wav, sr, _ = tts.infer(
        ref_file=ref_audio,
        ref_text=ref_text,
        gen_text=text,
        file_wave=output_path,
        speed=speed,
        remove_silence=remove_silence,
    )
    print(f"  Generated {len(wav)/sr:.2f}s of audio at {sr} Hz")
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Generate vintage-styled speech")
    parser.add_argument("text", help="Text to speak")
    parser.add_argument("--preset", default="transatlantic",
                        choices=list(PRESET_REFS.keys()),
                        help="Historical voice preset (default: transatlantic)")
    parser.add_argument("--model", default=None,
                        help="Path to fine-tuned VintageVoice checkpoint (.pt or .safetensors)")
    parser.add_argument("--vocab", default=None,
                        help="Path to vocab.txt matching the checkpoint")
    parser.add_argument("--ref-audio", default=None, help="Override reference audio path")
    parser.add_argument("--ref-text", default="",
                        help="Transcript of the reference audio (recommended; "
                             "skips Whisper auto-transcribe and reduces bleed)")
    parser.add_argument("--output", default="output.wav", help="Output WAV path")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--speed", type=float, default=0.9,
                        help="Playback speed; 0.9 suits measured vintage cadence")
    parser.add_argument("--keep-silence", action="store_true",
                        help="Do not trim leading/trailing silence")
    args = parser.parse_args()

    generate_speech(
        text=args.text,
        preset=args.preset,
        model_path=args.model,
        vocab_path=args.vocab,
        ref_audio=args.ref_audio,
        ref_text=args.ref_text,
        output_path=args.output,
        device=args.device,
        speed=args.speed,
        remove_silence=not args.keep_silence,
    )


if __name__ == "__main__":
    main()
