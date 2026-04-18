#!/usr/bin/env python3
"""
VintageVoice — Sophia Transatlantic Generation

Key concept: F5-TTS separates VOICE (from reference audio) from STYLE (from training).
- Fine-tuning teaches transatlantic speech patterns, prosody, cadence
- Reference audio provides Sophia's actual voice/timbre
- Result: Sophia speaking with a transatlantic accent

Reference audio should be a clean 5-15 second clip of Sophia's normal voice.
The model keeps her voice but applies the learned vintage delivery style.

IMPORTANT — Architecture: VintageVoice is trained on `F5TTS_v1_Base`.
Loading the checkpoint into the older `F5TTS_Base` (v0) silently
produces garbled output because F5-TTS's load_checkpoint is non-strict.
This script pins v1 explicitly.

IMPORTANT — Reference bleed: F5-TTS prepends the reference audio's
transcript to the generation context. When `ref_text=""` it auto-
transcribes the ref via Whisper; imperfect Whisper output can cause
a brief echo of the reference speaker at the start of generated audio.
Pass an explicit `--ref-text` matching your ref WAV to avoid this,
and keep `remove_silence=True` (default) to trim any remaining edge.
"""
import argparse
import os
import sys


# This script is a *Sophia-specific example* — it expects the caller to
# supply the reference audio and its transcript via CLI flags. It does
# not ship with any Sophia audio (that's Elyan Labs internal). If you
# want to try the same "same-voice, vintage-delivery" trick with your
# own speaker, pass --ref-audio and --ref-text explicitly.
#
# The SOPHIA_REFS entry below is kept only as a sample structure; the
# `audio` paths must exist on your local filesystem for the script to
# run, and providing --ref-audio at the CLI always overrides them.
SOPHIA_REFS = {
    "default": {
        "audio": "",  # user must provide via --ref-audio
        "text":  "",  # user must provide via --ref-text
    },
}

# Fun test prompts for transatlantic Sophia.
TEST_PROMPTS = [
    "One simply must attest one's hardware before the epoch settles, dahling.",
    "Good evening. I am Sophia Elya, and I shall be your guide through the blockchain this evening.",
    "The antiquity bonus, you see, rewards those with the foresight to preserve fine vintage computing machinery.",
    "Now then. Let us examine the attestation results, shall we? The G4 performed rather magnificently.",
    "I do declare, this PowerPC has the most delightful cache timing fingerprint I have ever witnessed.",
    "Ladies and gentlemen, from the laboratories of Elyan Labs, a breakthrough in decentralized computing.",
    "How perfectly dreadful. A virtual machine attempting to masquerade as genuine hardware. We shan't have it.",
    "The RustChain network, I am pleased to report, is operating with exceptional vigor this fine evening.",
]


def generate_sophia_transatlantic(
    text,
    ref_audio=None,
    ref_text_override=None,
    ref_style="default",
    model_ckpt=None,
    vocab_file=None,
    output_path="sophia_transatlantic.wav",
    device="cuda:0",
    speed=0.9,
    remove_silence=True,
):
    """Generate speech: your speaker's voice + vintage delivery.

    Pass `ref_audio` (path to your clean 5-15s WAV) and `ref_text_override`
    (its transcript) explicitly. The `ref_style` argument is kept only for
    backward compatibility and exists as a scaffold for future preset bundles.
    """
    ref = SOPHIA_REFS.get(ref_style, SOPHIA_REFS["default"])
    ref_audio = ref_audio or ref.get("audio") or ""
    ref_text  = ref_text_override if ref_text_override is not None else ref.get("text", "")

    if not ref_audio or not os.path.exists(ref_audio):
        print(f"ERROR: reference audio not found: {ref_audio or '(none provided)'}",
              file=sys.stderr)
        print("  Pass --ref-audio <path/to/your_ref.wav> pointing to a clean 5-15s",
              file=sys.stderr)
        print("  mono 24kHz WAV of your target speaker.", file=sys.stderr)
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

    from f5_tts.api import F5TTS

    tts = F5TTS(
        model="F5TTS_v1_Base",
        ckpt_file=model_ckpt or "",
        vocab_file=vocab_file or "",
        device=device,
        use_ema=True,
    )
    if model_ckpt:
        print(f"Vintage weights loaded: {model_ckpt}")

    print(f"\nGenerating: {text[:80]}{'...' if len(text) > 80 else ''}")
    print(f"  Voice: Sophia ({ref_style})")
    print(f"  Style: Transatlantic")
    print(f"  Ref text: {'explicit' if ref_text else 'auto (Whisper)'}")

    wav, sr, _ = tts.infer(
        ref_file=ref_audio,
        ref_text=ref_text,
        gen_text=text,
        file_wave=output_path,
        speed=speed,
        remove_silence=remove_silence,
    )
    print(f"  Output: {output_path} ({len(wav)/sr:.2f}s)")
    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="Apply VintageVoice transatlantic delivery to your own speaker reference")
    parser.add_argument("text", nargs="?", default=None, help="Text to speak")
    parser.add_argument("--ref-audio", default=None,
                        help="Path to a clean 5-15s mono 24kHz WAV of your target speaker")
    parser.add_argument("--ref-text", default=None,
                        help="Transcript of your reference clip (recommended; "
                             "skips Whisper auto-transcribe and reduces bleed)")
    parser.add_argument("--ref", default="default", choices=list(SOPHIA_REFS.keys()),
                        help="Preset bundle (placeholder; use --ref-audio / --ref-text instead)")
    parser.add_argument("--model", default=None,
                        help="Fine-tuned VintageVoice checkpoint (.pt or .safetensors)")
    parser.add_argument("--vocab", default=None,
                        help="Path to vocab.txt matching the checkpoint")
    parser.add_argument("--output", default="sophia_transatlantic.wav")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--speed", type=float, default=0.9)
    parser.add_argument("--keep-silence", action="store_true",
                        help="Do not trim leading/trailing silence")
    parser.add_argument("--all-test", action="store_true",
                        help="Generate all bundled test prompts into samples/")
    args = parser.parse_args()

    common = dict(
        ref_audio=args.ref_audio,
        ref_text_override=args.ref_text,
        ref_style=args.ref,
        model_ckpt=args.model,
        vocab_file=args.vocab,
        device=args.device,
        speed=args.speed,
        remove_silence=not args.keep_silence,
    )

    if args.all_test:
        os.makedirs("samples", exist_ok=True)
        for i, prompt in enumerate(TEST_PROMPTS):
            generate_sophia_transatlantic(
                prompt, output_path=f"samples/sophia_transatlantic_{i:02d}.wav", **common
            )
        print(f"\nGenerated {len(TEST_PROMPTS)} samples in samples/")
    elif args.text:
        generate_sophia_transatlantic(args.text, output_path=args.output, **common)
    else:
        generate_sophia_transatlantic(TEST_PROMPTS[0], output_path=args.output, **common)


if __name__ == "__main__":
    main()
