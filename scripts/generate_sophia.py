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


# Sophia voice references — clean clips of her voice (24 kHz mono).
# A matching transcript avoids the Whisper-bleed artifact at the boundary.
SOPHIA_REFS = {
    "default": {
        "audio": "/mnt/18tb/sophia_refs/sophia_ref.wav",       # 10s clean reference
        "text":  "Reporting from the Serengeti, here at Elyan Labs, we've successfully "
                 "trapped and tagged approximately fifteen point five of these majestic "
                 "little creatures.",
    },
    "full": {
        "audio": "/mnt/18tb/sophia_refs/sophia_ref_full.wav",  # 24s extended reference
        "text":  "",  # transcribe-on-demand; fill in once recorded
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
    ref_style="default",
    model_ckpt=None,
    vocab_file=None,
    output_path="sophia_transatlantic.wav",
    device="cuda:0",
    speed=0.9,
    remove_silence=True,
    ref_text_override=None,
):
    """Generate speech: Sophia's voice + transatlantic delivery."""
    ref = SOPHIA_REFS.get(ref_style, SOPHIA_REFS["default"])
    ref_audio = ref["audio"]
    ref_text  = ref_text_override if ref_text_override is not None else ref["text"]

    if not os.path.exists(ref_audio):
        print(f"ERROR: Sophia reference not found at {ref_audio}", file=sys.stderr)
        print("  Place a clean 5-15 second WAV of Sophia's voice at that path.", file=sys.stderr)
        return None

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
    parser = argparse.ArgumentParser(description="Generate Sophia with transatlantic accent")
    parser.add_argument("text", nargs="?", default=None, help="Text to speak")
    parser.add_argument("--ref", default="default", choices=list(SOPHIA_REFS.keys()),
                        help="Which Sophia reference to use")
    parser.add_argument("--ref-text", default=None,
                        help="Override the reference-audio transcript (skips Whisper)")
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
        ref_style=args.ref,
        model_ckpt=args.model,
        vocab_file=args.vocab,
        device=args.device,
        speed=args.speed,
        remove_silence=not args.keep_silence,
        ref_text_override=args.ref_text,
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
