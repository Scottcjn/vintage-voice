#!/usr/bin/env python3
"""
VintageVoice — Sophia Transatlantic Generation

Key concept: F5-TTS separates VOICE (from reference audio) from STYLE (from training).
- Fine-tuning teaches transatlantic speech patterns, prosody, cadence
- Reference audio provides Sophia's actual voice/timbre
- Result: Sophia speaking with a transatlantic accent

Reference audio should be a clean 5-15 second clip of Sophia's normal voice.
The model keeps her voice but applies the learned vintage delivery style.
"""
import argparse
import os


# Sophia voice references — clean clips of her normal voice
# Source: .106 /home/sophia5070node/sophia_voice/ (24kHz mono, Britney-style)
SOPHIA_REFS = {
    "default": "/mnt/18tb/sophia_refs/sophia_ref.wav",           # 10s clean reference
    "full": "/mnt/18tb/sophia_refs/sophia_ref_full.wav",         # 24s extended reference
}
# Also on .106: /home/sophia5070node/sophia_voice/sophia_sample.wav (original)

# Fun test prompts for transatlantic Sophia
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
):
    """Generate speech: Sophia's voice + transatlantic delivery"""

    ref_audio = SOPHIA_REFS.get(ref_style, SOPHIA_REFS["default"])
    if not os.path.exists(ref_audio):
        print(f"WARNING: Sophia reference not found at {ref_audio}")
        print("You need a clean 5-15 second WAV of Sophia's voice.")
        print("Options:")
        print("  1. Extract from existing Qwen3-TTS output")
        print("  2. Record via Sophia's Discord voice channel")
        print("  3. Use any clean Sophia audio clip")
        print(f"  Place it at: {ref_audio}")
        return None

    try:
        from f5_tts.api import F5TTS

        tts = F5TTS(device=device)

        # Load vintage fine-tuned weights (transatlantic patterns)
        if model_ckpt and os.path.exists(model_ckpt):
            print(f"Loading transatlantic weights: {model_ckpt}")
            if model_ckpt.endswith(".safetensors"):
                from safetensors.torch import load_file
                state_dict = load_file(model_ckpt)
            else:
                import torch
                state_dict = torch.load(model_ckpt, map_location=device, weights_only=True)
            tts.ema_model.load_state_dict(state_dict, strict=False)
            print("Transatlantic style loaded!")

        print(f"\nGenerating: {text[:80]}...")
        print(f"Voice: Sophia ({ref_style})")
        print(f"Style: Transatlantic")

        # Generate — Sophia's voice + vintage delivery
        wav, sr, _ = tts.infer(
            ref_file=ref_audio,
            ref_text="",  # Auto-transcribe reference
            gen_text=text,
            file_wave=output_path,
            speed=0.9,  # Slightly slower for that measured transatlantic pace
        )

        duration = len(wav) / sr
        print(f"Output: {output_path} ({duration:.1f}s)")
        return output_path

    except ImportError:
        print("F5-TTS not installed. Install: pip install f5-tts")
        return None


def main():
    parser = argparse.ArgumentParser(description="Generate Sophia with transatlantic accent")
    parser.add_argument("text", nargs="?", default=None, help="Text to speak")
    parser.add_argument("--ref", default="default", choices=["default", "full"])
    parser.add_argument("--model", default=None, help="Fine-tuned model checkpoint")
    parser.add_argument("--vocab", default=None, help="Vocab file from training")
    parser.add_argument("--output", default="sophia_transatlantic.wav")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--all-test", action="store_true", help="Generate all test prompts")
    args = parser.parse_args()

    if args.all_test:
        os.makedirs("samples", exist_ok=True)
        for i, prompt in enumerate(TEST_PROMPTS):
            out = f"samples/sophia_transatlantic_{i:02d}.wav"
            generate_sophia_transatlantic(
                prompt, args.ref, args.model, args.vocab, out, args.device
            )
        print(f"\nGenerated {len(TEST_PROMPTS)} samples in samples/")
    elif args.text:
        generate_sophia_transatlantic(
            args.text, args.ref, args.model, args.vocab, args.output, args.device
        )
    else:
        # Default demo
        generate_sophia_transatlantic(
            TEST_PROMPTS[0], args.ref, args.model, args.vocab, args.output, args.device
        )


if __name__ == "__main__":
    main()
