#!/usr/bin/env python3
"""
VintageVoice — CosyVoice2 zero-shot baseline (run in venv-cosy)

Measure-before-publish: generate French in Sophia's voice with STOCK
CosyVoice2-0.5B, before any fine-tuning. This isolates how much of the F5 v0's
"robotic/flat/fast" verdict was the model architecture vs our corpus.
"""
import sys
import os

BASE = "/home/scott/vintage-voice"
sys.path.insert(0, f"{BASE}/models/cosyvoice-repo")
sys.path.insert(0, f"{BASE}/models/cosyvoice-repo/third_party/Matcha-TTS")

import torchaudio
from cosyvoice.cli.cosyvoice import CosyVoice2

REF_AUDIO = "/home/scott/vintage-voice-samples-50ep/sophia_ref.wav"
REF_TEXT = ("reporting from the Serengeti. Here at Elion Labs, we've successfully "
            "trapped and tagged approximately 15.5 of these majestic little creatures.")
OUT = f"{BASE}/data/output/sophia_cajun"

LINES = [
    "Comment ça va, mon ami ? Ça fait longtemps que je t'ai pas vu, ouais.",
    "Laissez les bons temps rouler ! On va faire un bon gombo ce soir, cher.",
    "Mais regarde donc ça. La récolte de cannes est belle cette année, oui.",
]


def main():
    os.makedirs(OUT, exist_ok=True)
    model = CosyVoice2(f"{BASE}/models/CosyVoice2-0.5B", load_jit=False, load_trt=False, fp16=True)
    # repo HEAD API: prompt audio is passed as a PATH; frontend loads it itself
    for n, line in enumerate(LINES, 1):
        for i, out in enumerate(model.inference_zero_shot(line, REF_TEXT, REF_AUDIO, stream=False)):
            path = f"{OUT}/COSY_zeroshot_{n}.wav" if i == 0 else f"{OUT}/COSY_zeroshot_{n}_{i}.wav"
            torchaudio.save(path, out["tts_speech"], model.sample_rate)
            print("saved", path)


if __name__ == "__main__":
    main()
