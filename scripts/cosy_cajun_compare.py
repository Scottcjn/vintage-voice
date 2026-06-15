#!/usr/bin/env python3
"""
VintageVoice — CosyVoice2 Cajun French comparison: BASE vs ep2 vs ep6.
CPU-ONLY. Run with CUDA_VISIBLE_DEVICES="" so torch.cuda.is_available()==False,
which makes the repo fall back to device='cpu' and disable fp16 automatically.
"""
import sys
import os

BASE = "/home/scott/vintage-voice"
sys.path.insert(0, f"{BASE}/models/cosyvoice-repo")
sys.path.insert(0, f"{BASE}/models/cosyvoice-repo/third_party/Matcha-TTS")

import torch
import torchaudio
from cosyvoice.cli.cosyvoice import CosyVoice2

assert not torch.cuda.is_available(), "CUDA visible! Re-run with CUDA_VISIBLE_DEVICES=''"

REF_AUDIO = "/home/scott/vintage-voice-samples-50ep/sophia_ref.wav"
REF_TEXT = ("reporting from the Serengeti. Here at Elion Labs, we've successfully "
            "trapped and tagged approximately 15.5 of these majestic little creatures.")

LINES = [
    "Comment ça va, mon ami ? Ça fait longtemps que je t'ai pas vu, ouais.",
    "Laissez les bons temps rouler ! On va faire un bon gombo ce soir, cher.",
    "Mais regarde donc ça. La récolte de cannes est belle cette année, oui.",
]

MODELS = {
    "base": f"{BASE}/models/CosyVoice2-0.5B",
    "ep2":  f"{BASE}/models/CosyVoice2-cajun-ep2",
    "ep6":  f"{BASE}/models/CosyVoice2-cajun-ep6",
}

OUT_ROOT = f"{BASE}/data/output/sophia_cajun_v2"


def main():
    for tag, mdir in MODELS.items():
        out = f"{OUT_ROOT}/{tag}"
        os.makedirs(out, exist_ok=True)
        print(f"\n===== loading {tag}: {mdir} =====", flush=True)
        model = CosyVoice2(mdir, load_jit=False, load_trt=False, fp16=False)
        sr = model.sample_rate
        for n, line in enumerate(LINES, 1):
            for i, o in enumerate(model.inference_zero_shot(line, REF_TEXT, REF_AUDIO, stream=False)):
                path = f"{out}/line_{n}.wav" if i == 0 else f"{out}/line_{n}_{i}.wav"
                torchaudio.save(path, o["tts_speech"], sr)
                dur = o["tts_speech"].shape[-1] / sr
                print(f"  saved {path}  ({dur:.2f}s)", flush=True)
        # free before next model
        del model
        import gc; gc.collect()


if __name__ == "__main__":
    main()
