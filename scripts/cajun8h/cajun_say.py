#!/usr/bin/env python3
"""
Make Sophia speak Cajun French — CosyVoice2 epoch-2 finetune (8.1h corpus).

Usage (run with venv-cosy python; CPU by default so it won't touch the llamas):
  CUDA_VISIBLE_DEVICES="" /home/scott/vintage-voice/venv-cosy/bin/python \
      /home/scott/cajun_say.py "Mais comment ça va, cher ?" sortie

Args: <text> [out_basename]   -> writes /home/scott/vintage-voice/data/output/cajun_say/<out>.wav
Set CUDA_VISIBLE_DEVICES=0 to use the GPU (faster, but stop the llamas first).
"""
import sys, os
BASE = "/home/scott/vintage-voice"
sys.path.insert(0, f"{BASE}/models/cosyvoice-repo")
sys.path.insert(0, f"{BASE}/models/cosyvoice-repo/third_party/Matcha-TTS")
sys.path.insert(0, f"{BASE}/scripts/cajun8h")
import torch, torchaudio
from cosyvoice.cli.cosyvoice import CosyVoice2
from cajun_lexicon import respell  # Louisiana pronunciation lexicon

MODEL = f"{BASE}/models/CosyVoice2-cajun-ep2"          # epoch-2 = best CV loss
REF_AUDIO = "/home/scott/vintage-voice-samples-50ep/sophia_ref.wav"
REF_TEXT = ("reporting from the Serengeti. Here at Elion Labs, we've successfully "
            "trapped and tagged approximately 15.5 of these majestic little creatures.")
OUT = f"{BASE}/data/output/cajun_say"

def main():
    if len(sys.argv) < 2:
        print('usage: cajun_say.py "<cajun french text>" [out_basename]'); sys.exit(1)
    text = respell(sys.argv[1])  # apply Louisiana pronunciation lexicon
    name = sys.argv[2] if len(sys.argv) > 2 else "cajun"
    on_gpu = torch.cuda.is_available() and os.environ.get("CUDA_VISIBLE_DEVICES") not in (None, "")
    os.makedirs(OUT, exist_ok=True)
    print(f"loading epoch-2 Cajun model on {'GPU' if on_gpu else 'CPU'}...", flush=True)
    model = CosyVoice2(MODEL, load_jit=False, load_trt=False, fp16=on_gpu)
    import subprocess
    for i, out in enumerate(model.inference_zero_shot(text, REF_TEXT, REF_AUDIO, stream=False, speed=0.95)):
        path = f"{OUT}/{name}.wav" if i == 0 else f"{OUT}/{name}_{i}.wav"
        torchaudio.save(path, out["tts_speech"], model.sample_rate)
        # loudness boost — CosyVoice output is mastered quiet (~-26 LUFS); bring it up
        tmp = path + ".loud.wav"
        if subprocess.run(["ffmpeg","-y","-loglevel","error","-i",path,
                           "-af","loudnorm=I=-12:LRA=7:TP=-1.0","-ar","24000","-ac","1",tmp]).returncode == 0:
            os.replace(tmp, path)
        print("WROTE", path, flush=True)

if __name__ == "__main__":
    main()
