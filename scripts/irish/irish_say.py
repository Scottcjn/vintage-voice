#!/usr/bin/env python3
"""Make Sophia speak in her Irish (Dublin) accent.

CosyVoice2 finetune: llm epoch-0 + flow epoch-9, trained on Tadhg Hynes' 10.12h
public-domain LibriVox corpus (Dubliners + Portrait). Sophia's timbre comes from
the zero-shot reference; the brogue is baked into the model.

Usage (CPU by default so it won't touch the llamas):
  CUDA_VISIBLE_DEVICES="" /home/scott/vintage-voice/venv-cosy/bin/python \
      /home/scott/irish_say.py "Top of the morning to you." sortie
  -> writes /home/scott/vintage-voice/data/output/irish_say/<out>.wav
Set CUDA_VISIBLE_DEVICES=0 for GPU (stop the llamas first).
"""
import sys, os, subprocess
BASE = "/home/scott/vintage-voice"
sys.path.insert(0, f"{BASE}/models/cosyvoice-repo")
sys.path.insert(0, f"{BASE}/models/cosyvoice-repo/third_party/Matcha-TTS")
import torch, torchaudio
from cosyvoice.cli.cosyvoice import CosyVoice2

MODEL = f"{BASE}/models/CosyVoice2-irish"        # llm ep0 + flow ep9 keeper
REF_AUDIO = "/home/scott/vintage-voice-samples-50ep/sophia_ref.wav"
REF_TEXT = ("reporting from the Serengeti. Here at Elion Labs, we've successfully "
            "trapped and tagged approximately 15.5 of these majestic little creatures.")
OUT_DIR = f"{BASE}/data/output/irish_say"

def main():
    if len(sys.argv) < 2:
        print('usage: irish_say.py "<text>" [out_basename]'); sys.exit(1)
    text = sys.argv[1]
    name = sys.argv[2] if len(sys.argv) > 2 else "irish"
    on_gpu = torch.cuda.is_available() and os.environ.get("CUDA_VISIBLE_DEVICES") not in (None, "")
    os.makedirs(OUT_DIR, exist_ok=True)
    print(f"loading Irish Sophia on {'GPU' if on_gpu else 'CPU'}...", flush=True)
    m = CosyVoice2(MODEL, load_jit=False, load_trt=False, fp16=on_gpu)
    for i, out in enumerate(m.inference_zero_shot(text, REF_TEXT, REF_AUDIO, stream=False, speed=1.12)):
        path = f"{OUT_DIR}/{name}.wav" if i == 0 else f"{OUT_DIR}/{name}_{i}.wav"
        torchaudio.save(path, out["tts_speech"], m.sample_rate)
        tmp = path + ".l.wav"
        if subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", path,
                           "-af", "loudnorm=I=-12:LRA=7:TP=-1.0", "-ar", "24000", "-ac", "1", tmp]).returncode == 0:
            os.replace(tmp, path)
        print("WROTE", path, flush=True)

if __name__ == "__main__":
    main()
