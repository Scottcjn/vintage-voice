#!/usr/bin/env python3
"""Render trained Irish Sophia from an epoch checkpoint.
Usage: irish_render.py <epoch> ["text"] [out.wav]   (run with venv-cosy python, CPU)
Builds models/CosyVoice2-irish-ep<N> (symlink base + stripped llm.pt), then
inference_zero_shot with Sophia's reference -> Sophia's voice + learned accent."""
import sys, os, subprocess
BASE = "/home/scott/vintage-voice"
sys.path.insert(0, f"{BASE}/models/cosyvoice-repo")
sys.path.insert(0, f"{BASE}/models/cosyvoice-repo/third_party/Matcha-TTS")
import torch, torchaudio
from cosyvoice.cli.cosyvoice import CosyVoice2

EP   = sys.argv[1] if len(sys.argv) > 1 else "0"
TEXT = sys.argv[2] if len(sys.argv) > 2 else "Good evening, and welcome. I'm Sophia, and it's a genuine pleasure to be speaking with you tonight."
OUT  = sys.argv[3] if len(sys.argv) > 3 else f"/home/scott/irish_sophia_ep{EP}.wav"
BASEMODEL = f"{BASE}/models/CosyVoice2-0.5B"
CKPT = f"{BASE}/exp/irish/llm/epoch_{EP}_whole.pt"
MDIR = f"{BASE}/models/CosyVoice2-irish-ep{EP}"
REF_AUDIO = "/home/scott/vintage-voice-samples-50ep/sophia_ref.wav"
REF_TEXT  = ("reporting from the Serengeti. Here at Elion Labs, we've successfully "
             "trapped and tagged approximately 15.5 of these majestic little creatures.")

if not os.path.exists(f"{MDIR}/llm.pt"):
    os.makedirs(MDIR, exist_ok=True)
    for f in os.listdir(BASEMODEL):
        if f == "llm.pt":
            continue
        link = f"{MDIR}/{f}"
        if not os.path.lexists(link):
            os.symlink(f"{BASEMODEL}/{f}", link)
    sd = torch.load(CKPT, map_location="cpu")
    for k in ("epoch", "step"):
        sd.pop(k, None)
    torch.save(sd, f"{MDIR}/llm.pt")
    print(f"built {MDIR}", flush=True)

print(f"loading epoch {EP} (CPU)...", flush=True)
m = CosyVoice2(MDIR, load_jit=False, load_trt=False, fp16=False)
for i, out in enumerate(m.inference_zero_shot(TEXT, REF_TEXT, REF_AUDIO, stream=False, speed=1.12)):
    torchaudio.save(OUT, out["tts_speech"], m.sample_rate)
    tmp = OUT + ".loud.wav"
    if subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", OUT,
                       "-af", "loudnorm=I=-12:LRA=7:TP=-1.0", "-ar", "24000", "-ac", "1", tmp]).returncode == 0:
        os.replace(tmp, OUT)
    print("WROTE", OUT, flush=True)
    break
