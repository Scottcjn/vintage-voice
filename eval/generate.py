#!/usr/bin/env python3
"""Generate eval clips for each (model, respell) condition. Writes a manifest.
GPU if CUDA visible (run with llamas stopped); else CPU. Speed locked to 0.95
to match production. No loudnorm — analyze raw model output."""
import sys, os, csv, time
BASE = "/home/scott/vintage-voice"
sys.path.insert(0, f"{BASE}/models/cosyvoice-repo")
sys.path.insert(0, f"{BASE}/models/cosyvoice-repo/third_party/Matcha-TTS")
sys.path.insert(0, f"{BASE}/scripts/cajun8h")
sys.path.insert(0, f"{BASE}/eval")
import torch, torchaudio
from cosyvoice.cli.cosyvoice import CosyVoice2
from cajun_lexicon import respell
from test_set import SENTENCES

REF_AUDIO = "/home/scott/vintage-voice-samples-50ep/sophia_ref.wav"
REF_TEXT = ("reporting from the Serengeti. Here at Elion Labs, we've successfully "
            "trapped and tagged approximately 15.5 of these majestic little creatures.")
OUT = f"{BASE}/eval/out"
SPEED = 0.95

# (tag, model_dir, apply_respell)
CONDITIONS = [
    ("ep2",          f"{BASE}/models/CosyVoice2-cajun-ep2",          True),
    ("ep2_noresp",   f"{BASE}/models/CosyVoice2-cajun-ep2",          False),
    ("prairie",      f"{BASE}/models/CosyVoice2-cajun-prairie-ep1",  True),
    ("base",         f"{BASE}/models/CosyVoice2-0.5B",               True),
]

def main():
    on_gpu = torch.cuda.is_available() and os.environ.get("CUDA_VISIBLE_DEVICES") not in (None, "")
    print(f"device: {'GPU' if on_gpu else 'CPU'}", flush=True)
    os.makedirs(OUT, exist_ok=True)
    manifest = []
    for tag, mdir, do_resp in CONDITIONS:
        d = f"{OUT}/{tag}"; os.makedirs(d, exist_ok=True)
        print(f"\n=== {tag} ({'respell' if do_resp else 'raw'}) ===", flush=True)
        t0 = time.time()
        model = CosyVoice2(mdir, load_jit=False, load_trt=False, fp16=on_gpu)
        print(f"  loaded in {time.time()-t0:.1f}s", flush=True)
        for sid, text in SENTENCES:
            fed = respell(text) if do_resp else text
            path = f"{d}/{sid}.wav"
            try:
                for i, o in enumerate(model.inference_zero_shot(fed, REF_TEXT, REF_AUDIO, stream=False, speed=SPEED)):
                    torchaudio.save(path, o["tts_speech"], model.sample_rate)
                    break
                manifest.append({"tag": tag, "respell": int(do_resp), "id": sid,
                                 "fed_text": fed, "orig_text": text, "path": path})
                print(f"  {sid}", flush=True)
            except Exception as e:
                print(f"  {sid} FAILED: {e}", flush=True)
        del model
        import gc; gc.collect()
        if on_gpu: torch.cuda.empty_cache()
    with open(f"{OUT}/manifest.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["tag","respell","id","fed_text","orig_text","path"])
        w.writeheader(); w.writerows(manifest)
    print(f"\nDONE — {len(manifest)} clips, manifest at {OUT}/manifest.csv", flush=True)

if __name__ == "__main__":
    main()
