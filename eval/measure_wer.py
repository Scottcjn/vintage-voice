#!/usr/bin/env python3
"""Intelligibility: ASR back-transcription WER. whisper-medium --language fr on
each clip, jiwer WER vs the fed text (what the model was told to say).
Lower WER = more intelligible synthesis. Relative WER across models is the signal
(whisper itself is imperfect on Cajun, so treat absolutes with care)."""
import sys, os, csv, collections
BASE = "/home/scott/vintage-voice"
import whisper, jiwer

NORM = jiwer.Compose([
    jiwer.ToLowerCase(), jiwer.RemovePunctuation(),
    jiwer.RemoveMultipleSpaces(), jiwer.Strip(),
    jiwer.ReduceToListOfListOfWords(),
])

def main():
    rows = list(csv.DictReader(open(f"{BASE}/eval/out/manifest.csv")))
    import torch
    dev = "cuda" if torch.cuda.is_available() and os.environ.get("CUDA_VISIBLE_DEVICES") not in (None,"") else "cpu"
    print(f"loading whisper-medium on {dev}...", flush=True)
    model = whisper.load_model("medium", device=dev)
    out = []
    for r in rows:
        if not os.path.exists(r["path"]):
            continue
        res = model.transcribe(r["path"], language="fr", fp16=(dev=="cuda"), verbose=False)
        hyp = res["text"].strip()
        try:
            wer = jiwer.wer(r["fed_text"], hyp, truth_transform=NORM, hypothesis_transform=NORM)
        except Exception:
            wer = 1.0
        out.append({**r, "asr": hyp, "wer": round(wer, 4)})
        print(f"  [{r['tag']}] {r['id']}: WER={wer:.3f}", flush=True)
    with open(f"{BASE}/eval/out/wer.csv","w",newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0].keys())); w.writeheader(); w.writerows(out)
    # per-tag mean
    by = collections.defaultdict(list)
    for r in out: by[r["tag"]].append(r["wer"])
    print("\n=== MEAN WER by condition (lower=more intelligible) ===")
    for tag in sorted(by):
        v = by[tag]; print(f"  {tag:12} mean WER {sum(v)/len(v):.3f}  (n={len(v)})")

if __name__ == "__main__":
    main()
