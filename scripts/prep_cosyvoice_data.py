#!/usr/bin/env python3
"""
VintageVoice — Kaldi-style data prep for CosyVoice SFT

From the restored-French CSV (audio_file|text), produce the files CosyVoice's
recipe stages expect: wav.scp, text, utt2spk, spk2utt, in train/dev splits.

Speaker label = source recording stem (the part before _segNNNN) — each archive
recording is one (or one set of) Cajun speaker(s); that grouping is the best
speaker signal we have without diarization.
"""
import argparse
import csv
import os
import random
import re
from collections import defaultdict
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--dev-frac", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    rows = []
    with open(args.csv) as f:
        for row in csv.DictReader(f, delimiter="|"):
            path = os.path.abspath(row["audio_file"])
            if not os.path.exists(path):
                continue
            utt = Path(path).stem
            spk = re.sub(r"_seg\d+$", "", utt)
            rows.append((utt, spk, path, row["text"].strip()))

    random.Random(args.seed).shuffle(rows)
    n_dev = max(1, int(len(rows) * args.dev_frac))
    splits = {"dev": rows[:n_dev], "train": rows[n_dev:]}

    for name, items in splits.items():
        d = os.path.join(args.output_dir, name)
        os.makedirs(d, exist_ok=True)
        items = sorted(items)
        spk2utt = defaultdict(list)
        with open(f"{d}/wav.scp", "w") as w, open(f"{d}/text", "w") as t, open(f"{d}/utt2spk", "w") as u:
            for utt, spk, path, text in items:
                w.write(f"{utt} {path}\n")
                t.write(f"{utt} {text}\n")
                u.write(f"{utt} {spk}\n")
                spk2utt[spk].append(utt)
        with open(f"{d}/spk2utt", "w") as s:
            for spk in sorted(spk2utt):
                s.write(f"{spk} {' '.join(spk2utt[spk])}\n")
        print(f"{name}: {len(items)} utts, {len(spk2utt)} speakers -> {d}")


if __name__ == "__main__":
    main()
