#!/usr/bin/env python3
"""
VintageVoice — DeepFilterNet restoration pass for training segments

The v0 Cajun model learned the AM-radio scratch along with the voices ("flat").
This pass denoises each training segment before retraining. DeepFilterNet runs
at 48kHz internally; output is resampled back to 24kHz (F5/CosyVoice native).

Input:  pipe CSV (audio_file|text) from the Stage E builder
Output: restored wavs in --output-dir + a rewritten CSV pointing at them
Resume-safe: skips segments whose restored wav already exists.
"""
import argparse
import csv
import os
from pathlib import Path

import torch
import torchaudio


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", required=True, help="audio_file|text CSV")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--output-csv", required=True)
    p.add_argument("--sr", type=int, default=24000, help="output sample rate")
    args = p.parse_args()

    from df.enhance import enhance, init_df, load_audio

    os.makedirs(args.output_dir, exist_ok=True)
    model, df_state, _ = init_df()
    df_sr = df_state.sr()  # 48000

    rows = []
    with open(args.csv) as f:
        rows = list(csv.DictReader(f, delimiter="|"))
    print(f"{len(rows)} segments to restore")

    out_rows, done = [], 0
    for i, row in enumerate(rows):
        src = row["audio_file"]
        dst = os.path.join(args.output_dir, Path(src).name)
        if not os.path.exists(dst):
            try:
                audio, _ = load_audio(src, sr=df_sr)
                enhanced = enhance(model, df_state, audio)
                resampled = torchaudio.functional.resample(enhanced, df_sr, args.sr)
                # soundfile expects (T,) or (T,C); tensors come back (C,T)
                import soundfile as sf
                sf.write(dst, resampled.squeeze(0).numpy(), args.sr)
            except Exception as e:
                print(f"  [{i}] ERROR {Path(src).name}: {e}")
                continue
        out_rows.append({"audio_file": dst, "text": row["text"]})
        done += 1
        if done % 50 == 0:
            print(f"  [{done}/{len(rows)}] restored")

    with open(args.output_csv, "w") as f:
        f.write("audio_file|text\n")
        for r in out_rows:
            f.write(f"{r['audio_file']}|{r['text']}\n")
    print(f"Done: {done} restored -> {args.output_csv}")


if __name__ == "__main__":
    main()
