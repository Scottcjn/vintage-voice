#!/usr/bin/env python3
"""Fast manifest builder — uses soundfile instead of ffprobe"""
import soundfile as sf
import os
import glob
import sys

seg_dir = sys.argv[1] if len(sys.argv) > 1 else "/mnt/18tb/vintage_voice_processed/segments"
out_csv = sys.argv[2] if len(sys.argv) > 2 else "/mnt/18tb/vintage_voice_processed/manifest_37k.csv"

segments = sorted(glob.glob(os.path.join(seg_dir, "*.wav")))
print(f"Building manifest for {len(segments)} segments...")

with open(out_csv, "w") as f:
    f.write("path|duration|source\n")
    good = 0
    for i, seg in enumerate(segments):
        try:
            info = sf.info(seg)
            dur = info.duration
            if dur >= 2.0:
                f.write(f"{seg}|{dur:.2f}|{os.path.basename(seg)}\n")
                good += 1
        except Exception:
            pass
        if (i + 1) % 5000 == 0:
            print(f"  {i+1}/{len(segments)} processed, {good} valid")

print(f"Done! {good} valid segments in manifest")
