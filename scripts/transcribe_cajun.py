#!/usr/bin/env python3
"""
VintageVoice — Cajun French / Louisiana Transcription with QC signals

Uses the openai-whisper API (not the transformers pipeline) because it exposes
the per-segment quality signals we need to filter label noise at the source:
  - detected language (Louisiana recordings code-switch French/English constantly)
  - no_speech_prob   (hallucinated text over static/silence/music)
  - avg_logprob      (low-confidence garbage)
  - compression_ratio (repeated-n-gram hallucination loops)

Output JSON per input segment carries all signals so downstream filtering is a
re-run of a cheap pass, never a re-transcription. Also writes per-language
training manifests (cajun corpus splits into french/english subsets).
"""
import argparse
import csv
import json
import os
from pathlib import Path

# QC thresholds (from MMS / Emilia-style filtering; tune after first corpus pass)
MAX_NO_SPEECH = 0.5
MIN_AVG_LOGPROB = -1.0
MAX_COMPRESSION = 2.4
MIN_CHARS_PER_SEC = 4.0
MAX_CHARS_PER_SEC = 30.0


def qc_flags(seg, total_duration):
    """Return list of QC failure reasons for a whisper segment dict"""
    flags = []
    if seg.get("no_speech_prob", 0) > MAX_NO_SPEECH:
        flags.append("no_speech")
    if seg.get("avg_logprob", 0) < MIN_AVG_LOGPROB:
        flags.append("low_logprob")
    if seg.get("compression_ratio", 1) > MAX_COMPRESSION:
        flags.append("repeat_loop")
    text = seg.get("text", "").strip()

    # Use the actual duration of the sub-segment for chars_per_sec calculation
    seg_duration = seg.get("end", 0) - seg.get("start", 0)
    if seg_duration > 0:
        cps = len(text) / seg_duration
        if cps < MIN_CHARS_PER_SEC:
            flags.append("too_sparse")
        elif cps > MAX_CHARS_PER_SEC:
            flags.append("too_dense")
    return flags


def main():
    parser = argparse.ArgumentParser(description="Transcribe Louisiana audio with language + QC capture")
    parser.add_argument("--manifest", required=True, help="path|duration|source CSV (pipe-delimited)")
    parser.add_argument("--output", required=True, help="output dir for per-segment JSON + manifests")
    parser.add_argument("--model", default="medium", help="whisper model (medium fits 8GB alongside other loads; large-v3 needs ~10GB)")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--language", default=None, help="force language (default: auto-detect per file)")
    args = parser.parse_args()

    import whisper

    os.makedirs(args.output, exist_ok=True)

    segments = []
    with open(args.manifest) as f:
        reader = csv.DictReader(f, delimiter="|")
        for row in reader:
            if os.path.exists(row["path"]):
                segments.append(row)
    print(f"Loaded {len(segments)} segments from manifest")

    print(f"Loading Whisper {args.model} on {args.device}...")
    model = whisper.load_model(args.model, device=args.device)
    print("Whisper ready")

    results = []
    for i, seg in enumerate(segments):
        audio_path = seg["path"]
        duration = float(seg["duration"])
        stem = Path(audio_path).stem
        out_json = os.path.join(args.output, f"{stem}.json")

        if os.path.exists(out_json):
            with open(out_json) as f:
                results.append(json.load(f))
            continue

        try:
            r = model.transcribe(
                audio_path,
                language=args.language,
                fp16=(args.device != "cpu"),
                condition_on_previous_text=False,  # stops hallucination carry-over between segments
            )
        except Exception as e:
            print(f"  Error on {stem}: {e}")
            continue

        text = r["text"].strip()
        whisper_segs = r.get("segments", [])
        # Worst-case QC across sub-segments: one hallucinated stretch poisons the pair
        # Pass the total_duration here, but qc_flags will use seg["end"] - seg["start"] for its internal calculations.
        flags = sorted({f for s in whisper_segs for f in qc_flags(s, duration)})
        if not whisper_segs:
            flags.append("empty")

        record = {
            "audio_path": audio_path,
            "text": text,
            "duration": duration,
            "language": r.get("language", "unknown"),
            "qc_flags": flags,
            "no_speech_prob": max((s.get("no_speech_prob", 0) for s in whisper_segs), default=1.0),
            "avg_logprob": min((s.get("avg_logprob", 0) for s in whisper_segs), default=-10.0),
            "compression_ratio": max((s.get("compression_ratio", 1) for s in whisper_segs), default=0.0),
            "source": seg.get("source", ""),
        }
        with open(out_json, "w") as f:
            json.dump(record, f, indent=2, ensure_ascii=False)
        if text:
            results.append(record)

        if (i + 1) % 25 == 0:
            clean = sum(1 for x in results if not x["qc_flags"])
            print(f"  [{i+1}/{len(segments)}] transcribed={len(results)} clean={clean}")

    # Per-language training manifests, QC-clean entries only
    by_lang = {}
    for r in results:
        if r["qc_flags"]:
            continue
        by_lang.setdefault(r["language"], []).append(r)

    for lang, entries in sorted(by_lang.items()):
        path = os.path.join(args.output, f"train_{lang}.csv")
        with open(path, "w") as f:
            f.write("audio_path|text|duration\n")
            for r in entries:
                text = r["text"].replace("|", " ").replace("\n", " ")
                f.write(f"{r['audio_path']}|{text}|{r['duration']:.2f}\n")
        hours = sum(r["duration"] for r in entries) / 3600
        print(f"  {lang}: {len(entries)} clean segments, {hours:.1f}h -> {path}")

    flagged = [r for r in results if r["qc_flags"]]
    print(f"\nDone. {len(results)} transcribed, {len(flagged)} QC-flagged ({100*len(flagged)/max(len(results),1):.0f}%)")
    print("Flagged segments kept in JSON for review; excluded from train manifests.")


if __name__ == "__main__":
    main()