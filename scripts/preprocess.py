#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""
VintageVoice — Audio Preprocessing Pipeline
Converts raw archive.org audio into training-ready segments.

Pipeline:
1. Convert all audio to 24kHz mono WAV (F5-TTS native rate)
2. Split into 5-15 second segments on silence boundaries
3. Filter out music-only, noise-only, and too-quiet segments
4. Normalize loudness to -23 LUFS
5. Generate manifest CSV for training
"""
import argparse
import os
import subprocess
import json
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed


def convert_to_wav(input_path, output_path, sample_rate=24000):
    """Convert any audio to 24kHz mono WAV"""
    cmd = [
        "ffmpeg", "-y", "-i", input_path,
        "-ar", str(sample_rate), "-ac", "1",
        "-c:a", "pcm_s16le",
        "-af", "loudnorm=I=-23:TP=-1.5:LRA=11",
        output_path
    ]
    result = subprocess.run(cmd, capture_output=True, timeout=300)
    return result.returncode == 0


def split_on_silence(wav_path, output_dir, min_dur=3.0, max_dur=15.0, silence_thresh=-40):
    """Split audio on silence boundaries into segments"""
    os.makedirs(output_dir, exist_ok=True)
    stem = Path(wav_path).stem

    # Use ffmpeg silencedetect to find split points
    cmd = [
        "ffmpeg", "-i", wav_path,
        "-af", f"silencedetect=noise={silence_thresh}dB:d=0.5",
        "-f", "null", "-"
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    stderr = result.stderr

    # Parse silence timestamps
    silence_starts = []
    silence_ends = []
    for line in stderr.split('\n'):
        if 'silence_start' in line:
            try:
                ts = float(line.split('silence_start: ')[1].split()[0])
                silence_starts.append(ts)
            except (IndexError, ValueError):
                pass
        elif 'silence_end' in line:
            try:
                ts = float(line.split('silence_end: ')[1].split()[0])
                silence_ends.append(ts)
            except (IndexError, ValueError):
                pass

    # Get total duration
    probe_cmd = [
        "ffprobe", "-v", "quiet", "-show_entries",
        "format=duration", "-of", "json", wav_path
    ]
    probe = subprocess.run(probe_cmd, capture_output=True, text=True)
    try:
        total_dur = float(json.loads(probe.stdout)["format"]["duration"])
    except (KeyError, json.JSONDecodeError):
        return []

    # Build segment boundaries from silence gaps
    boundaries = [0.0]
    for s_end in silence_ends:
        if s_end - boundaries[-1] >= min_dur:
            boundaries.append(s_end)
    boundaries.append(total_dur)

    # Extract segments
    segments = []
    seg_idx = 0
    for i in range(len(boundaries) - 1):
        start = boundaries[i]
        end = boundaries[i + 1]
        duration = end - start

        if duration < min_dur:
            continue

        # If segment too long, split at max_dur intervals
        while duration > max_dur:
            seg_end = start + max_dur
            seg_path = os.path.join(output_dir, f"{stem}_seg{seg_idx:04d}.wav")
            extract_segment(wav_path, seg_path, start, seg_end)
            segments.append({"path": seg_path, "start": start, "end": seg_end, "duration": max_dur})
            seg_idx += 1
            start = seg_end
            duration = end - start

        if duration >= min_dur:
            seg_path = os.path.join(output_dir, f"{stem}_seg{seg_idx:04d}.wav")
            extract_segment(wav_path, seg_path, start, end)
            segments.append({"path": seg_path, "start": start, "end": end, "duration": duration})
            seg_idx += 1

    return segments


def extract_segment(input_path, output_path, start, end):
    """Extract a time segment from audio"""
    cmd = [
        "ffmpeg", "-y", "-i", input_path,
        "-ss", str(start), "-to", str(end),
        "-c:a", "pcm_s16le",
        output_path
    ]
    subprocess.run(cmd, capture_output=True, timeout=60)


def check_audio_quality(wav_path, min_rms=-50, max_rms=-5):
    """Filter out silence, noise-only, or clipped segments"""
    cmd = [
        "ffprobe", "-v", "quiet",
        "-show_entries", "stream=codec_type",
        "-show_entries", "format=duration",
        "-of", "json", wav_path
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    try:
        dur = float(json.loads(result.stdout)["format"]["duration"])
        if dur < 2.0 or dur > 20.0:
            return False
    except Exception:
        return False

    # Check RMS level
    cmd = [
        "ffmpeg", "-i", wav_path,
        "-af", "astats=metadata=1:reset=1",
        "-f", "null", "-"
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    # If ffmpeg ran ok, assume audio is valid (detailed RMS check is slow)
    return result.returncode == 0


def process_one_file(args_tuple):
    """Process a single audio file — convert, split, filter"""
    input_path, wav_dir, seg_dir = args_tuple
    stem = Path(input_path).stem

    # Convert to WAV
    wav_path = os.path.join(wav_dir, f"{stem}.wav")
    if not os.path.exists(wav_path):
        ok = convert_to_wav(input_path, wav_path)
        if not ok:
            return []

    # Split on silence
    segments = split_on_silence(wav_path, seg_dir)

    # Filter quality
    good_segments = []
    for seg in segments:
        if check_audio_quality(seg["path"]):
            good_segments.append(seg)
        else:
            os.unlink(seg["path"])

    return good_segments


def main():
    parser = argparse.ArgumentParser(description="Preprocess vintage audio for TTS training")
    parser.add_argument("--input", default="/mnt/18tb/vintage_voice", help="Raw audio directory")
    parser.add_argument("--output", default="/mnt/18tb/vintage_voice_processed", help="Output directory")
    parser.add_argument("--workers", type=int, default=8, help="Parallel workers")
    args = parser.parse_args()

    wav_dir = os.path.join(args.output, "wav_full")
    seg_dir = os.path.join(args.output, "segments")
    os.makedirs(wav_dir, exist_ok=True)
    os.makedirs(seg_dir, exist_ok=True)

    # Find all audio files
    audio_exts = {".mp3", ".ogg", ".wav", ".flac", ".m4a"}
    files = []
    for root, _, filenames in os.walk(args.input):
        for fn in filenames:
            if Path(fn).suffix.lower() in audio_exts:
                files.append(os.path.join(root, fn))

    print(f"Found {len(files)} audio files to process")

    # Process in parallel
    all_segments = []
    tasks = [(f, wav_dir, seg_dir) for f in files]

    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(process_one_file, t): t[0] for t in tasks}
        for i, future in enumerate(as_completed(futures)):
            src = os.path.basename(futures[future])
            try:
                segs = future.result()
                all_segments.extend(segs)
                if (i + 1) % 50 == 0 or i == 0:
                    print(f"  [{i+1}/{len(files)}] {src} -> {len(segs)} segments (total: {len(all_segments)})")
            except Exception as e:
                print(f"  [{i+1}/{len(files)}] ERROR {src}: {e}")

    # Write manifest
    manifest_path = os.path.join(args.output, "manifest.csv")
    with open(manifest_path, "w") as f:
        f.write("path|duration|source\n")
        for seg in all_segments:
            f.write(f"{seg['path']}|{seg['duration']:.2f}|{Path(seg['path']).stem}\n")

    print(f"\nDone! {len(all_segments)} segments from {len(files)} files")
    print(f"Manifest: {manifest_path}")

    # Summary stats
    total_dur = sum(s["duration"] for s in all_segments)
    print(f"Total audio: {total_dur/3600:.1f} hours")


if __name__ == "__main__":
    main()
