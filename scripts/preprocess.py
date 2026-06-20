#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""
VintageVoice — Audio Preprocessing Pipeline
Converts raw archive.org audio into training-ready segments.

Pipeline:
1. Convert all audio to 24kHz mono WAV (F5-TTS native rate) with two-pass loudnorm
2. Split into 5-15 second segments on silence boundaries
3. Filter out music-only, noise-only, and too-quiet/clipped segments (RMS+peak gate)
4. Normalize loudness to -23 LUFS
5. Generate manifest CSV for training

Security hardening (CWE-918 SSRF, CWE-59/CWE-367 symlink TOCTOU, CWE-1236 CSV
injection): all ffmpeg/ffprobe invocations are restricted to local file access,
inputs are content-sniffed before processing, outputs are written atomically, and
manifests are emitted with csv.writer.
"""
import argparse
import os
import subprocess
import json
import csv
import re
import hashlib
import tempfile
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed


# --- Security hardening ------------------------------------------------------
# Restrict FFmpeg/ffprobe to local file access only. Without this, a file whose
# *content* is an HLS/concat playlist (regardless of its extension) can make
# FFmpeg follow http/file URLs embedded in it, enabling SSRF and arbitrary file
# reads (CWE-918).
FFMPEG_PROTOCOLS = "file,pipe"

# Headers of playlist / markup payloads that must never be fed to FFmpeg even if
# they carry an audio extension. Real audio containers start with ID3/RIFF/OggS/
# fLaC/ftyp — never with these signatures.
_DANGEROUS_HEADERS = (b"#EXTM3U", b"#EXT-X", b"ffconcat", b"<?xml", b"<")

# Characters allowed in derived output filenames; everything else collapses to
# "_" so a crafted source name cannot inject CSV rows or path separators
# (CWE-1236 / CWE-507).
_SAFE_STEM_RE = re.compile(r"[^A-Za-z0-9._-]+")


def _safe_stem(name):
    """Sanitize a filename stem for safe use in output paths and the manifest.

    Distinct inputs MUST map to distinct outputs. Naive sanitization collapses
    e.g. "a:b", "a|b" and "a/b" all to "a_b", so unrelated files would overwrite
    each other. To guarantee uniqueness, when sanitization actually changes the
    stem we append a short hash of the *original* name. Names that are already
    safe pass through unchanged (preserves readable output filenames and keeps
    behaviour stable for the common case).
    """
    sanitized = _SAFE_STEM_RE.sub("_", name).strip("._")
    if sanitized == name:
        # Already filesystem/CSV-safe — leave it untouched.
        return sanitized or "segment"
    base = sanitized or "segment"
    # Short, stable suffix derived from the original (pre-sanitization) name so
    # that two different originals can never collide on the same output stem.
    digest = hashlib.sha1(name.encode("utf-8", "surrogatepass")).hexdigest()[:8]
    return f"{base}_{digest}"


def assert_safe_audio_input(path):
    """Reject inputs whose content is a playlist/markup disguised as audio."""
    try:
        with open(path, "rb") as fh:
            head = fh.read(64).lstrip()
    except OSError:
        raise ValueError(f"cannot read input: {path}")
    for sig in _DANGEROUS_HEADERS:
        if head.startswith(sig):
            raise ValueError(f"refusing non-audio/playlist payload: {path}")


def _parse_loudnorm_json(stderr):
    """Extract the loudnorm measurement dict from ffmpeg's stderr.

    With ``print_format=json`` ffmpeg emits a *multi-line* JSON object at the
    END of stderr, e.g.::

        [Parsed_loudnorm_0 @ 0x...]
        {
            "input_i" : "-27.61",
            "input_tp" : "-9.30",
            "input_lra" : "5.40",
            "input_thresh" : "-37.79",
            ...
        }

    A line-by-line ``json.loads`` never sees a complete object, so the whole
    two-pass feature silently degraded to single-pass. Here we locate the LAST
    balanced ``{ ... }`` block in stderr and parse it. Returns a dict with the
    ``input_*`` keys on success, or ``{}`` on any parse failure (caller falls
    back to single-pass).
    """
    if not stderr:
        return {}
    # ffmpeg's JSON block is the last top-level object in stderr. Find the last
    # '{' and walk forward tracking brace depth to its matching '}', which is
    # robust even if other braces appear earlier in the log.
    start = stderr.rfind('{')
    while start != -1:
        depth = 0
        for idx in range(start, len(stderr)):
            ch = stderr[idx]
            if ch == '{':
                depth += 1
            elif ch == '}':
                depth -= 1
                if depth == 0:
                    block = stderr[start:idx + 1]
                    try:
                        candidate = json.loads(block)
                    except json.JSONDecodeError:
                        break  # malformed block; try an earlier '{'
                    if isinstance(candidate, dict) and 'input_i' in candidate:
                        return candidate
                    break  # parsed but not the loudnorm object; look earlier
        # Either we broke out (malformed / wrong object) or never closed the
        # brace — search for an earlier '{' before this one.
        start = stderr.rfind('{', 0, start)
    return {}


def _ffmpeg_to_temp(output_path, cmd, timeout):
    """Run an ffmpeg command, writing the result atomically to output_path.

    FFmpeg writes to a freshly created temp file in the destination directory;
    on success it is os.replace()'d onto output_path. This defeats the symlink /
    TOCTOU arbitrary-write attack (CWE-367, CWE-59): even if output_path is a
    planted symlink, os.replace overwrites the link itself, never its target.
    The temp name carries no controlled extension, so callers must force the
    output format explicitly (e.g. "-f", "wav").

    Returns True on success, False on any failure (non-zero exit, empty output,
    timeout, etc.). Callers MUST check the return value before using output_path.
    """
    out_dir = os.path.dirname(output_path) or "."
    fd, tmp_path = tempfile.mkstemp(suffix=".tmp", dir=out_dir)
    os.close(fd)
    try:
        result = subprocess.run(cmd + [tmp_path], capture_output=True, timeout=timeout)
        if result.returncode != 0 or not os.path.getsize(tmp_path):
            os.unlink(tmp_path)
            return False
        os.replace(tmp_path, output_path)
        return True
    except Exception:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
        return False


def convert_to_wav(input_path, output_path, sample_rate=24000):
    """Convert any audio to 24kHz mono WAV with proper two-pass loudnorm.

    Two-pass loudnorm is essential for TTS training data:
    - Pass 1 measures the true integrated loudness of the full file.
    - Pass 2 applies a linear gain to hit exactly -23 LUFS, preventing the model
      from learning volume variations instead of speech patterns.
    - If measurement fails (malformed/short audio), falls back to single-pass.

    Security: the input is content-sniffed first, ffmpeg is locked to local
    protocols, and the result is written atomically (defeats symlink TOCTOU).
    """
    assert_safe_audio_input(input_path)

    # ── Pass 1: Measure true integrated loudness ──
    measure_cmd = [
        "ffmpeg", "-y", "-protocol_whitelist", FFMPEG_PROTOCOLS,
        "-i", input_path,
        "-ar", str(sample_rate), "-ac", "1",
        "-af", "loudnorm=I=-23:TP=-1.5:LRA=11:print_format=json",
        "-f", "null", "-"
    ]
    try:
        measure = subprocess.run(measure_cmd, capture_output=True, text=True, timeout=300)
    except subprocess.SubprocessError:
        measure = None

    # Parse the JSON measurement from stderr. ffmpeg emits a *multi-line* JSON
    # object at the end of stderr, so we extract the trailing balanced block
    # rather than scanning line-by-line (which never sees a complete object).
    measured = _parse_loudnorm_json(measure.stderr) if measure is not None else {}

    # ── Pass 2: Apply with measured values ──
    # Require every field pass-2 consumes; a partial measurement is treated as a
    # failure so we fall back to single-pass instead of emitting a broken filter.
    required = ("input_i", "input_tp", "input_lra", "input_thresh")
    if measured and all(k in measured for k in required):
        # linear=true: apply pure gain — no dynamic compression. Preserves the
        # original dynamic range while hitting the exact target loudness.
        loudnorm_filter = (
            f"loudnorm=I=-23:TP=-1.5:LRA=11:"
            f"measured_I={measured['input_i']}:"
            f"measured_TP={measured['input_tp']}:"
            f"measured_LRA={measured['input_lra']}:"
            f"measured_thresh={measured['input_thresh']}:"
            f"linear=true:print_format=summary"
        )
        # ffmpeg also reports the DC/normalization offset; pass it through when
        # present so pass-2 reproduces pass-1's gain decision exactly.
        if "target_offset" in measured:
            loudnorm_filter += f":offset={measured['target_offset']}"
    else:
        # Fallback: measurement failed (e.g. very short audio). Best-effort
        # single-pass — rare, but better than dropping the file.
        print(f"  [WARN] loudnorm measurement failed for {input_path}, "
              f"falling back to single-pass", flush=True)
        loudnorm_filter = "loudnorm=I=-23:TP=-1.5:LRA=11"

    cmd = [
        "ffmpeg", "-y", "-protocol_whitelist", FFMPEG_PROTOCOLS,
        "-i", input_path,
        "-ar", str(sample_rate), "-ac", "1",
        "-c:a", "pcm_s16le",
        "-af", loudnorm_filter,
        "-f", "wav",
    ]
    return _ffmpeg_to_temp(output_path, cmd, timeout=300)


def split_on_silence(wav_path, output_dir, min_dur=3.0, max_dur=15.0, silence_thresh=-40):
    """Split audio on silence boundaries into segments"""
    os.makedirs(output_dir, exist_ok=True)
    stem = _safe_stem(Path(wav_path).stem)

    # Use ffmpeg silencedetect to find split points
    cmd = [
        "ffmpeg", "-protocol_whitelist", FFMPEG_PROTOCOLS, "-i", wav_path,
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
        "ffprobe", "-v", "quiet", "-protocol_whitelist", FFMPEG_PROTOCOLS,
        "-show_entries", "format=duration", "-of", "json", wav_path
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
            # Only record the segment if extraction actually produced a file.
            if extract_segment(wav_path, seg_path, start, seg_end):
                segments.append({"path": seg_path, "start": start, "end": seg_end, "duration": max_dur})
                seg_idx += 1
            start = seg_end
            duration = end - start

        if duration >= min_dur:
            seg_path = os.path.join(output_dir, f"{stem}_seg{seg_idx:04d}.wav")
            if extract_segment(wav_path, seg_path, start, end):
                segments.append({"path": seg_path, "start": start, "end": end, "duration": duration})
                seg_idx += 1

    return segments


def extract_segment(input_path, output_path, start, end):
    """Extract a time segment from audio.

    Returns True only if ffmpeg succeeded and a non-empty output was written.
    Callers MUST honour the return value — a failed segment must not flow into
    quality checks or the manifest.
    """
    cmd = [
        "ffmpeg", "-y", "-protocol_whitelist", FFMPEG_PROTOCOLS,
        "-i", input_path,
        "-ss", str(start), "-to", str(end),
        "-c:a", "pcm_s16le",
        "-f", "wav",
    ]
    return _ffmpeg_to_temp(output_path, cmd, timeout=60)


def _parse_volume_db(stderr, field):
    """Return a volumedetect dB field from ffmpeg stderr."""
    marker = f"{field}:"
    for line in stderr.splitlines():
        if marker not in line:
            continue
        value_text = line.split(marker, 1)[1].strip().split()[0]
        try:
            return float(value_text)
        except ValueError:
            return None
    return None


def check_audio_quality(wav_path, min_rms=-50, max_rms=-5, max_peak=-0.1):
    """Filter out silence, noise-only, or clipped segments via a real RMS/peak gate."""
    cmd = [
        "ffprobe", "-v", "quiet", "-protocol_whitelist", FFMPEG_PROTOCOLS,
        "-show_entries", "stream=codec_type",
        "-show_entries", "format=duration",
        "-of", "json", wav_path
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    try:
        dur = float(json.loads(result.stdout)["format"]["duration"])
        if dur < 2.0 or dur > 20.0:
            return False
    except (KeyError, ValueError, json.JSONDecodeError):
        return False

    # Check RMS and peak level. The previous implementation accepted any file
    # where ffmpeg completed, which let silent or clipped segments into training.
    cmd = [
        "ffmpeg", "-protocol_whitelist", FFMPEG_PROTOCOLS, "-i", wav_path,
        "-af", "volumedetect",
        "-f", "null", "-"
    ]
    # A volumedetect timeout or ffmpeg error must reject only THIS segment, not
    # propagate out of the worker and drop the entire source file. Treat any
    # subprocess failure as "fails quality".
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    except (subprocess.SubprocessError, OSError):
        return False
    if result.returncode != 0:
        return False

    mean_volume = _parse_volume_db(result.stderr, "mean_volume")
    max_volume = _parse_volume_db(result.stderr, "max_volume")
    if mean_volume is None or max_volume is None:
        return False
    if mean_volume <= min_rms or mean_volume >= max_rms:
        return False
    if max_volume >= max_peak:
        return False
    return True


def process_one_file(args_tuple):
    """Process a single audio file — convert, split, filter"""
    input_path, wav_dir, seg_dir = args_tuple
    stem = _safe_stem(Path(input_path).stem)

    # Convert to WAV
    wav_path = os.path.join(wav_dir, f"{stem}.wav")
    # Refuse to read or overwrite through a symlink planted at the output path
    # (CWE-59): otherwise ffmpeg would follow it to an arbitrary target.
    if os.path.islink(wav_path):
        return []
    if not os.path.exists(wav_path):
        try:
            ok = convert_to_wav(input_path, wav_path)
        except ValueError:
            # Rejected non-audio / playlist payload — skip silently.
            return []
        if not ok:
            return []
    else:
        # Resume / pre-populated wav_dir path: the WAV already exists, so
        # convert_to_wav (which sniffs the input) is skipped. Sniff the existing
        # WAV here too — otherwise a .wav whose *content* is a playlist/markup
        # payload would flow straight into ffmpeg via split_on_silence(),
        # re-opening the SSRF/local-read hole (CWE-918) on the resume path.
        try:
            assert_safe_audio_input(wav_path)
        except ValueError:
            return []

    # Split on silence
    segments = split_on_silence(wav_path, seg_dir)

    # Filter quality
    good_segments = []
    for seg in segments:
        if check_audio_quality(seg["path"]):
            good_segments.append(seg)
        else:
            try:
                os.unlink(seg["path"])
            except OSError:
                pass

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

    # Write manifest. Use csv.writer (not raw string formatting) so any field
    # containing a pipe, quote or newline is properly quoted — this prevents
    # manifest row injection / ML data poisoning (CWE-1236) while staying
    # compatible with the downstream csv.DictReader(delimiter="|") consumers.
    manifest_path = os.path.join(args.output, "manifest.csv")
    with open(manifest_path, "w", newline="") as f:
        writer = csv.writer(f, delimiter="|", quoting=csv.QUOTE_MINIMAL)
        writer.writerow(["path", "duration", "source"])
        for seg in all_segments:
            writer.writerow([seg["path"], f"{seg['duration']:.2f}", Path(seg["path"]).stem])

    print(f"\nDone! {len(all_segments)} segments from {len(files)} files")
    print(f"Manifest: {manifest_path}")

    # Summary stats
    total_dur = sum(s["duration"] for s in all_segments)
    print(f"Total audio: {total_dur/3600:.1f} hours")


if __name__ == "__main__":
    main()
