# SPDX-License-Identifier: MIT
"""Tests for the two-pass loudnorm JSON parsing + resume-path hardening."""
from types import SimpleNamespace

import pytest

from scripts import preprocess


# A realistic ffmpeg stderr: normal banner/log lines, then the multi-line JSON
# block loudnorm prints at the very end with print_format=json.
REALISTIC_STDERR = """ffmpeg version 6.1.1 Copyright (c) 2000-2023 the FFmpeg developers
  built with gcc 13 (GCC)
Input #0, mp3, from 'in.mp3':
  Duration: 00:03:21.50, start: 0.025057, bitrate: 128 kb/s
Stream mapping:
  Stream #0:0 -> #0:0 (mp3 (mp3float) -> pcm_s16le (native))
[Parsed_loudnorm_0 @ 0x55d1f0]
{
	"input_i" : "-27.61",
	"input_tp" : "-9.30",
	"input_lra" : "5.40",
	"input_thresh" : "-37.79",
	"output_i" : "-23.00",
	"output_tp" : "-1.50",
	"output_lra" : "5.30",
	"output_thresh" : "-33.18",
	"normalization_type" : "dynamic",
	"target_offset" : "0.21"
}
"""


def test_parse_loudnorm_json_extracts_trailing_block():
    measured = preprocess._parse_loudnorm_json(REALISTIC_STDERR)
    assert measured["input_i"] == "-27.61"
    assert measured["input_tp"] == "-9.30"
    assert measured["input_lra"] == "5.40"
    assert measured["input_thresh"] == "-37.79"
    assert measured["target_offset"] == "0.21"


def test_parse_loudnorm_json_returns_empty_on_garbage():
    assert preprocess._parse_loudnorm_json("no json here at all") == {}
    assert preprocess._parse_loudnorm_json("") == {}
    # A truncated/partial block (crashed pass 1) must not parse.
    assert preprocess._parse_loudnorm_json('[loudnorm]\n{\n\t"input_i" : "-27') == {}


def test_parse_loudnorm_json_ignores_unrelated_json_object():
    # An earlier non-loudnorm JSON object must not be mistaken for the result.
    stderr = '{"foo": 1}\n[Parsed_loudnorm_0]\n{\n  "input_i": "-20.0",\n  "input_tp": "-3.0",\n  "input_lra": "4.0",\n  "input_thresh": "-30.0"\n}\n'
    measured = preprocess._parse_loudnorm_json(stderr)
    assert measured["input_i"] == "-20.0"
    assert "foo" not in measured


def test_convert_to_wav_runs_two_pass_when_measured(monkeypatch):
    """With a valid measurement, pass 2 must use the measured_* values."""
    captured = {}

    def fake_run(cmd, **_kwargs):
        # First call = measurement pass (-f null), returns JSON in stderr.
        if "null" in cmd:
            return SimpleNamespace(returncode=0, stdout="", stderr=REALISTIC_STDERR)
        raise AssertionError(f"unexpected measurement command: {cmd}")

    def fake_to_temp(output_path, cmd, timeout):
        # Capture the pass-2 ffmpeg command (the one writing the WAV).
        captured["cmd"] = cmd
        return True

    monkeypatch.setattr(preprocess, "assert_safe_audio_input", lambda p: None)
    monkeypatch.setattr(preprocess.subprocess, "run", fake_run)
    monkeypatch.setattr(preprocess, "_ffmpeg_to_temp", fake_to_temp)

    assert preprocess.convert_to_wav("in.mp3", "out.wav") is True

    filt = captured["cmd"][captured["cmd"].index("-af") + 1]
    assert "measured_I=-27.61" in filt
    assert "measured_TP=-9.30" in filt
    assert "measured_LRA=5.40" in filt
    assert "measured_thresh=-37.79" in filt
    assert "linear=true" in filt
    assert "offset=0.21" in filt


def test_convert_to_wav_falls_back_on_unparseable_measurement(monkeypatch):
    """Genuine parse failure must drop to single-pass loudnorm, not crash."""
    captured = {}

    def fake_run(cmd, **_kwargs):
        return SimpleNamespace(returncode=0, stdout="", stderr="no json here")

    def fake_to_temp(output_path, cmd, timeout):
        captured["cmd"] = cmd
        return True

    monkeypatch.setattr(preprocess, "assert_safe_audio_input", lambda p: None)
    monkeypatch.setattr(preprocess.subprocess, "run", fake_run)
    monkeypatch.setattr(preprocess, "_ffmpeg_to_temp", fake_to_temp)

    assert preprocess.convert_to_wav("in.mp3", "out.wav") is True
    filt = captured["cmd"][captured["cmd"].index("-af") + 1]
    assert filt == "loudnorm=I=-23:TP=-1.5:LRA=11"
    assert "measured_" not in filt


def test_existing_wav_is_sniffed_on_resume(monkeypatch, tmp_path):
    """An existing .wav whose content is a playlist must be rejected (CWE-918)."""
    wav_dir = tmp_path / "wav"
    seg_dir = tmp_path / "seg"
    wav_dir.mkdir()
    seg_dir.mkdir()
    # Pre-populate wav_dir with a malicious "wav" that is actually an HLS playlist.
    poisoned = wav_dir / "track.wav"
    poisoned.write_bytes(b"#EXTM3U\nhttp://169.254.169.254/latest/meta-data\n")

    # split_on_silence must NEVER be reached for the poisoned file.
    def boom(*_a, **_k):
        raise AssertionError("split_on_silence reached a non-audio payload")

    monkeypatch.setattr(preprocess, "split_on_silence", boom)

    result = preprocess.process_one_file((str(tmp_path / "track.mp3"), str(wav_dir), str(seg_dir)))
    assert result == []


def test_check_audio_quality_rejects_segment_on_volumedetect_timeout(monkeypatch):
    """A volumedetect timeout must reject the segment, not propagate."""
    def fake_run(cmd, **_kwargs):
        if cmd[0] == "ffprobe":
            return SimpleNamespace(
                returncode=0,
                stdout='{"format": {"duration": "6.0"}}',
                stderr="",
            )
        if cmd[0] == "ffmpeg":
            raise preprocess.subprocess.TimeoutExpired(cmd, 30)
        raise AssertionError(f"unexpected command: {cmd}")

    monkeypatch.setattr(preprocess.subprocess, "run", fake_run)

    # Must return False (segment rejected), NOT raise out of the worker.
    assert preprocess.check_audio_quality("slow.wav") is False
