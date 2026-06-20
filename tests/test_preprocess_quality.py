# SPDX-License-Identifier: MIT
from types import SimpleNamespace

from scripts import preprocess


def _patch_probe_and_volume(monkeypatch, *, duration="6.0", mean="-23.4", peak="-3.2", rc=0):
    def fake_run(cmd, **_kwargs):
        if cmd[0] == "ffprobe":
            return SimpleNamespace(
                returncode=0,
                stdout=f'{{"format": {{"duration": "{duration}"}}}}',
                stderr="",
            )
        if cmd[0] == "ffmpeg":
            stderr = (
                "[Parsed_volumedetect_0 @ 0x1] n_samples: 144000\n"
                f"[Parsed_volumedetect_0 @ 0x1] mean_volume: {mean} dB\n"
                f"[Parsed_volumedetect_0 @ 0x1] max_volume: {peak} dB\n"
            )
            return SimpleNamespace(returncode=rc, stdout="", stderr=stderr)
        raise AssertionError(f"unexpected command: {cmd}")

    monkeypatch.setattr(preprocess.subprocess, "run", fake_run)


def test_parse_volume_db_reads_ffmpeg_volumedetect_output():
    stderr = "[Parsed_volumedetect_0 @ 0x1] mean_volume: -23.4 dB\n"

    assert preprocess._parse_volume_db(stderr, "mean_volume") == -23.4


def test_check_audio_quality_accepts_well_leveled_segment(monkeypatch):
    _patch_probe_and_volume(monkeypatch, mean="-23.4", peak="-3.2")

    assert preprocess.check_audio_quality("sample.wav")


def test_check_audio_quality_rejects_silent_segment(monkeypatch):
    _patch_probe_and_volume(monkeypatch, mean="-91.0", peak="-90.2")

    assert not preprocess.check_audio_quality("silent.wav")


def test_check_audio_quality_rejects_clipped_segment(monkeypatch):
    _patch_probe_and_volume(monkeypatch, mean="-12.0", peak="0.0")

    assert not preprocess.check_audio_quality("clipped.wav")


def test_check_audio_quality_rejects_too_hot_segment(monkeypatch):
    _patch_probe_and_volume(monkeypatch, mean="-4.8", peak="-1.0")

    assert not preprocess.check_audio_quality("too_hot.wav")


def test_check_audio_quality_rejects_missing_volume_metrics(monkeypatch):
    def fake_run(cmd, **_kwargs):
        if cmd[0] == "ffprobe":
            return SimpleNamespace(
                returncode=0,
                stdout='{"format": {"duration": "6.0"}}',
                stderr="",
            )
        if cmd[0] == "ffmpeg":
            return SimpleNamespace(returncode=0, stdout="", stderr="no volume here")
        raise AssertionError(f"unexpected command: {cmd}")

    monkeypatch.setattr(preprocess.subprocess, "run", fake_run)

    assert not preprocess.check_audio_quality("unknown.wav")
