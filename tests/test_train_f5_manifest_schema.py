# SPDX-License-Identifier: MIT
"""
Regression tests for ``VintageVoiceDataset`` manifest parsing (Refs #181).

Bug: the F5 fine-tune dataset loader read a hard-coded ``audio_path`` column
(and a mandatory ``duration`` column), but no producer in the pipeline emits
that schema:

- ``preprocess.py`` / ``fast_manifest.py`` / ``auto_pipeline.sh`` write
  ``path|duration|source``.
- ``build_f5_csv.py`` (the F5 training CSV that ``auto_pipeline.sh`` builds)
  writes ``audio_file|text``.

So ``VintageVoiceDataset(manifest)`` raised ``KeyError: 'audio_path'`` on the
first row and fine-tuning never started. These tests pin the fix: the loader
resolves the audio column flexibly, requires paired ``text``, and probes the
clip duration when the manifest has no ``duration`` column.
"""
import csv

import pytest

# The trainer pulls in torch/torchaudio; skip cleanly where they are absent
# (the project's training environment has them — that is where this runs).
pytest.importorskip("torch")
pytest.importorskip("torchaudio")
np = pytest.importorskip("numpy")
sf = pytest.importorskip("soundfile")

from scripts.train_f5 import VintageVoiceDataset


def _wav(path, seconds=3.0, sr=24000):
    n = int(seconds * sr)
    sf.write(str(path), np.zeros(n, dtype="float32"), sr)


def _write_manifest(path, header, rows):
    with open(path, "w", newline="") as f:
        w = csv.writer(f, delimiter="|")
        w.writerow(header)
        w.writerows(rows)


def test_f5_train_csv_schema_loads(tmp_path):
    """audio_file|text (build_f5_csv.py output) — no duration column."""
    wav = tmp_path / "clip1.wav"
    _wav(wav)
    manifest = tmp_path / "f5_train.csv"
    _write_manifest(manifest, ["audio_file", "text"], [[str(wav), "bonjour mes amis"]])

    ds = VintageVoiceDataset(str(manifest))

    assert len(ds) == 1, "audio_file|text manifest must load (was KeyError before)"
    item = ds[0]
    assert item["text"] == "bonjour mes amis"
    assert item["duration"] == pytest.approx(3.0, abs=0.1)  # probed from the wav


def test_path_duration_text_schema_loads(tmp_path):
    """path|duration|text — pipeline manifest extended with transcription."""
    wav = tmp_path / "clip2.wav"
    _wav(wav, seconds=4.0)
    manifest = tmp_path / "manifest.csv"
    _write_manifest(
        manifest, ["path", "duration", "text"], [[str(wav), "4.00", "ça c'est bon"]]
    )

    ds = VintageVoiceDataset(str(manifest))

    assert len(ds) == 1
    assert ds[0]["duration"] == pytest.approx(4.0, abs=0.01)


def test_textless_and_short_rows_are_skipped_not_fatal(tmp_path):
    """A text-less manifest (path|duration|source) and sub-2s clips are
    skipped gracefully instead of crashing or being trained on."""
    good = tmp_path / "good.wav"
    short = tmp_path / "short.wav"
    _wav(good, seconds=3.0)
    _wav(short, seconds=1.0)
    manifest = tmp_path / "mixed.csv"
    _write_manifest(
        manifest,
        ["audio_file", "duration", "text"],
        [
            [str(good), "3.00", "valid line"],
            [str(short), "1.00", "too short"],          # < 2.0s -> skipped
            [str(good), "3.00", ""],                    # no text -> skipped
            [str(tmp_path / "missing.wav"), "3.00", "x"],  # missing file -> skipped
        ],
    )

    ds = VintageVoiceDataset(str(manifest))

    assert len(ds) == 1, "only the one valid, transcribed, >=2s row should load"


def test_unprobeable_duration_is_skipped_not_admitted_as_zero(tmp_path):
    """A clip whose duration is neither in the manifest nor probeable from the
    file (corrupt/unsupported audio) must be dropped, not admitted as 0.0 — a
    0.0 entry would slip past the MIN_DURATION guard and poison downstream
    length-based bucketing."""
    good = tmp_path / "good.wav"
    _wav(good, seconds=3.0)
    # A path that exists (passes the os.path.exists check) but is not decodable
    # audio, so torchaudio.info raises and no duration column is present.
    bogus = tmp_path / "bogus.wav"
    bogus.write_bytes(b"not really audio")
    manifest = tmp_path / "noduration.csv"
    _write_manifest(
        manifest,
        ["audio_file", "text"],
        [
            [str(good), "valid line"],
            [str(bogus), "unprobeable clip"],  # duration unknown -> skipped
        ],
    )

    ds = VintageVoiceDataset(str(manifest))

    assert len(ds) == 1, "the unprobeable-duration row must be skipped, not kept as 0.0"
    assert ds[0]["duration"] == pytest.approx(3.0, abs=0.1)
