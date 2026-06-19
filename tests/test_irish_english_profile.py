# SPDX-License-Identifier: MIT
from pathlib import Path
import wave

from scripts.generate import PRESET_REFS, resolve_ref


PROFILE_CONFIG = Path("configs/profiles/irish_english_librivox.yaml")
PROFILE_DOC = Path("docs/profiles/IRISH_ENGLISH_LIBRIVOX.md")
REF_WAV = Path("refs/irish_english_librivox_ref.wav")


def test_irish_english_preset_resolves_reference_clip():
    expected = str(REF_WAV)

    assert PRESET_REFS["irish_english"] == expected
    assert resolve_ref("irish_english", override=None, model_dir=None) == expected


def test_irish_english_reference_clip_is_short_24k_mono_wav():
    with wave.open(str(REF_WAV), "rb") as wav:
        assert wav.getnchannels() == 1
        assert wav.getframerate() == 24000
        assert wav.getsampwidth() == 2
        duration = wav.getnframes() / wav.getframerate()

    assert 11.9 <= duration <= 12.1


def test_irish_english_profile_records_public_domain_source():
    config = PROFILE_CONFIG.read_text(encoding="utf-8")
    doc = PROFILE_DOC.read_text(encoding="utf-8")

    assert "dialect_accent_0909_librivox" in config
    assert "dialectaccent_vol_01_02poh.mp3" in config
    assert "license: Public Domain" in config
    assert "refs/irish_english_librivox_ref.wav" in config
    assert "https://archive.org/details/dialect_accent_0909_librivox" in doc
    assert "This is **not** an Irish/Gaeilge language model" in doc
