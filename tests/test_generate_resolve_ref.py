from pathlib import Path

from scripts.generate import resolve_ref


def test_resolve_ref_prefers_explicit_override(tmp_path):
    override = tmp_path / "custom_ref.wav"

    assert resolve_ref("transatlantic", str(override), model_dir=None) == str(override)


def test_resolve_ref_uses_repo_relative_reference_when_present(tmp_path, monkeypatch):
    ref = tmp_path / "refs" / "transatlantic_ref.wav"
    ref.parent.mkdir()
    ref.touch()
    monkeypatch.chdir(tmp_path)

    assert resolve_ref("transatlantic", override=None, model_dir=None) == str(
        Path("refs") / "transatlantic_ref.wav"
    )


def test_resolve_ref_uses_model_dir_reference_when_repo_relative_missing(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    model_dir = tmp_path / "model"
    model_ref = model_dir / "refs" / "newsreel_narrator_ref.wav"
    model_ref.parent.mkdir(parents=True)
    model_ref.touch()

    assert resolve_ref("newsreel", override=None, model_dir=str(model_dir)) == str(model_ref)


def test_resolve_ref_returns_relative_reference_for_caller_error_message(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    assert resolve_ref("fireside", override=None, model_dir=None) == str(
        Path("refs") / "fdr_fireside_ref.wav"
    )


def test_resolve_ref_returns_none_for_unknown_preset_without_override(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    assert resolve_ref("unknown", override=None, model_dir=str(tmp_path / "model")) is None
