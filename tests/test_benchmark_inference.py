# SPDX-License-Identifier: MIT
from argparse import Namespace

from scripts.benchmark_inference import run_benchmark


def make_args(tmp_path, **overrides):
    values = {
        "text": "portable test",
        "preset": "transatlantic",
        "model": None,
        "vocab": None,
        "ref_audio": None,
        "ref_text": "",
        "device": "cpu",
        "runs": 2,
        "warmup": 1,
        "output_dir": str(tmp_path),
        "json": None,
        "dry_run": False,
    }
    values.update(overrides)
    return Namespace(**values)


def test_dry_run_returns_plan_without_generating(tmp_path):
    called = False

    def generate(**kwargs):
        nonlocal called
        called = True

    result = run_benchmark(make_args(tmp_path, dry_run=True), generate=generate)

    assert result["dry_run"] is True
    assert result["plan"]["device"] == "cpu"
    assert result["runs"] == []
    assert called is False


def test_benchmark_skips_warmup_and_records_outputs(tmp_path):
    calls = []

    def generate(**kwargs):
        calls.append(kwargs)
        return kwargs["output_path"]

    result = run_benchmark(make_args(tmp_path), generate=generate)

    assert len(calls) == 3
    assert len(result["runs"]) == 2
    assert result["runs"][0]["output"].endswith("portable_inference_2.wav")
    assert all(call["device"] == "cpu" for call in calls)


def test_benchmark_rejects_empty_run_count(tmp_path):
    try:
        run_benchmark(make_args(tmp_path, runs=0))
    except ValueError as exc:
        assert "--runs" in str(exc)
    else:
        raise AssertionError("expected ValueError")
