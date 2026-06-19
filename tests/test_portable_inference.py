# SPDX-License-Identifier: MIT
from types import SimpleNamespace

from scripts.portable_inference import choose_inference_plan, is_cpu_first_arch


def fake_torch(*, cuda=False, mps=False):
    return SimpleNamespace(
        cuda=SimpleNamespace(is_available=lambda: cuda),
        backends=SimpleNamespace(mps=SimpleNamespace(is_available=lambda: mps)),
    )


def test_explicit_device_is_preserved():
    plan = choose_inference_plan("cpu", machine="x86_64", torch_module=fake_torch(cuda=True))

    assert plan.device == "cpu"
    assert plan.reason == "explicit device override"


def test_apple_silicon_prefers_mps_when_available():
    plan = choose_inference_plan(
        "auto",
        system="Darwin",
        machine="arm64",
        torch_module=fake_torch(mps=True),
    )

    assert plan.device == "mps"


def test_apple_silicon_falls_back_to_cpu_without_mps():
    plan = choose_inference_plan(
        "auto",
        system="Darwin",
        machine="arm64",
        torch_module=fake_torch(mps=False, cuda=True),
    )

    assert plan.device == "cpu"


def test_powerpc_targets_cpu_even_when_cuda_is_visible():
    plan = choose_inference_plan(
        "auto",
        system="Linux",
        machine="ppc64le",
        torch_module=fake_torch(cuda=True),
    )

    assert plan.device == "cpu"


def test_conventional_cuda_host_uses_cuda():
    plan = choose_inference_plan(
        "auto",
        system="Linux",
        machine="x86_64",
        torch_module=fake_torch(cuda=True),
    )

    assert plan.device == "cuda:0"


def test_cpu_first_arch_prefixes_cover_power_and_arm_variants():
    assert is_cpu_first_arch("powerpc64")
    assert is_cpu_first_arch("aarch64")
    assert not is_cpu_first_arch("x86_64")
