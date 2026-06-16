# SPDX-License-Identifier: MIT
import types

from scripts import portable_infer


class _Cuda:
    def __init__(self, available):
        self._available = available

    def is_available(self):
        return self._available


class _Mps:
    def __init__(self, available):
        self._available = available

    def is_available(self):
        return self._available


def fake_torch(cuda=False, mps=False):
    return types.SimpleNamespace(
        cuda=_Cuda(cuda),
        backends=types.SimpleNamespace(mps=_Mps(mps)),
    )


def test_portable_target_detects_power_and_arm_arches():
    assert portable_infer.is_portable_target("ppc64le")
    assert portable_infer.is_portable_target("armv7l")
    assert portable_infer.is_portable_target("aarch64")
    assert not portable_infer.is_portable_target("x86_64")


def test_choose_device_honors_explicit_request():
    assert portable_infer.choose_device("cpu", machine="x86_64") == "cpu"
    assert portable_infer.choose_device("mps", machine="arm64") == "mps"


def test_choose_device_uses_mps_on_apple_silicon(monkeypatch):
    monkeypatch.setattr(portable_infer.platform, "system", lambda: "Darwin")
    monkeypatch.setitem(__import__("sys").modules, "torch", fake_torch(mps=True))

    assert portable_infer.choose_device("auto", machine="arm64") == "mps"


def test_choose_device_keeps_portable_arches_on_cpu_even_if_cuda_visible(monkeypatch):
    monkeypatch.setattr(portable_infer.platform, "system", lambda: "Linux")
    monkeypatch.setitem(__import__("sys").modules, "torch", fake_torch(cuda=True))

    assert portable_infer.choose_device("auto", machine="ppc64le") == "cpu"


def test_choose_device_uses_cuda_on_non_portable_hosts(monkeypatch):
    monkeypatch.setattr(portable_infer.platform, "system", lambda: "Linux")
    monkeypatch.setitem(__import__("sys").modules, "torch", fake_torch(cuda=True))

    assert portable_infer.choose_device("auto", machine="x86_64") == "cuda:0"
