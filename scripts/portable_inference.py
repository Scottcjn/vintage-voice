# SPDX-License-Identifier: MIT
"""Portable inference device planning for VintageVoice.

The helpers here avoid importing F5-TTS at module import time so architecture
selection and benchmark dry-runs stay testable on machines without model deps.
"""

from __future__ import annotations

from dataclasses import dataclass
import importlib
import platform
from typing import Any


CPU_FIRST_ARCH_PREFIXES = ("arm", "aarch", "ppc", "powerpc", "power")
APPLE_SILICON_ARCHES = {"arm64", "aarch64"}


@dataclass(frozen=True)
class InferencePlan:
    system: str
    machine: str
    device: str
    reason: str


def normalize_machine(machine: str | None = None) -> str:
    return (machine or platform.machine() or "").strip().lower().replace("-", "_")


def normalize_system(system: str | None = None) -> str:
    return (system or platform.system() or "").strip().lower()


def is_cpu_first_arch(machine: str | None = None) -> bool:
    normalized = normalize_machine(machine)
    return normalized.startswith(CPU_FIRST_ARCH_PREFIXES)


def _load_torch() -> Any | None:
    try:
        return importlib.import_module("torch")
    except ImportError:
        return None


def _torch_mps_available(torch_module: Any | None) -> bool:
    backends = getattr(torch_module, "backends", None)
    mps = getattr(backends, "mps", None)
    checker = getattr(mps, "is_available", None)
    return bool(checker and checker())


def _torch_cuda_available(torch_module: Any | None) -> bool:
    cuda = getattr(torch_module, "cuda", None)
    checker = getattr(cuda, "is_available", None)
    return bool(checker and checker())


def choose_inference_plan(
    requested_device: str = "auto",
    *,
    machine: str | None = None,
    system: str | None = None,
    torch_module: Any | None = None,
) -> InferencePlan:
    """Resolve an F5-TTS device string for a host.

    Explicit devices are preserved. ``auto`` prefers Apple Silicon MPS, keeps
    PPC/POWER/non-mac ARM on CPU, then falls back to CUDA on conventional hosts.
    """

    normalized_machine = normalize_machine(machine)
    normalized_system = normalize_system(system)
    requested = (requested_device or "auto").strip()
    if requested.lower() != "auto":
        return InferencePlan(
            system=normalized_system,
            machine=normalized_machine,
            device=requested,
            reason="explicit device override",
        )

    torch_for_probe = torch_module if torch_module is not None else _load_torch()
    if (
        normalized_system == "darwin"
        and normalized_machine in APPLE_SILICON_ARCHES
        and _torch_mps_available(torch_for_probe)
    ):
        return InferencePlan(
            system=normalized_system,
            machine=normalized_machine,
            device="mps",
            reason="Apple Silicon MPS is available",
        )

    if is_cpu_first_arch(normalized_machine):
        return InferencePlan(
            system=normalized_system,
            machine=normalized_machine,
            device="cpu",
            reason="portable CPU-first target architecture",
        )

    if _torch_cuda_available(torch_for_probe):
        return InferencePlan(
            system=normalized_system,
            machine=normalized_machine,
            device="cuda:0",
            reason="CUDA is available on this host",
        )

    return InferencePlan(
        system=normalized_system,
        machine=normalized_machine,
        device="cpu",
        reason="no supported accelerator detected",
    )
