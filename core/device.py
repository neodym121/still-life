"""Backend selection: CUDA (incl. ROCm builds) -> DirectML -> CPU.

PyTorch has no Vulkan backend, so DirectML is the GPU path for AMD and Intel
cards on Windows. Every probe is lazy and failure-tolerant: a missing or broken
backend simply counts as "not available".
"""

from __future__ import annotations

import logging
import platform
from dataclasses import dataclass, replace
from typing import Any, Optional

import config

log = logging.getLogger(__name__)

DEVICE_CHOICES = ("auto", "cuda", "dml", "cpu")


@dataclass(frozen=True)
class DeviceInfo:
    kind: str                      # "cuda" | "dml" | "cpu"
    name: str                      # adapter / processor name
    backend: str                   # "CUDA" | "ROCm" | "DirectML" | "CPU"
    torch_device: Any = "cpu"      # whatever torch's .to() accepts
    unet_dtype: str = "float32"    # "float16" | "float32"
    vram_gb: Optional[float] = None
    note: str = ""                 # why the requested device was not used, etc.

    @property
    def is_gpu(self) -> bool:
        return self.kind != "cpu"

    @property
    def label(self) -> str:
        return f"{self.backend} · {self.name}" if self.name else self.backend


class TorchProbe:
    """Real hardware probe. Tests replace it with a fake."""

    def cuda(self) -> Optional[dict]:
        import torch

        if not torch.cuda.is_available() or torch.cuda.device_count() == 0:
            return None
        props = torch.cuda.get_device_properties(0)
        return {
            "name": props.name,
            "vram_gb": props.total_memory / 1024 ** 3,
            "rocm": getattr(torch.version, "hip", None) is not None,
            "device": torch.device("cuda", 0),
        }

    def dml(self) -> Optional[dict]:
        if platform.system() != "Windows":
            return None
        import torch_directml

        if not torch_directml.is_available():
            return None
        names = [torch_directml.device_name(i).strip("\x00 ") for i in range(torch_directml.device_count())]
        # Skip the software rasterizer if a real adapter is present.
        candidates = [i for i, n in enumerate(names) if "basic render" not in n.lower()] or list(range(len(names)))
        if not candidates:
            return None
        index = candidates[0]
        return {"name": names[index], "device": torch_directml.device(index)}

    def cpu_name(self) -> str:
        if platform.system() == "Windows":
            try:
                import winreg

                key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE,
                                     r"HARDWARE\DESCRIPTION\System\CentralProcessor\0")
                return str(winreg.QueryValueEx(key, "ProcessorNameString")[0]).strip()
            except OSError:
                pass
        return platform.processor() or "CPU"


def _safe(fn, what: str):
    try:
        return fn()
    except Exception as exc:  # a broken backend must not take the app down
        log.info("%s probe failed: %s", what, exc)
        return None


def cpu_device(probe=None) -> DeviceInfo:
    probe = probe or TorchProbe()
    name = _safe(probe.cpu_name, "CPU") or "CPU"
    return DeviceInfo(kind="cpu", name=name, backend="CPU", torch_device="cpu", unet_dtype="float32")


def select_device(pref: str = "auto", probe=None) -> DeviceInfo:
    """Pick the inference device. `pref` is one of DEVICE_CHOICES."""
    pref = (pref or "auto").lower()
    if pref not in DEVICE_CHOICES:
        raise ValueError(f"unknown device {pref!r}, expected one of {', '.join(DEVICE_CHOICES)}")
    probe = probe or TorchProbe()

    def cuda_info() -> Optional[DeviceInfo]:
        found = _safe(probe.cuda, "CUDA")
        if not found:
            return None
        return DeviceInfo(kind="cuda", name=found["name"], backend="ROCm" if found.get("rocm") else "CUDA",
                          torch_device=found["device"], unet_dtype="float16", vram_gb=found.get("vram_gb"))

    def dml_info() -> Optional[DeviceInfo]:
        found = _safe(probe.dml, "DirectML")
        if not found:
            return None
        dtype = "float16" if config.DML_TRY_FP16 else "float32"
        return DeviceInfo(kind="dml", name=found["name"], backend="DirectML",
                          torch_device=found["device"], unet_dtype=dtype)

    if pref == "cpu":
        return cpu_device(probe)

    note = ""
    if pref in ("cuda", "dml"):
        info = cuda_info() if pref == "cuda" else dml_info()
        if info:
            return info
        note = f"{'CUDA' if pref == 'cuda' else 'DirectML'} was requested but is not available"
        log.warning(note)

    for finder in (cuda_info, dml_info):
        info = finder()
        if info:
            return replace(info, note=note)
    return replace(cpu_device(probe), note=note)


def reset_peak_memory(device: DeviceInfo) -> None:
    if device.kind == "cuda":
        import torch

        torch.cuda.reset_peak_memory_stats()


def peak_memory_gb(device: DeviceInfo) -> Optional[float]:
    """Peak VRAM since the last reset, or None where the backend can't tell."""
    if device.kind == "cuda":
        import torch

        return torch.cuda.max_memory_allocated() / 1024 ** 3
    return None
