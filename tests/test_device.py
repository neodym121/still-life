import pytest

from core.device import select_device


class Probe:
    def __init__(self, cuda=None, dml=None, broken=()):
        self._cuda, self._dml, self._broken = cuda, dml, broken

    def cuda(self):
        if "cuda" in self._broken:
            raise RuntimeError("driver exploded")
        return self._cuda

    def dml(self):
        if "dml" in self._broken:
            raise ImportError("No module named 'torch_directml'")
        return self._dml

    def cpu_name(self):
        return "Ryzen 5 5600"


NVIDIA = {"name": "GTX 1650", "vram_gb": 4.0, "rocm": False, "device": "cuda:0"}
ROCM = {"name": "RX 7900", "vram_gb": 24.0, "rocm": True, "device": "cuda:0"}
RADEON = {"name": "AMD Radeon RX 7600", "device": "privateuseone:0"}


def test_auto_prefers_cuda():
    info = select_device("auto", Probe(cuda=NVIDIA, dml=RADEON))
    assert (info.kind, info.backend, info.unet_dtype, info.vram_gb) == ("cuda", "CUDA", "float16", 4.0)


def test_rocm_counts_as_cuda():
    info = select_device("auto", Probe(cuda=ROCM))
    assert info.kind == "cuda" and info.backend == "ROCm"


def test_auto_uses_directml_without_cuda():
    info = select_device("auto", Probe(dml=RADEON))
    assert (info.kind, info.backend, info.unet_dtype) == ("dml", "DirectML", "float32")
    assert info.label == "DirectML · AMD Radeon RX 7600"


def test_auto_falls_back_to_cpu():
    info = select_device("auto", Probe())
    assert (info.kind, info.name, info.unet_dtype) == ("cpu", "Ryzen 5 5600", "float32")
    assert not info.is_gpu


def test_broken_backends_are_skipped():
    info = select_device("auto", Probe(cuda=NVIDIA, dml=RADEON, broken=("cuda",)))
    assert info.kind == "dml"
    assert select_device("auto", Probe(broken=("cuda", "dml"))).kind == "cpu"


def test_explicit_choices():
    probe = Probe(cuda=NVIDIA, dml=RADEON)
    assert select_device("cpu", probe).kind == "cpu"
    assert select_device("dml", probe).kind == "dml"
    assert select_device("cuda", probe).kind == "cuda"
    assert select_device("DML", probe).kind == "dml"


def test_unavailable_choice_falls_back_with_a_note():
    info = select_device("cuda", Probe(dml=RADEON))
    assert info.kind == "dml" and "CUDA was requested" in info.note
    info = select_device("dml", Probe())
    assert info.kind == "cpu" and "DirectML was requested" in info.note


def test_unknown_choice():
    with pytest.raises(ValueError):
        select_device("vulkan", Probe())
