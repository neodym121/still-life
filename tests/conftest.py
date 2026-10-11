import os
import sys
import threading
from pathlib import Path

import numpy as np
import pytest
from PIL import Image, ImageDraw

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
APP_DIR = Path(__file__).resolve().parents[1]
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from core.device import DeviceInfo  # noqa: E402

CPU = DeviceInfo(kind="cpu", name="Test CPU", backend="CPU")
DML = DeviceInfo(kind="dml", name="Test Radeon", backend="DirectML", torch_device="dml-test")
CUDA = DeviceInfo(kind="cuda", name="Test GeForce", backend="CUDA", torch_device="cuda-test",
                  unet_dtype="float16", vram_gb=8.0)


def cartoon(size=(320, 240)) -> Image.Image:
    """Small flat-colour picture with straight lines and a 'face'."""
    w, h = size
    im = Image.new("RGB", size, (120, 190, 230))
    d = ImageDraw.Draw(im)
    for x in range(0, w, 32):
        d.line([(x, 0), (x, h)], fill=(90, 160, 210), width=2)
    d.rectangle([w * 0.3, h * 0.2, w * 0.7, h * 0.8], fill=(250, 230, 70), outline=(30, 30, 30), width=3)
    for cx in (0.42, 0.58):
        d.ellipse([w * cx - 12, h * 0.35, w * cx + 12, h * 0.5], fill=(255, 255, 255), outline=(0, 0, 0), width=2)
    d.arc([w * 0.4, h * 0.5, w * 0.6, h * 0.7], 20, 160, fill=(30, 30, 30), width=3)
    return im


@pytest.fixture
def picture() -> Image.Image:
    return cartoon()


@pytest.fixture
def picture_array(picture) -> np.ndarray:
    return np.asarray(picture, dtype=np.uint8).copy()


class FakeEngine:
    """Engine stand-in: optional failures per call, optional gate to hold a run."""

    instances: list = []

    def __init__(self, device, fail=None, gate: threading.Event | None = None):
        self.device = device
        self.low_memory = False
        self.loaded = False
        self.released = False
        self.calls: list[tuple[str, int, bool]] = []
        self._fail = fail or (lambda engine, image: None)
        self._gate = gate
        FakeEngine.instances.append(self)

    def load(self, status):
        self.loaded = True

    def enable_low_memory(self):
        if self.device.kind != "cuda":
            return False
        self.low_memory = True
        return True

    def img2img(self, image, num_inference_steps, strength, seed, tick):
        self.calls.append((self.device.kind, max(image.shape[:2]), self.low_memory))
        error = self._fail(self, image)
        if error is not None:
            raise error
        tick()
        for _ in range(int(num_inference_steps * strength)):
            if self._gate is not None:
                self._gate.wait(5)
            tick()
        tick()
        return image

    def free_memory(self):
        pass

    def release(self):
        self.released = True


@pytest.fixture
def fake_engines():
    FakeEngine.instances = []
    yield FakeEngine
    FakeEngine.instances = []


@pytest.fixture(scope="session")
def qapp():
    from PyQt6.QtWidgets import QApplication

    app = QApplication.instance() or QApplication(["pytest"])
    yield app
