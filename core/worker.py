"""Background threads: inference never runs on the GUI thread.

The worker objects live in a QThread (the moveToThread pattern). A QThread
subclass cannot declare its own `finished(object)` signal without hiding
QThread.finished, so the signals sit on a QObject instead.
"""

from __future__ import annotations

import logging
import threading
from typing import Callable

from PyQt6.QtCore import QObject, Qt, QThread, pyqtSignal, pyqtSlot

import config
from core.pipeline import Cancelled, Reconstructor

log = logging.getLogger(__name__)


def describe_error(exc: BaseException) -> str:
    """Short, human-readable reason for the status line."""
    if isinstance(exc, ImportError):
        return "PyTorch / diffusers are not installed in this environment (see README)"
    text = str(exc).strip().splitlines()[0] if str(exc).strip() else type(exc).__name__
    lowered = text.lower()
    if any(s in lowered for s in ("connection", "resolve host", "timed out", "offline", "max retries")):
        return "Could not download the SD-Turbo weights. Check the internet connection"
    if len(text) > 160:
        text = text[:157] + "…"
    return f"{type(exc).__name__}: {text}"


class ReconstructionWorker(QObject):
    progress = pyqtSignal(int)
    status = pyqtSignal(str)
    finished = pyqtSignal(object)   # PIL.Image.Image
    failed = pyqtSignal(str)
    cancelled = pyqtSignal()
    done = pyqtSignal()             # always emitted last

    def __init__(self, reconstructor: Reconstructor, image, weirdness: float, seed: int):
        super().__init__()
        self._reconstructor = reconstructor
        self._image = image
        self._weirdness = weirdness
        self._seed = seed
        self._cancel = threading.Event()

    def cancel(self) -> None:
        self._cancel.set()

    @pyqtSlot()
    def run(self) -> None:
        try:
            result = self._reconstructor.run(
                self._image, self._weirdness, self._seed,
                on_progress=self.progress.emit, on_status=self.status.emit,
                is_cancelled=self._cancel.is_set,
            )
        except Cancelled:
            log.info("Generation cancelled")
            self.cancelled.emit()
        except Exception as exc:
            log.exception("Generation failed")
            self.failed.emit(f"Generation failed: {describe_error(exc)}. Details: {config.LOG_FILE}")
        else:
            self.finished.emit(result)
        finally:
            self.done.emit()


class CallWorker(QObject):
    """Runs one function off the GUI thread (device detection, model loading)."""

    result = pyqtSignal(object)
    status = pyqtSignal(str)
    error = pyqtSignal(str)
    done = pyqtSignal()

    def __init__(self, fn: Callable[..., object], pass_status: bool = False):
        super().__init__()
        self._fn = fn
        self._pass_status = pass_status

    @pyqtSlot()
    def run(self) -> None:
        try:
            value = self._fn(self.status.emit) if self._pass_status else self._fn()
        except Exception as exc:
            log.exception("Background call failed")
            self.error.emit(describe_error(exc))
        else:
            self.result.emit(value)
        finally:
            self.done.emit()


def start_in_thread(worker: QObject, on_finished: Callable[[], None]) -> QThread:
    """Move `worker` to a new QThread and start `worker.run` there.

    Everything is connected before start() so a fast run cannot slip past.
    `done -> quit` is a direct connection so the thread can stop even while
    the GUI thread is blocked in QThread.wait() (e.g. when the window closes).
    """
    thread = QThread()
    worker.moveToThread(thread)
    thread.started.connect(worker.run)
    worker.done.connect(thread.quit, Qt.ConnectionType.DirectConnection)
    thread.finished.connect(on_finished)
    thread.start()
    return thread
