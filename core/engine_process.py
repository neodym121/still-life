"""SD-Turbo in a separate process.

DirectML (and the CUDA caching allocator) keep GPU memory reserved for as long
as the process lives, even after every tensor is gone: measured on an RX 7600,
dropping the model left ~5.9 GB of 6 GB allocated. So the model lives in a
child process, and unloading it means ending that process, which makes
Windows reclaim all of its video memory. The window process never imports
torch at all.

    probe_device(pref)     device detection in a short-lived child process
    ProcessEngine(device)  the engine interface (see core/pipeline.py),
                           forwarded to a child that runs SDTurboEngine
"""

from __future__ import annotations

import importlib
import logging
import logging.handlers
import multiprocessing as mp
import os
import sys
import traceback
from dataclasses import asdict, replace
from typing import Callable, Optional

import numpy as np

import config
from core.device import DeviceInfo, cpu_device

log = logging.getLogger(__name__)

DEFAULT_BUILDER = "core.pipeline:SDTurboEngine"
PROBE_TIMEOUT_S = 180
DRAIN_TIMEOUT_S = 600


class EngineProcessError(RuntimeError):
    """The engine process failed or died; the message is the remote error."""


def _noop(*_args) -> None:
    pass


# --------------------------------------------------------------------------
# Child side
# --------------------------------------------------------------------------

def _prepare_child(log_name: str) -> None:
    # Under pythonw.exe the child has no console either.
    for name in ("stdout", "stderr"):
        if getattr(sys, name) is None:
            setattr(sys, name, open(os.devnull, "w", encoding="utf-8"))
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
    try:
        config.LOG_DIR.mkdir(parents=True, exist_ok=True)
        handler = logging.handlers.RotatingFileHandler(config.LOG_DIR / log_name, maxBytes=2_000_000,
                                                       backupCount=2, encoding="utf-8")
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)-7s [engine] %(name)s: %(message)s"))
        root = logging.getLogger()
        root.setLevel(logging.INFO)
        root.addHandler(handler)
        for noisy in ("PIL", "urllib3", "httpx", "httpcore", "filelock"):
            logging.getLogger(noisy).setLevel(logging.WARNING)
        logging.captureWarnings(True)
    except OSError:
        pass


def _plain_device(device: DeviceInfo) -> dict:
    """DeviceInfo without the torch device object (the window must not need torch)."""
    data = asdict(device)
    data["torch_device"] = str(device.torch_device)
    return data


def _probe_main(conn, pref: str) -> None:
    _prepare_child("engine.log")
    try:
        from core.device import select_device

        conn.send(("ok", _plain_device(select_device(pref))))
    except Exception as exc:  # pragma: no cover - reported to the parent
        conn.send(("error", type(exc).__name__, str(exc), traceback.format_exc()))
    finally:
        conn.close()


def _build(builder: str, device: DeviceInfo):
    module, _, attr = builder.partition(":")
    return getattr(importlib.import_module(module), attr)(device)


def _engine_main(conn, kind: str, builder: str) -> None:
    _prepare_child("engine.log")
    engine = None
    while True:
        try:
            message = conn.recv()
        except (EOFError, OSError):
            break  # the window process is gone
        command = message[0]
        try:
            if command == "quit":
                conn.send(("ok", None))
                break
            if command == "load":
                from core.device import select_device

                device = cpu_device() if kind == "cpu" else select_device(kind)
                if device.kind != kind:
                    raise RuntimeError(f"{kind} is not available in the engine process")
                engine = _build(builder, device)
                engine.load(lambda text: conn.send(("status", text)))
                conn.send(("ok", {"low_memory": bool(engine.low_memory)}))
            elif command == "img2img":
                _, image, steps, strength, seed = message
                result = engine.img2img(image, steps, strength, seed, lambda: conn.send(("tick",)))
                conn.send(("ok", np.ascontiguousarray(result)))
            elif command == "low_memory":
                conn.send(("ok", bool(engine.enable_low_memory())))
            elif command == "free":
                engine.free_memory()
                conn.send(("ok", None))
            elif command == "reset_peak":
                reset = getattr(engine, "reset_peak_memory", None)
                if reset is not None:
                    reset()
                conn.send(("ok", None))
            elif command == "peak":
                peak = getattr(engine, "peak_memory_gb", None)
                conn.send(("ok", peak() if peak is not None else None))
            else:
                raise ValueError(f"unknown command {command!r}")
        except Exception as exc:
            logging.getLogger(__name__).exception("Engine command %s failed", command)
            try:
                conn.send(("error", type(exc).__name__, str(exc), traceback.format_exc()))
            except (OSError, ValueError):
                break


# --------------------------------------------------------------------------
# Parent side
# --------------------------------------------------------------------------

def _remote_error(name: str, message: str, tb: str) -> Exception:
    log.error("Engine process error (%s):\n%s", name, tb.rstrip())
    if name in ("ImportError", "ModuleNotFoundError"):
        return ImportError(message)
    if name in ("MemoryError", "OutOfMemoryError") and "memory" not in message.lower():
        message = f"out of memory {message}".strip()
    return EngineProcessError(f"{name}: {message}")


def probe_device(pref: str = "auto") -> DeviceInfo:
    """Pick the device in a throwaway child process (no torch in the window)."""
    ctx = mp.get_context("spawn")
    receiver, sender = ctx.Pipe(duplex=False)
    process = ctx.Process(target=_probe_main, args=(sender, pref), daemon=True, name="StillLifeProbe")
    process.start()
    sender.close()
    try:
        if not receiver.poll(PROBE_TIMEOUT_S):
            raise TimeoutError("device detection timed out")
        reply = receiver.recv()
    except (EOFError, OSError, TimeoutError) as exc:
        log.warning("Device probe failed (%s), using the CPU", exc)
        return replace(cpu_device(), note=f"device detection failed: {exc}")
    finally:
        receiver.close()
        process.join(5)
        if process.is_alive():
            process.terminate()
    if reply[0] != "ok":
        error = _remote_error(*reply[1:])
        if isinstance(error, ImportError):
            raise error
        log.warning("Device probe failed (%s), using the CPU", error)
        return replace(cpu_device(), note=f"device detection failed: {error}")
    return DeviceInfo(**reply[1])


class ProcessEngine:
    """Engine interface backed by a child process. release() ends the process."""

    def __init__(self, device: DeviceInfo, builder: str = DEFAULT_BUILDER):
        self.device = device
        self.low_memory = False
        self._builder = builder
        self._process = None
        self._conn = None
        self._pending = False  # a reply is still on its way (the run was cancelled)

    @property
    def alive(self) -> bool:
        return self._process is not None and self._process.is_alive()

    @property
    def pid(self) -> Optional[int]:
        return self._process.pid if self._process is not None else None

    def load(self, status: Callable[[str], None] = _noop) -> None:
        self.release()
        ctx = mp.get_context("spawn")
        conn, child = ctx.Pipe()
        self._process = ctx.Process(target=_engine_main, args=(child, self.device.kind, self._builder),
                                    daemon=True, name="StillLifeEngine")
        self._process.start()
        child.close()
        self._conn = conn
        log.info("Engine process %s started for %s", self._process.pid, self.device.label)
        try:
            reply = self._call(("load",), status=status)
        except BaseException:
            self.release()
            raise
        self.low_memory = bool(reply.get("low_memory"))

    def img2img(self, image: np.ndarray, num_inference_steps: int, strength: float, seed: int,
                tick: Callable[[], None]) -> np.ndarray:
        return self._call(("img2img", np.ascontiguousarray(image), int(num_inference_steps),
                           float(strength), int(seed)), tick=tick)

    def enable_low_memory(self) -> bool:
        if self.device.kind != "cuda" or not self.alive:
            return False
        enabled = bool(self._call(("low_memory",)))
        self.low_memory = self.low_memory or enabled
        return enabled

    def free_memory(self) -> None:
        if self.alive:
            self._call(("free",))

    def reset_peak_memory(self) -> None:
        if self.alive:
            self._call(("reset_peak",))

    def peak_memory_gb(self) -> Optional[float]:
        return self._call(("peak",)) if self.alive else None

    def release(self) -> None:
        """End the child process; the OS frees all of its GPU memory."""
        process, conn = self._process, self._conn
        self._process = self._conn = None
        self._pending = False
        if process is None:
            return
        try:
            if process.is_alive() and conn is not None:
                conn.send(("quit",))
            process.join(5)
        except (OSError, ValueError):
            pass
        if process.is_alive():
            process.terminate()
            process.join(5)
        if conn is not None:
            conn.close()
        log.info("Engine process %s stopped", process.pid)

    # -- messaging -----------------------------------------------------------

    def _receive(self):
        try:
            return self._conn.recv()
        except (EOFError, OSError) as exc:
            self._pending = False
            raise EngineProcessError(f"the engine process stopped unexpectedly ({exc or 'no reply'})") from exc

    def _drain(self) -> None:
        """Swallow the reply of a call whose caller gave up (cancelled run)."""
        while self._pending:
            if not self._conn.poll(DRAIN_TIMEOUT_S):
                self.release()
                raise EngineProcessError("the engine process did not answer")
            if self._receive()[0] in ("ok", "error"):
                self._pending = False

    def _call(self, message, status: Callable[[str], None] = _noop, tick: Optional[Callable[[], None]] = None):
        if self._conn is None:
            raise EngineProcessError("the engine process is not running")
        self._drain()
        try:
            self._conn.send(message)
        except (OSError, ValueError) as exc:
            raise EngineProcessError(f"the engine process is not reachable ({exc})") from exc
        self._pending = True
        while True:
            reply = self._receive()
            kind = reply[0]
            if kind == "status":
                status(reply[1])
            elif kind == "tick":
                if tick is not None:
                    tick()  # may raise Cancelled; the reply is drained by the next call
            elif kind == "ok":
                self._pending = False
                return reply[1]
            elif kind == "error":
                self._pending = False
                raise _remote_error(*reply[1:])
