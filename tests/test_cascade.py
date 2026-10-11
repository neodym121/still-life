import threading
from functools import partial

import pytest

from conftest import CPU, CUDA, DML
from core.pipeline import Cancelled, PassthroughEngine, Reconstructor, is_oom_error, plan_attempts


def oom_dml():
    return RuntimeError("Could not allocate tensor with 52428800 bytes. "
                        "There is not enough GPU video memory available!")


class OutOfMemoryError(RuntimeError):  # same name as torch.cuda.OutOfMemoryError
    pass


def make(device, factory, **kw):
    return Reconstructor(engine_factory=factory, device_selector=lambda pref: device,
                         cpu_selector=lambda: CPU, **kw)


def run(rec, picture, w=0.5, **kw):
    statuses, progress = [], []
    result = rec.run(picture, w, 11, on_status=statuses.append, on_progress=progress.append, **kw)
    return result, statuses, progress


def test_is_oom_error():
    assert is_oom_error(oom_dml())
    assert is_oom_error(OutOfMemoryError("CUDA error"))
    assert is_oom_error(MemoryError())
    assert is_oom_error(RuntimeError("CUDA out of memory. Tried to allocate 20.00 MiB"))
    assert is_oom_error(RuntimeError("DefaultCPUAllocator: not enough memory: you tried to allocate"))
    assert is_oom_error(RuntimeError("0x8007000E E_OUTOFMEMORY"))
    try:
        try:
            raise oom_dml()
        except RuntimeError as inner:
            raise ValueError("wrapped") from inner
    except ValueError as outer:
        assert is_oom_error(outer)
    assert not is_oom_error(RuntimeError("shape mismatch"))
    assert not is_oom_error(ValueError("bad value"))


def test_attempt_order():
    sides = (448, 384, 320)
    dml = [(a.device.kind, a.long_side, a.low_memory) for a in plan_attempts(DML, CPU, sides)]
    assert dml == [("dml", 448, False), ("dml", 384, False), ("dml", 320, False),
                   ("cpu", 448, False), ("cpu", 384, False), ("cpu", 320, False)]
    cuda = [(a.device.kind, a.long_side, a.low_memory) for a in plan_attempts(CUDA, CPU, sides)]
    assert cuda[:4] == [("cuda", 448, False), ("cuda", 448, True), ("cuda", 384, True), ("cuda", 320, True)]
    assert cuda[4][0] == "cpu"
    cpu = [(a.device.kind, a.long_side) for a in plan_attempts(CPU, CPU, sides)]
    assert cpu == [("cpu", 448), ("cpu", 384), ("cpu", 320)]


def test_success_on_first_try(picture, fake_engines):
    rec = make(DML, fake_engines)
    result, statuses, progress = run(rec, picture)
    engine = fake_engines.instances[0]
    assert engine.loaded and all(call[:2] == ("dml", 448) for call in engine.calls)
    assert max(result.size) == 448
    assert progress[0] == 0 and progress[-1] == 100
    assert progress == sorted(progress)


def test_oom_retries_smaller_then_succeeds(picture, fake_engines):
    fail = lambda engine, image: oom_dml() if max(image.shape[:2]) > 384 else None  # noqa: E731
    rec = make(DML, partial(fake_engines, fail=fail))
    result, statuses, _ = run(rec, picture)
    assert max(result.size) == 384
    # first OOM: free memory and retry the same size once, then shrink
    assert [c[:2] for c in fake_engines.instances[0].calls] == [("dml", 448), ("dml", 448), ("dml", 384)]
    assert any("trying again" in s for s in statuses)
    assert any("Retrying at 384 px" in s for s in statuses)
    assert rec.device.kind == "dml"


def test_oom_everywhere_on_gpu_falls_back_to_cpu(picture, fake_engines):
    fail = lambda engine, image: oom_dml() if engine.device.kind == "dml" else None  # noqa: E731
    rec = make(DML, partial(fake_engines, fail=fail))
    result, statuses, _ = run(rec, picture)
    gpu, cpu = fake_engines.instances
    assert [c[1] for c in gpu.calls] == [448, 448, 384, 320]
    assert gpu.released
    assert cpu.device.kind == "cpu" and cpu.calls[0][:2] == ("cpu", 448)
    assert max(result.size) == 448
    assert any("Falling back to the CPU" in s for s in statuses)
    # the next run goes straight to the CPU
    assert rec.device.kind == "cpu" and "fell back" in rec.device.note
    run(rec, picture)
    assert len(fake_engines.instances) == 2


def test_cuda_tries_offload_before_shrinking(picture, fake_engines):
    def fail(engine, image):
        if engine.device.kind == "cuda" and not (engine.low_memory and max(image.shape[:2]) <= 384):
            return OutOfMemoryError("CUDA out of memory")
        return None

    rec = make(CUDA, partial(fake_engines, fail=fail))
    result, statuses, _ = run(rec, picture)
    calls = fake_engines.instances[0].calls
    assert calls[:4] == [("cuda", 448, False), ("cuda", 448, False), ("cuda", 448, True), ("cuda", 384, True)]
    assert max(result.size) == 384
    assert any("offload" in s for s in statuses)


def test_non_oom_gpu_error_goes_straight_to_cpu(picture, fake_engines):
    fail = lambda engine, image: RuntimeError("unsupported op") if engine.device.is_gpu else None  # noqa: E731
    rec = make(DML, partial(fake_engines, fail=fail))
    result, statuses, _ = run(rec, picture)
    gpu, cpu = fake_engines.instances
    assert len(gpu.calls) == 1 and cpu.calls[0][:2] == ("cpu", 448)
    assert any("inference failed" in s for s in statuses)


def test_gpu_load_failure_goes_to_cpu(picture, fake_engines):
    class BrokenLoad(fake_engines):
        def load(self, status):
            if self.device.is_gpu:
                raise RuntimeError("DirectML device lost")
            super().load(status)

    rec = make(DML, BrokenLoad)
    result, statuses, _ = run(rec, picture)
    assert rec.engine.device.kind == "cpu"
    assert any("Could not load the model" in s for s in statuses)


def test_cpu_error_is_raised(picture, fake_engines):
    fail = lambda engine, image: ValueError("broken weights")  # noqa: E731
    rec = make(CPU, partial(fake_engines, fail=fail))
    with pytest.raises(ValueError):
        run(rec, picture)


def test_cpu_oom_shrinks_then_gives_up(picture, fake_engines):
    fail = lambda engine, image: MemoryError()  # noqa: E731
    rec = make(CPU, partial(fake_engines, fail=fail))
    with pytest.raises(MemoryError):
        run(rec, picture)
    assert [c[1] for c in fake_engines.instances[0].calls] == [448, 448, 384, 320]


def test_cancel_stops_without_fallback(picture, fake_engines):
    cancel = threading.Event()

    def fail(engine, image):
        cancel.set()
        return None

    rec = make(DML, partial(fake_engines, fail=fail))
    with pytest.raises(Cancelled):
        run(rec, picture, is_cancelled=cancel.is_set)
    assert len(fake_engines.instances) == 1
    assert rec.device.kind == "dml"


def test_passthrough_engine_end_to_end(picture):
    rec = make(CPU, PassthroughEngine)
    for w in (0.0, 0.3, 1.0):
        result, _, progress = run(rec, picture, w)
        assert result.mode == "RGB" and max(result.size) == 448
        assert progress[-1] == 100
