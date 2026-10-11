"""The model runs in a child process: unloading ends that process."""

import numpy as np
import pytest

pytest.importorskip("torch")
pytest.importorskip("diffusers")

from conftest import CPU  # noqa: E402
from core.device import select_device  # noqa: E402
from core.engine_process import EngineProcessError, ProcessEngine, probe_device  # noqa: E402
from core.pipeline import Cancelled, is_oom_error  # noqa: E402

IMAGE = np.random.default_rng(0).integers(0, 255, (64, 96, 3), dtype=np.uint8)


def devices():
    found = [pytest.param(CPU, id="cpu")]
    dml = select_device("dml")
    if dml.kind == "dml":
        found.append(pytest.param(dml, id="directml"))
    return found


@pytest.fixture
def engine_for():
    engines = []

    def make(device, builder="tiny_engine:build"):
        engine = ProcessEngine(device, builder=builder)
        engines.append(engine)
        return engine

    yield make
    for engine in engines:
        engine.release()


@pytest.mark.parametrize("device", devices())
def test_runs_in_a_child_and_release_ends_it(engine_for, device):
    engine = engine_for(device)
    statuses = []
    engine.load(statuses.append)
    assert engine.alive and "tiny model loading" in statuses
    process = engine._process
    ticks = []
    out = engine.img2img(IMAGE, 4, 0.5, 3, lambda: ticks.append(1))
    assert out.shape == IMAGE.shape and out.dtype == np.uint8
    assert len(ticks) == 2 + 2  # encode + 2 steps + decode
    engine.release()
    assert not engine.alive and not process.is_alive()  # the OS reclaims its GPU memory
    assert process.exitcode is not None


def test_same_seed_same_result_across_processes(engine_for):
    first = engine_for(CPU)
    first.load()
    a = first.img2img(IMAGE, 4, 0.5, 7, lambda: None)
    first.release()
    second = engine_for(CPU)
    second.load()
    assert np.array_equal(a, second.img2img(IMAGE, 4, 0.5, 7, lambda: None))


def test_remote_oom_is_still_recognised(engine_for):
    engine = engine_for(CPU, builder="tiny_engine:build_oom")
    engine.load()
    with pytest.raises(EngineProcessError) as info:
        engine.img2img(IMAGE, 4, 0.5, 1, lambda: None)
    assert is_oom_error(info.value)
    assert engine.alive  # a failed call does not kill the process


def test_cancel_then_next_call_still_works(engine_for):
    engine = engine_for(CPU)
    engine.load()

    def stop():
        raise Cancelled()

    with pytest.raises(Cancelled):
        engine.img2img(IMAGE, 4, 0.5, 1, stop)
    ticks = []
    out = engine.img2img(IMAGE, 4, 0.5, 1, lambda: ticks.append(1))  # drains the stale reply first
    assert out.shape == IMAGE.shape and len(ticks) == 4


def test_dead_process_is_an_error_not_a_hang(engine_for):
    engine = engine_for(CPU)
    engine.load()
    engine._process.terminate()
    engine._process.join(5)
    with pytest.raises(EngineProcessError):
        engine.img2img(IMAGE, 4, 0.5, 1, lambda: None)


def test_broken_builder_fails_load_and_cleans_up(engine_for):
    engine = engine_for(CPU, builder="tiny_engine:does_not_exist")
    with pytest.raises(EngineProcessError):
        engine.load()
    assert not engine.alive


def test_probe_device_runs_in_a_child():
    info = probe_device("cpu")
    assert info.kind == "cpu" and isinstance(info.torch_device, str)
    auto = probe_device("auto")
    assert isinstance(auto.torch_device, str)
    assert auto.kind == select_device("auto").kind
