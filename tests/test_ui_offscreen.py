"""Headless window check: open -> generate -> controls locked -> save.

Runs with QT_QPA_PLATFORM=offscreen and a fake engine instead of SD-Turbo.
"""

import threading
import time
from functools import partial

import pytest
from PIL import Image

from conftest import DML, cartoon


def wait_until(app, predicate, timeout=10.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        app.processEvents()
        if predicate():
            return True
        time.sleep(0.01)
    app.processEvents()
    return predicate()


@pytest.fixture
def make_window(qapp, tmp_path):
    from PyQt6.QtCore import QSettings

    from core.pipeline import Reconstructor
    from ui.main_window import MainWindow

    windows = []

    def factory(engine_factory):
        settings = QSettings(str(tmp_path / "settings.ini"), QSettings.Format.IniFormat)
        rec = Reconstructor(engine_factory=engine_factory, device_selector=lambda pref: DML,
                            cpu_selector=lambda: DML)
        window = MainWindow(rec, settings=settings, theme_name="light")
        window.show()
        windows.append(window)
        wait_until(qapp, lambda: window._probe is None)
        return window

    yield factory
    for window in windows:
        window._closing = True
        window.close()
        window.deleteLater()
    qapp.processEvents()


@pytest.fixture
def image_file(tmp_path):
    path = tmp_path / "profile.png"
    cartoon((300, 200)).save(path)
    return path


def test_open_generate_lock_save(qapp, make_window, fake_engines, image_file, tmp_path):
    gate = threading.Event()
    window = make_window(partial(fake_engines, gate=gate))

    assert window.device_badge.text() == "DirectML · Test Radeon"
    assert not window.generate_button.isEnabled()
    assert not window.save_button.isEnabled()
    assert not window.result_view.has_image()

    assert window.load_image_path(image_file)
    assert window.status_label.text() == \
        "Loaded Profile: profile.png. Set the weirdness and press Generate."
    assert window.original_view.has_image()
    assert window.original_meta.text() == "300 × 200"
    assert window.generate_button.isEnabled()

    from core.misread import PRESETS

    assert [b.text() for b in window.preset_buttons] == [p.name for p in PRESETS]
    assert window.preset.name == "Almost correct" and window.preset_buttons[0].isChecked()
    window.preset_buttons[2].click()
    assert window.preset.name == "Copy of a copy" and window.preset.weirdness == 0.70
    assert window.preset_hint.text() == PRESETS[2].hint
    assert sum(b.isChecked() for b in window.preset_buttons) == 1

    window.generate_button.click()
    assert window.busy
    for widget in (window.open_button, window.generate_button, window.save_button,
                   window.model_button, *window.preset_buttons):
        assert not widget.isEnabled(), widget
    assert wait_until(qapp, lambda: (window.generate_button.progress or 0) > 0)
    assert window.generate_button.display_text().startswith("Generating ")
    assert window.generate_button.display_text().endswith("%")

    gate.set()
    assert wait_until(qapp, lambda: not window.busy)
    assert window.result is not None and window.result_view.has_image()
    assert window.status_label.text().startswith("Reconstruction ready")
    assert "seed" not in window.status_label.text() and "Save it" not in window.status_label.text()
    assert window.model_button.text() == "Unload model"  # the run loaded the model
    assert window.generate_button.progress is None
    assert window.save_button.isEnabled() and window.generate_button.isEnabled()
    assert all(b.isEnabled() for b in window.preset_buttons) and window.open_button.isEnabled()

    out_png = tmp_path / "out.png"
    assert window.save_result_to(out_png)
    assert Image.open(out_png).size == window.result.size
    out_jpg = tmp_path / "out.jpg"
    assert window.save_result_to(out_jpg)
    assert Image.open(out_jpg).format == "JPEG"
    assert window.save_result_to(tmp_path / "noext")
    assert (tmp_path / "noext.png").exists()


def test_bad_file_keeps_window_usable(qapp, make_window, fake_engines, tmp_path):
    window = make_window(fake_engines)
    bad = tmp_path / "broken.png"
    bad.write_bytes(b"\x89PNG not really")
    assert not window.load_image_path(bad)
    assert "Could not open" in window.status_label.text()
    assert "stilllife.log" in window.status_label.text()
    assert not window.generate_button.isEnabled()


def test_failure_points_to_the_log(qapp, make_window, fake_engines, image_file):
    fail = lambda engine, image: ValueError("weights are corrupted")  # noqa: E731
    window = make_window(partial(fake_engines, fail=fail))
    window._closing = True  # no message box in tests
    window.load_image_path(image_file)
    window.generate()
    assert wait_until(qapp, lambda: not window.busy)
    text = window.status_label.text()
    assert "Generation failed" in text and "weights are corrupted" in text and "stilllife.log" in text
    assert window.result is None and not window.save_button.isEnabled()
    assert window.generate_button.isEnabled()


def test_seed_is_always_random(qapp, make_window, fake_engines, image_file):
    window = make_window(fake_engines)
    assert not hasattr(window, "seed_box") and not hasattr(window, "random_seed")
    window.load_image_path(image_file)
    seeds = set()
    for _ in range(3):
        window.generate()
        assert wait_until(qapp, lambda: not window.busy)
        seeds.add(window._run_seed)
    assert len(seeds) > 1


def test_close_during_generation_does_not_crash(qapp, make_window, fake_engines, image_file):
    gate = threading.Event()  # never set: the run only stops through cancellation
    window = make_window(partial(fake_engines, gate=gate))
    window.load_image_path(image_file)
    window.generate()
    assert wait_until(qapp, lambda: (window.generate_button.progress or 0) > 0)
    thread = window._thread
    started = time.monotonic()
    window.close()
    assert not thread.isRunning()
    assert time.monotonic() - started < 15
    qapp.processEvents()


def test_model_load_and_unload_button(qapp, make_window, fake_engines):
    window = make_window(fake_engines)
    assert window.model_button.text() == "Load model" and window.model_button.isEnabled()
    window.model_button.click()
    assert wait_until(qapp, lambda: not window.busy)
    assert window.reconstructor.model_loaded
    engine = window.reconstructor.engine
    assert engine.loaded and window.model_button.text() == "Unload model"
    assert window.status_label.text().startswith("Model ready on")
    window.model_button.click()
    assert not window.reconstructor.model_loaded and engine.released
    assert window.model_button.text() == "Load model"
    assert "unloaded" in window.status_label.text()


def test_model_button_is_locked_while_generating(qapp, make_window, fake_engines, image_file):
    gate = threading.Event()
    window = make_window(partial(fake_engines, gate=gate))
    window.load_image_path(image_file)
    window.generate()
    assert wait_until(qapp, lambda: (window.generate_button.progress or 0) > 0)
    assert not window.model_button.isEnabled()
    window.toggle_model()  # ignored while busy
    gate.set()
    assert wait_until(qapp, lambda: not window.busy)
    assert window.model_button.isEnabled() and window.reconstructor.model_loaded


def test_theme_toggle(qapp, make_window, fake_engines):
    from ui import theme

    window = make_window(fake_engines)
    assert theme.palette().name == "light" and window.theme_button.text() == "Dark"
    window.toggle_theme()
    assert theme.palette().name == "dark" and window.theme_button.text() == "Light"
    window.toggle_theme()
    assert theme.palette().name == "light"
