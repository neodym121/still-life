"""Main window: Original / Reconstruction, the Weirdness presets and actions."""

from __future__ import annotations

import logging
import os
import random
import time
from pathlib import Path
from typing import Optional

from PIL import Image
from PyQt6.QtCore import QSettings, Qt, QThread
from PyQt6.QtGui import QIcon, QKeySequence, QShortcut
from PyQt6.QtWidgets import (QButtonGroup, QFileDialog, QFrame, QGridLayout, QHBoxLayout, QLabel,
                             QMainWindow, QMessageBox, QPushButton, QSizePolicy, QVBoxLayout, QWidget)

import config
from core.imaging import ImageLoadError, load_image
from core.misread import DEFAULT_PRESET, PRESETS, Preset, preset_named
from core.pipeline import Reconstructor
from core.worker import CallWorker, ReconstructionWorker, start_in_thread
from ui import theme
from ui.widgets import ImageView, ProgressButton, logo_pixmap

log = logging.getLogger(__name__)

LOGO_PATH = config.ASSETS_DIR / "logo.png"
RESULT_PLACEHOLDER = "Your reconstruction will appear here"
CLOSE_WAIT_MS = 20000


def _card() -> QFrame:
    card = QFrame()
    card.setObjectName("Card")
    return card


def _label(text: str, name: str) -> QLabel:
    label = QLabel(text)
    label.setObjectName(name)
    return label


class MainWindow(QMainWindow):
    def __init__(self, reconstructor: Reconstructor, settings: Optional[QSettings] = None,
                 theme_name: Optional[str] = None):
        super().__init__()
        self.reconstructor = reconstructor
        self.settings = settings or QSettings(config.APP_ID, config.APP_ID)
        self.original: Optional[Image.Image] = None
        self.original_path: Optional[Path] = None
        self.result: Optional[Image.Image] = None
        self._thread: Optional[QThread] = None
        self._worker: Optional[ReconstructionWorker | CallWorker] = None
        self._probe: Optional[tuple[QThread, CallWorker]] = None
        self._loading_model = False
        self._started_at = 0.0
        self._run_seed = 0
        self._closing = False

        self.setWindowTitle(config.APP_NAME)
        self.setWindowIcon(QIcon(str(LOGO_PATH)))
        self.setAcceptDrops(True)
        self.resize(1120, 740)
        self.setMinimumSize(880, 600)

        self._theme_name = theme_name or str(self.settings.value("theme", "light"))
        self._build()
        self._apply_theme(self._theme_name)
        self._restore_settings()
        self._update_preset()
        self._update_actions()
        self.set_status("Open an image to begin.")
        self._probe_device()

    # -- layout ---------------------------------------------------------------

    def _build(self) -> None:
        canvas = QFrame()
        canvas.setObjectName("Canvas")
        outer = QVBoxLayout(canvas)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        content = QWidget()
        body = QVBoxLayout(content)
        body.setContentsMargins(20, 14, 20, 14)
        body.setSpacing(14)
        body.addLayout(self._build_header())
        body.addLayout(self._build_views(), 1)
        body.addWidget(self._build_controls())
        outer.addWidget(content, 1)
        outer.addWidget(self._build_status_band())
        self.setCentralWidget(canvas)

        QShortcut(QKeySequence.StandardKey.Open, self, activated=self.open_image)
        QShortcut(QKeySequence.StandardKey.Save, self, activated=self.save_result)
        QShortcut(QKeySequence("Ctrl+Return"), self, activated=self.generate)

    def _build_header(self) -> QHBoxLayout:
        row = QHBoxLayout()
        row.setSpacing(12)
        self.logo = QLabel()
        self.logo.setFixedSize(32, 32)
        row.addWidget(self.logo)

        titles = QVBoxLayout()
        titles.setSpacing(0)
        titles.addWidget(_label(config.APP_NAME, "AppTitle"))
        titles.addWidget(_label("The Complex tries to remember your picture. It almost gets it right.",
                                "AppSubtitle"))
        row.addLayout(titles)
        row.addStretch(1)

        self.device_badge = _label("Detecting device…", "Badge")
        row.addWidget(self.device_badge, 0, Qt.AlignmentFlag.AlignVCenter)
        self.model_button = QPushButton("Load model")
        self.model_button.setObjectName("HeaderButton")
        self.model_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self.model_button.clicked.connect(self.toggle_model)
        row.addWidget(self.model_button, 0, Qt.AlignmentFlag.AlignVCenter)
        self.theme_button = QPushButton()
        self.theme_button.setObjectName("Nav")
        self.theme_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self.theme_button.clicked.connect(self.toggle_theme)
        row.addWidget(self.theme_button, 0, Qt.AlignmentFlag.AlignVCenter)
        return row

    def _image_card(self, title: str, view: ImageView) -> tuple[QFrame, QLabel]:
        card = _card()
        layout = QVBoxLayout(card)
        layout.setContentsMargins(14, 10, 14, 14)
        layout.setSpacing(8)
        head = QHBoxLayout()
        head.addWidget(_label(title, "CardTitle"))
        head.addStretch(1)
        meta = _label("", "Meta")
        head.addWidget(meta)
        layout.addLayout(head)
        layout.addWidget(view, 1)
        return card, meta

    def _build_views(self) -> QHBoxLayout:
        row = QHBoxLayout()
        row.setSpacing(14)
        self.original_view = ImageView("Open an image or drop it here")
        self.result_view = ImageView(RESULT_PLACEHOLDER)
        original_card, self.original_meta = self._image_card("Original", self.original_view)
        result_card, self.result_meta = self._image_card("Reconstruction", self.result_view)
        row.addWidget(original_card, 1)
        row.addWidget(result_card, 1)
        return row

    def _build_controls(self) -> QFrame:
        card = _card()
        row = QHBoxLayout(card)
        row.setContentsMargins(18, 14, 18, 14)
        row.setSpacing(24)

        left = QVBoxLayout()
        left.setSpacing(8)
        self.open_button = QPushButton("Open image...")
        self.open_button.setObjectName("Secondary")
        self.open_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self.open_button.clicked.connect(self.open_image)
        self.open_button.setMinimumWidth(150)
        left.addWidget(self.open_button)
        left.addStretch(1)
        row.addLayout(left)

        middle = QVBoxLayout()
        middle.setSpacing(6)
        middle.addWidget(_label("Weirdness", "FieldLabel"))
        segments = QFrame()
        segments.setObjectName("Segments")
        grid = QGridLayout(segments)
        grid.setContentsMargins(3, 3, 3, 3)
        grid.setSpacing(3)
        self.preset_group = QButtonGroup(self)
        self.preset_group.setExclusive(True)
        self.preset_buttons: list[QPushButton] = []
        # Two rows that fill the width evenly: the grid has a cell for every
        # combination, buttons in each row span an equal share of it.
        first_row = (len(PRESETS) + 1) // 2
        rows = (PRESETS[:first_row], PRESETS[first_row:])
        cells = first_row * max(1, len(rows[1]))
        for index, preset in enumerate(PRESETS):
            button = QPushButton(preset.name)
            button.setObjectName("Segment")
            button.setCheckable(True)
            button.setCursor(Qt.CursorShape.PointingHandCursor)
            button.setToolTip(preset.hint)
            self.preset_group.addButton(button, index)
            self.preset_buttons.append(button)
            r = 0 if index < first_row else 1
            pos = index if r == 0 else index - first_row
            span = cells // len(rows[r])
            grid.addWidget(button, r, pos * span, 1, span)
        for column in range(cells):
            grid.setColumnStretch(column, 1)
        self.preset_buttons[DEFAULT_PRESET].setChecked(True)
        self.preset_group.idToggled.connect(lambda _id, on: on and self._update_preset())
        middle.addWidget(segments)
        self.preset_hint = _label("", "StageHint")
        middle.addWidget(self.preset_hint)
        row.addLayout(middle, 1)

        right = QVBoxLayout()
        right.setSpacing(8)
        self.generate_button = ProgressButton("Generate")
        self.generate_button.clicked.connect(self.generate)
        self.generate_button.setMinimumWidth(170)
        right.addWidget(self.generate_button)
        self.save_button = QPushButton("Save result")
        self.save_button.setObjectName("Secondary")
        self.save_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self.save_button.clicked.connect(self.save_result)
        right.addWidget(self.save_button)
        right.addStretch(1)
        row.addLayout(right)
        return card

    def _build_status_band(self) -> QFrame:
        band = QFrame()
        band.setObjectName("StatusBand")
        row = QHBoxLayout(band)
        row.setContentsMargins(20, 6, 20, 6)
        row.setSpacing(16)
        self.status_label = _label("", "StatusText")
        self.status_label.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        self.status_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        row.addWidget(self.status_label, 1)
        self.status_meta = _label(f"Log: {config.LOG_FILE.name}", "StatusMeta")
        row.addWidget(self.status_meta)
        return band

    # -- theme & settings -------------------------------------------------------

    def _apply_theme(self, name: str) -> None:
        from PyQt6.QtWidgets import QApplication

        self._theme_name = theme.apply(QApplication.instance(), name).name
        self.theme_button.setText("Dark" if self._theme_name == "light" else "Light")
        self.theme_button.setToolTip(f"Switch to the {'dark' if self._theme_name == 'light' else 'light'} theme")
        self.logo.setPixmap(logo_pixmap(LOGO_PATH, 32, self.devicePixelRatioF()))
        for widget in (self.original_view, self.result_view, self.generate_button):
            widget.update()

    def toggle_theme(self) -> None:
        self._apply_theme("dark" if self._theme_name == "light" else "light")
        self.settings.setValue("theme", self._theme_name)

    def _restore_settings(self) -> None:
        try:
            saved = preset_named(str(self.settings.value("preset", "")))
            if saved is not None:
                self.preset_buttons[PRESETS.index(saved)].setChecked(True)
        except (TypeError, ValueError):
            pass

    def _save_settings(self) -> None:
        self.settings.setValue("theme", self._theme_name)
        self.settings.setValue("preset", self.preset.name)
        self.settings.sync()

    # -- state ------------------------------------------------------------------

    @property
    def busy(self) -> bool:
        return self._thread is not None

    def set_status(self, text: str) -> None:
        self.status_label.setText(text)
        self.status_label.setToolTip(text)

    @property
    def preset(self) -> Preset:
        index = self.preset_group.checkedId()
        return PRESETS[index if index >= 0 else DEFAULT_PRESET]

    def select_preset(self, name: str) -> None:
        preset = preset_named(name)
        if preset is None:
            raise ValueError(f"unknown preset {name!r}")
        self.preset_buttons[PRESETS.index(preset)].setChecked(True)

    def _update_preset(self) -> None:
        self.preset_hint.setText(self.preset.hint)

    def _update_actions(self) -> None:
        busy = self.busy
        self.open_button.setEnabled(not busy)
        for button in self.preset_buttons:
            button.setEnabled(not busy)
        self.generate_button.setEnabled(not busy and self.original is not None)
        self.save_button.setEnabled(not busy and self.result is not None)
        self.generate_button.set_progress(self.generate_button.progress if busy and not self._loading_model else None)
        self._update_model_button()

    def _update_model_button(self) -> None:
        if self._loading_model:
            self.model_button.setText("Loading model…")
            self.model_button.setEnabled(False)
            return
        loaded = self.reconstructor.model_loaded
        self.model_button.setText("Unload model" if loaded else "Load model")
        self.model_button.setToolTip("Free the GPU memory used by SD-Turbo" if loaded
                                     else "Load SD-Turbo now (otherwise it loads on the first Generate)")
        self.model_button.setEnabled(not self.busy)

    def _set_device_badge(self) -> None:
        device = self.reconstructor.device
        if device is None:
            return
        self.device_badge.setText(device.label)
        tip = f"Inference device: {device.label}"
        if device.note:
            tip += f"\n{device.note}"
        self.device_badge.setToolTip(tip)

    def _probe_device(self) -> None:
        worker = CallWorker(self.reconstructor.ensure_device)
        worker.result.connect(lambda _device: self._set_device_badge())
        worker.error.connect(lambda msg: self.device_badge.setText("Device: unknown"))
        self._probe = (start_in_thread(worker, self._on_probe_finished), worker)

    def _on_probe_finished(self) -> None:
        if self._probe is not None:
            self._probe[0].wait()  # finished() fires just before the thread really ends
        self._probe = None

    # -- actions ----------------------------------------------------------------

    def open_image(self) -> None:
        if self.busy:
            return
        start = str(self.settings.value("last_dir", str(Path.home())))
        path, _ = QFileDialog.getOpenFileName(self, "Open image", start, config.IMAGE_FILTER)
        if path:
            self.load_image_path(path)

    def load_image_path(self, path: str | os.PathLike) -> bool:
        if self.busy:
            return False
        path = Path(path)
        try:
            image = load_image(path)
        except ImageLoadError as exc:
            log.warning("Could not open %s: %s", path, exc)
            self.set_status(f"Could not open {exc}. Details: {config.LOG_FILE}")
            return False
        self.original, self.original_path = image, path
        self.result = None
        self.original_view.set_image(image)
        self.original_meta.setText(f"{image.width} × {image.height}")
        self.result_view.set_image(None)
        self.result_view.set_placeholder(RESULT_PLACEHOLDER)
        self.result_meta.setText("")
        self.settings.setValue("last_dir", str(path.parent))
        log.info("Loaded %s (%dx%d)", path, image.width, image.height)
        self.set_status(f"Loaded Profile: {path.name}. Set the weirdness and press Generate.")
        self._update_actions()
        return True

    def toggle_model(self) -> None:
        """Load SD-Turbo in the background, or unload it to free the GPU memory."""
        if self.busy:
            return
        if self.reconstructor.model_loaded:
            self.reconstructor.unload()
            self.set_status("Model unloaded. It loads again on Load model or Generate.")
            self._update_actions()
            return
        worker = CallWorker(self.reconstructor.preload, pass_status=True)
        worker.status.connect(self.set_status)
        worker.result.connect(lambda _: self.set_status(f"Model ready on {self.reconstructor.engine.device.label}."))
        worker.error.connect(lambda msg: self._on_failed(f"Could not load the model: {msg}. Details: {config.LOG_FILE}"))
        self._worker = worker
        self._loading_model = True
        self._thread = start_in_thread(worker, self._on_thread_finished)
        self._update_actions()

    def generate(self) -> None:
        if self.busy or self.original is None:
            return
        self._run_seed = random.randrange(0, 2_147_483_647)
        weirdness = self.preset.weirdness

        worker = ReconstructionWorker(self.reconstructor, self.original, weirdness, self._run_seed)
        worker.progress.connect(self._on_progress)
        worker.status.connect(self.set_status)
        worker.finished.connect(self._on_finished)
        worker.failed.connect(self._on_failed)
        worker.cancelled.connect(lambda: self.set_status("Cancelled."))
        self._worker = worker
        self._started_at = time.perf_counter()
        self.generate_button.set_progress(0)
        self.result_view.set_image(None)
        self.result_view.set_placeholder("Remembering…")
        self.result_meta.setText("")
        self._thread = start_in_thread(worker, self._on_thread_finished)
        self._update_actions()

    def _on_progress(self, value: int) -> None:
        if self.busy:
            self.generate_button.set_progress(value)

    def _on_finished(self, image: Image.Image) -> None:
        self.result = image
        self.result_view.set_image(image)
        self.result_meta.setText(f"{image.width} × {image.height}")
        elapsed = time.perf_counter() - self._started_at
        device = self.reconstructor.device
        where = f" · {device.backend}" if device else ""
        self.set_status(f"Reconstruction ready · {elapsed:.1f} s{where}")

    def _on_failed(self, message: str) -> None:
        self.result_view.set_placeholder(RESULT_PLACEHOLDER)
        self.set_status(message)
        if not self._closing:
            box = QMessageBox(QMessageBox.Icon.Warning, config.APP_NAME, message, parent=self)
            box.setWindowModality(Qt.WindowModality.NonModal)
            box.open()

    def _on_thread_finished(self) -> None:
        if self._thread is not None:
            self._thread.wait()  # finished() fires just before the thread really ends
        self._thread = None
        self._worker = None
        self._loading_model = False
        self.generate_button.set_progress(None)
        if self.result is None:
            self.result_view.set_placeholder(RESULT_PLACEHOLDER)
        self._set_device_badge()
        self._update_actions()

    def save_result(self) -> None:
        if self.result is None or self.busy:
            return
        stem = self.original_path.stem if self.original_path else "still_life"
        start = Path(str(self.settings.value("last_save_dir", self.settings.value("last_dir", str(Path.home())))))
        suggested = start / f"{stem}_{self.preset.name.lower().replace(' ', '_')}.png"
        path, _ = QFileDialog.getSaveFileName(self, "Save result", str(suggested),
                                              "PNG image (*.png);;JPEG image (*.jpg *.jpeg)")
        if path:
            self.save_result_to(path)

    def save_result_to(self, path: str | os.PathLike) -> bool:
        if self.result is None:
            return False
        path = Path(path)
        if path.suffix.lower() not in (".png", ".jpg", ".jpeg"):
            path = path.with_suffix(".png")
        try:
            if path.suffix.lower() == ".png":
                self.result.save(path, "PNG")
            else:
                self.result.convert("RGB").save(path, "JPEG", quality=95)
        except OSError as exc:
            log.exception("Saving %s failed", path)
            self.set_status(f"Could not save {path.name}: {exc}. Details: {config.LOG_FILE}")
            return False
        self.settings.setValue("last_save_dir", str(path.parent))
        log.info("Saved %s", path)
        self.set_status(f"Saved {path.name}.")
        return True

    # -- drag & drop ------------------------------------------------------------

    @staticmethod
    def _dropped_path(event) -> Optional[str]:
        mime = event.mimeData()
        if not mime.hasUrls():
            return None
        for url in mime.urls():
            local = url.toLocalFile()
            if local and local.lower().endswith(config.IMAGE_EXTENSIONS):
                return local
        return None

    def dragEnterEvent(self, event) -> None:
        if not self.busy and self._dropped_path(event):
            event.acceptProposedAction()

    def dropEvent(self, event) -> None:
        path = self._dropped_path(event)
        if path:
            self.load_image_path(path)
            event.acceptProposedAction()

    # -- shutdown ---------------------------------------------------------------

    def closeEvent(self, event) -> None:
        self._closing = True
        self._save_settings()
        if self._thread is not None:
            log.info("Window closed during generation, stopping the worker")
            cancel = getattr(self._worker, "cancel", None)
            if cancel is not None:
                cancel()
            self.hide()
            # Cancellation is checked between steps, but loading weights cannot be
            # interrupted. Wait a while, then leave without letting Qt destroy a
            # running thread (which would abort the process).
            if not self._thread.wait(CLOSE_WAIT_MS):
                log.warning("Worker still busy after %d s, exiting anyway", CLOSE_WAIT_MS // 1000)
                logging.shutdown()
                os._exit(0)
        if self._probe is not None:
            self._probe[0].wait(CLOSE_WAIT_MS)
        self.reconstructor.unload()  # ends the engine process
        event.accept()
