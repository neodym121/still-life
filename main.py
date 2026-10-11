"""Still Life AI: the Complex tries to remember your picture.

    python main.py [--device auto|cuda|dml|cpu] [--theme light|dark]
"""

from __future__ import annotations

import argparse
import logging
import logging.handlers
import multiprocessing
import os
import sys
import threading
import warnings

import config
from core.device import DEVICE_CHOICES

log = logging.getLogger("stilllife")


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=config.APP_NAME)
    parser.add_argument("--device", choices=DEVICE_CHOICES, default="auto",
                        help="inference device (default: auto = CUDA, then DirectML, then CPU)")
    parser.add_argument("--theme", choices=("light", "dark"), default=None,
                        help="override the saved theme")
    return parser.parse_args(argv)


def detach_missing_streams() -> None:
    """Under pythonw.exe there is no console: stdout/stderr are None and the
    Hugging Face download progress bars would crash writing to them."""
    for name in ("stdout", "stderr"):
        if getattr(sys, name) is None:
            setattr(sys, name, open(os.devnull, "w", encoding="utf-8"))


def setup_logging() -> None:
    config.LOG_DIR.mkdir(parents=True, exist_ok=True)
    fmt = logging.Formatter("%(asctime)s %(levelname)-7s [%(threadName)s] %(name)s: %(message)s")
    file_handler = logging.handlers.RotatingFileHandler(config.LOG_FILE, maxBytes=2_000_000,
                                                        backupCount=3, encoding="utf-8")
    file_handler.setFormatter(fmt)
    console = logging.StreamHandler()
    console.setFormatter(fmt)
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    root.addHandler(file_handler)
    root.addHandler(console)
    for noisy in ("PIL", "urllib3", "httpx", "httpcore", "filelock", "huggingface_hub.file_download"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    logging.captureWarnings(True)
    warnings.filterwarnings("ignore", message=".*CUDA is not available.*")


def install_excepthooks() -> None:
    # PyQt6 aborts the process on an unhandled exception in a slot unless
    # sys.excepthook is replaced.
    def hook(exc_type, exc, tb):
        log.critical("Unhandled exception", exc_info=(exc_type, exc, tb))

    def thread_hook(args):
        log.critical("Unhandled exception in %s", args.thread, exc_info=(args.exc_type, args.exc_value,
                                                                          args.exc_traceback))

    sys.excepthook = hook
    threading.excepthook = thread_hook


def set_windows_app_id() -> None:
    """Own taskbar icon instead of python.exe's."""
    if sys.platform == "win32":
        try:
            import ctypes

            ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(config.APP_ID)
        except (AttributeError, OSError):
            pass


def main(argv=None) -> int:
    multiprocessing.freeze_support()  # SD-Turbo runs in a child process
    detach_missing_streams()
    args = parse_args(argv)
    setup_logging()
    install_excepthooks()
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
    log.info("Starting %s (device=%s, python %s)", config.APP_NAME, args.device, sys.version.split()[0])
    set_windows_app_id()

    from PyQt6.QtGui import QIcon
    from PyQt6.QtWidgets import QApplication

    from core.pipeline import Reconstructor
    from ui.main_window import MainWindow

    app = QApplication(sys.argv[:1])
    app.setApplicationName(config.APP_NAME)
    app.setOrganizationName(config.APP_ID)
    app.setWindowIcon(QIcon(str(config.ASSETS_DIR / "logo.png")))

    window = MainWindow(Reconstructor(device_pref=args.device), theme_name=args.theme)
    window.show()
    code = app.exec()
    log.info("Exit")
    return code


if __name__ == "__main__":
    sys.exit(main())
