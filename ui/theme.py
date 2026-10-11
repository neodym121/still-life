"""Visual system from `дизайн/DESIGN.md`: warm parchment, ink, one clay accent.

Light is the reference theme. Dark keeps the same warm neutrals inverted (for
the liminal night look) and is a toggle in the header.
"""

from __future__ import annotations

from dataclasses import dataclass

from PyQt6.QtGui import QColor, QFont, QFontDatabase
from PyQt6.QtWidgets import QApplication

import config


@dataclass(frozen=True)
class Palette:
    name: str
    canvas: str          # page background
    card: str            # elevated cards
    nested: str          # image wells, hover fills, badges
    ink: str             # primary text
    graphite: str        # secondary text, button labels
    ashen: str           # helper text
    pebble: str          # tertiary text
    mist: str            # hairline borders on controls
    chalk: str           # card borders
    clay: str            # the only chromatic accent, decorative only
    segment_on: str      # selected preset button
    primary_bg: str      # filled action button
    primary_fg: str
    primary_hover: str
    primary_progress: str
    disabled_bg: str
    disabled_fg: str
    band: str            # status band at the bottom
    band_text: str


LIGHT = Palette(
    name="light",
    canvas="#f8f8f6", card="#ffffff", nested="#efeeeb",
    ink="#121212", graphite="#373734", ashen="#7b7974", pebble="#9c9a92",
    mist="#b7b7b5", chalk="#e7e6e1", clay="#d97757", segment_on="#ffffff",
    primary_bg="#121212", primary_fg="#f8f8f6", primary_hover="#373734", primary_progress="#4a4945",
    disabled_bg="#e7e6e1", disabled_fg="#9c9a92",
    band="#000000", band_text="#9c9a92",
)

DARK = Palette(
    name="dark",
    canvas="#0e0e0d", card="#171716", nested="#21211f",
    ink="#f2f0eb", graphite="#d9d7d0", ashen="#9c9a92", pebble="#7b7974",
    mist="#4a4945", chalk="#2c2b29", clay="#d97757", segment_on="#3a3936",
    primary_bg="#f2f0eb", primary_fg="#121212", primary_hover="#ffffff", primary_progress="#c9c6bd",
    disabled_bg="#2c2b29", disabled_fg="#7b7974",
    band="#000000", band_text="#9c9a92",
)

THEMES = {"light": LIGHT, "dark": DARK}

# Anthropic Serif / Sans are not public; these are the documented substitutes.
SERIF_FAMILIES = ("Anthropic Serif", "Source Serif 4", "Source Serif Pro", "Charter", "Georgia",
                  "Cambria", "Times New Roman")
SANS_FAMILIES = ("Anthropic Sans", "Inter", "IBM Plex Sans", "Segoe UI Variable Text", "Segoe UI",
                 "Helvetica Neue", "Arial")

_current = LIGHT
_fonts: dict[str, str] = {}


def palette() -> Palette:
    return _current


def color(name: str) -> QColor:
    return QColor(getattr(_current, name))


def _pick_family(candidates) -> str:
    available = set(QFontDatabase.families())
    for family in candidates:
        if family in available:
            return family
    return QApplication.font().family()


def sans_family() -> str:
    if "sans" not in _fonts:
        _fonts["sans"] = _pick_family(SANS_FAMILIES)
    return _fonts["sans"]


def serif_family() -> str:
    if "serif" not in _fonts:
        _fonts["serif"] = _pick_family(SERIF_FAMILIES)
    return _fonts["serif"]


def sans(px: int, weight: QFont.Weight = QFont.Weight.Normal) -> QFont:
    font = QFont(sans_family())
    font.setPixelSize(px)
    font.setWeight(weight)
    return font


def stylesheet(p: Palette) -> str:
    sans_name, serif_name = sans_family(), serif_family()
    return f"""
    QMainWindow, QFrame#Canvas {{ background: {p.canvas}; }}
    QWidget {{ color: {p.ink}; font-family: "{sans_name}"; font-size: 14px; }}
    QLabel {{ background: transparent; }}
    QToolTip {{ background: {p.ink}; color: {p.canvas}; border: none; padding: 6px 8px; }}

    QFrame#Card {{ background: {p.card}; border: 1px solid {p.chalk}; border-radius: 14px; }}

    QLabel#AppTitle {{ font-family: "{serif_name}"; font-size: 20px; color: {p.ink}; }}
    QLabel#AppSubtitle {{ color: {p.ashen}; font-size: 12px; }}
    QLabel#CardTitle {{ color: {p.graphite}; font-size: 13px; font-weight: 500; }}
    QLabel#Meta {{ color: {p.pebble}; font-size: 12px; }}
    QLabel#FieldLabel {{ color: {p.graphite}; font-size: 13px; font-weight: 500; }}
    QLabel#Value {{ color: {p.ink}; font-size: 15px; font-weight: 500; }}
    QLabel#StageHint {{ color: {p.ashen}; font-size: 12px; }}
    QLabel#Badge {{ background: {p.nested}; color: {p.graphite}; border-radius: 5px;
                    padding: 4px 8px; font-size: 12px; font-weight: 500; }}

    QPushButton#Secondary {{ background: transparent; color: {p.graphite}; border: 1px solid {p.mist};
                             border-radius: 6px; padding: 7px 16px; font-size: 14px; font-weight: 500; }}
    QPushButton#Secondary:hover {{ background: {p.nested}; color: {p.ink}; }}
    QPushButton#Secondary:pressed {{ background: {p.chalk}; }}
    QPushButton#Secondary:focus {{ border-color: {p.graphite}; }}
    QPushButton#Secondary:disabled {{ color: {p.pebble}; border-color: {p.chalk}; background: transparent; }}

    QPushButton#Nav {{ background: transparent; color: {p.graphite}; border: none; border-radius: 5px;
                       padding: 4px 10px; font-size: 13px; font-weight: 500; }}
    QPushButton#Nav:hover {{ background: {p.nested}; color: {p.ink}; }}

    QPushButton#HeaderButton {{ background: transparent; color: {p.graphite}; border: 1px solid {p.mist};
                                border-radius: 5px; padding: 3px 10px; font-size: 12px; font-weight: 500; }}
    QPushButton#HeaderButton:hover {{ background: {p.nested}; color: {p.ink}; }}
    QPushButton#HeaderButton:disabled {{ color: {p.pebble}; border-color: {p.chalk}; }}

    QFrame#Segments {{ background: {p.nested}; border-radius: 7px; }}
    QPushButton#Segment {{ background: transparent; color: {p.graphite}; border: 1px solid transparent;
                           border-radius: 5px; padding: 5px 10px; font-size: 13px; font-weight: 500; }}
    QPushButton#Segment:hover {{ color: {p.ink}; }}
    QPushButton#Segment:checked {{ background: {p.segment_on}; color: {p.ink}; border-color: {p.chalk}; }}
    QPushButton#Segment:disabled {{ color: {p.pebble}; }}
    QPushButton#Segment:checked:disabled {{ color: {p.ashen}; }}

    QFrame#StatusBand {{ background: {p.band}; }}
    QLabel#StatusText {{ color: {p.band_text}; font-size: 12px; }}
    QLabel#StatusMeta {{ color: {p.ashen}; font-size: 12px; }}

    QMessageBox {{ background: {p.card}; }}
    QMessageBox QLabel {{ color: {p.ink}; }}
    """


def apply(app: QApplication, name: str = "light") -> Palette:
    global _current
    _current = THEMES.get(name, LIGHT)
    app.setFont(sans(14))
    app.setStyleSheet(stylesheet(_current))
    return _current
