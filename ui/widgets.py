"""Custom-painted widgets: image wells, the progress button, the logo tile."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from PIL import Image
from PyQt6.QtCore import QRectF, QSize, Qt
from PyQt6.QtGui import QColor, QFont, QImage, QPainter, QPainterPath, QPixmap
from PyQt6.QtWidgets import QPushButton, QSizePolicy, QWidget

from ui import theme


def pil_to_qimage(image: Image.Image) -> QImage:
    rgb = image.convert("RGB")
    data = rgb.tobytes("raw", "RGB")
    qimage = QImage(data, rgb.width, rgb.height, rgb.width * 3, QImage.Format.Format_RGB888)
    return qimage.copy()  # detach from the Python buffer


class ImageView(QWidget):
    """Nested 'paper' well that shows an image scaled with its aspect ratio."""

    RADIUS = 10.0
    INSET = 8.0

    def __init__(self, placeholder: str, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._placeholder = placeholder
        self._pixmap: Optional[QPixmap] = None
        self._scaled: Optional[QPixmap] = None
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setMinimumSize(220, 180)

    def set_placeholder(self, text: str) -> None:
        self._placeholder = text
        self.update()

    def set_image(self, image: Optional[Image.Image]) -> None:
        self._pixmap = QPixmap.fromImage(pil_to_qimage(image)) if image is not None else None
        self._scaled = None
        self.update()

    def has_image(self) -> bool:
        return self._pixmap is not None

    def resizeEvent(self, event) -> None:
        self._scaled = None
        super().resizeEvent(event)

    def _target(self) -> QRectF:
        area = QRectF(self.rect()).adjusted(self.INSET, self.INSET, -self.INSET, -self.INSET)
        size = self._pixmap.size().scaled(area.size().toSize(), Qt.AspectRatioMode.KeepAspectRatio)
        x = area.x() + (area.width() - size.width()) / 2
        y = area.y() + (area.height() - size.height()) / 2
        return QRectF(x, y, size.width(), size.height())

    def paintEvent(self, event) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        well = QPainterPath()
        well.addRoundedRect(QRectF(self.rect()), self.RADIUS, self.RADIUS)
        p.fillPath(well, theme.color("nested"))

        if self._pixmap is None:
            p.setPen(theme.color("ashen"))
            p.setFont(theme.sans(14))
            p.drawText(QRectF(self.rect()).adjusted(24, 24, -24, -24),
                       Qt.AlignmentFlag.AlignCenter | Qt.TextFlag.TextWordWrap, self._placeholder)
            return

        target = self._target()
        dpr = self.devicePixelRatioF()
        want = QSize(max(1, round(target.width() * dpr)), max(1, round(target.height() * dpr)))
        if self._scaled is None or self._scaled.size() != want:
            self._scaled = self._pixmap.scaled(want, Qt.AspectRatioMode.IgnoreAspectRatio,
                                               Qt.TransformationMode.SmoothTransformation)
            self._scaled.setDevicePixelRatio(dpr)
        p.drawPixmap(target.topLeft(), self._scaled)  # square corners: the picture is shown as is


class ProgressButton(QPushButton):
    """Filled dark action button that shows progress on itself while busy."""

    def __init__(self, text: str, busy_text: str = "Generating", parent: Optional[QWidget] = None):
        super().__init__(text, parent)
        self._busy_text = busy_text
        self._progress: Optional[int] = None
        self.setAttribute(Qt.WidgetAttribute.WA_Hover, True)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setMinimumHeight(36)
        self.setFont(theme.sans(14, QFont.Weight.Medium))

    @property
    def progress(self) -> Optional[int]:
        return self._progress

    def set_progress(self, value: Optional[int]) -> None:
        self._progress = None if value is None else max(0, min(100, int(value)))
        self.update()

    def display_text(self) -> str:
        if self._progress is None:
            return self.text()
        return f"{self._busy_text} {self._progress}%"

    def sizeHint(self) -> QSize:
        metrics = self.fontMetrics()
        width = max(metrics.horizontalAdvance(self.text()),
                    metrics.horizontalAdvance(f"{self._busy_text} 100%")) + 48
        return QSize(width, 36)

    def paintEvent(self, event) -> None:
        pal = theme.palette()
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        rect = QRectF(self.rect()).adjusted(0.5, 0.5, -0.5, -0.5)
        path = QPainterPath()
        path.addRoundedRect(rect, 6.0, 6.0)
        busy = self._progress is not None

        if busy:
            bg, fg = pal.primary_bg, pal.primary_fg
        elif not self.isEnabled():
            bg, fg = pal.disabled_bg, pal.disabled_fg
        elif self.isDown():
            bg, fg = pal.primary_progress, pal.primary_fg
        elif self.underMouse():
            bg, fg = pal.primary_hover, pal.primary_fg
        else:
            bg, fg = pal.primary_bg, pal.primary_fg
        p.fillPath(path, QColor(bg))

        if busy and self._progress:
            p.save()
            p.setClipPath(path)
            p.fillRect(QRectF(rect.x(), rect.y(), rect.width() * self._progress / 100.0, rect.height()),
                       QColor(pal.primary_progress))
            p.restore()
        if self.hasFocus() and self.isEnabled() and not busy:
            p.setPen(QColor(pal.clay))
            p.drawPath(path)

        p.setPen(QColor(fg))
        p.setFont(self.font())
        p.drawText(rect, Qt.AlignmentFlag.AlignCenter, self.display_text())


def logo_pixmap(path: Path, size: int, dpr: float = 1.0, radius_ratio: float = 0.22) -> QPixmap:
    """The app mark as a rounded tile (the PNG has a square black field)."""
    source = QPixmap(str(path))
    px = max(1, round(size * dpr))
    tile = QPixmap(px, px)
    tile.fill(Qt.GlobalColor.transparent)
    if source.isNull():
        return tile
    p = QPainter(tile)
    p.setRenderHint(QPainter.RenderHint.Antialiasing)
    p.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
    clip = QPainterPath()
    clip.addRoundedRect(QRectF(0, 0, px, px), px * radius_ratio, px * radius_ratio)
    p.setClipPath(clip)
    p.drawPixmap(0, 0, source.scaled(px, px, Qt.AspectRatioMode.IgnoreAspectRatio,
                                     Qt.TransformationMode.SmoothTransformation))
    p.end()
    tile.setDevicePixelRatio(dpr)
    return tile
