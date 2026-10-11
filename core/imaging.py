"""Loading and resizing input images."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

import config


class ImageLoadError(Exception):
    """The file is missing, unreadable or not an image."""


def load_image(path: str | Path) -> Image.Image:
    """Open an image as RGB: EXIF rotation applied, transparency flattened."""
    path = Path(path)
    try:
        with Image.open(path) as im:
            im.load()
            im = ImageOps.exif_transpose(im)
            return to_rgb(im)
    except FileNotFoundError as exc:
        raise ImageLoadError(f"{path.name}: file not found") from exc
    except (UnidentifiedImageError, Image.DecompressionBombError) as exc:
        raise ImageLoadError(f"{path.name}: not a supported image") from exc
    except (OSError, SyntaxError, ValueError, EOFError) as exc:
        raise ImageLoadError(f"{path.name}: the file is damaged or unreadable") from exc


def to_rgb(im: Image.Image, background=config.ALPHA_BACKGROUND) -> Image.Image:
    if im.mode in ("I", "I;16", "I;16B", "I;16L", "I;16N"):
        # 16/32-bit greyscale: scale down instead of clipping to white.
        arr = np.asarray(im, dtype=np.float64)
        peak = arr.max() if arr.size else 0
        scale = 255.0 / (65535.0 if peak > 255 else max(peak, 1.0))
        im = Image.fromarray(np.clip(arr * scale, 0, 255).astype(np.uint8), "L")
    elif im.mode == "F":
        arr = np.asarray(im, dtype=np.float64)
        im = Image.fromarray(np.clip(arr * (255.0 if arr.max(initial=0) <= 1.0 else 1.0), 0, 255).astype(np.uint8), "L")

    if im.mode == "P":
        im = im.convert("RGBA" if "transparency" in im.info else "RGB")
    elif im.mode in ("L", "1") and "transparency" in im.info:
        im = im.convert("RGBA")
    if im.mode in ("LA", "PA", "La", "RGBa"):
        im = im.convert("RGBA")

    if im.mode == "RGBA":
        canvas = Image.new("RGB", im.size, tuple(background))
        canvas.paste(im, mask=im.getchannel("A"))
        return canvas
    return im.convert("RGB")


def working_size(width: int, height: int, long_side: int = config.WORK_LONG_SIDE,
                 multiple: int = config.SIZE_MULTIPLE, min_side: int = config.MIN_SIDE) -> tuple[int, int]:
    """Size with the long side ~= `long_side`, both sides multiples of `multiple`."""
    if width <= 0 or height <= 0:
        raise ValueError("image has no pixels")
    scale = long_side / max(width, height)

    def snap(v: float) -> int:
        return max(min_side, int(round(v / multiple)) * multiple)

    return snap(width * scale), snap(height * scale)


def prepare_array(im: Image.Image, long_side: int = config.WORK_LONG_SIDE) -> np.ndarray:
    """RGB uint8 HxWx3 array at the working resolution."""
    size = working_size(im.width, im.height, long_side)
    if im.size != size:
        im = im.resize(size, Image.Resampling.LANCZOS)
    return np.ascontiguousarray(np.asarray(im.convert("RGB"), dtype=np.uint8))
