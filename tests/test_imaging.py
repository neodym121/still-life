import numpy as np
import pytest
from PIL import Image

import config
from core.imaging import ImageLoadError, load_image, prepare_array, working_size


def test_exif_rotation_is_applied(tmp_path):
    im = Image.new("RGB", (40, 20), (200, 10, 10))
    im.paste((10, 10, 200), (0, 0, 10, 20))  # blue stripe on the left
    exif = Image.Exif()
    exif[0x0112] = 6  # rotate 90 degrees clockwise when displayed
    path = tmp_path / "rotated.jpg"
    im.save(path, exif=exif, quality=95)
    loaded = load_image(path)
    assert loaded.size == (20, 40)
    r, g, b = loaded.getpixel((10, 2))
    assert b > r  # the left stripe is now on top


def test_rgba_transparency_is_flattened_on_background(tmp_path):
    im = Image.new("RGBA", (16, 16), (255, 0, 0, 0))
    im.paste((0, 0, 255, 255), (0, 0, 8, 16))
    path = tmp_path / "alpha.png"
    im.save(path)
    loaded = load_image(path)
    assert loaded.mode == "RGB"
    assert loaded.getpixel((12, 8)) == tuple(config.ALPHA_BACKGROUND)
    assert loaded.getpixel((2, 8)) == (0, 0, 255)


def test_palette_with_transparency(tmp_path):
    im = Image.new("P", (8, 8), 0)
    im.putpalette([255, 0, 0, 0, 255, 0] + [0] * 762)
    im.paste(1, (0, 0, 4, 8))
    path = tmp_path / "palette.png"
    im.save(path, transparency=0)
    loaded = load_image(path)
    assert loaded.getpixel((6, 4)) == tuple(config.ALPHA_BACKGROUND)
    assert loaded.getpixel((1, 4)) == (0, 255, 0)


def test_grey_alpha_and_16_bit(tmp_path):
    la = Image.new("LA", (8, 8), (40, 0))
    la.save(tmp_path / "la.png")
    assert load_image(tmp_path / "la.png").getpixel((0, 0)) == tuple(config.ALPHA_BACKGROUND)

    deep = Image.fromarray(np.full((8, 8), 30000, dtype=np.uint16))
    deep.save(tmp_path / "deep.png")
    value = load_image(tmp_path / "deep.png").getpixel((0, 0))
    assert 100 < value[0] < 140  # scaled down, not clipped to white


@pytest.mark.parametrize("ext", ["png", "jpg", "webp", "bmp"])
def test_supported_formats(tmp_path, ext):
    path = tmp_path / f"pic.{ext}"
    Image.new("RGB", (30, 20), (10, 120, 200)).save(path)
    assert load_image(path).size == (30, 20)


def test_broken_files(tmp_path):
    garbage = tmp_path / "garbage.png"
    garbage.write_bytes(b"definitely not an image" * 10)
    with pytest.raises(ImageLoadError):
        load_image(garbage)

    empty = tmp_path / "empty.jpg"
    empty.write_bytes(b"")
    with pytest.raises(ImageLoadError):
        load_image(empty)

    good = tmp_path / "good.png"
    Image.new("RGB", (200, 200), (1, 2, 3)).save(good)
    truncated = tmp_path / "truncated.png"
    truncated.write_bytes(good.read_bytes()[:60])
    with pytest.raises(ImageLoadError):
        load_image(truncated)

    with pytest.raises(ImageLoadError):
        load_image(tmp_path / "missing.png")


@pytest.mark.parametrize("size,expected", [
    ((1000, 500), (448, 224)),
    ((500, 1000), (224, 448)),
    ((448, 448), (448, 448)),
    ((100, 80), (448, 360)),       # small images are scaled up
    ((5000, 40), (448, 64)),       # very thin images keep a minimum side
])
def test_working_size(size, expected):
    assert working_size(*size) == expected


def test_working_size_is_multiple_of_8():
    for w in range(50, 2000, 37):
        for h in (77, 333, 1024):
            ww, hh = working_size(w, h)
            assert ww % 8 == 0 and hh % 8 == 0
            assert max(ww, hh) in range(440, 457)


def test_prepare_array(picture):
    arr = prepare_array(picture, 384)
    assert arr.dtype == np.uint8 and arr.shape == (288, 384, 3)
    assert arr.flags["C_CONTIGUOUS"]
