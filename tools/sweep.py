"""Render one image at several weirdness levels side by side.

    python tools/sweep.py photo.png                 # the six presets from the window
    python tools/sweep.py photo.png --levels 1 25 50 75 100 --seed 1984 --device dml
    python tools/sweep.py photo.png --no-model      # procedural part only, no SD-Turbo

Prints time and peak VRAM per level and writes <image>_sweep.png next to the
input (or --out). Use it to tune the curves in core/misread.py:plan_for().
Targets: 1% ~ the original; 25% still recognizable but lines wavy and details
shifted; 100% heavily distorted but clearly made from the original.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

APP_DIR = Path(__file__).resolve().parents[1]
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from PIL import Image, ImageDraw, ImageFont  # noqa: E402

import config  # noqa: E402
from core.device import DEVICE_CHOICES  # noqa: E402
from core.imaging import ImageLoadError, load_image, prepare_array  # noqa: E402
from core.misread import PRESETS, plan_for  # noqa: E402
from core.pipeline import PassthroughEngine, Reconstructor  # noqa: E402

CAPTION_H = 44
GAP = 8


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Weirdness sweep for Still Life AI")
    parser.add_argument("image", type=Path)
    parser.add_argument("--levels", type=int, nargs="+",
                        default=[round(p.weirdness * 100) for p in PRESETS],
                        help="weirdness levels in percent (default: the window's presets)")
    parser.add_argument("--seed", type=int, default=config.DEFAULT_SEED)
    parser.add_argument("--device", choices=DEVICE_CHOICES, default="auto")
    parser.add_argument("--no-model", action="store_true",
                        help="skip SD-Turbo: misread + anchor only (fast, for tuning the procedural part)")
    parser.add_argument("--out", type=Path, default=None, help="output PNG (default: <image>_sweep.png)")
    return parser.parse_args(argv)


def _font(size: int):
    try:
        return ImageFont.load_default(size=size)
    except TypeError:  # Pillow < 10.1
        return ImageFont.load_default()


def level_name(level: int) -> str:
    for preset in PRESETS:
        if round(preset.weirdness * 100) == level:
            return preset.name
    return ""


def compose(tiles: list[tuple[str, str, Image.Image]]) -> Image.Image:
    height = max(t[2].height for t in tiles)
    width = sum(t[2].width for t in tiles) + GAP * (len(tiles) + 1)
    sheet = Image.new("RGB", (width, height + CAPTION_H + GAP * 2), (248, 248, 246))
    draw = ImageDraw.Draw(sheet)
    title_font, meta_font = _font(16), _font(12)
    x = GAP
    for title, meta, image in tiles:
        sheet.paste(image, (x, GAP + (height - image.height) // 2))
        draw.text((x + 2, GAP + height + 6), title, fill=(18, 18, 18), font=title_font)
        draw.text((x + 2, GAP + height + 26), meta, fill=(123, 121, 116), font=meta_font)
        x += image.width + GAP
    return sheet


def main(argv=None) -> int:
    args = parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    try:
        original = load_image(args.image)
    except ImageLoadError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    factory = PassthroughEngine if args.no_model else None
    reconstructor = Reconstructor(device_pref=args.device, engine_factory=factory)
    device = reconstructor.ensure_device()
    print(f"Device: {device.backend} ({device.name}){', no model' if args.no_model else ''}")
    started = time.perf_counter()
    reconstructor.preload(on_status=lambda s: print(f"  {s}"))
    print(f"Model ready in {time.perf_counter() - started:.1f} s\n")

    tiles = [("Original", f"{original.width}x{original.height}", Image.fromarray(prepare_array(original)))]
    print(f"{'level':>6} {'preset':<21} {'passes':>6} {'steps':>5} {'strength':>8} {'time':>8} {'peak VRAM':>10}")
    for level in args.levels:
        level = max(0, min(100, level))
        plan = plan_for(level / 100)
        engine = reconstructor.engine
        if hasattr(engine, "reset_peak_memory"):
            engine.reset_peak_memory()
        t0 = time.perf_counter()
        result = reconstructor.run(original, level / 100, args.seed)
        elapsed = time.perf_counter() - t0
        engine = reconstructor.engine
        active = engine.device if engine else device
        peak = engine.peak_memory_gb() if hasattr(engine, "peak_memory_gb") else None
        vram = f"{peak:.2f} GB" if peak is not None else f"n/a ({active.backend})"
        print(f"{level:>5}% {level_name(level):<21} {plan.passes:>6} {plan.denoise_steps:>5} "
              f"{plan.strength:>8.2f} {elapsed:>7.2f}s {vram:>10}")
        tiles.append((f"{level}%  {level_name(level)}".strip(), f"{elapsed:.1f} s · VRAM {vram}", result))

    out = args.out or args.image.with_name(f"{args.image.stem}_sweep.png")
    compose(tiles).save(out)
    print(f"\nSaved {out}")
    reconstructor.unload()
    return 0


if __name__ == "__main__":
    sys.exit(main())
