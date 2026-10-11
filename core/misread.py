"""How the Complex "misreads" a picture.

Everything that depends on the weirdness w (0..1) lives in `plan_for(w)`: one
table of monotonic curves (`CURVES`) that is easy to tune in a single place.
The window offers fixed `PRESETS` on that scale instead of a slider.
The rest of the module is procedural damage done with numpy / OpenCV around
the neural pass:

    misread()      half-remember the picture before SD-Turbo sees it: whole
                   elements (eyes, teeth, spots, letters...) are found by
                   colour segmentation and moved, swapped, turned, mirrored,
                   resized, duplicated or forgotten. Edges stay crisp; there
                   is no blur and no ripple warp.
    apply_anchor() pull the original's coarse layout and colours back in

All random structure is drawn from the seed only, never from `w`, so a stronger
preset grows the *same* disorder instead of rolling a new one.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, fields

import cv2
import numpy as np

import config


# --------------------------------------------------------------------------
# Plan: every weirdness-dependent number in one place
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class Curve:
    """Monotonic ramp from `at0` (w <= start) to `at1` (w >= end).

    kind: "float"; "int" (rounded half-up); "geo_int" (geometric interpolation,
    rounded half-up, for quantities like posterize levels).
    """

    at0: float
    at1: float
    start: float = 0.0
    end: float = 1.0
    gamma: float = 1.0
    kind: str = "float"

    def __call__(self, w: float) -> float:
        t = (w - self.start) / (self.end - self.start)
        t = min(1.0, max(0.0, t)) ** self.gamma
        if self.kind == "geo_int":
            value = self.at0 * (self.at1 / self.at0) ** t
        else:
            value = self.at0 + (self.at1 - self.at0) * t
        if self.kind in ("int", "geo_int"):
            return math.floor(value + 0.5)
        return value


# Spatial values are fractions of the working image's long side.
# Tuned on the real SD-Turbo: above strength ~0.35 it starts inventing objects,
# so the disorder comes from the element shuffle, not from the denoiser.
CURVES: dict[str, Curve] = {
    # Misreading: elements land in the wrong place (crisp, no blur)
    "shuffle_count": Curve(0, 10, 0.12, 1.0, 0.8, "int"),  # elements touched
    "shuffle_shift": Curve(0.0, 0.06, 0.12, 1.0, 0.8),     # max move
    "shuffle_turn": Curve(0.0, 40.0, 0.15),                # max rotation, degrees
    "shuffle_resize": Curve(0.0, 0.35, 0.15),              # max scale change
    "shuffle_flip": Curve(0.0, 0.30, 0.30),                # chance to be mirrored
    "shuffle_swap": Curve(0.0, 0.30, 0.20),                # chance to trade places
    "shuffle_drop": Curve(0.0, 0.20, 0.35),                # chance to be forgotten
    "shuffle_copy": Curve(0.0, 0.08, 0.45),                # chance of a duplicate
    # Misreading: slow, large-scale geometry only (no ripples)
    "warp_large": Curve(0.0, 0.02, 0.35, 1.0, 1.2),
    "squash": Curve(0.0, 0.16, 0.20),                      # flattened "head" lens
    # Misreading: detail loss without blur
    "simplify": Curve(0, 1, 0.60, 1.0, 1.0, "int"),       # median filter radius
    # Misreading: colour decay (neutral, no tint)
    "desaturate": Curve(0.0, 0.10, 0.50),
    "fade": Curve(0.0, 0.05, 0.60),                        # contrast towards grey
    # Reconstruction (SD-Turbo img2img)
    "strength": Curve(0.10, 0.33, 0.0, 1.0, 0.6),
    "denoise_steps": Curve(1, 3, 0.20, 1.0, 1.0, "int"),   # executed steps per pass
    "passes": Curve(1, 2, 0.60, 1.0, 1.0, "int"),         # copy of a copy
    "repass_scale": Curve(0.30, 0.45, 0.60),               # misread between passes
    # Anchor, fine part: restores the original's detail where nothing was
    # moved (masked out around shuffled elements, so they leave no ghost).
    "anchor_weight": Curve(0.90, 0.12, 0.0, 1.0, 0.9),
    "anchor_sigma": Curve(0.003, 0.010),
    # Anchor, coarse part: layout and colours everywhere. Never reaches zero,
    # so even 100% stays derived from the original.
    "anchor_coarse": Curve(0.35, 0.22),
}

MAX_DENOISE_STEPS = 5
MAX_PASSES = 3


@dataclass(frozen=True)
class Plan:
    weirdness: float
    long_side: int
    num_inference_steps: int
    shuffle_count: int
    shuffle_shift: float
    shuffle_turn: float
    shuffle_resize: float
    shuffle_flip: float
    shuffle_swap: float
    shuffle_drop: float
    shuffle_copy: float
    warp_large: float
    squash: float
    simplify: int
    desaturate: float
    fade: float
    strength: float
    denoise_steps: int
    passes: int
    repass_scale: float
    anchor_weight: float
    anchor_sigma: float
    anchor_coarse: float


def inference_steps_for(denoise_steps: int, strength: float) -> int:
    """Smallest num_inference_steps that makes img2img run `denoise_steps`.

    diffusers runs int(num_inference_steps * strength) steps; zero means no
    denoising at all, so the product must stay >= 1.
    """
    n = max(1, math.ceil(denoise_steps / strength - 1e-9))
    while int(n * strength) < denoise_steps:
        n += 1
    return n


def plan_for(w: float) -> Plan:
    """All stage parameters for weirdness w in [0, 1]."""
    w = min(1.0, max(0.0, float(w)))
    values = {name: curve(w) for name, curve in CURVES.items()}
    for name in ("shuffle_count", "simplify", "denoise_steps", "passes"):
        values[name] = int(values[name])
    values["denoise_steps"] = min(MAX_DENOISE_STEPS, max(1, values["denoise_steps"]))
    values["passes"] = min(MAX_PASSES, max(1, values["passes"]))
    steps = inference_steps_for(values["denoise_steps"], values["strength"])
    return Plan(weirdness=w, long_side=config.WORK_LONG_SIDE, num_inference_steps=steps, **values)


PLAN_FIELDS = tuple(f.name for f in fields(Plan))


# --------------------------------------------------------------------------
# Presets shown as buttons (the weakest one is 25%: below that the change is
# hardly visible)
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class Preset:
    name: str
    weirdness: float
    hint: str


PRESETS = (
    Preset("Almost correct", 0.25, "Recognizable, but a few details have moved."),
    Preset("Misremembered", 0.40, "Small parts trade places. Something is off."),
    Preset("Copy of a copy", 0.70, "Parts are swapped, turned around or lost."),
    Preset("Complex decay", 0.85, "Most details are wrong now, the shape still holds."),
    Preset("Still life facsimile", 1.00, "A mannequin of the original. Only the outline remembers."),
)
DEFAULT_PRESET = 0


def preset_named(name: str) -> Preset | None:
    return next((p for p in PRESETS if p.name == name), None)


# --------------------------------------------------------------------------
# Procedural damage
# --------------------------------------------------------------------------

_MAX_ELEMENTS = 24
_SEGMENT_COLOURS = 12
_LARGE_ELEMENT = 0.025       # area fraction above which elements never swap
_ANCHOR_COARSE_SIGMA = 0.06  # fraction of the long side
_FADE_TO = np.float32(0.72)  # neutral grey: fading must not tint the picture
_LUMA = np.array([0.299, 0.587, 0.114], dtype=np.float32)


def _streams(seed: int) -> dict[str, np.random.Generator]:
    names = ("warp", "segment", "elements")
    children = np.random.SeedSequence(int(seed) & 0xFFFFFFFF).spawn(len(names))
    return {name: np.random.default_rng(child) for name, child in zip(names, children)}


def _to_float(img: np.ndarray) -> np.ndarray:
    return img.astype(np.float32) / 255.0


def _to_uint8(x: np.ndarray) -> np.ndarray:
    return (np.clip(x, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)


def _smooth_field(rng: np.random.Generator, h: int, w: int, cells: float) -> np.ndarray:
    """Smooth noise in [-1, 1]: a coarse random grid upsampled bicubically."""
    long_side = max(h, w)
    gh = max(2, round(cells * h / long_side)) + 1
    gw = max(2, round(cells * w / long_side)) + 1
    grid = rng.standard_normal((gh, gw)).astype(np.float32)
    field = cv2.resize(grid, (w, h), interpolation=cv2.INTER_CUBIC)
    peak = float(np.abs(field).max())
    return field / peak if peak > 1e-6 else field


def _warp(img: np.ndarray, plan: Plan, rng: np.random.Generator, scale: float) -> np.ndarray:
    """Slow large-scale bend plus a squashing lens: proportions, not ripples."""
    h, w = img.shape[:2]
    long_side = max(h, w)
    large = (_smooth_field(rng, h, w, 2.0), _smooth_field(rng, h, w, 2.0))
    lens_center = (rng.normal(0.5, 0.10), rng.normal(0.42, 0.10))
    lens_side = rng.uniform(-0.45, 0.25)

    amplitude = plan.warp_large * scale * long_side
    k = plan.squash * scale
    if amplitude < 0.05 and k < 1e-3:
        return img
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    dx = amplitude * large[0]
    dy = amplitude * large[1]
    if k > 0:
        cx, cy = lens_center[0] * w, lens_center[1] * h
        radius = 0.38 * long_side
        g = np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * radius * radius))
        dy += (yy - cy) * k * g  # sample further out: content squashed vertically
        dx += (xx - cx) * k * lens_side * g
    return cv2.remap(img, xx + dx, yy + dy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)


# -- element shuffle ---------------------------------------------------------

@dataclass
class _Element:
    x0: int
    y0: int
    mask: np.ndarray   # uint8 0/1, crop starting at (x0, y0)
    cx: float          # centroid, image coordinates
    cy: float
    area: int
    score: float

    @property
    def diameter(self) -> float:
        return 2.0 * math.sqrt(self.area / math.pi)

    def paint(self, target: np.ndarray, value: int = 1) -> None:
        h, w = self.mask.shape
        region = target[self.y0:self.y0 + h, self.x0:self.x0 + w]
        region[self.mask > 0] = value


def _kmeans(samples: np.ndarray, k: int, rng: np.random.Generator, iterations: int = 10) -> np.ndarray:
    """Tiny seeded k-means (k-means++ init), deterministic for a given rng."""
    first = int(rng.integers(len(samples)))
    centers = [samples[first]]
    d2 = ((samples - samples[first]) ** 2).sum(1)
    for _ in range(1, k):
        total = float(d2.sum())
        pick = int(rng.choice(len(samples), p=d2 / total)) if total > 0 else int(rng.integers(len(samples)))
        centers.append(samples[pick])
        d2 = np.minimum(d2, ((samples - samples[pick]) ** 2).sum(1))
    c = np.array(centers, dtype=np.float32)
    for _ in range(iterations):
        labels = ((samples[:, None, :] - c[None]) ** 2).sum(-1).argmin(1)
        for j in range(k):
            members = samples[labels == j]
            if len(members):
                c[j] = members.mean(0)
    return c


def _segment(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Label map of flat colour regions (edge-preserving smoothing + k-means in Lab)."""
    h, w = img.shape[:2]
    smooth = cv2.bilateralFilter(img, 7, 40, 5)
    lab = cv2.cvtColor(smooth, cv2.COLOR_RGB2LAB).reshape(-1, 3).astype(np.float32)
    picks = rng.choice(len(lab), size=min(len(lab), 6000), replace=False)
    centers = _kmeans(lab[picks], _SEGMENT_COLOURS, rng)
    labels = np.empty(len(lab), dtype=np.uint8)
    for start in range(0, len(lab), 65536):
        chunk = lab[start:start + 65536]
        labels[start:start + 65536] = ((chunk[:, None, :] - centers[None]) ** 2).sum(-1).argmin(1)
    return labels.reshape(h, w)


def _fill_holes(mask: np.ndarray) -> np.ndarray:
    padded = np.pad(mask, 1).astype(np.uint8)
    outside = padded.copy()
    cv2.floodFill(outside, None, (0, 0), 2)
    return ((padded == 1) | (outside == 0))[1:-1, 1:-1].astype(np.uint8)


def _find_elements(img: np.ndarray, rng: np.random.Generator) -> list[_Element]:
    """Medium-sized, contrasting colour regions that do not touch the border."""
    h, w = img.shape[:2]
    total, long_side = h * w, max(h, w)
    labels = _segment(img, rng)
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB).astype(np.float32)
    grow = max(1, round(long_side / 224))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * grow + 1, 2 * grow + 1))
    margin = 3 * grow + 2
    found: list[_Element] = []
    for colour in range(int(labels.max()) + 1):
        n, cc, stats, _ = cv2.connectedComponentsWithStats((labels == colour).astype(np.uint8),
                                                           connectivity=8)
        if n <= 1:
            continue
        x, y, bw, bh, area = (stats[1:, i] for i in range(5))
        keep = ((area >= 0.0012 * total) & (area <= 0.07 * total)
                & (x > 0) & (y > 0) & (x + bw < w) & (y + bh < h)
                & (area >= 0.25 * bw * bh) & (np.maximum(bw, bh) <= 5 * np.minimum(bw, bh)))
        for i in np.flatnonzero(keep) + 1:
            cx0, cy0, cw, ch = (int(v) for v in stats[i, :4])
            x0, y0 = max(0, cx0 - margin), max(0, cy0 - margin)
            x1, y1 = min(w, cx0 + cw + margin), min(h, cy0 + ch + margin)
            body = _fill_holes((cc[y0:y1, x0:x1] == i).astype(np.uint8))
            mask = cv2.dilate(body, kernel)  # take the outline ring along
            ring = cv2.dilate(mask, kernel, iterations=2) - mask
            if not ring.any():
                continue
            patch = lab[y0:y1, x0:x1]
            contrast = float(np.linalg.norm(patch[mask > 0].mean(0) - patch[ring > 0].mean(0)))
            ys, xs = np.nonzero(mask)
            ex, ey = float(xs.mean()) + x0, float(ys.mean()) + y0
            centre = 1.0 - 0.6 * min(1.0, math.hypot(ex - w / 2, ey - h / 2) / (0.5 * long_side))
            size = float(mask.sum())
            # Prefer details (eyes, teeth, letters) over big parts (jacket, belt).
            score = contrast * math.sqrt(size) * centre * math.exp(-size / total / 0.02)
            if size > _LARGE_ELEMENT * total:
                score *= 0.25
            found.append(_Element(x0, y0, mask, ex, ey, int(mask.sum()), score))
    jitter = rng.uniform(0.6, 1.4, size=len(found))
    for element, j in zip(found, jitter):
        element.score *= float(j)
    found.sort(key=lambda e: e.score, reverse=True)
    return found


def _select(elements: list[_Element], count: int, shape) -> list[_Element]:
    """Best-scoring elements that do not overlap each other."""
    taken = np.zeros(shape, dtype=np.uint8)
    chosen: list[_Element] = []
    for element in elements:
        if len(chosen) >= count:
            break
        h, w = element.mask.shape
        overlap = taken[element.y0:element.y0 + h, element.x0:element.x0 + w][element.mask > 0].sum()
        if overlap > 0.15 * element.area:
            continue
        element.paint(taken)
        chosen.append(element)
    return chosen


def _affine(element: _Element, tx: float, ty: float, angle: float, scale: float, flip: bool) -> np.ndarray:
    """Crop -> image transform: flip/turn/resize around the centroid, then move it to (tx, ty)."""
    lcx, lcy = element.cx - element.x0, element.cy - element.y0
    rot = np.vstack([cv2.getRotationMatrix2D((lcx, lcy), angle, scale), [0, 0, 1]])
    mirror = np.array([[-1, 0, 2 * lcx], [0, 1, 0], [0, 0, 1]], dtype=np.float64) if flip else np.eye(3)
    m = (rot @ mirror)[:2]
    m[:, 2] += (tx - lcx, ty - lcy)
    return m


def _plan_ops(chosen: list[_Element], draws: np.ndarray, plan: Plan, scale: float,
              long_side: int, total: int) -> list[dict]:
    p_drop, p_swap, p_copy = plan.shuffle_drop * scale, plan.shuffle_swap * scale, plan.shuffle_copy * scale
    ops = []
    for element, u in zip(chosen, draws):
        if u[0] < p_drop:
            kind = "drop"
        elif u[0] < p_drop + p_swap:
            kind = "swap"
        elif u[0] < p_drop + p_swap + p_copy:
            kind = "copy"
        else:
            kind = "move"
        ops.append({"element": element, "kind": kind, "u": u, "partner": None})

    # A swap trades centres with the closest element of a similar size.
    for op in ops:
        if op["kind"] != "swap" or op["partner"] is not None:
            continue
        e = op["element"]
        reach = min(0.25 * long_side, max(3.0 * e.diameter, 0.12 * long_side))
        partners = [o for o in ops if o is not op and o["kind"] in ("move", "swap") and o["partner"] is None
                    and 1 / 3 <= o["element"].area / e.area <= 3
                    and max(e.area, o["element"].area) <= _LARGE_ELEMENT * total
                    and math.hypot(o["element"].cx - e.cx, o["element"].cy - e.cy) <= reach]
        if not partners:
            op["kind"] = "move"
            continue
        other = min(partners, key=lambda o: math.hypot(o["element"].cx - e.cx, o["element"].cy - e.cy))
        other["kind"] = "swap"
        op["partner"], other["partner"] = other, op
    return ops


def _shuffle(img: np.ndarray, plan: Plan, rng_segment: np.random.Generator,
             rng: np.random.Generator, scale: float) -> tuple[np.ndarray, np.ndarray]:
    """Move, swap, turn, mirror, resize, duplicate or forget whole elements.

    Returns the new image and a 0..1 mask of everything that was touched.
    """
    count = min(_MAX_ELEMENTS, int(round(plan.shuffle_count * scale)))
    draws = rng.random((_MAX_ELEMENTS, 6))  # drawn up front: same structure for every w
    h, w = img.shape[:2]
    touched = np.zeros((h, w), dtype=np.float32)
    if count <= 0:
        return img, touched
    long_side = max(h, w)
    chosen = _select(_find_elements(img, rng_segment), count, (h, w))
    if not chosen:
        return img, touched
    ops = _plan_ops(chosen, draws, plan, scale, long_side, h * w)

    removed = np.zeros((h, w), dtype=np.uint8)
    for op in ops:
        if op["kind"] != "copy":
            op["element"].paint(removed)
    touched = np.maximum(touched, removed.astype(np.float32))
    plate = img
    if removed.any():
        holes = cv2.dilate(removed, np.ones((3, 3), np.uint8)) * 255
        plate = cv2.inpaint(img, holes, max(3, round(long_side / 112)), cv2.INPAINT_TELEA)
    canvas = plate.astype(np.float32)

    for op in sorted(ops, key=lambda o: -o["element"].area):  # small elements end up on top
        if op["kind"] == "drop":
            continue
        e, u = op["element"], op["u"]
        if op["kind"] == "swap":
            tx, ty = op["partner"]["element"].cx, op["partner"]["element"].cy
        else:
            direction = 2 * math.pi * u[1]
            dist = plan.shuffle_shift * scale * long_side * (0.35 + 0.65 * u[2])
            # Big parts drift less; small ones stay near where they belong.
            dist *= min(1.0, 0.08 * long_side / e.diameter)
            dist = min(dist, max(1.0 * e.diameter, 6.0))
            if e.area > _LARGE_ELEMENT * h * w:
                dist = min(dist, 0.5 * e.diameter)
            if op["kind"] == "copy":
                dist = max(dist, 0.9 * e.diameter)
            tx, ty = e.cx + dist * math.cos(direction), e.cy + dist * math.sin(direction)
        radius = 0.5 * e.diameter
        tx = float(np.clip(tx, radius, w - radius))
        ty = float(np.clip(ty, radius, h - radius))
        angle = (2 * u[3] - 1) * plan.shuffle_turn * scale
        size = 1.0 + (2 * u[4] - 1) * plan.shuffle_resize * scale
        flip = bool(u[5] < plan.shuffle_flip * scale)
        m = _affine(e, tx, ty, angle, size, flip)
        mh, mw = e.mask.shape
        source = img[e.y0:e.y0 + mh, e.x0:e.x0 + mw].astype(np.float32)
        warped = cv2.warpAffine(source, m, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        alpha = cv2.warpAffine(e.mask.astype(np.float32), m, (w, h), flags=cv2.INTER_LINEAR,
                               borderMode=cv2.BORDER_CONSTANT)[..., None]
        canvas = canvas * (1.0 - alpha) + warped * alpha
        touched = np.maximum(touched, alpha[..., 0])
    # Grow the mask a little so the seams SD-Turbo heals are not anchored back.
    grow = max(2, round(long_side / 75))
    touched = cv2.dilate(touched, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * grow + 1, 2 * grow + 1)))
    touched = np.clip(cv2.GaussianBlur(touched, (0, 0), grow * 0.75), 0.0, 1.0)
    return _to_uint8(canvas / 255.0), touched


def _simplify(img: np.ndarray, radius: int) -> np.ndarray:
    """Forget fine details (thin lines, specks) while keeping edges sharp."""
    return cv2.medianBlur(img, 2 * radius + 1) if radius > 0 else img


def _decay_color(x: np.ndarray, desaturate: float, fade: float) -> np.ndarray:
    if desaturate > 0:
        lum = (x @ _LUMA)[..., None]
        x = x + (lum - x) * desaturate
    if fade > 0:
        x = x * (1.0 - fade) + _FADE_TO * fade
    return x.astype(np.float32)


@dataclass
class Misreading:
    image: np.ndarray       # what SD-Turbo receives (uint8 HxWx3)
    reference: np.ndarray   # the original under the same geometry: the anchor target
    touched: np.ndarray     # float32 HxW, 1 where elements were moved, copied or lost


def misread_detailed(img: np.ndarray, plan: Plan, seed: int, scale: float = 1.0) -> Misreading:
    """Half-remembered copy of `img` (uint8 HxWx3) before the neural pass.

    `scale` < 1 gives the lighter damage used between repeated passes.
    """
    rng = _streams(seed)
    reference = _warp(np.ascontiguousarray(img), plan, rng["warp"], scale)
    out, touched = _shuffle(reference, plan, rng["segment"], rng["elements"], scale)
    out = _simplify(out, int(round(plan.simplify * scale)))
    if plan.desaturate > 0 or plan.fade > 0:
        out = _to_uint8(_decay_color(_to_float(out), plan.desaturate * scale, plan.fade * scale))
    return Misreading(out, reference, touched)


def misread(img: np.ndarray, plan: Plan, seed: int, scale: float = 1.0) -> np.ndarray:
    """Just the damaged image (see misread_detailed)."""
    return misread_detailed(img, plan, seed, scale).image


def apply_anchor(result: np.ndarray, reference: np.ndarray, plan: Plan,
                 touched: np.ndarray | None = None) -> np.ndarray:
    """Pull the original back in, in two parts.

    Fine: the reference's detail (small blur) wherever nothing was shuffled,
    so untouched areas stay faithful and moved elements leave no ghost.
    Coarse: layout and colours everywhere. Its weight never reaches zero, so
    even 100% keeps the original's coarse layout and colours.
    """
    if result.shape != reference.shape:
        result = cv2.resize(result, (reference.shape[1], reference.shape[0]), interpolation=cv2.INTER_LINEAR)
    r = _to_float(result)
    o = _to_float(reference)
    long_side = max(r.shape[:2])
    keep = 1.0 if touched is None else (1.0 - np.clip(touched, 0.0, 1.0))[..., None]
    fine = max(0.5, plan.anchor_sigma * long_side)
    r = r + plan.anchor_weight * keep * (cv2.GaussianBlur(o, (0, 0), fine) - cv2.GaussianBlur(r, (0, 0), fine))
    coarse = _ANCHOR_COARSE_SIGMA * long_side
    r = r + plan.anchor_coarse * (cv2.GaussianBlur(o, (0, 0), coarse) - cv2.GaussianBlur(r, (0, 0), coarse))
    return _to_uint8(r)
