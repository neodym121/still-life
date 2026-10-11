import numpy as np
import pytest

from core.misread import apply_anchor, misread, misread_detailed, plan_for


def diff(a, b) -> float:
    return float(np.abs(a.astype(np.float32) - b.astype(np.float32)).mean())


def test_zero_weirdness_is_identity(picture_array):
    plan = plan_for(0.0)
    assert np.array_equal(misread(picture_array, plan, seed=1), picture_array)


def test_no_colour_cast_and_no_scan_lines():
    # A flat grey picture stays grey (no yellow tint) and has no row pattern.
    grey = np.full((96, 128, 3), 128, dtype=np.uint8)
    out = misread(grey, plan_for(1.0), seed=4).astype(np.int16)
    assert np.abs(out[..., 0] - out[..., 2]).max() <= 1
    assert np.abs(out[..., 1] - out[..., 2]).max() <= 1
    rows = out.mean(axis=(1, 2))
    assert rows.max() - rows.min() <= 2


@pytest.mark.parametrize("w", [0.25, 0.5, 1.0])
def test_shape_and_dtype_preserved(picture_array, w):
    out = misread(picture_array, plan_for(w), seed=3)
    assert out.shape == picture_array.shape and out.dtype == np.uint8


def test_odd_sizes(picture_array):
    for shape in [(64, 448, 3), (448, 64, 3), (72, 80, 3)]:
        img = np.random.default_rng(0).integers(0, 255, shape, dtype=np.uint8)
        assert misread(img, plan_for(1.0), seed=5).shape == shape


def test_deterministic_per_seed(picture_array):
    plan = plan_for(0.6)
    a = misread(picture_array, plan, seed=42)
    b = misread(picture_array, plan, seed=42)
    c = misread(picture_array, plan, seed=43)
    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)


def test_damage_grows_with_weirdness(picture_array):
    levels = [0.01, 0.25, 0.5, 1.0]
    damage = [np.mean([diff(misread(picture_array, plan_for(w), seed=s), picture_array) for s in range(4)])
              for w in levels]
    assert damage[0] < 1.0, "1% should be almost untouched"
    assert damage[0] < damage[1] < damage[2] < damage[3]


def test_repass_scale_is_lighter(picture_array):
    plan = plan_for(1.0)
    full = diff(misread(picture_array, plan, seed=9), picture_array)
    light = diff(misread(picture_array, plan, seed=9, scale=plan.repass_scale), picture_array)
    assert light < full


def test_anchor_pulls_back_towards_original(picture_array):
    plan = plan_for(1.0)
    wrecked = misread(picture_array, plan, seed=7)
    anchored = apply_anchor(wrecked, picture_array, plan)
    assert diff(anchored, picture_array) < diff(wrecked, picture_array)
    assert not np.array_equal(anchored, wrecked)


def test_anchor_restores_colours_at_low_weirdness(picture_array):
    # The anchor replaces low frequencies: a colour / contrast drift is undone.
    plan = plan_for(0.01)
    drifted = np.clip(picture_array.astype(np.float32) * 0.6 + [60, 20, -10], 0, 255).astype(np.uint8)
    anchored = apply_anchor(drifted, picture_array, plan)
    assert diff(anchored, picture_array) < diff(drifted, picture_array) * 0.3


def test_touched_mask_marks_only_moved_elements(picture_array):
    calm = misread_detailed(picture_array, plan_for(0.0), seed=2)
    assert calm.touched.max() == 0 and np.array_equal(calm.image, picture_array)
    busy = misread_detailed(picture_array, plan_for(0.7), seed=2)
    assert busy.touched.shape == picture_array.shape[:2]
    assert 0 < busy.touched.mean() < 0.6
    changed = np.abs(busy.image.astype(int) - busy.reference.astype(int)).sum(-1) > 30
    assert busy.touched[changed].mean() > 0.9  # every real change is inside the mask


def test_anchor_leaves_moved_elements_alone(picture_array):
    # Where elements were moved the fine anchor is off, so they are not pulled
    # back towards the original positions (no ghosts).
    plan = plan_for(0.7)
    m = misread_detailed(picture_array, plan, seed=2)
    image = m.image.astype(int)
    masked = np.abs(apply_anchor(m.image, m.reference, plan, m.touched).astype(int) - image).sum(-1)
    unmasked = np.abs(apply_anchor(m.image, m.reference, plan).astype(int) - image).sum(-1)
    inside = m.touched > 0.99
    assert inside.any()
    assert masked[inside].mean() < 0.5 * unmasked[inside].mean()


def test_anchor_handles_size_mismatch(picture_array):
    plan = plan_for(0.5)
    small = picture_array[::2, ::2]
    assert apply_anchor(small, picture_array, plan).shape == picture_array.shape
