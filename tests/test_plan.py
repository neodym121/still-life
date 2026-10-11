import math

import pytest

import config
from core.misread import (CURVES, DEFAULT_PRESET, MAX_DENOISE_STEPS, MAX_PASSES, PRESETS,
                          inference_steps_for, plan_for, preset_named)

GRID = [i / 400 for i in range(401)]


@pytest.mark.parametrize("name", sorted(CURVES))
def test_every_curve_is_monotonic_and_bounded(name):
    curve = CURVES[name]
    values = [getattr(plan_for(w), name) for w in GRID]
    lo, hi = min(curve.at0, curve.at1), max(curve.at0, curve.at1)
    assert all(lo - 1e-9 <= v <= hi + 1e-9 for v in values), name
    pairs = list(zip(values, values[1:]))
    if curve.at1 >= curve.at0:
        assert all(b >= a - 1e-12 for a, b in pairs), f"{name} must not decrease"
    else:
        assert all(b <= a + 1e-12 for a, b in pairs), f"{name} must not increase"


def test_endpoints():
    zero, full = plan_for(0.0), plan_for(1.0)
    for name, curve in CURVES.items():
        assert getattr(zero, name) == pytest.approx(curve.at0), name
        assert getattr(full, name) == pytest.approx(curve.at1), name
    assert zero.warp_large == zero.squash == 0
    assert zero.shuffle_count == 0 and zero.simplify == 0
    assert zero.desaturate == 0 and zero.fade == 0 and zero.passes == 1


def test_out_of_range_weirdness_is_clamped():
    assert plan_for(-3) == plan_for(0)
    assert plan_for(7) == plan_for(1)


@pytest.mark.parametrize("w", GRID)
def test_img2img_always_runs_the_planned_steps(w):
    plan = plan_for(w)
    executed = min(int(plan.num_inference_steps * plan.strength), plan.num_inference_steps)
    assert executed == plan.denoise_steps
    assert 1 <= plan.denoise_steps <= MAX_DENOISE_STEPS
    assert plan.num_inference_steps * plan.strength >= 1
    assert 0 < plan.strength < 1
    assert 1 <= plan.passes <= MAX_PASSES


@pytest.mark.parametrize("steps", range(1, MAX_DENOISE_STEPS + 1))
@pytest.mark.parametrize("strength", [0.05, 0.1, 0.123, 1 / 3, 0.3333, 0.45, 0.5, 0.55, 0.99])
def test_inference_steps_for(steps, strength):
    n = inference_steps_for(steps, strength)
    assert int(n * strength) == steps
    assert int((n - 1) * strength) < steps  # smallest such n


@pytest.mark.parametrize("w", GRID)
def test_anchor_never_reaches_zero(w):
    plan = plan_for(w)
    assert plan.anchor_weight >= 0.12 and plan.anchor_coarse >= 0.2
    assert plan.anchor_sigma > 0


def test_resolution_does_not_depend_on_weirdness():
    assert {plan_for(w).long_side for w in GRID} == {config.WORK_LONG_SIDE}


def test_reconstruction_gets_stronger_with_weirdness():
    low, mid, high = plan_for(0.01), plan_for(0.5), plan_for(1.0)
    assert low.strength < mid.strength < high.strength
    assert low.passes <= mid.passes <= high.passes >= 2
    assert low.anchor_weight > mid.anchor_weight > high.anchor_weight
    assert low.shuffle_count < mid.shuffle_count < high.shuffle_count


def test_strength_stays_where_sd_turbo_keeps_the_picture():
    # Tested on the real model: above ~0.35 it starts inventing new objects.
    assert all(plan_for(w).strength <= 0.35 for w in GRID)


def test_presets():
    levels = [p.weirdness for p in PRESETS]
    assert levels == sorted(levels) and len(set(levels)) == len(levels)
    assert levels == [0.25, 0.40, 0.70, 0.85, 1.00]
    assert preset_named("Distorted memory") is None
    assert len({p.name for p in PRESETS}) == len(PRESETS)
    assert all(p.hint and "%" not in p.name for p in PRESETS)
    assert PRESETS[DEFAULT_PRESET].weirdness == 0.25
    assert preset_named("Copy of a copy").weirdness == 0.70
    assert preset_named("nope") is None


def test_presets_grow_the_damage():
    plans = [plan_for(p.weirdness) for p in PRESETS]
    counts = [plan.shuffle_count for plan in plans]
    assert counts == sorted(counts) and counts[0] >= 1


def test_curves_have_sane_shape():
    for name, curve in CURVES.items():
        assert 0 <= curve.start < curve.end <= 1, name
        assert curve.gamma > 0 and math.isfinite(curve.gamma), name
