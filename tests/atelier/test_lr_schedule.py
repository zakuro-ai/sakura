"""sakura.atelier.lr_schedule: step counts and the learning-rate multiplier, with no training stack."""

import math

import pytest

from sakura.atelier.lr_schedule import lr_factor, optimizer_steps, warmup_steps


def test_one_optimiser_step_per_accumulation_group_and_the_last_partial_group_counts():
    assert optimizer_steps(rows=9, batch_size=2, grad_accum=1, epochs=2) == 10  # 5 batches an epoch
    assert optimizer_steps(rows=9, batch_size=2, grad_accum=2, epochs=2) == 6  # groups of 2, 2 and 1
    assert optimizer_steps(rows=8, batch_size=2, grad_accum=4, epochs=3) == 3  # one full group an epoch
    assert optimizer_steps(rows=41888, batch_size=4, grad_accum=4, epochs=3) == 7854
    assert optimizer_steps(rows=0, batch_size=2, grad_accum=1, epochs=3) == 0


def test_limits_max_steps_shortens_the_plan_and_never_lengthens_it():
    assert optimizer_steps(rows=100, batch_size=2, grad_accum=1, epochs=2, max_steps=30) == 30
    assert optimizer_steps(rows=100, batch_size=2, grad_accum=1, epochs=2, max_steps=500) == 100


def test_the_warmup_is_the_ceiling_of_its_share_of_the_steps():
    assert warmup_steps(100, 0.03) == 3 and warmup_steps(7854, 0.03) == 236  # 235.62 rounds up
    assert warmup_steps(10, 0.25) == 3 and warmup_steps(10, 0.0) == 0 and warmup_steps(10, 0.5) == 5
    assert warmup_steps(100, 0.07) == 7  # 0.07 * 100 is 7.000000000000001 in floating point: still 7


@pytest.mark.parametrize("schedule", ["constant", "linear", "cosine"])
def test_the_warmup_climbs_linearly_from_zero_to_the_peak(schedule):
    values = [lr_factor(s, 100, schedule, warmup=10, min_ratio=0.1) for s in range(11)]
    assert values == pytest.approx([s / 10 for s in range(11)])  # 0 at the first step, the peak at step 10


def test_cosine_falls_from_the_peak_to_the_floor():
    f = [lr_factor(s, 100, "cosine", warmup=10, min_ratio=0.1) for s in range(101)]
    assert f[10] == 1.0 and f[100] == pytest.approx(0.1)
    assert f[55] == pytest.approx(0.1 + 0.9 * 0.5)  # half way down at the middle of the decay
    assert all(a > b for a, b in zip(f[10:], f[11:]))  # strictly down after the peak
    assert f[99] == pytest.approx(0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * 89 / 90)))  # the last step taken


def test_linear_falls_in_equal_steps_to_the_floor():
    f = [lr_factor(s, 50, "linear", warmup=0, min_ratio=0.2) for s in range(51)]
    assert f[0] == 1.0 and f[50] == pytest.approx(0.2) and f[25] == pytest.approx(0.6)
    assert [round(a - b, 9) for a, b in zip(f, f[1:])] == [0.016] * 50


def test_constant_stays_at_the_peak_after_the_warmup_whatever_the_floor():
    assert [lr_factor(s, 20, "constant", warmup=2, min_ratio=0.5) for s in (0, 1, 2, 10, 19, 40)] == [0, 0.5, 1, 1, 1, 1]


def test_a_floor_of_one_is_a_flat_rate_and_a_step_past_the_plan_stays_on_the_floor():
    assert lr_factor(7, 10, "cosine", 0, 1.0) == 1.0
    assert lr_factor(25, 10, "cosine", 2, 0.1) == pytest.approx(0.1) == lr_factor(25, 10, "linear", 2, 0.1)


def test_an_unknown_schedule_is_refused():
    with pytest.raises(ValueError, match="lr_schedule must be one of"):
        lr_factor(0, 10, "exponential")


def test_the_open_clip_backend_uses_the_same_curve():
    from sakura.atelier.backends import open_clip

    for step in (0, 3, 10, 40, 99, 100):
        assert open_clip.warmup_cosine(step, 10, 100) == lr_factor(step, 100, "cosine", 10, 0.0)
