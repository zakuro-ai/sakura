"""The decision template, option order, and the calibration arithmetic. Pure Python: no model."""

from __future__ import annotations

import math
import random

import pytest

from sakura.atelier.decision import (
    Instance,
    Scored,
    build_prompt,
    expected_calibration_error,
    fit_temperature,
    fit_temperatures,
    identity_order,
    report_blocks,
    shuffled_order,
    softmax,
    split_metrics,
    to_label_order,
    to_letter_order,
)

CHOICE = Instance(state="Hi, I was charged twice for March.", qid="topic", kind="choice",
                  text="What is this about?", labels=["billing", "bug", "other"], target=[1.0, 0.0, 0.0])


# ---------------------------------------------------------------- template

def test_decision_v1_is_pinned_byte_for_byte():
    # The template is fixed and versioned: a change here is a new version, and a model trained on
    # v1 is only read correctly with v1.
    prefix, suffix = build_prompt(CHOICE, [1, 0, 2])
    assert prefix == "### State\nHi, I was charged twice for March.\n\n"
    assert suffix == "### Question\nWhat is this about?\nA. bug\nB. billing\nC. other\nAnswer:"


def test_a_noul_is_two_lettered_options_true_then_false():
    q = Instance(state="x", qid="urgent", kind="noul", text="The customer needs a reply today.",
                 labels=["true", "false"], target=[1.0, 0.0])
    _, suffix = build_prompt(q, identity_order(2))
    assert suffix == ("### Question\nIs the following statement true or false?\nThe customer needs a reply today.\n"
                      "A. true\nB. false\nAnswer:")


def test_a_choice_without_instructions_still_asks_something():
    q = Instance(state="x", qid="t", kind="choice", text="", labels=["a", "b"], target=[1.0, 0.0])
    assert "Choose the best option.\nA. a\nB. b\n" in build_prompt(q, identity_order(2))[1]


def test_a_json_state_is_rendered_with_sorted_keys_and_a_string_state_as_is():
    q = Instance(state={"b": [1, 2], "a": "é"}, qid="t", kind="choice", text="?", labels=["a", "b"], target=[1.0, 0.0])
    assert build_prompt(q, identity_order(2))[0] == '### State\n{"a": "é", "b": [1, 2]}\n\n'


def test_the_prefix_depends_on_the_state_only_so_questions_can_share_it():
    other = Instance(state=CHOICE.state, qid="u", kind="noul", text="s", labels=["true", "false"], target=[1.0, 0.0])
    assert build_prompt(CHOICE, [0, 1, 2])[0] == build_prompt(other, [0, 1])[0]


# ------------------------------------------------------------- option order

def test_inference_order_is_the_given_order():
    assert identity_order(4) == [0, 1, 2, 3]


def test_shuffled_order_is_a_permutation_that_changes_between_draws_and_with_the_seed():
    rng = random.Random(0)
    draws = [shuffled_order(rng, 5) for _ in range(20)]
    assert all(sorted(d) == [0, 1, 2, 3, 4] for d in draws)
    assert len({tuple(d) for d in draws}) > 5  # reshuffled per example per epoch
    assert [shuffled_order(random.Random(1), 5) for _ in range(3)] == [shuffled_order(random.Random(1), 5) for _ in range(3)]


def test_letter_order_and_label_order_are_inverse():
    order = [2, 0, 1]
    assert to_letter_order([0.1, 0.2, 0.7], order) == [0.7, 0.1, 0.2]  # letter A shows label 2
    assert to_label_order([0.7, 0.1, 0.2], order) == [0.1, 0.2, 0.7]


def test_letter_position_carries_no_signal_in_the_training_orders():
    # the gold label's letter should be uniform over positions across many shuffles
    rng = random.Random(3)
    counts = [0, 0, 0]
    for _ in range(3000):
        letters = to_letter_order([1.0, 0.0, 0.0], shuffled_order(rng, 3))
        counts[letters.index(1.0)] += 1
    assert all(900 < c < 1100 for c in counts)


# ------------------------------------------------------------- arithmetic

def test_softmax_is_a_distribution_and_temperature_flattens_it():
    p1, p2 = softmax([2.0, 0.0, -1.0]), softmax([2.0, 0.0, -1.0], temperature=2.0)
    assert sum(p1) == pytest.approx(1.0) and sum(p2) == pytest.approx(1.0)
    assert p2[0] < p1[0] and p2[2] > p1[2]
    assert softmax([1000.0, 0.0]) == pytest.approx([1.0, 0.0])  # no overflow


def test_ece_of_a_known_miscalibration():
    # all 0.9 confident, 60% right: |0.6 - 0.9| in the single occupied bin
    assert expected_calibration_error([0.9] * 10, [True] * 6 + [False] * 4) == pytest.approx(0.3)
    # a confidence of exactly 1.0 lands in the last of the 15 bins, not off the end
    assert expected_calibration_error([1.0, 1.0], [True, True]) == pytest.approx(0.0)
    assert expected_calibration_error([], []) == 0.0


def test_ece_uses_fifteen_equal_width_bins():
    # 0.50 and 0.55 are different bins of width 1/15 (0.4667-0.5333 and 0.5333-0.6): each is its own gap
    got = expected_calibration_error([0.5, 0.56], [True, True])
    assert got == pytest.approx((abs(1 - 0.5) + abs(1 - 0.56)) / 2)


def _overconfident(n, factor, seed=0, k=3):
    """Items whose logits are `factor` times too large: labels are drawn from the true softmax."""
    rng = random.Random(seed)
    items = []
    for _ in range(n):
        z = [rng.gauss(0, 1.5) for _ in range(k)]
        label = rng.choices(range(k), weights=softmax(z))[0]
        items.append(Scored(qid="q", kind="choice", logits=[v * factor for v in z],
                            target=[1.0 if i == label else 0.0 for i in range(k)]))
    return items


def test_the_temperature_fit_recovers_an_overconfident_model():
    t = fit_temperature(_overconfident(4000, factor=3.0))
    assert t == pytest.approx(3.0, rel=0.15)


def test_the_temperature_fit_recovers_an_underconfident_model():
    t = fit_temperature(_overconfident(4000, factor=0.5))
    assert t == pytest.approx(0.5, rel=0.15)


def test_the_temperature_fit_minimises_the_nll_it_is_given():
    items = _overconfident(500, factor=2.0, seed=2)
    best = fit_temperature(items)

    def nll(t):
        return -sum(sum(tg * math.log(max(p, 1e-12)) for tg, p in zip(i.target, softmax(i.logits, t))) for i in items)

    assert nll(best) <= nll(best * 1.1) and nll(best) <= nll(best / 1.1)


def test_the_temperature_fit_handles_soft_labels():
    rng = random.Random(0)
    items = []
    for _ in range(300):
        z = [rng.gauss(0, 1) for _ in range(3)]
        items.append(Scored("q", "choice", [v * 2 for v in z], softmax(z)))  # target IS the calibrated softmax
    assert fit_temperature(items) == pytest.approx(2.0, rel=0.05)


def test_the_fit_stays_inside_its_bounds():
    sure = [Scored("q", "choice", [50.0, 0.0], [1.0, 0.0])] * 100  # always right, hugely confident
    t = fit_temperature(sure)
    assert 0.05 <= t <= 20.0


def test_split_metrics_accuracy_log_loss_ece():
    items = [Scored("q", "choice", [math.log(9), 0.0], [1.0, 0.0]),   # p = .9, right
             Scored("q", "choice", [math.log(9), 0.0], [0.0, 1.0])]   # p = .9 for the wrong one
    m = split_metrics(items, {"choice": 1.0})
    assert m["n"] == 2 and m["accuracy"] == 0.5
    assert m["log_loss"] == pytest.approx((-math.log(0.9) - math.log(0.1)) / 2, abs=1e-5)
    assert m["ece"] == pytest.approx(abs(0.5 - 0.9), abs=1e-5)


def test_split_metrics_applies_the_temperature_of_each_type():
    a = Scored("q", "choice", [math.log(9), 0.0], [1.0, 0.0])
    b = Scored("u", "noul", [math.log(9), 0.0], [1.0, 0.0])
    cool = split_metrics([a, b], {"choice": 1.0, "noul": 1.0})["log_loss"]
    warm = split_metrics([a, b], {"choice": 1.0, "noul": 10.0})["log_loss"]
    assert warm > cool  # the noul item was flattened toward 0.5


def test_split_metrics_of_soft_labels_use_the_cross_entropy_and_the_modal_label():
    item = Scored("q", "choice", [0.0, 0.0, 0.0], [0.5, 0.3, 0.2])
    m = split_metrics([item], {"choice": 1.0})
    assert m["log_loss"] == pytest.approx(-(0.5 * math.log(1 / 3) + 0.3 * math.log(1 / 3) + 0.2 * math.log(1 / 3)))
    assert m["accuracy"] == 1.0  # ties break to the first label, which is the modal one here


# -------------------------------------------------------------- calibration

def test_one_temperature_per_answer_type_from_the_calibration_items_only():
    calib = [*_overconfident(60, 3.0, seed=1), *[Scored("u", "noul", i.logits[:2], i.target[:2])
                                                 for i in _overconfident(60, 1.0, seed=2, k=2)]]
    temps, n_calib, warnings = fit_temperatures(calib)
    assert set(temps) == {"choice", "noul", "score"} and n_calib == {"choice": 60, "noul": 60, "score": 0}
    assert temps["choice"] > 1.5 and temps["noul"] == pytest.approx(1.0, abs=0.5)
    assert temps["score"] == 1.0
    assert warnings == []  # a type that is not in the data at all is not a warning


def test_a_type_with_under_thirty_calibration_answers_keeps_t_one_and_warns():
    temps, n_calib, warnings = fit_temperatures(_overconfident(29, 5.0))
    assert temps["choice"] == 1.0 and n_calib["choice"] == 29
    assert any("choice" in w and "29 calibration answers" in w and "kept at 1" in w for w in warnings)
    temps, _, warnings = fit_temperatures(_overconfident(30, 5.0))
    assert temps["choice"] > 1.5 and not any("choice" in w for w in warnings)


def test_eval_items_never_enter_the_fit():
    calib = _overconfident(200, 3.0, seed=1)
    ev = _overconfident(200, 3.0, seed=9)
    a = report_blocks(calib, ev, [])
    # hand the eval split to something that would move T if it were used: flip every eval label
    flipped = [Scored(i.qid, i.kind, i.logits, list(reversed(i.target))) for i in ev]
    b = report_blocks(calib, flipped, [])
    assert a["calibration"]["temperatures"] == b["calibration"]["temperatures"]
    assert a["calibration"]["ece_before"] != b["calibration"]["ece_before"]  # ...but the eval numbers do move


def test_ece_after_is_measured_on_eval_and_is_lower_for_an_overconfident_model():
    out = report_blocks(_overconfident(1500, 3.0, seed=1), _overconfident(1500, 3.0, seed=2), [])
    cal = out["calibration"]
    assert cal["method"] == "temperature" and cal["ece_after"] < cal["ece_before"]
    assert cal["ece_before"] > 0.1 and cal["ece_after"] < 0.05
    assert out["metrics"]["ece"] == cal["ece_after"]  # metrics are the calibrated eval numbers


def test_the_report_blocks_have_the_documented_keys():
    out = report_blocks(_overconfident(100, 2.0, seed=1), _overconfident(100, 2.0, seed=2), ["only 50 usable rows"])
    assert set(out) == {"metrics", "evaluation", "calibration"}
    assert set(out["metrics"]) == {"accuracy", "log_loss", "ece"}
    ev = out["evaluation"]
    assert ev["warnings"][0] == "only 50 usable rows"
    eval_split = ev["splits"]["eval"]
    assert {"n", "accuracy", "log_loss", "ece", "by_type", "by_question"} <= set(eval_split)
    assert eval_split["n"] == 100 and set(eval_split["by_type"]) == {"choice"} and set(eval_split["by_question"]) == {"q"}
    assert {"n", "accuracy", "log_loss", "ece"} <= set(eval_split["by_type"]["choice"])
    assert set(out["calibration"]) == {"method", "temperatures", "n_calib", "ece_before", "ece_after"}
    assert out["calibration"]["n_calib"]["choice"] == 100


def test_small_calibration_warnings_land_in_the_evaluation_warnings():
    out = report_blocks(_overconfident(10, 2.0), _overconfident(40, 2.0, seed=2), [])
    assert any("10 calibration answers" in w for w in out["evaluation"]["warnings"])


def test_unfitted_types_leave_ece_after_equal_to_before():
    out = report_blocks(_overconfident(10, 4.0), _overconfident(200, 4.0, seed=2), [])
    assert out["calibration"]["temperatures"]["choice"] == 1.0
    assert out["calibration"]["ece_after"] == out["calibration"]["ece_before"]
