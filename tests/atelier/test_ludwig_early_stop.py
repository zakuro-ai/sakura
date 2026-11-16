"""What stopped a Ludwig run, in epochs (BE-19).

The preset's patience never reached Ludwig's trainer (it stopped on its own
default of 5 rounds), so "patience 3, best at epoch 15, ran to epoch 20" did
not add up. The compiled config now carries it, and the report says which
metric the stop watched, where the best epoch was and why training ended.
Needs no Ludwig: the statistics are plain `(epoch, step, value)` series (the
shape of Ludwig's `TrainerMetric`) or, as Ludwig 0.17's `model.train()`
actually returns them, bare floats with an `evaluation_frequency`."""

from __future__ import annotations

from collections import namedtuple

from sakura.atelier.backends.ludwig import compile_config, early_stop_summary, epoch_metrics
from sakura.atelier.registry import resolve
from sakura.atelier.spec import load_spec

TrainerMetric = namedtuple("TrainerMetric", ["epoch", "step", "value"])

SPEC = """
atelier: 1
name: cifar-mini
task: image_classification
data: {uri: file:///x, format: image_folder, split: {validation_fraction: 0.2}}
features:
  inputs: [{name: img, type: image}]
  output: {name: label, type: category}
model: resnet18
hp: image_classification/fast@1
overrides: {epochs: 30, early_stop: 3}
export: [onnx]
"""


def _series(values, steps_per_epoch=7):
    return [TrainerMetric(e, (e + 1) * steps_per_epoch, v) for e, v in enumerate(values)]


def test_the_presets_patience_reaches_ludwigs_trainer():
    config = compile_config(resolve(load_spec(SPEC)))
    assert config["trainer"]["early_stop"] == 3 and config["trainer"]["epochs"] == 30
    no_stop = compile_config(resolve(load_spec(SPEC.replace("early_stop: 3", "early_stop: -1"))))
    assert no_stop["trainer"]["early_stop"] == -1
    tuned = compile_config(resolve(load_spec(SPEC.replace("early_stop: 3}", "early_stop: 3, validation_metric: accuracy}"))))
    assert tuned["trainer"]["validation_metric"] == "accuracy"
    # backend_config still has the last word
    raw = SPEC + "backend_config: {trainer: {early_stop: 9}}\n"
    assert compile_config(resolve(load_spec(raw)))["trainer"]["early_stop"] == 9


def test_early_stop_reports_best_and_stopped_epochs_in_epochs():
    loss = [2.0, 1.5, 1.2, 1.1, 1.0, 1.05, 1.2, 1.3]  # best at epoch 5, patience 3 -> stops after epoch 8
    stats = {"validation": {"label": {"loss": _series(loss)}}}
    out = early_stop_summary(stats, field="label", metric="loss", patience=3, max_epochs=30)
    assert out == {"metric": "loss", "field": "label", "mode": "min", "patience": 3, "enabled": True,
                   "best_epoch": 5, "best_step": 35, "best_value": 1.0, "stopped_epoch": 8, "stopped_step": 56,
                   "max_epochs": 30, "reason": "early_stop"}


def test_accuracy_is_maximised_and_the_budget_can_end_a_run():
    acc = [0.2, 0.3, 0.42, 0.41]
    stats = {"validation": {"combined": {"accuracy": _series(acc)}}}
    out = early_stop_summary(stats, field="combined", metric="accuracy", patience=-1, max_epochs=4)
    assert out["mode"] == "max" and out["best_epoch"] == 3 and out["stopped_epoch"] == 4
    assert out["reason"] == "max_epochs" and out["enabled"] is False and out["patience"] is None


def test_first_best_wins_ties_and_missing_series_is_none():
    stats = {"validation": {"label": {"loss": _series([1.0, 0.5, 0.5, 0.7])}}}
    assert early_stop_summary(stats, field="label", metric="loss", patience=2, max_epochs=10)["best_epoch"] == 2
    assert early_stop_summary({}, field="label", metric="loss", patience=2, max_epochs=10) is None
    nan = {"validation": {"label": {"loss": _series([float("nan")])}}}
    assert early_stop_summary(nan, field="label", metric="loss", patience=2, max_epochs=10) is None


def test_epoch_metrics_are_one_row_per_epoch_with_both_splits():
    stats = {
        "training": {"label": {"loss": _series([1.0, 0.5]), "accuracy": _series([0.5, 0.9])},
                     "combined": {"loss": _series([1.0, 0.5])}},
        "validation": {"label": {"loss": _series([1.2, 1.1]), "accuracy": _series([0.3, 0.42])}},
    }
    rows = epoch_metrics(stats, "label")
    assert rows == [
        {"epoch": 1, "step": 7, "train": {"loss": 1.0, "accuracy": 0.5}, "validation": {"loss": 1.2, "accuracy": 0.3}},
        {"epoch": 2, "step": 14, "train": {"loss": 0.5, "accuracy": 0.9}, "validation": {"loss": 1.1, "accuracy": 0.42}},
    ]


def test_epoch_metrics_keep_the_last_point_of_an_epoch_evaluated_twice():
    pts = [TrainerMetric(0, 3, 0.9), TrainerMetric(0, 7, 0.8), TrainerMetric(1, 14, 0.7)]
    rows = epoch_metrics({"validation": {"y": {"root_mean_squared_error": pts}}}, "y")
    assert [(r["epoch"], r["step"], r["validation"]["rmse"]) for r in rows] == [(1, 7, 0.8), (2, 14, 0.7)]


# Ludwig 0.17's `model.train()` statistics hold bare floats, one per
# evaluation, with `evaluation_frequency` saying how often it evaluated
# (staging job 245ced5b: `{"frequency": 1, "period": "epoch"}`, 8 rounds).
# Read as (epoch, step, value) tuples they were all dropped, so every real
# run reported no early_stop and no epoch_metrics.
BARE = {
    "evaluation_frequency": {"frequency": 1, "period": "epoch"},
    "training": {"label": {"loss": [2.7, 1.9, 1.4, 1.0, 0.7, 0.5, 0.3, 0.2]}},
    "validation": {"label": {"loss": [3.1, 2.4, 2.3, 2.0, 1.98, 2.1, 2.3, 2.4],
                             "accuracy": [0.13, 0.24, 0.27, 0.37, 0.34, 0.34, 0.29, 0.3]},
                   "combined": {"loss": [3.1, 2.4, 2.3, 2.0, 1.98, 2.1, 2.3, 2.4]}},
}


def test_bare_float_series_are_numbered_by_epoch():
    out = early_stop_summary(BARE, field="combined", metric="loss", patience=3, max_epochs=30)
    assert out["best_epoch"] == 5 and out["stopped_epoch"] == 8 and out["reason"] == "early_stop"
    assert out["best_step"] is None and out["stopped_step"] is None, "steps are not in the statistics"
    rows = epoch_metrics(BARE, "label")
    assert [r["epoch"] for r in rows] == list(range(1, 9))
    assert rows[0]["train"]["loss"] == 2.7 and rows[3]["validation"]["accuracy"] == 0.37
    assert "step" not in rows[0]


def test_bare_float_series_every_k_epochs_and_as_attributes():
    class Stats:  # the object form of the same statistics
        evaluation_frequency = {"frequency": 2, "period": "epoch"}
        validation = {"label": {"loss": [2.0, 1.5, 1.6]}}
        training = {}
    out = early_stop_summary(Stats(), field="label", metric="loss", patience=None, max_epochs=6)
    assert out["best_epoch"] == 4 and out["stopped_epoch"] == 6 and out["reason"] == "max_epochs"


def test_bare_floats_evaluated_by_step_have_no_epochs_to_report():
    stats = {**BARE, "evaluation_frequency": {"frequency": 50, "period": "step"}}
    assert early_stop_summary(stats, field="combined", metric="loss", patience=3, max_epochs=30) is None
    assert epoch_metrics(stats, "label") == []
