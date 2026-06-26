"""Workload dataclass tests."""
from __future__ import annotations

from sakura.bench.harness import Workload


def test_workload_carries_required_fields():
    w = Workload(
        name="dummy",
        tier="smoke",
        make_model=object,
        make_train_loader=lambda: [(1, 2)],
        make_val_loader=lambda: [(3, 4)],
        eval_fn=lambda model, payload: {"loss": 0.0},
        epochs=1,
    )
    assert w.name == "dummy"
    assert w.tier == "smoke"
    assert w.epochs == 1
    assert w.metric_target is None


def test_workload_metric_target_optional():
    w = Workload(
        name="x", tier="ci",
        make_model=lambda: None,
        make_train_loader=lambda: [],
        make_val_loader=lambda: [],
        eval_fn=lambda m, p: {},
        epochs=3,
        metric_target=("val_acc", 0.85),
    )
    assert w.metric_target == ("val_acc", 0.85)
