"""DistilBERT + GLUE multi-task workload (perf tier — stub).

The CI-tier sakura/bench/workloads/distilbert.py runs SST-2 only with a
~1 minute footprint. This perf-tier glue.py runs the full GLUE benchmark
(MNLI, QQP, SST-2, etc.) and is too heavy for CI.

Stub for now — real implementation in v1.x with the perf runner.
"""
from __future__ import annotations

from sakura.bench.harness import Workload


def _make_model():
    raise NotImplementedError("glue.make_workload is a perf-tier stub for v1.x.")


def _make_loader():
    raise NotImplementedError("glue loader is a stub.")


def _eval_fn(model, loader):
    raise NotImplementedError("glue eval_fn is a stub.")


def make_workload(*, batch_size: int = 32, epochs: int = 3) -> Workload:
    return Workload(
        name="distilbert-glue",
        tier="perf",
        make_model=_make_model,
        make_train_loader=_make_loader,
        make_val_loader=_make_loader,
        eval_fn=_eval_fn,
        epochs=epochs,
    )


__all__ = ["make_workload"]
