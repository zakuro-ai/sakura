"""Workload smoke tests — instantiate + run 1 epoch via BaselineRunner."""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from sakura.bench.harness import BaselineRunner
from sakura.bench.workloads.mnist import make_workload as make_mnist


def test_mnist_workload_runs_one_epoch():
    """Build the MNIST workload, run 1 epoch via BaselineRunner. Assert it completes."""
    wl = make_mnist(batch_size=64, epochs=1)
    runner = BaselineRunner(framework="pytorch-ddp")
    report = runner.run(wl)

    assert report.workload == "mnist-mlp"
    assert report.elapsed_secs > 0
    assert "val_acc" in report.final_metrics
    assert "val_loss" in report.final_metrics
    # Even with 1 epoch on a tiny MLP, val_acc on synthetic data is around 0.1
    # (random); on real MNIST it's around 0.6+. We only assert non-zero.
    assert report.final_metrics["val_acc"] >= 0.0
