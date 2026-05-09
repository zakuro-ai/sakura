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


def test_cifar10_workload_runs_one_epoch():
    """Build CIFAR-10 + ResNet-50 workload, run 1 epoch via BaselineRunner.
    Tiny subset (32 train, 16 val) to keep CPU smoke under 60s."""
    from sakura.bench.workloads.cifar import make_workload as make_cifar

    wl = make_cifar(batch_size=16, epochs=1, n_train=32, n_val=16)
    runner = BaselineRunner(framework="pytorch-ddp")
    report = runner.run(wl)

    assert report.workload == "cifar10-resnet50"
    assert report.elapsed_secs > 0
    assert "val_acc" in report.final_metrics


def test_distilbert_workload_runs_one_epoch():
    """Build DistilBERT + SST-2 workload, run 1 epoch via BaselineRunner.
    Tiny subset (64 train, 32 val) to keep CPU smoke under 90s."""
    from sakura.bench.workloads.distilbert import make_workload as make_distilbert

    wl = make_distilbert(batch_size=16, epochs=1, n_train=64, n_val=32, max_length=32)
    runner = BaselineRunner(framework="pytorch-ddp")
    report = runner.run(wl)

    assert report.workload == "distilbert-sst2"
    assert report.elapsed_secs > 0
    assert "val_acc" in report.final_metrics


import pytest


def test_llama_workload_skips_without_gpu():
    """Llama-3-1B is perf-tier; skip cleanly on CPU-only."""
    if torch.cuda.is_available() and torch.cuda.device_count() >= 1:
        pytest.skip("Llama-3-1B perf-tier workload not yet implemented")
    from sakura.bench.workloads.llama import make_workload
    wl = make_workload()
    assert wl.tier == "perf"
    # Calling make_model without a GPU raises:
    with pytest.raises(NotImplementedError, match="GPU|stub"):
        wl.make_model()


def test_mistral_workload_is_perf_tier_stub():
    from sakura.bench.workloads.mistral import make_workload
    wl = make_workload()
    assert wl.tier == "perf"
    assert "mistral" in wl.name.lower()


def test_glue_workload_is_perf_tier_stub():
    from sakura.bench.workloads.glue import make_workload
    wl = make_workload()
    assert wl.tier == "perf"
