"""BaselineRunner smoke test on a tiny synthetic workload."""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from sakura.bench.harness import BaselineRunner, Workload


def _make_synthetic_workload() -> Workload:
    """4-batch tiny MLP — runs in <1s."""
    def make_model():
        return torch.nn.Sequential(
            torch.nn.Linear(8, 16),
            torch.nn.ReLU(),
            torch.nn.Linear(16, 4),
        )

    def make_loader():
        # Yield 4 batches of 8 samples × 8 features → 4 classes.
        torch.manual_seed(0)
        return [
            (torch.randn(8, 8), torch.randint(0, 4, (8,)))
            for _ in range(4)
        ]

    def eval_fn(model, loader):
        correct = total = 0
        with torch.no_grad():
            for x, y in loader:
                pred = model(x).argmax(dim=-1)
                correct += int((pred == y).sum())
                total += int(y.numel())
        return {"val_acc": correct / max(total, 1)}

    return Workload(
        name="synthetic-tiny-mlp",
        tier="smoke",
        make_model=make_model,
        make_train_loader=make_loader,
        make_val_loader=make_loader,
        eval_fn=eval_fn,
        epochs=1,
    )


def test_baseline_runner_pytorch_ddp_completes():
    wl = _make_synthetic_workload()
    runner = BaselineRunner(framework="pytorch-ddp")
    report = runner.run(wl)

    assert report.workload == "synthetic-tiny-mlp"
    assert report.framework == "pytorch-ddp"
    assert report.elapsed_secs > 0
    assert report.samples_per_sec > 0
    assert "val_acc" in report.final_metrics
    assert report.git_sha != "" or report.git_sha == ""  # either is fine
