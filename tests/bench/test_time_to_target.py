"""Time-to-target run-mode tests (pytorch-ddp only)."""
from __future__ import annotations

import json
import subprocess
import sys

import pytest

torch = pytest.importorskip("torch")

from sakura.bench.harness import BaselineRunner, SakuraRunner, Workload


def _tiny_workload(metric_target=None, epochs=1, eval_metrics_seq=None) -> Workload:
    """A 4-batch tiny-MLP workload.

    When `eval_metrics_seq` is given, eval_fn returns successive dicts from
    that list (clamped at the last element), making the early-stop logic
    deterministic regardless of training dynamics — the loop control is what
    Task 5 tests, not model convergence.
    """
    state = {"i": 0}

    def make_model():
        return torch.nn.Sequential(
            torch.nn.Linear(8, 16), torch.nn.ReLU(), torch.nn.Linear(16, 4),
        )

    def make_loader():
        torch.manual_seed(0)
        ds = torch.utils.data.TensorDataset(
            torch.randn(16, 8), torch.randint(0, 4, (16,))
        )
        return torch.utils.data.DataLoader(ds, batch_size=8, shuffle=False)

    def eval_fn(model, loader):
        if eval_metrics_seq is None:
            return {"val_loss": 1.0, "val_acc": 0.5}
        i = state["i"]
        state["i"] += 1
        return dict(eval_metrics_seq[min(i, len(eval_metrics_seq) - 1)])

    return Workload(
        name="ttt-tiny", tier="smoke",
        make_model=make_model, make_train_loader=make_loader,
        make_val_loader=make_loader, eval_fn=eval_fn,
        epochs=epochs, metric_target=metric_target,
    )


def test_target_reached_direction():
    f = BaselineRunner._target_reached
    # lower-is-better
    assert f("val_loss", 0.4, 0.5) is True
    assert f("val_loss", 0.6, 0.5) is False
    assert f("perplexity", 18.0, 20.0) is True
    # higher-is-better
    assert f("val_acc", 0.9, 0.85) is True
    assert f("val_acc", 0.8, 0.85) is False
    assert f("f1_score", 0.7, 0.6) is True
    # unknown name defaults to lower-is-better (LM headline metric is loss)
    assert f("mystery", 0.4, 0.5) is True


def test_time_to_target_requires_pytorch_ddp():
    wl = _tiny_workload(metric_target=("val_loss", 0.5))
    for fw in ("lightning", "hf-trainer"):
        runner = BaselineRunner(framework=fw, mode="time-to-target")
        with pytest.raises(ValueError, match="pytorch-ddp"):
            runner.run(wl)


def test_baseline_time_to_target_stops_early():
    wl = _tiny_workload(
        metric_target=("val_loss", 0.5),
        eval_metrics_seq=[{"val_loss": 2.0}, {"val_loss": 1.0},
                          {"val_loss": 0.4}, {"val_loss": 0.1}],
    )
    runner = BaselineRunner(framework="pytorch-ddp",
                            mode="time-to-target", max_epochs=10)
    report = runner.run(wl)
    assert report.reached_target is True
    assert report.epochs_to_target == 3       # 2.0, 1.0, 0.4<=0.5 -> stop on 3rd
    assert report.epochs_to_target < 10


def test_baseline_time_to_target_unreachable():
    wl = _tiny_workload(
        metric_target=("val_loss", 0.5),
        eval_metrics_seq=[{"val_loss": 2.0}],   # clamps -> never improves
    )
    runner = BaselineRunner(framework="pytorch-ddp",
                            mode="time-to-target", max_epochs=3)
    report = runner.run(wl)
    assert report.reached_target is False
    assert report.epochs_to_target is None


def test_baseline_fixed_mode_leaves_reached_target_none():
    # Regression guard: fixed mode never sets the time-to-target fields even
    # when the (ignored) target would be met.
    wl = _tiny_workload(
        metric_target=("val_loss", 0.5),
        eval_metrics_seq=[{"val_loss": 0.1}],
        epochs=2,
    )
    report = BaselineRunner(framework="pytorch-ddp").run(wl)
    assert report.reached_target is None
    assert report.epochs_to_target is None


def test_sakura_time_to_target_stops_early():
    wl = _tiny_workload(
        metric_target=("val_loss", 0.5),
        eval_metrics_seq=[{"val_loss": 2.0}, {"val_loss": 0.4}, {"val_loss": 0.1}],
    )
    runner = SakuraRunner(framework="pytorch-ddp", services=[],
                          mode="time-to-target", max_epochs=10)
    report = runner.run(wl)
    assert report.reached_target is True
    assert report.epochs_to_target == 2       # 2.0, 0.4<=0.5 -> stop on 2nd
    assert report.epochs_to_target < 10


def test_sakura_time_to_target_rejects_async_eval():
    from sakura.bench.__main__ import _build_services
    wl = _tiny_workload(metric_target=("val_loss", 0.5))
    services = _build_services(["async_eval:in_thread"], wl)
    runner = SakuraRunner(framework="pytorch-ddp", services=services,
                          mode="time-to-target", max_epochs=2)
    with pytest.raises(ValueError, match="async_eval"):
        runner.run(wl)


def test_cli_time_to_target_reachable(tmp_path):
    out_dir = tmp_path / "reports"
    out_dir.mkdir()
    rc = subprocess.run(
        [sys.executable, "-m", "sakura.bench", "run",
         "--workload", "mnist-mlp", "--runner", "baseline",
         "--framework", "pytorch-ddp",
         "--mode", "time-to-target",
         "--target-metric", "val_acc", "--target-value", "0.0",
         "--max-epochs", "1",
         "--output", str(out_dir)],
        capture_output=True, text=True,
    )
    assert rc.returncode == 0, f"stderr:\n{rc.stderr}"
    report = json.loads(next(out_dir.glob("*.json")).read_text())
    assert report["reached_target"] is True
    assert report["epochs_to_target"] == 1


def test_cli_time_to_target_unreachable(tmp_path):
    out_dir = tmp_path / "reports"
    out_dir.mkdir()
    rc = subprocess.run(
        [sys.executable, "-m", "sakura.bench", "run",
         "--workload", "mnist-mlp", "--runner", "baseline",
         "--framework", "pytorch-ddp",
         "--mode", "time-to-target",
         "--target-metric", "val_acc", "--target-value", "2.0",
         "--max-epochs", "1",
         "--output", str(out_dir)],
        capture_output=True, text=True,
    )
    assert rc.returncode == 0, f"stderr:\n{rc.stderr}"
    report = json.loads(next(out_dir.glob("*.json")).read_text())
    assert report["reached_target"] is False
    assert report["epochs_to_target"] is None
