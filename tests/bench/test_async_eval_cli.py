"""sakura-bench CLI: --service async_eval bridges to workload.eval_fn."""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from sakura.bench.__main__ import _build_services, _make_async_eval
from sakura.bench.harness import SakuraRunner, Workload


def _make_synthetic_overlap_workload() -> Workload:
    """Tiny workload (3 epochs × 16 batches × eval=64 batches) — exercises
    the overlap path without making the test slow."""
    def make_model():
        return torch.nn.Sequential(
            torch.nn.Linear(8, 16),
            torch.nn.ReLU(),
            torch.nn.Linear(16, 4),
        )

    def make_loader(n=128):
        torch.manual_seed(0)
        ds = torch.utils.data.TensorDataset(
            torch.randn(n, 8), torch.randint(0, 4, (n,))
        )
        return torch.utils.data.DataLoader(ds, batch_size=8, shuffle=False)

    def eval_fn(model, loader):
        # Reference impl — used by baseline path only when async_eval is
        # NOT installed. The CLI bridge replaces this with its own
        # tensor-slice path inside the eval thread.
        model.eval()
        correct = total = 0
        with torch.no_grad():
            for x, y in loader:
                logits = model(x)
                correct += int((logits.argmax(dim=-1) == y).sum())
                total += int(y.numel())
        return {"val_acc": correct / max(total, 1)}

    return Workload(
        name="async-eval-bridge-test",
        tier="smoke",
        make_model=make_model,
        make_train_loader=make_loader,
        make_val_loader=make_loader,
        eval_fn=eval_fn,
        epochs=3,
    )


def test_make_async_eval_thread_dispatcher():
    wl = _make_synthetic_overlap_workload()
    svc = _make_async_eval("thread", wl)
    assert svc.name == "async_eval"
    assert hasattr(svc, "_bench_snapshot")
    assert svc._bench_snapshot["state_dict"] is None
    assert svc._bench_snapshot["val_loader"] is None


def test_make_async_eval_unknown_kind_raises():
    wl = _make_synthetic_overlap_workload()
    with pytest.raises(ValueError, match=r"thread.*local.*in_thread"):
        _make_async_eval("bogus", wl)


def test_build_services_passes_workload_to_async_eval_factory():
    wl = _make_synthetic_overlap_workload()
    services = _build_services(["telemetry", "async_eval:in_thread"], wl)
    assert [s.name for s in services] == ["telemetry", "async_eval"]
    assert hasattr(services[1], "_bench_snapshot")


def test_sakura_runner_with_async_eval_completes_and_reports_metrics():
    """End-to-end: sakura+async_eval bridges through to workload.eval_fn,
    final metrics on the RunReport are populated, and the report carries
    'async_eval' as an installed service."""
    wl = _make_synthetic_overlap_workload()
    svc = _make_async_eval("in_thread", wl)  # synchronous for test determinism
    runner = SakuraRunner(framework="pytorch-ddp", services=[svc])
    report = runner.run(wl)

    assert report.workload == "async-eval-bridge-test"
    assert "async_eval" in report.sakura_services
    assert "val_acc" in report.final_metrics
    assert isinstance(report.final_metrics["val_acc"], float)
    # AsyncEval ran one eval per epoch (3 total).
    assert len(svc.history) == 3
    assert all("val_acc" in h for h in svc.history)
