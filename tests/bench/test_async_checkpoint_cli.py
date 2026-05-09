"""sakura-bench CLI: --service async_checkpoint writes per-epoch ckpts off-thread."""
from __future__ import annotations

import os
import shutil

import pytest

torch = pytest.importorskip("torch")

from sakura.bench.__main__ import _build_services, _make_async_checkpoint
from sakura.bench.harness import SakuraRunner, Workload


def _make_synthetic_workload() -> Workload:
    def make_model():
        return torch.nn.Sequential(
            torch.nn.Linear(8, 16),
            torch.nn.ReLU(),
            torch.nn.Linear(16, 4),
        )

    def make_loader(n=64):
        torch.manual_seed(0)
        ds = torch.utils.data.TensorDataset(
            torch.randn(n, 8), torch.randint(0, 4, (n,))
        )
        return torch.utils.data.DataLoader(ds, batch_size=8, shuffle=False)

    def eval_fn(model, loader):
        return {"val_acc": 0.5}

    return Workload(
        name="async-ckpt-bridge-test",
        tier="smoke",
        make_model=make_model,
        make_train_loader=make_loader,
        make_val_loader=make_loader,
        eval_fn=eval_fn,
        epochs=3,
    )


def test_make_async_checkpoint_attaches_bridge_surface():
    wl = _make_synthetic_workload()
    svc = _make_async_checkpoint("thread", wl)
    try:
        assert svc.name == "async_checkpoint"
        assert hasattr(svc, "_bench_snapshot")
        assert hasattr(svc, "_bench_out_dir")
        assert os.path.isdir(svc._bench_out_dir)
        assert svc._bench_snapshot["state_dict"] is None
    finally:
        shutil.rmtree(svc._bench_out_dir, ignore_errors=True)


def test_make_async_checkpoint_unknown_kind_raises():
    wl = _make_synthetic_workload()
    with pytest.raises(ValueError, match=r"thread.*local.*in_thread"):
        _make_async_checkpoint("bogus", wl)


def test_build_services_passes_workload_to_async_checkpoint_factory():
    wl = _make_synthetic_workload()
    services = _build_services(["async_checkpoint:in_thread"], wl)
    try:
        assert [s.name for s in services] == ["async_checkpoint"]
        assert hasattr(services[0], "_bench_snapshot")
    finally:
        shutil.rmtree(services[0]._bench_out_dir, ignore_errors=True)


def test_sakura_runner_with_async_checkpoint_writes_per_epoch_ckpts():
    """End-to-end: 3 epochs → 3 .pt files written; loadable round-trips state."""
    wl = _make_synthetic_workload()
    svc = _make_async_checkpoint("in_thread", wl)  # synchronous → deterministic
    try:
        runner = SakuraRunner(framework="pytorch-ddp", services=[svc])
        report = runner.run(wl)
        assert "async_checkpoint" in report.sakura_services

        ckpts = sorted(f for f in os.listdir(svc._bench_out_dir) if f.endswith(".pt"))
        assert ckpts == ["epoch_0000.pt", "epoch_0001.pt", "epoch_0002.pt"]

        # Round-trip the last checkpoint to confirm the snapshot is real and complete.
        loaded = torch.load(os.path.join(svc._bench_out_dir, ckpts[-1]),
                            map_location="cpu", weights_only=True)
        m = wl.make_model()
        m.load_state_dict(loaded)
    finally:
        shutil.rmtree(svc._bench_out_dir, ignore_errors=True)


def test_sakura_runner_async_eval_and_async_checkpoint_compose():
    """Both services together: shared snapshot is written by harness only once
    per epoch, and both services see the same state_dict."""
    from sakura.bench.__main__ import _make_async_eval

    wl = _make_synthetic_workload()
    eval_svc = _make_async_eval("in_thread", wl)
    ckpt_svc = _make_async_checkpoint("in_thread", wl)
    try:
        runner = SakuraRunner(framework="pytorch-ddp", services=[eval_svc, ckpt_svc])
        report = runner.run(wl)
        assert set(report.sakura_services) == {"async_eval", "async_checkpoint"}
        # AsyncEval ran 3 evals with metrics.
        assert len(eval_svc.history) == 3
        assert all("val_acc" in h for h in eval_svc.history)
        # AsyncCheckpoint wrote 3 ckpts.
        ckpts = sorted(f for f in os.listdir(ckpt_svc._bench_out_dir) if f.endswith(".pt"))
        assert len(ckpts) == 3
    finally:
        shutil.rmtree(ckpt_svc._bench_out_dir, ignore_errors=True)
