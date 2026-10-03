"""sakura-bench CLI wiring fixes (P0 bug #2 AC factory + bug #1 adaptive gate).

Exercises the two `_build_services` wiring fixes in sakura/bench/__main__.py:
  - activation_checkpoint must target the workload's Block type (not ()).
  - async_eval must arm the adaptive no-net-negative gate (sync_eval_fn +
    device_eval_fn + adaptive=True + total_epochs), so AsyncEval starts in the
    warmup 'profile' mode and only goes async when overlap measurably helps.
"""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from sakura.bench.__main__ import _build_services, _make_async_eval
from sakura.bench.workloads.gpt2 import Block, make_workload


def _gpt2_smoke_workload():
    # make_workload returns *callables*; no model is built here, so this is
    # cheap and CPU-only. seq_len/batch/batches kept tiny on purpose.
    return make_workload(size="124m", seq_len=64, batch_size=2, epochs=3,
                         synthetic=True, n_train_batches=2, n_val_batches=1)


# ---- bug #2: activation_checkpoint factory targets the workload's Block ----
def test_build_activation_checkpoint_targets_block_type():
    wl = _gpt2_smoke_workload()
    services = _build_services(["activation_checkpoint"], wl)
    assert [s.name for s in services] == ["activation_checkpoint"]
    ac = services[0]
    assert ac._target_types == (Block,)


def test_build_activation_checkpoint_empty_block_types_raises():
    wl = _gpt2_smoke_workload()
    wl.block_types = ()  # Workload is a non-frozen dataclass; simulate a forgotten declaration
    with pytest.raises(ValueError, match=r"block_types"):
        _build_services(["activation_checkpoint"], wl)


# ---- bug #1: async_eval arms the adaptive no-net-negative gate ----
def test_build_async_eval_enables_adaptive_gate():
    wl = _gpt2_smoke_workload()
    services = _build_services(["async_eval"], wl)
    assert [s.name for s in services] == ["async_eval"]
    svc = services[0]
    # A sync_eval_fn is present + adaptive=True -> _can_gate True (async_eval.py).
    assert svc._can_gate is True
    # The bench factory also supplies a device_eval_fn, so the warmup
    # resource-allocation profiler is armed and the service starts by measuring
    # (train + eval-on-device vs eval-on-CPU) before committing to sync_device
    # or async — instead of assuming overlap helps.
    assert svc._can_profile is True
    assert svc.mode == "profile"
    assert svc._total_epochs == wl.epochs
    svc._dispatcher.shutdown()  # default 'thread' kind spawns a ThreadDispatcher


def test_make_async_eval_sync_bridge_runs_workload_eval_fn():
    """The sync_eval_fn bridge rebuilds a CPU copy and calls workload.eval_fn.

    sync_eval_fn creates a CPU copy of the live model (via state_dict) before
    calling workload.eval_fn, so calibration works on GPU training runs where
    the workload's data lives on CPU.  We swap make_model for a cheap factory
    to avoid building GPT-2 124m weights in a unit test.
    """
    wl = _gpt2_smoke_workload()
    seen = {}

    def spy_eval_fn(model, loader):
        seen["called"] = True
        return {"val_loss": 1.0, "perplexity": 2.7}

    wl.eval_fn = spy_eval_fn  # non-frozen dataclass
    # Cheap make_model so sync_eval_fn can build its CPU copy quickly.
    wl.make_model = lambda: torch.nn.Linear(2, 2)
    svc = _make_async_eval("in_thread", wl)  # InThreadDispatcher: no thread to reap
    real_model = torch.nn.Linear(2, 2)
    out = svc._sync_eval_fn(real_model, None)
    assert seen.get("called") is True
    assert out == {"val_loss": 1.0, "perplexity": 2.7}
