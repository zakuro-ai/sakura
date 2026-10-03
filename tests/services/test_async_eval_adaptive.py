"""AsyncEval adaptive A/B gate — the no-net-negative guarantee.

The gate measures both modes and keeps the faster one, so it cannot end up
net-negative even when async *runs* but fails to overlap (CPU saturation / a
synchronous dispatcher).
"""
from __future__ import annotations

import threading
import time

import pytest
from concurrent.futures import Future as _ConcFuture

import torch

from sakura.dispatch import ThreadDispatcher
from sakura.dispatch.in_thread import InThreadDispatcher
from sakura.events import OnEpochEnd, OnTrainBegin, OnTrainEnd
from sakura.runtime import SakuraRuntime
from sakura.services.async_eval import AsyncEval


def _model(n=512):
    return torch.nn.Linear(n, n)


class _SpyInThread(InThreadDispatcher):
    """Synchronous dispatcher (runs eval inline) — async gets NO overlap."""

    def __init__(self):
        self.submits = 0

    def submit(self, callable, *args, **kwargs):
        self.submits += 1
        return super().submit(callable, *args, **kwargs)


class _SpyThread(ThreadDispatcher):
    def __init__(self, **kw):
        super().__init__(**kw)
        self.submits = 0

    def submit(self, callable, *args, **kwargs):
        self.submits += 1
        return super().submit(callable, *args, **kwargs)


class _SlowInThread(InThreadDispatcher):
    """Synchronous dispatcher with a fixed extra latency: async eval runs inline
    AND pays the latency, so the async epoch is clearly slower than sync — the
    A/B gate must measure that and revert. (Deterministic timescale separation.)"""

    def __init__(self, extra=0.02):
        self.submits = 0
        self._extra = extra

    def submit(self, callable, *args, **kwargs):
        self.submits += 1
        r = super().submit(callable, *args, **kwargs)
        time.sleep(self._extra)
        return r


def _drive(svc, n_epochs, model, train_sleep=0.0):
    rt = SakuraRuntime()
    rt.install(svc)
    rt.dispatch(OnTrainBegin(model=model, optimizer="o", train_loader=None,
                             val_loader="vl", rank=0, world_size=1))
    for e in range(n_epochs):
        if train_sleep:
            time.sleep(train_sleep)  # simulate epoch training time
        rt.dispatch(OnEpochEnd(epoch=e, model=model, optimizer="o", metrics={},
                               rank=0, world_size=1))
    rt.dispatch(OnTrainEnd(model=model, history=[], rank=0, world_size=1))


class TestAdaptiveGate:
    def test_cheap_eval_stays_synchronous_never_dispatches_per_epoch(self):
        """Trivial eval ≪ overhead → pre-filter keeps it synchronous; the only
        submit ever made is the one-time overhead probe (never net-negative)."""
        model = _model()
        disp = _SpyInThread()
        svc = AsyncEval(eval_fn=lambda e, p: {"val_loss": 0.5}, eval_payload=None,
                        dispatcher=disp, sync_eval_fn=lambda m, vl: {"val_loss": 0.5},
                        adaptive=True, calibration_epochs=2)
        _drive(svc, n_epochs=8, model=model)
        assert svc.mode == "locked_sync"
        assert disp.submits == 1  # overhead probe only — no per-epoch async overhead
        assert len(svc.history) == 8

    @pytest.mark.timing
    def test_overlap_helps_keeps_async(self):
        """Heavy eval + a real thread (async returns immediately) → async epoch
        period is far below sync period → gate locks to async.

        Updated for gate-aware backpressure: once the gate decides async, new
        epochs may be skipped (not blocked) when the queue is saturated, so
        "async" need not appear in per-epoch decisions; the terminal mode is
        the authoritative check.
        """
        model = _model()
        disp = _SpyThread(max_workers=1)

        def slow_eval(*_a):
            time.sleep(0.03)  # 30ms eval
            return {"val_loss": 0.1}

        svc = AsyncEval(eval_fn=lambda e, p: slow_eval(), eval_payload=None,
                        dispatcher=disp, sync_eval_fn=lambda m, vl: slow_eval(),
                        adaptive=True, calibration_epochs=2, trial_epochs=2)
        _drive(svc, n_epochs=10, model=model)
        assert svc.mode == "async"
        modes = [d[1] for d in svc.decisions]
        # Gate went through trial calibration before deciding.
        assert "trial" in modes
        disp.shutdown()

    def test_async_without_overlap_reverts_to_sync(self):
        """The crucial guarantee: a synchronous dispatcher means async eval runs
        inline (no overlap), so its epoch period is NOT faster than sync — the
        A/B gate must measure that and revert to sync, never staying net-negative."""
        model = _model()
        disp = _SlowInThread(extra=0.02)  # async runs inline + 20ms -> clearly slower

        def slow_eval(*_a):
            time.sleep(0.1)               # 100ms sync eval; async epoch ~120ms > 100ms
            return {"val_loss": 0.1}

        svc = AsyncEval(eval_fn=lambda e, p: slow_eval(), eval_payload=None,
                        dispatcher=disp, sync_eval_fn=lambda m, vl: slow_eval(),
                        adaptive=True, calibration_epochs=2, trial_epochs=2, min_gain=0.03)
        _drive(svc, n_epochs=10, model=model)
        assert svc.mode == "locked_sync"  # async measurably slower -> reverted

    def test_epochs_remaining_guard_skips_trial_on_short_runs(self):
        model = _model()
        disp = _SpyThread(max_workers=1)

        def slow_eval(*_a):
            time.sleep(0.02)
            return {"val_loss": 0.1}

        # total_epochs=3, calibration=2, trial=2 -> never enough room to trial.
        svc = AsyncEval(eval_fn=lambda e, p: slow_eval(), eval_payload=None,
                        dispatcher=disp, sync_eval_fn=lambda m, vl: slow_eval(),
                        adaptive=True, calibration_epochs=2, trial_epochs=2, total_epochs=3)
        _drive(svc, n_epochs=3, model=model)
        assert svc.mode in ("sync", "locked_sync")
        assert disp.submits == 1  # only the probe; never trialled
        disp.shutdown()

    def test_cuda_model_takes_gpu_fastpath_to_async_no_trial(self):
        """On a GPU model, the gate skips the noisy CPU A/B trial and goes async
        directly (GPU train and CPU-worker eval are on disjoint hardware)."""
        class _P:
            is_cuda = True

        class _FakeCuda:
            def parameters(self):
                yield _P()

            def state_dict(self):
                return {}

        disp = _SpyInThread()

        def slow(*_a):
            time.sleep(0.01)
            return {"v": 1}

        svc = AsyncEval(eval_fn=lambda e, p: slow(), eval_payload=None, dispatcher=disp,
                        sync_eval_fn=lambda m, vl: slow(), adaptive=True, calibration_epochs=2)
        _drive(svc, n_epochs=6, model=_FakeCuda())
        assert svc.mode == "async"
        assert "trial" not in [d[1] for d in svc.decisions]  # trial skipped

    def test_no_sync_fn_is_backward_compatible_always_async(self):
        model = _model()
        disp = _SpyInThread()
        svc = AsyncEval(eval_fn=lambda e, p: {"val_loss": float(e)}, eval_payload=None,
                        dispatcher=disp, adaptive=True)
        assert svc.mode == "async"
        _drive(svc, n_epochs=3, model=model)
        assert disp.submits == 3
        assert [r["epoch"] for r in svc.history] == [0, 1, 2]

    def test_adaptive_false_is_always_async(self):
        model = _model()
        disp = _SpyInThread()
        svc = AsyncEval(eval_fn=lambda e, p: {"v": float(e)}, eval_payload=None,
                        dispatcher=disp, sync_eval_fn=lambda m, vl: {"v": 0.0}, adaptive=False)
        assert svc.mode == "async"
        _drive(svc, n_epochs=3, model=model)
        assert disp.submits == 3


class TestBackpressureGateAware:
    """Gate-aware backpressure: during calibration/trial BLOCK; locked-async SKIP.

    Regression guard for the prior bug where skip fired unconditionally on
    backpressure, corrupting the trial→async measurement so the gate could never
    lock into steady async.
    """

    def _make_svc(self, **kw) -> AsyncEval:
        defaults = dict(
            eval_fn=lambda e, p: {"val_loss": 0.1},
            eval_payload=None,
            dispatcher=InThreadDispatcher(),
            sync_eval_fn=lambda m, vl: {"val_loss": 0.1},
            adaptive=True,
            calibration_epochs=1,
            trial_epochs=2,
            max_pending=1,
            on_backpressure="skip",  # would skip in old code regardless of mode
        )
        defaults.update(kw)
        return AsyncEval(**defaults)

    def test_trial_mode_blocks_on_full_queue(self):
        """Full pending queue + mode='trial' → BLOCK (drain + dispatch), not skip.

        The old unconditional-skip broke this: skipping during trial records a
        short epoch cost (no eval ran) that corrupts the async-vs-sync comparison
        and prevents the gate from ever locking to async.
        """
        svc = self._make_svc()
        # Force the gate into trial mode (skip calibration handshake).
        svc._mode = "trial"
        svc._sync_evals = 1
        svc._sync_costs = [50.0]

        # Pre-fill the pending queue (max_pending=1) with a future that resolves
        # after a brief pause — _block_oldest() will call .result() and drain it.
        blocking_fut = _ConcFuture()

        def _resolve():
            time.sleep(0.005)  # 5 ms
            blocking_fut.set_result({"val_loss": 0.99})

        resolver = threading.Thread(target=_resolve, daemon=True)

        model = _model()
        svc.on_train_begin(
            OnTrainBegin(model=model, optimizer="o", train_loader=None,
                         val_loader="vl", rank=0, world_size=1)
        )
        # Inject a not-yet-done future so _collect_done() won't drain it.
        svc._pending.append((-1, blocking_fut))  # type: ignore[arg-type]

        resolver.start()
        svc.on_epoch_end(
            OnEpochEnd(epoch=5, model=model, optimizer="o",
                       metrics={}, rank=0, world_size=1)
        )
        resolver.join()

        decisions = svc.decisions  # list[tuple[int, str]]
        assert not any(d[1] == "skip" for d in decisions), (
            f"trial mode must BLOCK on backpressure, not skip; decisions={decisions}"
        )
        assert any(d[1] == "trial" for d in decisions), (
            f"epoch 5 should have been dispatched as 'trial'; decisions={decisions}"
        )

        # Drain remaining pending on cleanup.
        svc.on_train_end(OnTrainEnd(model=model, history=[], rank=0, world_size=1))

    def test_async_mode_skips_on_full_queue(self):
        """Full pending queue + mode='async' (locked) → SKIP, never stall training.

        Once the gate has decided async is beneficial, training throughput must
        not be compromised: skip the eval and record a 'skip' decision.
        """
        # Use a dispatcher that returns futures that never resolve on their own,
        # simulating a backlogged thread pool.
        class _NeverDoneDispatcher(InThreadDispatcher):
            def submit(self, callable, *args, **kwargs):
                return _ConcFuture()  # never resolves until we call set_result

        svc = self._make_svc(dispatcher=_NeverDoneDispatcher())
        # Force locked-async mode.
        svc._mode = "async"

        model = _model()
        svc.on_train_begin(
            OnTrainBegin(model=model, optimizer="o", train_loader=None,
                         val_loader="vl", rank=0, world_size=1)
        )
        # Pre-fill the pending queue with a hanging future (never done).
        hanging = _ConcFuture()
        svc._pending.append((0, hanging))  # type: ignore[arg-type]

        svc.on_epoch_end(
            OnEpochEnd(epoch=1, model=model, optimizer="o",
                       metrics={}, rank=0, world_size=1)
        )

        decisions = svc.decisions
        assert any(d[1] == "skip" for d in decisions), (
            f"async mode must SKIP on full queue; decisions={decisions}"
        )
        skip_hist = [h for h in svc.history if h.get("skipped")]
        assert skip_hist, "skip must be recorded in history"
        assert skip_hist[0]["reason"] == "backpressure"

        # Resolve the hanging future so on_train_end doesn't deadlock.
        hanging.set_result({"val_loss": 0.5})
        svc.on_train_end(OnTrainEnd(model=model, history=[], rank=0, world_size=1))


class TestWarmupProfiler:
    """The warmup resource-allocation profiler: when a `device_eval_fn` is
    supplied it MEASURES train + eval-on-device + eval-on-CPU during a short
    warmup, then commits to `sync_device` or `async` — instead of assuming
    overlap helps. Grounded in the ASR case where CPU eval ≫ device eval."""

    def test_picks_sync_device_when_cpu_eval_dominates(self):
        # device eval is cheap; the CPU-eval estimate is huge (RNN-on-CPU case)
        # -> overlapping the CPU eval can't beat just evaluating on-device.
        model = _model()
        disp = _SpyThread(max_workers=1)
        svc = AsyncEval(
            eval_fn=lambda e, p: {"val_loss": 0.1}, eval_payload=None, dispatcher=disp,
            sync_eval_fn=lambda m, vl: {"val_loss": 0.1},
            device_eval_fn=lambda m, vl: {"val_loss": 0.1},  # fast on-device eval
            cpu_probe_fn=lambda m: 1000.0,                   # CPU eval ≈ 1s (estimate)
            adaptive=True, calibration_epochs=2, total_epochs=8,
        )
        assert svc.mode == "profile"
        _drive(svc, n_epochs=8, model=model, train_sleep=0.005)
        assert svc.mode == "sync_device"
        assert svc.allocation["choice"] == "sync_device"
        assert svc.allocation["t_eval_cpu_ms"] == 1000.0
        assert disp.submits == 1  # only the one-time overhead probe; no per-epoch async
        assert len(svc.history) == 8
        disp.shutdown()

    @pytest.mark.timing
    def test_picks_async_when_overlap_wins(self):
        # device eval is non-trivial (sync_device would pay it every epoch) but
        # the CPU eval is cheap enough to fully hide under a longer train epoch.
        model = _model()
        disp = _SpyThread(max_workers=1)

        def dev_eval(*_a):
            time.sleep(0.02)  # 20ms on-device eval
            return {"val_loss": 0.1}

        svc = AsyncEval(
            eval_fn=lambda e, p: {"val_loss": 0.1}, eval_payload=None, dispatcher=disp,
            sync_eval_fn=lambda m, vl: {"val_loss": 0.1},
            device_eval_fn=lambda m, vl: dev_eval(),
            cpu_probe_fn=lambda m: 20.0,  # 20ms CPU eval << 50ms train -> hides
            adaptive=True, calibration_epochs=2, total_epochs=10,
        )
        _drive(svc, n_epochs=10, model=model, train_sleep=0.05)
        assert svc.mode == "async"
        assert svc.allocation["choice"] == "async"
        disp.shutdown()
