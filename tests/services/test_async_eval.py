"""AsyncEval: dispatches eval_fn at epoch end, gathers result, records to history."""
from __future__ import annotations

import threading
import time

import pytest

from sakura.dispatch import ThreadDispatcher
from sakura.dispatch.in_thread import InThreadDispatcher
from sakura.events import OnEpochEnd, OnTrainEnd
from sakura.runtime import SakuraRuntime
from sakura.services.async_eval import AsyncEval


def _eval_fn(epoch: int, payload: dict):
    return {"val_loss": 1.0 / (epoch + 1), "epoch": epoch, **payload}


class TestAsyncEval:
    def test_dispatches_eval_at_epoch_end_and_records_result(self):
        dispatcher = InThreadDispatcher()
        svc = AsyncEval(eval_fn=_eval_fn, eval_payload={"tag": "v"}, dispatcher=dispatcher)
        rt = SakuraRuntime()
        rt.install(svc)
        rt.dispatch(OnEpochEnd(epoch=0, model="m", optimizer="o", metrics={},
                                rank=0, world_size=1))
        rt.dispatch(OnEpochEnd(epoch=1, model="m", optimizer="o", metrics={},
                                rank=0, world_size=1))
        rt.dispatch(OnTrainEnd(model="m", history=[], rank=0, world_size=1))
        h = svc.history
        assert len(h) == 2
        assert h[0]["val_loss"] == pytest.approx(1.0)
        assert h[1]["val_loss"] == pytest.approx(0.5)
        assert h[0]["tag"] == "v"

    def test_rank_nonzero_is_noop(self):
        dispatcher = InThreadDispatcher()
        svc = AsyncEval(eval_fn=_eval_fn, eval_payload={}, dispatcher=dispatcher)
        rt = SakuraRuntime()
        rt.install(svc)
        rt.dispatch(OnEpochEnd(epoch=0, model="m", optimizer="o", metrics={},
                                rank=1, world_size=4))
        assert svc.history == []

    def test_every_n_skips_intermediate_epochs(self):
        dispatcher = InThreadDispatcher()
        svc = AsyncEval(eval_fn=_eval_fn, eval_payload={}, dispatcher=dispatcher, every=2)
        rt = SakuraRuntime()
        rt.install(svc)
        for e in range(5):
            rt.dispatch(OnEpochEnd(epoch=e, model="m", optimizer="o", metrics={},
                                    rank=0, world_size=1))
        rt.dispatch(OnTrainEnd(model="m", history=[], rank=0, world_size=1))
        # Only epochs 0, 2, 4 dispatched.
        assert [r["epoch"] for r in svc.history] == [0, 2, 4]

    def test_priority_is_80(self):
        dispatcher = InThreadDispatcher()
        svc = AsyncEval(eval_fn=_eval_fn, eval_payload={}, dispatcher=dispatcher)
        assert svc.priority == 80
        assert svc.name == "async_eval"

    def test_thread_dispatcher_overlaps_eval_with_caller_work(self):
        """AsyncEval+ThreadDispatcher: while eval runs on a background thread,
        the caller can do work in parallel. Wallclock < (train_work + eval_work).
        """
        eval_started = threading.Event()
        eval_done = threading.Event()

        def slow_eval(epoch: int, payload):
            eval_started.set()
            time.sleep(0.15)  # 150ms "eval"
            eval_done.set()
            return {"val_loss": float(epoch), "epoch": epoch}

        dispatcher = ThreadDispatcher(max_workers=1)
        svc = AsyncEval(eval_fn=slow_eval, eval_payload={}, dispatcher=dispatcher,
                        max_pending=1, on_backpressure="block")
        rt = SakuraRuntime()
        rt.install(svc)

        t0 = time.perf_counter()
        # Submit one eval at end of epoch 0.
        rt.dispatch(OnEpochEnd(epoch=0, model="m", optimizer="o", metrics={},
                                rank=0, world_size=1))
        # Wait for the eval thread to actually start so this isn't a race.
        assert eval_started.wait(timeout=1.0), "eval did not start within 1s"
        # The caller (us) is free to do work — simulate 100ms of "training".
        time.sleep(0.1)
        caller_resumed = time.perf_counter()
        # Drain at train_end — blocks for the remaining eval time.
        rt.dispatch(OnTrainEnd(model="m", history=[], rank=0, world_size=1))
        total = time.perf_counter() - t0

        # The caller resumed after ~100ms (its sleep), not after waiting for
        # the full 150ms eval — that's the overlap.
        caller_elapsed = caller_resumed - t0
        assert caller_elapsed < 0.13, (
            f"caller did not overlap with eval: returned after {caller_elapsed*1000:.0f}ms "
            f"(threshold 130ms)"
        )
        # Total wallclock < serial sum (100ms + 150ms = 250ms). Allow slack.
        assert total < 0.22, f"total {total*1000:.0f}ms exceeded overlap budget (220ms)"
        assert eval_done.is_set()
        assert svc.history[0]["epoch"] == 0
        dispatcher.shutdown()

    def test_backpressure_skip_records_skip_marker(self):
        """Backpressure='skip' drops the eval and records the skip in history."""

        class _BackpressuredDispatcher(InThreadDispatcher):
            def __init__(self):
                self._n = 0

            def submit(self, callable, *args, **kwargs):
                self._n += 1
                # Simulate "saturated after the first submission" — second call raises
                # the canonical backpressure exception.
                if self._n >= 2:
                    raise BackpressureSaturatedError()
                return super().submit(callable, *args, **kwargs)

        from sakura.services.async_eval import BackpressureSaturatedError

        dispatcher = _BackpressuredDispatcher()
        svc = AsyncEval(eval_fn=_eval_fn, eval_payload={}, dispatcher=dispatcher,
                        on_backpressure="skip")
        rt = SakuraRuntime()
        rt.install(svc)
        rt.dispatch(OnEpochEnd(epoch=0, model="m", optimizer="o", metrics={},
                                rank=0, world_size=1))
        rt.dispatch(OnEpochEnd(epoch=1, model="m", optimizer="o", metrics={},
                                rank=0, world_size=1))
        rt.dispatch(OnTrainEnd(model="m", history=[], rank=0, world_size=1))
        # First epoch: real eval. Second epoch: skipped.
        assert len(svc.history) == 2
        assert "val_loss" in svc.history[0]
        assert svc.history[1].get("skipped") is True
        assert svc.history[1]["epoch"] == 1
