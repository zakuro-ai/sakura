"""AsyncEval: dispatches eval_fn at epoch end, gathers result, records to history."""
from __future__ import annotations

import pytest

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
