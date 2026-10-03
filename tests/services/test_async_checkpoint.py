"""AsyncCheckpoint: dispatches state-dict writes at configured triggers."""
from __future__ import annotations

import pytest

from sakura.dispatch.in_thread import InThreadDispatcher
from sakura.events import OnEpochEnd, OnTrainEnd
from sakura.runtime import SakuraRuntime
from sakura.services.async_checkpoint import AsyncCheckpoint


def _capture_write(state, path):
    """Toy 'writer' that returns where it would have written."""
    return {"path": str(path), "state_keys": sorted(state.keys()) if isinstance(state, dict) else None}


class TestAsyncCheckpoint:
    def test_writes_every_epoch(self, tmp_path):
        dispatcher = InThreadDispatcher()
        state_provider = lambda: {"weights": [1, 2, 3]}
        svc = AsyncCheckpoint(
            dir=str(tmp_path),
            every="epoch",
            dispatcher=dispatcher,
            writer=_capture_write,
            state_provider=state_provider,
        )
        rt = SakuraRuntime()
        rt.install(svc)
        rt.dispatch(OnEpochEnd(epoch=0, model="m", optimizer="o", metrics={},
                                rank=0, world_size=1))
        rt.dispatch(OnEpochEnd(epoch=1, model="m", optimizer="o", metrics={},
                                rank=0, world_size=1))
        rt.dispatch(OnTrainEnd(model="m", history=[], rank=0, world_size=1))
        assert len(svc.history) == 2
        assert svc.history[0]["state_keys"] == ["weights"]

    def test_writes_every_n(self, tmp_path):
        dispatcher = InThreadDispatcher()
        state_provider = lambda: {"weights": []}
        svc = AsyncCheckpoint(
            dir=str(tmp_path),
            every=2,
            dispatcher=dispatcher,
            writer=_capture_write,
            state_provider=state_provider,
        )
        rt = SakuraRuntime()
        rt.install(svc)
        for e in range(5):
            rt.dispatch(OnEpochEnd(epoch=e, model="m", optimizer="o", metrics={},
                                    rank=0, world_size=1))
        rt.dispatch(OnTrainEnd(model="m", history=[], rank=0, world_size=1))
        assert len(svc.history) == 3  # epochs 0, 2, 4

    def test_writes_only_when_metric_improves(self, tmp_path):
        """`every='best'` writes when the named metric improves (mode='min')."""
        dispatcher = InThreadDispatcher()
        state_provider = lambda: {"w": []}
        svc = AsyncCheckpoint(
            dir=str(tmp_path),
            every="best",
            metric="val_loss",
            mode="min",
            dispatcher=dispatcher,
            writer=_capture_write,
            state_provider=state_provider,
        )
        rt = SakuraRuntime()
        rt.install(svc)
        # Simulate metrics in events:
        for epoch, val_loss in [(0, 1.0), (1, 0.8), (2, 0.9), (3, 0.5)]:
            rt.dispatch(OnEpochEnd(epoch=epoch, model="m", optimizer="o",
                                    metrics={"val_loss": val_loss},
                                    rank=0, world_size=1))
        rt.dispatch(OnTrainEnd(model="m", history=[], rank=0, world_size=1))
        # Best at epochs 0, 1, 3 (each is a new minimum).
        assert len(svc.history) == 3

    def test_priority_is_85(self, tmp_path):
        dispatcher = InThreadDispatcher()
        svc = AsyncCheckpoint(
            dir=str(tmp_path),
            dispatcher=dispatcher,
            writer=_capture_write,
            state_provider=lambda: {},
        )
        assert svc.priority == 85
        assert svc.name == "async_checkpoint"

    def test_rank_nonzero_is_noop(self, tmp_path):
        dispatcher = InThreadDispatcher()
        svc = AsyncCheckpoint(
            dir=str(tmp_path),
            dispatcher=dispatcher,
            writer=_capture_write,
            state_provider=lambda: {"w": 0},
        )
        rt = SakuraRuntime()
        rt.install(svc)
        rt.dispatch(OnEpochEnd(epoch=0, model="m", optimizer="o", metrics={},
                                rank=2, world_size=4))
        assert svc.history == []


class TestKeepAndTimeCadence:
    def _svc(self, tmp_path, **kw):
        return AsyncCheckpoint(
            dir=str(tmp_path),
            dispatcher=InThreadDispatcher(),
            state_provider=lambda: {"w": 1},
            **kw,
        )

    def _epoch_end(self, svc, epoch):
        svc.on_epoch_end(
            OnEpochEnd(epoch=epoch, model=None, optimizer=None, metrics={}, rank=0, world_size=1)
        )

    def test_keep_retains_only_newest(self, tmp_path):
        svc = self._svc(tmp_path, every="epoch", keep=2)
        for e in range(5):
            self._epoch_end(svc, e)
        svc.on_train_end(None)
        assert sorted(p.name for p in tmp_path.iterdir()) == ["epoch_0003.pt", "epoch_0004.pt"]

    def test_keep_none_keeps_everything(self, tmp_path):
        svc = self._svc(tmp_path, every="epoch", keep=None)
        for e in range(4):
            self._epoch_end(svc, e)
        svc.on_train_end(None)
        assert len(list(tmp_path.iterdir())) == 4

    def test_every_seconds_throttles_writes(self, tmp_path):
        svc = self._svc(tmp_path, every_seconds=3600, keep=None)
        for e in range(5):
            self._epoch_end(svc, e)
        svc.on_train_end(None)
        assert [p.name for p in tmp_path.iterdir()] == ["epoch_0000.pt"]

    def test_every_seconds_writes_again_after_interval(self, tmp_path):
        import time

        svc = self._svc(tmp_path, every_seconds=0.05, keep=None)
        self._epoch_end(svc, 0)
        self._epoch_end(svc, 1)  # too soon
        time.sleep(0.08)
        self._epoch_end(svc, 2)
        svc.on_train_end(None)
        assert sorted(p.name for p in tmp_path.iterdir()) == ["epoch_0000.pt", "epoch_0002.pt"]

    def test_every_seconds_must_be_positive(self, tmp_path):
        with pytest.raises(ValueError):
            self._svc(tmp_path, every_seconds=0)


class TestBackpressure:
    def test_skips_instead_of_queueing_when_writes_are_in_flight(self, tmp_path):
        import threading

        from sakura.dispatch.thread import ThreadDispatcher

        gate = threading.Event()

        def slow_writer(state, path):
            gate.wait(5)
            return {"path": path}

        calls = []
        svc = AsyncCheckpoint(
            dir=str(tmp_path),
            dispatcher=ThreadDispatcher(max_workers=1),
            state_provider=lambda: calls.append(1) or {"w": 1},
            writer=slow_writer,
            every="epoch",
            max_pending=1,
            keep=None,
        )
        for e in range(4):
            svc.on_epoch_end(
                OnEpochEnd(epoch=e, model=None, optimizer=None, metrics={}, rank=0, world_size=1)
            )
        assert len(calls) == 1  # the state is not even built for skipped triggers
        skipped = [h for h in svc.history if h.get("reason") == "backpressure"]
        assert len(skipped) == 3
        gate.set()
        svc.on_train_end(None)

    def test_max_pending_must_be_positive(self, tmp_path):
        with pytest.raises(ValueError):
            AsyncCheckpoint(
                dir=str(tmp_path),
                dispatcher=InThreadDispatcher(),
                state_provider=lambda: {},
                max_pending=0,
            )
