"""AsyncCheckpoint: dispatches state-dict writes at configured triggers."""
from __future__ import annotations

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
