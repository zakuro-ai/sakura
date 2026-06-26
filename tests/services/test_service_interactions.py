"""Cross-service integration tests.

Tests the integration matrix from spec §6.8 on the cases Plan 3 ships:
- Telemetry observes events from every service.
- AsyncEval + AsyncCheckpoint can co-exist; AsyncCheckpoint best-mode reads
  metrics that adapter events deliver.
"""
from __future__ import annotations

from sakura.dispatch.in_thread import InThreadDispatcher
from sakura.events import OnEpochEnd, OnTrainEnd
from sakura.runtime import SakuraRuntime
from sakura.services.async_checkpoint import AsyncCheckpoint
from sakura.services.async_eval import AsyncEval
from sakura.services.telemetry import Telemetry


def _eval_fn(epoch, payload):
    return {"val_loss": 1.0 / (epoch + 1), "epoch": epoch}


def _writer(state, path):
    return {"path": str(path)}


class TestInteractions:
    def test_telemetry_records_all_service_events(self):
        sink: list[dict] = []
        rt = SakuraRuntime()
        rt.install(Telemetry(output=sink.append))
        rt.install(AsyncEval(eval_fn=_eval_fn, eval_payload={},
                              dispatcher=InThreadDispatcher()))
        rt.dispatch(OnEpochEnd(epoch=0, model="m", optimizer="o", metrics={},
                                rank=0, world_size=1))
        # Telemetry observed at least the OnEpochEnd dispatch.
        assert len(sink) >= 1
        assert sink[0]["event"] == "OnEpochEnd"

    def test_async_eval_and_async_checkpoint_coexist(self, tmp_path):
        dispatcher = InThreadDispatcher()
        eval_svc = AsyncEval(eval_fn=_eval_fn, eval_payload={},
                              dispatcher=dispatcher)
        ckpt = AsyncCheckpoint(
            dir=str(tmp_path),
            every="epoch",
            dispatcher=dispatcher,
            writer=_writer,
            state_provider=lambda: {"w": []},
        )
        rt = SakuraRuntime()
        rt.install(eval_svc)
        rt.install(ckpt)
        for e in range(3):
            rt.dispatch(OnEpochEnd(epoch=e, model="m", optimizer="o", metrics={},
                                    rank=0, world_size=1))
        rt.dispatch(OnTrainEnd(model="m", history=[], rank=0, world_size=1))
        assert len(eval_svc.history) == 3
        assert len(ckpt.history) == 3

    def test_async_checkpoint_best_uses_event_metrics(self, tmp_path):
        dispatcher = InThreadDispatcher()
        ckpt = AsyncCheckpoint(
            dir=str(tmp_path),
            every="best",
            metric="val_loss",
            mode="min",
            dispatcher=dispatcher,
            writer=_writer,
            state_provider=lambda: {"w": []},
        )
        rt = SakuraRuntime()
        rt.install(ckpt)
        for epoch, vl in [(0, 1.0), (1, 0.5), (2, 0.7), (3, 0.3)]:
            rt.dispatch(OnEpochEnd(epoch=epoch, model="m", optimizer="o",
                                    metrics={"val_loss": vl},
                                    rank=0, world_size=1))
        rt.dispatch(OnTrainEnd(model="m", history=[], rank=0, world_size=1))
        # Best at 0, 1, 3 — three writes.
        assert len(ckpt.history) == 3

    def test_priority_ordering_runs_telemetry_first(self):
        """Telemetry priority=0 must run before AsyncEval priority=80."""
        sequence: list[str] = []

        from sakura.service import BaseService

        class _Marker(BaseService):
            def __init__(self, name, priority):
                self.name = name
                self.priority = priority
                self.requires = ()
                super().__init__()

            def on_epoch_end(self, event):
                sequence.append(self.name)

        rt = SakuraRuntime()
        rt.install(_Marker("late", 80))
        rt.install(_Marker("early", 0))
        rt.dispatch(OnEpochEnd(epoch=0, model="m", optimizer="o", metrics={},
                                rank=0, world_size=1))
        assert sequence == ["early", "late"]
