"""Tests for the typed event payloads emitted by adapters and consumed by services."""
from __future__ import annotations

from sakura.events import (
    Event,
    OnEpochBegin,
    OnEpochEnd,
    OnError,
    OnOptimizerStep,
    OnSave,
    OnTrainBegin,
    OnTrainEnd,
    OnTrainStepBegin,
)


class TestEventTypes:
    def test_on_train_begin_carries_required_fields(self):
        evt = OnTrainBegin(model="model_obj", optimizer="optim_obj", train_loader="loader",
                           val_loader=None, rank=0, world_size=1)
        assert evt.model == "model_obj"
        assert evt.optimizer == "optim_obj"
        assert evt.train_loader == "loader"
        assert evt.val_loader is None
        assert evt.rank == 0
        assert evt.world_size == 1
        assert isinstance(evt, Event)

    def test_on_epoch_end_carries_metrics(self):
        evt = OnEpochEnd(epoch=3, model="m", optimizer="o", metrics={"val_loss": 0.21}, rank=0, world_size=2)
        assert evt.epoch == 3
        assert evt.metrics == {"val_loss": 0.21}
        assert evt.rank == 0
        assert evt.world_size == 2

    def test_on_train_step_begin(self):
        evt = OnTrainStepBegin(model="m", batch=("x", "y"), step=42, rank=1, world_size=4)
        assert evt.step == 42
        assert evt.batch == ("x", "y")

    def test_on_optimizer_step(self):
        evt = OnOptimizerStep(optimizer="o", rank=0, world_size=1)
        assert evt.optimizer == "o"

    def test_on_save_carries_path_and_state_dict(self):
        evt = OnSave(path="/tmp/ckpt.pt", state_dict={"weights": 0}, rank=0, world_size=1)
        assert evt.path == "/tmp/ckpt.pt"
        assert evt.state_dict == {"weights": 0}

    def test_on_train_end_carries_history(self):
        history = [{"epoch": 0, "val_loss": 0.5}, {"epoch": 1, "val_loss": 0.4}]
        evt = OnTrainEnd(model="m", history=history, rank=0, world_size=1)
        assert evt.history == history

    def test_on_error_carries_exc_and_context(self):
        exc = RuntimeError("boom")
        evt = OnError(exc=exc, context={"hook": "lightning"}, rank=0, world_size=1)
        assert evt.exc is exc
        assert evt.context["hook"] == "lightning"

    def test_event_name_classmethod_returns_consistent_string(self):
        assert OnEpochEnd.name() == "on_epoch_end"
        assert OnTrainBegin.name() == "on_train_begin"
        assert OnError.name() == "on_error"

    def test_events_are_hashable_via_id(self):
        # Two distinct constructions must compare non-equal (content-identified).
        a = OnEpochEnd(epoch=0, model="m", optimizer="o", metrics={}, rank=0, world_size=1)
        b = OnEpochEnd(epoch=1, model="m", optimizer="o", metrics={}, rank=0, world_size=1)
        assert a != b
