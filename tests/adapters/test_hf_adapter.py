"""HFAdapter: maps transformers.TrainerCallback hooks to runtime events."""
from __future__ import annotations

import pytest

transformers = pytest.importorskip("transformers")

from sakura.adapters.huggingface import HFAdapter
from sakura.runtime import SakuraRuntime
from sakura.service import BaseService


class _EventCollector(BaseService):
    name = "collector"
    priority = 10

    def __init__(self):
        super().__init__()
        self.events: list[str] = []
        self.metrics: list[dict] = []

    def on_train_begin(self, event):
        self.events.append("OnTrainBegin")

    def on_train_step_begin(self, event):
        self.events.append("OnTrainStepBegin")

    def on_optimizer_step(self, event):
        self.events.append("OnOptimizerStep")

    def on_epoch_end(self, event):
        self.events.append("OnEpochEnd")
        self.metrics.append(dict(event.metrics))

    def on_train_end(self, event):
        self.events.append("OnTrainEnd")


def test_hf_adapter_subclasses_trainer_callback():
    rt = SakuraRuntime()
    adapter = HFAdapter(rt)
    assert isinstance(adapter, transformers.TrainerCallback)


def test_hf_adapter_emits_lifecycle_events():
    rt = SakuraRuntime()
    collector = _EventCollector()
    rt.install(collector)
    adapter = HFAdapter(rt)

    class _State:
        epoch = 1.0
        log_history = [{"val_loss": 0.21, "step": 100}]
        global_step = 100

    args = object()
    state = _State()
    control = object()
    fake_model = object()
    fake_optimizer = object()
    fake_loader = object()

    # Lightning calls these in sequence:
    adapter.on_train_begin(args, state, control,
                            model=fake_model, optimizer=fake_optimizer,
                            train_dataloader=fake_loader)
    adapter.on_step_begin(args, state, control,
                           model=fake_model, inputs={"x": 1})
    adapter.on_pre_optimizer_step(args, state, control,
                                    optimizer=fake_optimizer)
    adapter.on_epoch_end(args, state, control,
                          model=fake_model, optimizer=fake_optimizer)
    adapter.on_train_end(args, state, control, model=fake_model)

    assert collector.events == [
        "OnTrainBegin", "OnTrainStepBegin", "OnOptimizerStep",
        "OnEpochEnd", "OnTrainEnd",
    ]
    # Last log_history entry is exposed as metrics on epoch_end
    assert collector.metrics[0].get("val_loss") == 0.21
