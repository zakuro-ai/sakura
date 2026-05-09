"""DDPAdapter: explicit hooks for raw PyTorch DDP loops."""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from sakura.adapters.ddp import DDPAdapter
from sakura.runtime import SakuraRuntime
from sakura.service import BaseService


class _EventCollector(BaseService):
    name = "collector"
    priority = 10

    def __init__(self):
        super().__init__()
        self.events: list[tuple[str, int]] = []  # (event_type, rank)

    def on_event(self, event):
        self.events.append((type(event).__name__, event.rank))


def test_ddp_adapter_emits_events_with_correct_rank_and_world_size():
    rt = SakuraRuntime()
    collector = _EventCollector()
    rt.install(collector)
    adapter = DDPAdapter(rt, rank=0, world_size=4)

    fake_model = object()
    fake_opt = object()
    fake_loader = object()

    adapter.on_train_begin(fake_model, fake_opt, fake_loader)
    adapter.on_epoch_begin(0)
    adapter.on_train_step_begin(fake_model, batch=("x", "y"), step=0)
    adapter.on_optimizer_step(fake_opt)
    adapter.on_epoch_end(0, fake_model, fake_opt, metrics={"val_loss": 0.3})
    adapter.on_train_end(fake_model)

    expected = [
        ("OnTrainBegin", 0),
        ("OnEpochBegin", 0),
        ("OnTrainStepBegin", 0),
        ("OnOptimizerStep", 0),
        ("OnEpochEnd", 0),
        ("OnTrainEnd", 0),
    ]
    assert collector.events == expected


def test_ddp_adapter_passes_world_size_through():
    rt = SakuraRuntime()
    seen_world: list[int] = []

    class _W(BaseService):
        name = "w"
        priority = 10

        def on_epoch_end(self, event):
            seen_world.append(event.world_size)

    rt.install(_W())
    adapter = DDPAdapter(rt, rank=2, world_size=8)
    adapter.on_epoch_end(epoch=0, model=None, optimizer=None, metrics={})
    assert seen_world == [8]
