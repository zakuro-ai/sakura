"""LightningAdapter: maps lightning.Callback hooks to runtime events."""
from __future__ import annotations

import pytest

lightning = pytest.importorskip("lightning")
torch = pytest.importorskip("torch")

from sakura.adapters.lightning import LightningAdapter
from sakura.events import OnEpochEnd, OnTrainBegin, OnTrainEnd
from sakura.runtime import SakuraRuntime
from sakura.service import BaseService


class _EventCollector(BaseService):
    name = "collector"
    priority = 10

    def __init__(self):
        super().__init__()
        self.events: list[str] = []

    def on_event(self, event):
        self.events.append(type(event).__name__)


def test_lightning_adapter_subclasses_callback():
    rt = SakuraRuntime()
    adapter = LightningAdapter(rt)
    assert isinstance(adapter, lightning.pytorch.Callback)


def test_lightning_adapter_translates_lifecycle_to_events():
    """Direct invocation of the adapter's callback methods emits the right events."""
    rt = SakuraRuntime()
    collector = _EventCollector()
    rt.install(collector)
    adapter = LightningAdapter(rt)

    # Build a minimal fake "trainer" + "module" so the adapter can extract optim/loaders.
    class _FakeOptim:
        pass
    class _FakeLoader:
        pass
    class _FakeTrainer:
        current_epoch = 0
        optimizers = [_FakeOptim()]
        train_dataloader = _FakeLoader()
        val_dataloaders = None
        callback_metrics = {"val_loss": 0.5}
    class _FakeModule(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.l = torch.nn.Linear(2, 1)

    trainer = _FakeTrainer()
    module = _FakeModule()

    # Lightning calls these in sequence:
    adapter.on_train_start(trainer, module)
    adapter.on_train_epoch_start(trainer, module)
    adapter.on_train_epoch_end(trainer, module)
    adapter.on_train_end(trainer, module)

    # We should have collected: OnTrainBegin, OnEpochBegin, OnEpochEnd, OnTrainEnd.
    expected = ["OnTrainBegin", "OnEpochBegin", "OnEpochEnd", "OnTrainEnd"]
    assert collector.events == expected


def test_lightning_adapter_carries_metrics_on_epoch_end():
    rt = SakuraRuntime()
    collected: list[dict] = []

    class _Capturer(BaseService):
        name = "capturer"
        priority = 10

        def on_epoch_end(self, event):
            collected.append(dict(event.metrics))

    rt.install(_Capturer())
    adapter = LightningAdapter(rt)

    class _FakeTrainer:
        current_epoch = 5
        optimizers = []
        callback_metrics = {"val_loss": 0.123, "val_acc": 0.95}
    class _FakeModule(torch.nn.Module):
        pass

    adapter.on_train_epoch_end(_FakeTrainer(), _FakeModule())
    assert len(collected) == 1
    assert collected[0]["val_loss"] == 0.123
    assert collected[0]["val_acc"] == 0.95
