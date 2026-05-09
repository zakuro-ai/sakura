"""LightningAdapter — lightning.Callback that emits Sakura runtime events.

Hook mapping (per spec §10.1):
  setup-after / on_train_start  → on_train_begin
  on_train_epoch_start          → on_epoch_begin
  on_train_batch_start          → on_train_step_begin
  on_before_optimizer_step      → on_optimizer_step
  on_train_epoch_end            → on_epoch_end (carries trainer.callback_metrics)
  on_train_end                  → on_train_end
  on_exception                  → on_error
"""
from __future__ import annotations

from typing import Any

try:
    from lightning.pytorch import Callback
except ImportError:  # pragma: no cover
    class Callback:  # type: ignore[no-redef]
        pass

from sakura.adapters.base import Adapter
from sakura.events import (
    OnEpochBegin,
    OnEpochEnd,
    OnError,
    OnOptimizerStep,
    OnTrainBegin,
    OnTrainEnd,
    OnTrainStepBegin,
)
from sakura.runtime import SakuraRuntime


class LightningAdapter(Callback, Adapter):
    """Lightning callback that translates framework hooks into Sakura events."""

    def __init__(self, runtime: SakuraRuntime, *, rank: int = 0, world_size: int = 1):
        # NOTE: we call Adapter.__init__ explicitly because Callback's __init__
        # may not accept positional args.
        Callback.__init__(self)
        Adapter.__init__(self, runtime)
        self._rank = rank
        self._world_size = world_size
        self._collected: list[dict] = []

    # ........................................................... lifecycle

    def on_train_start(self, trainer, pl_module):
        opt = trainer.optimizers[0] if trainer.optimizers else None
        self.emit(OnTrainBegin(
            model=pl_module,
            optimizer=opt,
            train_loader=getattr(trainer, "train_dataloader", None),
            val_loader=getattr(trainer, "val_dataloaders", None),
            rank=self._rank,
            world_size=self._world_size,
        ))

    def on_train_epoch_start(self, trainer, pl_module):
        self.emit(OnEpochBegin(
            epoch=int(trainer.current_epoch),
            rank=self._rank,
            world_size=self._world_size,
        ))

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        self.emit(OnTrainStepBegin(
            model=pl_module, batch=batch, step=int(batch_idx),
            rank=self._rank, world_size=self._world_size,
        ))

    def on_before_optimizer_step(self, trainer, pl_module, optimizer):
        self.emit(OnOptimizerStep(
            optimizer=optimizer, rank=self._rank, world_size=self._world_size,
        ))

    def on_train_epoch_end(self, trainer, pl_module):
        opt = trainer.optimizers[0] if getattr(trainer, "optimizers", None) else None
        metrics = dict(getattr(trainer, "callback_metrics", {}) or {})
        # Convert any tensor metrics to float for telemetry serializability.
        clean = {}
        for k, v in metrics.items():
            try:
                clean[k] = float(v)
            except Exception:
                clean[k] = v
        self.emit(OnEpochEnd(
            epoch=int(trainer.current_epoch),
            model=pl_module, optimizer=opt, metrics=clean,
            rank=self._rank, world_size=self._world_size,
        ))

    def on_train_end(self, trainer, pl_module):
        self.emit(OnTrainEnd(
            model=pl_module, history=list(self._collected),
            rank=self._rank, world_size=self._world_size,
        ))

    def on_exception(self, trainer, pl_module, exception):
        self.emit(OnError(
            exc=exception, context={"hook": "lightning"},
            rank=self._rank, world_size=self._world_size,
        ))


__all__ = ["LightningAdapter"]
