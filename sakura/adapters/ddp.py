"""DDPAdapter — explicit-hook adapter for raw PyTorch DDP loops.

Unlike LightningAdapter / HFAdapter, DDPAdapter has no callback subclass.
Users invoke its methods directly from their training loop. Embeds rank
and world_size in every emitted event.
"""
from __future__ import annotations

from typing import Any, Optional

from sakura.adapters.base import Adapter
from sakura.events import (
    OnEpochBegin,
    OnEpochEnd,
    OnOptimizerStep,
    OnTrainBegin,
    OnTrainEnd,
    OnTrainStepBegin,
)
from sakura.runtime import SakuraRuntime


class DDPAdapter(Adapter):
    """Explicit-hook adapter for raw PyTorch DDP training loops."""

    def __init__(self, runtime: SakuraRuntime, *, rank: int, world_size: int):
        super().__init__(runtime)
        self._rank = int(rank)
        self._world_size = int(world_size)

    @property
    def rank(self) -> int:
        return self._rank

    @property
    def world_size(self) -> int:
        return self._world_size

    def on_train_begin(self, model: Any, optimizer: Any, train_loader: Any,
                        val_loader: Optional[Any] = None) -> None:
        self.emit(OnTrainBegin(
            model=model, optimizer=optimizer, train_loader=train_loader,
            val_loader=val_loader, rank=self._rank, world_size=self._world_size,
        ))

    def on_epoch_begin(self, epoch: int) -> None:
        self.emit(OnEpochBegin(epoch=int(epoch), rank=self._rank, world_size=self._world_size))

    def on_train_step_begin(self, model: Any, batch: Any, step: int) -> None:
        self.emit(OnTrainStepBegin(
            model=model, batch=batch, step=int(step),
            rank=self._rank, world_size=self._world_size,
        ))

    def on_optimizer_step(self, optimizer: Any) -> None:
        self.emit(OnOptimizerStep(
            optimizer=optimizer, rank=self._rank, world_size=self._world_size,
        ))

    def on_epoch_end(self, epoch: int, model: Any, optimizer: Any,
                      metrics: dict) -> None:
        self.emit(OnEpochEnd(
            epoch=int(epoch), model=model, optimizer=optimizer, metrics=dict(metrics),
            rank=self._rank, world_size=self._world_size,
        ))

    def on_train_end(self, model: Any) -> None:
        self.emit(OnTrainEnd(
            model=model, history=[], rank=self._rank, world_size=self._world_size,
        ))


__all__ = ["DDPAdapter"]
