"""HFAdapter — transformers.TrainerCallback that emits Sakura runtime events."""
from __future__ import annotations

try:
    from transformers import TrainerCallback
except ImportError:  # pragma: no cover
    class TrainerCallback:  # type: ignore[no-redef]
        pass

from sakura.adapters.base import Adapter
from sakura.events import (
    OnEpochEnd,
    OnOptimizerStep,
    OnTrainBegin,
    OnTrainEnd,
    OnTrainStepBegin,
)
from sakura.runtime import SakuraRuntime


class HFAdapter(TrainerCallback, Adapter):
    """transformers.TrainerCallback that translates HF Trainer hooks into Sakura events."""

    min_transformers_version: str = "4.38"

    def __init__(self, runtime: SakuraRuntime, *, rank: int = 0, world_size: int = 1):
        TrainerCallback.__init__(self)
        Adapter.__init__(self, runtime)
        self._rank = rank
        self._world_size = world_size

    # ........................................................... lifecycle

    def on_train_begin(self, args, state, control, **kw):
        self.emit(OnTrainBegin(
            model=kw.get("model"),
            optimizer=kw.get("optimizer"),
            train_loader=kw.get("train_dataloader"),
            val_loader=kw.get("eval_dataloader"),
            rank=self._rank, world_size=self._world_size,
        ))

    def on_step_begin(self, args, state, control, **kw):
        self.emit(OnTrainStepBegin(
            model=kw.get("model"),
            batch=kw.get("inputs"),
            step=int(getattr(state, "global_step", 0)),
            rank=self._rank, world_size=self._world_size,
        ))

    def on_pre_optimizer_step(self, args, state, control, **kw):
        self.emit(OnOptimizerStep(
            optimizer=kw.get("optimizer"),
            rank=self._rank, world_size=self._world_size,
        ))

    def on_epoch_end(self, args, state, control, **kw):
        # Pull most-recent metrics from state.log_history if present.
        metrics: dict = {}
        log = getattr(state, "log_history", None)
        if log:
            metrics = dict(log[-1])
        self.emit(OnEpochEnd(
            epoch=int(state.epoch) if state.epoch is not None else 0,
            model=kw.get("model"),
            optimizer=kw.get("optimizer"),
            metrics=metrics,
            rank=self._rank, world_size=self._world_size,
        ))

    def on_train_end(self, args, state, control, **kw):
        self.emit(OnTrainEnd(
            model=kw.get("model"),
            history=list(getattr(state, "log_history", []) or []),
            rank=self._rank, world_size=self._world_size,
        ))


__all__ = ["HFAdapter"]
