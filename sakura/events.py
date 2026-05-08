"""Typed event payloads emitted by adapters and consumed by services.

Every event carries (rank, world_size) so DDP-aware services can branch on
event.rank without each adapter doing the bookkeeping. Single-process runs
get rank=0, world_size=1.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass(frozen=True)
class Event:
    """Base type for all event payloads. Every event carries rank/world_size."""
    rank: int
    world_size: int

    @classmethod
    def name(cls) -> str:
        """Convert ClassName like OnEpochEnd -> on_epoch_end."""
        out = []
        for i, c in enumerate(cls.__name__):
            if c.isupper() and i > 0:
                out.append("_")
            out.append(c.lower())
        return "".join(out)


@dataclass(frozen=True)
class OnTrainBegin(Event):
    model: Any
    optimizer: Any
    train_loader: Any
    val_loader: Optional[Any] = None


@dataclass(frozen=True)
class OnEpochBegin(Event):
    epoch: int


@dataclass(frozen=True)
class OnTrainStepBegin(Event):
    model: Any
    batch: Any
    step: int


@dataclass(frozen=True)
class OnOptimizerStep(Event):
    optimizer: Any


@dataclass(frozen=True)
class OnEpochEnd(Event):
    epoch: int
    model: Any
    optimizer: Any
    metrics: dict = field(default_factory=dict)


@dataclass(frozen=True)
class OnSave(Event):
    path: str
    state_dict: Any


@dataclass(frozen=True)
class OnTrainEnd(Event):
    model: Any
    history: list


@dataclass(frozen=True)
class OnError(Event):
    exc: BaseException
    context: dict = field(default_factory=dict)


__all__ = [
    "Event",
    "OnTrainBegin",
    "OnEpochBegin",
    "OnTrainStepBegin",
    "OnOptimizerStep",
    "OnEpochEnd",
    "OnSave",
    "OnTrainEnd",
    "OnError",
]
