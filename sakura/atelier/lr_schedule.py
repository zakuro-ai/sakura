"""Learning-rate schedules shared by the training backends: the multiplier of the peak rate at each
optimiser step, as a plain function (for ``torch.optim.lr_scheduler.LambdaLR``).

A linear warmup from 0, then a constant rate, or a linear or cosine decay to a floor (CONTRACTS §2.2).
Nothing here imports a training stack.
"""

from __future__ import annotations

import math

SCHEDULES = ("constant", "linear", "cosine")


def optimizer_steps(rows: int, batch_size: int, grad_accum: int, epochs: int, max_steps: int | None = None) -> int:
    """How many optimiser steps a run takes: one per ``grad_accum`` batches, the last and shorter group of an
    epoch included. ``max_steps`` (``limits.max_steps``) lowers it, so a capped run plans a whole schedule."""
    batches = -(-rows // batch_size)
    total = epochs * -(-batches // max(1, grad_accum))
    return total if max_steps is None else min(total, max_steps)


def warmup_steps(total: int, warmup_ratio: float) -> int:
    """The steps of the linear warmup: ``ceil(warmup_ratio * total)``."""
    return math.ceil(warmup_ratio * total - 1e-9)


def lr_factor(step: int, total: int, schedule: str = "cosine", warmup: int = 0, min_ratio: float = 0.0) -> float:
    """The multiplier of the peak learning rate at optimiser step ``step`` (counted from 0).

    ``step / warmup`` during the warmup. After it, with ``p`` the share of the remaining steps done:
    ``constant`` stays at 1, ``linear`` and ``cosine`` go from 1 down to ``min_ratio``, which they reach at
    ``step == total`` (the last step a run takes, ``total - 1``, is one notch above it)."""
    if schedule not in SCHEDULES:
        raise ValueError(f"lr_schedule must be one of {list(SCHEDULES)}, got {schedule!r}")
    if step < warmup:
        return step / max(warmup, 1)
    if schedule == "constant":
        return 1.0
    progress = min(1.0, (step - warmup) / max(total - warmup, 1))
    shape = 1.0 - progress if schedule == "linear" else 0.5 * (1 + math.cos(math.pi * progress))
    return min_ratio + (1.0 - min_ratio) * shape


__all__ = ["SCHEDULES", "lr_factor", "optimizer_steps", "warmup_steps"]
