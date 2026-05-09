"""ZeRO1 — optimizer-state sharding (stage 1) — Plan 3 single-rank stub.

Plan 3 ships the wrap structure + correctness for world_size=1 (passthrough).
Multi-rank sharding (cyclic dealing of param groups across ranks + all_gather
of updated weights) is Plan 5 work where real DDP is exercised.
"""
from __future__ import annotations

from typing import Any, Optional

from sakura.events import OnOptimizerStep, OnTrainBegin, OnTrainEnd
from sakura.service import BaseService


class ZeRO1(BaseService):
    name = "zero1"
    priority = 30

    def __init__(
        self,
        *,
        process_group: Optional[Any] = None,
        bucket_size_mb: int = 16,
        cpu_offload: bool = False,
    ):
        super().__init__()
        self._process_group = process_group
        self._bucket_size_mb = bucket_size_mb
        self._cpu_offload = cpu_offload
        self._original_optimizer: Optional[Any] = None

    def on_train_begin(self, event: OnTrainBegin):
        # World-size==1 fast path: passthrough, no sharding.
        if event.world_size <= 1:
            self._original_optimizer = event.optimizer
            return
        # Multi-rank: shard optimizer state cyclically across ranks.
        # Plan 3 placeholder: full implementation in Plan 5.
        # For now we still install but no actual sharding.
        self._original_optimizer = event.optimizer

    def on_optimizer_step(self, event: OnOptimizerStep):
        # World-size==1: just passthrough to opt.step().
        # Multi-rank: would step on local shard then all_gather updated weights.
        opt = event.optimizer
        if hasattr(opt, "step"):
            opt.step()

    def on_train_end(self, event: OnTrainEnd):
        # Restore unsharded optimizer (no-op for world_size==1).
        pass

    def gather_state_dict(self, model) -> dict:
        """Return the full (gathered) state dict. World_size==1 returns model.state_dict() directly."""
        return dict(model.state_dict())


__all__ = ["ZeRO1"]
