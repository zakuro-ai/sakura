"""ZeRO1 — optimizer-state sharding stage 1.

Plan 3 shipped single-rank passthrough. Plan 5 adds the multi-rank path
via sakura.zero.ShardedOptimizer (cyclic param dealing + broadcast).
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
        self._sharded: Optional[Any] = None

    def on_train_begin(self, event: OnTrainBegin):
        self._original_optimizer = event.optimizer
        if event.world_size > 1:
            from sakura.zero.sharded_optimizer import ShardedOptimizer
            self._sharded = ShardedOptimizer(
                event.optimizer, process_group=self._process_group
            )

    def on_optimizer_step(self, event: OnOptimizerStep):
        if self._sharded is not None:
            self._sharded.step()
        else:
            opt = event.optimizer
            if hasattr(opt, "step"):
                opt.step()

    def on_train_end(self, event: OnTrainEnd):
        # Nothing to restore — underlying optimizer was never replaced; we
        # just kept a wrapper alongside.
        self._sharded = None

    def gather_state_dict(self, model) -> dict:
        return dict(model.state_dict())


__all__ = ["ZeRO1"]
