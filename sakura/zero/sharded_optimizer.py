"""ShardedOptimizer — wraps a torch.optim.Optimizer for ZeRO stage-1 sharding.

Each rank holds optimizer state for 1/world_size of the parameters.
Cyclic dealing: param i goes to rank (i % world_size). At step time, each
rank does local optimizer.step on its shard, then broadcasts the updated
weights so every rank has the full updated parameter set.

Plan 3 shipped a single-rank passthrough; this Plan 5 implementation
adds the cyclic dealing + per-shard step + all-rank broadcast.
"""
from __future__ import annotations

from typing import Any, Optional

from sakura._optional import load


class ShardedOptimizer:
    """ZeRO stage-1 sharded optimizer wrapper.

    Wraps an arbitrary torch.optim.Optimizer; each rank owns a 1/world_size
    cyclic shard of the parameters and performs optimizer.step only on its
    shard. After each step, updated parameters are broadcast across all
    ranks so every rank has the full updated parameter set.

    For world_size==1, this is a passthrough — opt.step() runs as usual,
    no broadcasting. Same for runs where torch.distributed isn't initialized.
    """

    def __init__(
        self,
        optimizer: Any,
        *,
        process_group: Optional[Any] = None,
    ):
        # torch is an optional ("training") dependency; surface a friendly
        # install hint instead of a bare ModuleNotFoundError.
        load("torch", extra="training")
        import torch.distributed as dist
        self._opt = optimizer
        self._pg = process_group
        if dist.is_available() and dist.is_initialized():
            self._world_size = dist.get_world_size(group=process_group)
            self._rank = dist.get_rank(group=process_group)
        else:
            self._world_size = 1
            self._rank = 0
        self._param_owner = self._partition_params()

    def _partition_params(self) -> dict:
        """Cyclic dealing: param i -> rank (i % world_size).

        Returns a dict mapping id(param) -> owning rank.
        """
        owners = {}
        idx = 0
        for group in self._opt.param_groups:
            for p in group["params"]:
                owners[id(p)] = idx % self._world_size
                idx += 1
        return owners

    def _all_params(self) -> list:
        out = []
        for group in self._opt.param_groups:
            out.extend(group["params"])
        return out

    def step(self, closure=None):
        """Local step on this rank's shard, then broadcast updated weights."""
        if self._world_size <= 1:
            # Single-rank passthrough.
            return self._opt.step(closure)

        import torch.distributed as dist

        # Save grads of params NOT owned by this rank, zero them so the local
        # optimizer doesn't update those params, then restore after step.
        saved_grads: dict[int, Any] = {}
        for p in self._all_params():
            if self._param_owner[id(p)] != self._rank:
                saved_grads[id(p)] = p.grad
                p.grad = None  # Tell the optimizer to skip this param.

        result = self._opt.step(closure)

        # Restore the grads we hid (other ranks need them for their step too if
        # the user calls step multiple times before zero_grad).
        for p in self._all_params():
            if id(p) in saved_grads:
                p.grad = saved_grads[id(p)]

        # Broadcast each parameter from its owning rank to all other ranks.
        for p in self._all_params():
            owner = self._param_owner[id(p)]
            dist.broadcast(p.data, src=owner, group=self._pg)

        return result

    def zero_grad(self, set_to_none: bool = True):
        return self._opt.zero_grad(set_to_none=set_to_none)

    @property
    def param_groups(self):
        return self._opt.param_groups

    def state_dict(self):
        return self._opt.state_dict()

    def load_state_dict(self, state):
        return self._opt.load_state_dict(state)


__all__ = ["ShardedOptimizer"]
