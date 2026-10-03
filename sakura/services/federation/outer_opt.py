"""Federation outer optimizer.

Applies a momentum optimizer to the aggregated pseudo-gradient ``dbar`` (the
sample-weighted average of the workers' anchor deltas) before it is committed
to the canonical anchor. Closes gap G1: DiLoCo's Nesterov / SlowMo outer step
was design-only — the leader previously committed ``dbar`` directly (plain
averaging, η_out=1, μ=0). Kinds:

  * ``plain``    — update = lr · dbar                          (FedAvg, the old behaviour)
  * ``sgdm``     — m = μ·m + dbar ;  update = lr · m           (heavy-ball / SlowMo-style)
  * ``slowmo``   — alias of ``sgdm``
  * ``nesterov`` — m = μ·m + dbar ;  update = lr · (dbar + μ·m)  (DiLoCo outer, η_out≈0.7, μ≈0.9)

The momentum buffer ``m`` is owned by the leader (full-length, sliced per
streaming shard) and is mutated in place so it persists across rounds.
"""
from __future__ import annotations

from typing import Optional

import torch

OUTER_KINDS = ("plain", "sgdm", "slowmo", "nesterov")


def outer_step(
    dbar: torch.Tensor,
    m: Optional[torch.Tensor],
    kind: str = "plain",
    lr: float = 1.0,
    momentum: float = 0.9,
) -> torch.Tensor:
    """Return the update to add to the anchor. ``m`` (mutated in place) is the
    persistent outer-momentum buffer; pass ``None`` for ``plain``."""
    if kind == "plain":
        return dbar if lr == 1.0 else dbar * lr
    if m is None:
        raise ValueError(f"outer optimizer {kind!r} requires a momentum buffer")
    m.mul_(momentum).add_(dbar)  # m = μ·m + dbar
    if kind == "nesterov":
        return (dbar + momentum * m) * lr
    if kind in ("sgdm", "slowmo"):
        return m * lr
    raise ValueError(kind)
