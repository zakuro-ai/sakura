"""Optimizer-state policy applied when a FULL_SYNC lands on a worker (SAK-0014).

Optimizer state is local: the leader never sees it. A FULL_SYNC hard-sets the
worker's parameters to the consensus anchor, but the inner optimizer still
holds the momentum it built on the worker's own trajectory (for example while
it trained alone during a disconnection). The policy decides what happens to
that momentum:

  * ``noop``    keep it unchanged (the behaviour before the policy existed and
    the default; see the measurement recorded in SAK-0014).
  * ``reset``   zero every first-moment buffer, as if the optimizer were new.
  * ``average`` scale every first-moment buffer by ``decay`` (default 0.5):
    the element-wise mean of the stale local buffer and a fresh zero buffer,
    a point between ``noop`` (decay 1) and ``reset`` (decay 0). There is no
    fleet-mean buffer to average with, because no momentum leaves a worker.

"First-moment buffers" are the state tensors stored under ``momentum_buffer``
(``torch.optim.SGD``) and ``exp_avg`` (Adam, AdamW and similar). Second
moments (``exp_avg_sq`` and friends) and step counters are left alone: they
set the per-coordinate step scale rather than a direction, and zeroing them
while the step counter keeps its value would make Adam's next update blow up.
"""

from __future__ import annotations

import logging
from typing import Any

import torch

log = logging.getLogger("sakura.federation")

REJOIN_POLICIES = ("noop", "reset", "average")
DEFAULT_REJOIN_POLICY = "noop"
DEFAULT_AVERAGE_DECAY = 0.5
FIRST_MOMENT_KEYS = ("momentum_buffer", "exp_avg")


def check_policy(policy: str, decay: float = DEFAULT_AVERAGE_DECAY) -> None:
    """Raise ``ValueError`` for an unknown policy or a decay outside [0, 1]."""
    if policy not in REJOIN_POLICIES:
        raise ValueError(
            f"unknown rejoin momentum policy {policy!r}; expected one of {REJOIN_POLICIES}"
        )
    if not 0.0 <= decay <= 1.0:
        raise ValueError(f"rejoin momentum decay must be in [0, 1], got {decay}")


def apply_rejoin_policy(
    optimizer: Any, policy: str, decay: float = DEFAULT_AVERAGE_DECAY
) -> int:
    """Apply ``policy`` to ``optimizer``'s first-moment buffers in place.

    Returns how many buffers were changed (0 for ``noop``, for an optimizer
    with no state yet, or for an object without a ``state`` mapping, such as
    a wrapper that hides it). Call it on the thread that steps the optimizer.
    """
    check_policy(policy, decay)
    if policy == "noop" or optimizer is None:
        return 0
    state: Any = getattr(optimizer, "state", None)
    if state is None or not hasattr(state, "values"):
        log.debug("rejoin momentum policy %s: optimizer exposes no state", policy)
        return 0
    factor = 0.0 if policy == "reset" else float(decay)
    n = 0
    with torch.no_grad():
        for st in state.values():
            if not isinstance(st, dict):
                continue
            for key in FIRST_MOMENT_KEYS:
                buf = st.get(key)
                if isinstance(buf, torch.Tensor):
                    if factor == 0.0:
                        buf.zero_()
                    else:
                        buf.mul_(factor)
                    n += 1
    return n
