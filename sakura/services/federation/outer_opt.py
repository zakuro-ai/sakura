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
  * ``delayed_nesterov`` — buffered (Delayed) Nesterov, HALoS (arXiv 2506.04531)
    / Liu et al. 2024 "DN": dbar is accumulated into a buffer ``b`` and the
    momentum buffer is only refreshed every ``delay`` = N global updates::

        b += dbar ; t += 1
        t % N == 0:  m = μ·m + b/N ; b = 0 ; update = lr · (dbar + μ·m)
        otherwise:                           update = lr · dbar   (m frozen)

    With N = 1 it is exactly ``nesterov``. Under async aggregation each global
    update carries ONE worker's (possibly stale) delta; refreshing ``m`` per
    arrival lets stale deltas compound through the 1/(1-μ) momentum gain,
    which is why naive async + Nesterov degrades. Averaging N arrivals before
    the momentum step (N ≈ number of workers ≈ one sync round's worth) is the
    fix HALoS reports recovers sync-level quality.
  * ``ef``       — update = lr · dbar ; the caller re-adds μ·e (not e) of the
    codec's broadcast error-feedback residual before encoding
    (SparseLoCo, arXiv 2508.15706). No momentum buffer: the EF accumulator
    *is* the momentum. With an aggressive sparsifier (``topk2``) most
    coordinates sit in the residual for many rounds and integrate
    ``lr·dbar`` geometrically (gain ≈ 1/(1-μ)) until they are sent. With a
    lossless codec (no residual) it degenerates to ``plain`` at ``lr``.

The momentum buffer ``m`` (and, for ``delayed_nesterov``, the pseudo-gradient
buffer ``buf``) is owned by the leader (full-length, sliced per streaming
shard) and is mutated in place so it persists across rounds.
``plain`` and ``ef`` take ``m=None`` (see :func:`needs_momentum_buffer`).

Staleness-weighted async aggregation: an async aggregator commits each
upload as soon as it arrives, so an upload computed from an anchor that is
``tau`` global updates old is down-weighted by
``staleness_weight(tau, a) = (1 + tau) ** -a`` (a = 0 disables it; a = 0.5 is
the FedAsync / HALoS-style polynomial default) before the outer step.
"""
from __future__ import annotations

from typing import Optional

import torch

OUTER_KINDS = ("plain", "sgdm", "slowmo", "nesterov", "ef", "delayed_nesterov")


def needs_momentum_buffer(kind: str) -> bool:
    """True for kinds that keep a persistent full-length momentum buffer."""
    return kind not in ("plain", "ef")


def ef_decay(kind: str, momentum: float) -> float:
    """Factor applied to the broadcast error-feedback residual before it is
    re-added to the update: ``μ`` for ``ef`` (the residual is the momentum
    accumulator, decayed like one), ``1`` for every other kind (classic,
    undecayed error feedback)."""
    return momentum if kind == "ef" else 1.0


def outer_step(
    dbar: torch.Tensor,
    m: Optional[torch.Tensor],
    kind: str = "plain",
    lr: float = 1.0,
    momentum: float = 0.9,
    buf: Optional[torch.Tensor] = None,
    step: int = 1,
    delay: int = 1,
) -> torch.Tensor:
    """Return the update to add to the anchor. ``m`` (mutated in place) is the
    persistent outer-momentum buffer; pass ``None`` for ``plain`` and ``ef``.
    For ``ef`` the momentum lives in the caller's error-feedback residual,
    folded in with :func:`ef_decay`.

    ``delayed_nesterov`` additionally needs ``buf`` (pseudo-gradient buffer,
    mutated in place), ``step`` (1-based count of global updates applied to
    this slice, including this one) and ``delay`` (N >= 1)."""
    if kind in ("plain", "ef"):
        return dbar if lr == 1.0 else dbar * lr
    if m is None:
        raise ValueError(f"outer optimizer {kind!r} requires a momentum buffer")
    if kind == "delayed_nesterov":
        if buf is None:
            raise ValueError("delayed_nesterov requires a pseudo-gradient buffer")
        if delay < 1 or step < 1:
            raise ValueError(f"delayed_nesterov needs delay>=1, step>=1 (got {delay}, {step})")
        buf.add_(dbar)
        if step % delay != 0:
            return dbar * lr  # momentum frozen between refreshes
        m.mul_(momentum).add_(buf, alpha=1.0 / delay)  # m = μ·m + b/N
        buf.zero_()
        return (dbar + momentum * m) * lr
    m.mul_(momentum).add_(dbar)  # m = μ·m + dbar
    if kind == "nesterov":
        return (dbar + momentum * m) * lr
    if kind in ("sgdm", "slowmo"):
        return m * lr
    raise ValueError(kind)


def staleness_weight(tau: int, alpha: float = 0.5) -> float:
    """Polynomial staleness discount ``(1 + tau) ** -alpha`` for an upload
    computed against an anchor ``tau`` >= 0 global updates old. ``alpha=0``
    gives 1.0 (no discount); fresh uploads (tau=0) always weigh 1.0."""
    if tau < 0:
        raise ValueError(f"staleness must be >= 0, got {tau}")
    if alpha < 0:
        raise ValueError(f"staleness alpha must be >= 0, got {alpha}")
    return float((1.0 + tau) ** -alpha)


def staleness_weighted_delta(
    delta: torch.Tensor, share: float, tau: int, alpha: float = 0.5
) -> torch.Tensor:
    """Async aggregation of ONE arriving upload: ``share`` is the worker's
    sample share (ema_n_w / Σ ema_n, so one arrival per worker sums to the
    sync weighted mean) and the staleness discount is applied on top, NOT
    renormalised — a stale upload moves the anchor less, it does not take
    weight from anyone else. The result is the ``dbar`` fed to
    :func:`outer_step`."""
    return delta * (share * staleness_weight(tau, alpha))
