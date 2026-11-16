"""Scope advisory (SAK-0017): warn when a fleet is unlikely to beat its own
fastest member training alone.

Federation trades staleness and local drift for aggregate throughput. That
trade only pays when the job is big enough that one GPU cannot finish it
quickly. Two measured cases bound the rule (both are test fixtures):

* E5, ResNet-50@224 — fleet 2 363 img/s vs solo 4090 1 648 (+43%):
  federation wins.
* E6, CIFAR-10 / ResNet-18 — fastest member ~4 460 img/s in-fleet (6 054
  solo); one GPU sweeps the 50k-image set every ~11 s and the solo control
  beat the fleet by ~2 pp at equal wall-clock: federation loses.

The rule is advisory only (warn, never refuse): two data points cannot
support a hard gate. It also assumes IID shards and the same architecture
on every member; non-IID shards add client drift the rule does not model, so
an "ok" verdict says nothing about them.
"""

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional

# The fleet must add at least this much over its fastest member; below it the
# staleness/drift cost is paid for (almost) no extra throughput. A floor from
# first principles, not fitted: neither fixture is near it (E5 1.46, E6 2.07).
MIN_FLEET_SPEEDUP = 1.2
# Warn when the fastest member alone would sweep the whole dataset in less
# than this. E6 (warn) sits at ~11 s, E5 on an ImageNet-1k-sized set (ok) at
# ~790 s; the log-midpoint is ~94 s, rounded down to 60 s to favour fewer
# false warnings.
MIN_SOLO_EPOCH_S = 60.0


def scope_advisory(
    rates: Mapping[str, float], dataset_size: Optional[int] = None
) -> Dict[str, Any]:
    """Classify a fleet from per-worker measured throughput (samples/s).

    Inputs: ``rates`` (worker_id -> samples/s over the first round's window)
    and ``dataset_size`` (training samples across all shards; ``None`` skips
    the epoch check). Returns a JSON-able dict whose ``verdict`` is ``warn``
    (a check failed), ``ok`` (both checks ran and passed) or ``unchecked``
    (not enough inputs to run every check)."""
    rates = {w: float(r) for w, r in rates.items() if r and r > 0}
    adv: Dict[str, Any] = {
        "verdict": "unchecked",
        "reasons": [],
        "rates": {w: round(r, 1) for w, r in rates.items()},
        "assumes": "IID shards, same architecture on every worker",
    }
    if not rates:
        adv["reasons"].append("no worker reported a measurable throughput")
        return adv
    fastest = max(rates, key=lambda w: rates[w])
    r_max = rates[fastest]
    fleet = sum(rates.values())
    speedup = fleet / r_max
    adv.update(
        fastest=fastest,
        fastest_rate=round(r_max, 1),
        fleet_rate=round(fleet, 1),
        fleet_speedup=round(speedup, 3),
        skew=round(r_max / min(rates.values()), 1),
        solo_epoch_s=None,
    )
    warn = False
    if speedup < MIN_FLEET_SPEEDUP:
        warn = True
        adv["reasons"].append(
            f"fleet adds only {100 * (speedup - 1):.0f}% over its fastest worker "
            f"{fastest!r} (< {100 * (MIN_FLEET_SPEEDUP - 1):.0f}%)"
        )
    if dataset_size and dataset_size > 0:
        t_ep = dataset_size / r_max
        adv["solo_epoch_s"] = round(t_ep, 1)
        if t_ep < MIN_SOLO_EPOCH_S:
            warn = True
            adv["reasons"].append(
                f"{fastest!r} alone would sweep all {dataset_size} samples every "
                f"{t_ep:.1f}s (< {MIN_SOLO_EPOCH_S:.0f}s): one GPU can finish this "
                "job, and the fleet's staleness cost likely exceeds its gain"
            )
    else:
        adv["reasons"].append(
            "solo-epoch check skipped: dataset size unknown (pass Leader("
            "dataset_size=...) or FederationService(shard_size=...))"
        )
    if warn:
        adv["verdict"] = "warn"
    elif adv["solo_epoch_s"] is not None:
        adv["verdict"] = "ok"
    return adv


def format_advisory(adv: Mapping[str, Any]) -> str:
    """One log line for an advisory dict."""
    head = (
        f"federation scope advisory: {str(adv.get('verdict')).upper()}"
        f" (fleet {adv.get('fleet_rate')} samples/s = "
        f"{adv.get('fleet_speedup')}x fastest worker {adv.get('fastest')!r}; "
        f"solo epoch {adv.get('solo_epoch_s')}s; assumes {adv.get('assumes')})"
    )
    reasons = "; ".join(str(r) for r in adv.get("reasons") or [])
    return f"{head}: {reasons}" if reasons else head
