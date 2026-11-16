"""Bootstrap link probe (SAK-0013): pick the fleet codec and shard count K
from measured link bandwidth/RTT instead of operator settings.

The leader measures every worker link once, at bootstrap, from the FULL_SYNC
it already sends (full fp32 anchor + BN, timed until the worker's ack) plus
the kernel's smoothed RTT where the platform exposes it. `plan_links` then
walks a fixed ladder of (codec, K) rungs, most bytes first, and keeps the
first rung whose estimated per-round transfer time on the SLOWEST link fits
the round budget. One codec for the broadcast keeps SAK-0002's single byte
string; each worker's upload codec may be less aggressive than the broadcast
codec when its own link has room (the leader decodes each upload with the
codec it assigned, so consensus only depends on the broadcast bytes).

The plan is re-chosen when a worker joins (or leaves) a running fleet and is
persisted with the leader's state, so a restarted leader resumes it
(`LinkPlan.to_state` / `from_state`). `LinkPlan.link_estimate()` is the
SAK-0010 hand-off: a `LinkEstimate` that builds a `sakura.wire.FlowControl`
with BDP-scaled QUIC windows.

Everything here is pure and injectable: tests feed `LinkSample`s directly.
"""

from __future__ import annotations

import math
import socket
import struct
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Tuple

from .codec import BUCKET, _codec_base, _topk_density, _topk_k

AUTO_CODEC = "auto"  # Leader(codec=AUTO_CODEC): codec chosen by the probe
AUTO_SHARDS = 0  # Leader(shards=AUTO_SHARDS): K chosen by the probe
# Fraction of the round period the transfer legs may occupy when the
# operator set no link_duty (same meaning as Leader.link_duty).
DEFAULT_DUTY = 0.5
# Auto codecs, least to most aggressive (int2/topk2 stay operator opt-in:
# their convergence cost is unmeasured on a WAN rung, see SAK-0013).
CODEC_LADDER: Tuple[str, ...] = ("int8", "int4", "topk")
SHARD_LADDER: Tuple[int, ...] = (1, 2, 4, 8, 16)


@dataclass(frozen=True)
class LinkSample:
    """One measured worker link. `bytes_per_s` is one-way goodput (assumed
    symmetric); `rtt_s` is 0.0 when the platform does not expose it."""

    worker_id: str
    bytes_per_s: float
    rtt_s: float

    @property
    def bdp_bytes(self) -> float:
        return self.bytes_per_s * self.rtt_s


@dataclass(frozen=True)
class LinkEstimate:
    """The SAK-0010 `LinkEstimate { rtt, bandwidth_bytes_per_s }` hand-off
    type, on the Python side. `flow_control()` builds the native
    `sakura.wire.FlowControl` (QUIC windows = clamp(2 x BDP)); the import is
    lazy so the federation package keeps working without the extension."""

    rtt_s: float
    bandwidth_bytes_per_s: float

    @property
    def rtt_ms(self) -> float:
        return self.rtt_s * 1e3

    @property
    def bdp_bytes(self) -> float:
        return self.bandwidth_bytes_per_s * self.rtt_s

    def flow_control(
        self, min_window: Optional[int] = None, max_window: Optional[int] = None
    ) -> Any:
        from sakura.wire import FlowControl

        return FlowControl(
            rtt_ms=self.rtt_ms,
            bandwidth_bytes_per_s=max(1, int(self.bandwidth_bytes_per_s)),
            min_window=min_window,
            max_window=max_window,
        )


@dataclass(frozen=True)
class LinkPlan:
    codec: str
    shards: int
    upload_codecs: Dict[str, str]
    budget_s: float
    est_round_s: float  # slowest link's estimated transfer time under the plan
    bottleneck: str
    fits: bool  # False: even the last rung overruns the budget
    samples: List[LinkSample] = field(default_factory=list)

    def as_record(self) -> Dict[str, object]:
        return {
            "codec": self.codec,
            "shards": self.shards,
            "upload_codecs": dict(sorted(self.upload_codecs.items())),
            "budget_s": round(self.budget_s, 4),
            "est_round_s": round(self.est_round_s, 4),
            "bottleneck": self.bottleneck,
            "fits": self.fits,
            "links": {
                s.worker_id: {
                    "mbps": round(s.bytes_per_s * 8 / 1e6, 3),
                    "rtt_ms": round(s.rtt_s * 1e3, 3),
                    "bdp_kb": round(s.bdp_bytes / 1e3, 3),
                }
                for s in sorted(self.samples, key=lambda s: s.worker_id)
            },
        }

    def link_estimate(self, worker_id: Optional[str] = None) -> Optional[LinkEstimate]:
        """`worker_id`'s measured link, or (None) the fleet's largest-BDP
        link: one window sized for it never throttles a smaller-BDP link.
        None without a sample or without a measured RTT (no BDP to size)."""
        links = [s for s in self.samples
                 if worker_id is None or s.worker_id == worker_id]
        links = [s for s in links if s.rtt_s > 0 and s.bytes_per_s > 0]
        if not links:
            return None
        s = max(links, key=lambda s: (s.bdp_bytes, s.worker_id))
        return LinkEstimate(s.rtt_s, s.bytes_per_s)

    def same_decision(self, other: Optional["LinkPlan"]) -> bool:
        """True when `other` puts the same bytes on the wire (codec, K and
        every upload codec); measurements alone may differ."""
        return other is not None and (
            (self.codec, self.shards, self.upload_codecs)
            == (other.codec, other.shards, other.upload_codecs)
        )

    def to_state(self) -> Dict[str, Any]:
        """Lossless, torch-free form for the leader's `persist_path` file
        (plain str / int / float / bool containers: `weights_only` loads)."""
        return {
            "codec": self.codec,
            "shards": int(self.shards),
            "upload_codecs": dict(self.upload_codecs),
            "budget_s": float(self.budget_s),
            "est_round_s": float(self.est_round_s),
            "bottleneck": self.bottleneck,
            "fits": bool(self.fits),
            "samples": [[s.worker_id, float(s.bytes_per_s), float(s.rtt_s)]
                        for s in self.samples],
        }

    @classmethod
    def from_state(cls, st: Mapping[str, Any]) -> "LinkPlan":
        return cls(
            str(st["codec"]),
            int(st["shards"]),
            {str(k): str(v) for k, v in dict(st["upload_codecs"]).items()},
            float(st["budget_s"]),
            float(st["est_round_s"]),
            str(st["bottleneck"]),
            bool(st["fits"]),
            [LinkSample(str(w), float(b), float(r)) for w, b, r in st["samples"]],
        )


def codec_payload_bytes(kind: str, numel: int) -> int:
    """Exact `codec_encode` payload size for a `numel`-element slice (pinned
    against the encoder by tests/services/test_link_probe.py)."""
    base = _codec_base(kind)
    if base == "fp32raw":
        return 4 * numel
    nb = (numel + BUCKET - 1) // BUCKET
    if base == "int8":
        return nb * 4 + nb * BUCKET
    if base == "int4":
        return nb * 4 + nb * BUCKET // 2
    if base == "int2":
        return nb * 4 + nb * BUCKET // 4
    k = _topk_k(_topk_density(kind))
    if base == "topk":  # u32 k || fp32 scales || u16 idx || int8 vals
        return 4 + nb * 4 + nb * k * 3
    # topk2: u32 k || fp32 scales || u16 idx || packed crumbs
    return 4 + nb * 4 + nb * k * 2 + (nb * k + 3) // 4


def round_bytes(kind: str, numel: int, shards: int, bn_numel: int) -> int:
    """Bytes one leg (upload or broadcast) of a round moves for one worker:
    the largest shard's payload plus the fp32 BN sideband."""
    per = -(-numel // max(1, shards))
    return codec_payload_bytes(kind, per) + 4 * bn_numel


def est_round_s(sample: LinkSample, up_bytes: int, down_bytes: int) -> float:
    """Gather leg + broadcast leg: one RTT and one serialization each."""
    return 2.0 * sample.rtt_s + (up_bytes + down_bytes) / max(sample.bytes_per_s, 1e-9)


def round_budget_s(gap_s: float, link_duty: float) -> float:
    return gap_s * (link_duty if link_duty > 0 else DEFAULT_DUTY)


def _rungs(
    numel: int, bn_numel: int, codec: Optional[str], shards: Optional[int]
) -> List[Tuple[str, int, int]]:
    """(codec, K, bytes per leg), most bytes first; ties prefer the smaller K
    (every parameter fresh each round), then the less aggressive codec."""
    codecs = (codec,) if codec is not None else CODEC_LADDER
    ks = (shards,) if shards is not None else SHARD_LADDER
    out = [
        (c, k, round_bytes(c, numel, k, bn_numel)) for k in ks for c in codecs
    ]
    order = {c: i for i, c in enumerate(codecs)}
    out.sort(key=lambda r: (-r[2], r[1], order[r[0]]))
    return out


def plan_links(
    samples: Mapping[str, LinkSample],
    numel: int,
    bn_numel: int,
    budget_s: float,
    codec: Optional[str] = None,
    shards: Optional[int] = None,
) -> Optional[LinkPlan]:
    """Pick (codec, K) from the slowest link; `codec` / `shards` not None are
    operator settings and are never overridden. Returns None without samples."""
    if not samples:
        return None
    links = list(samples.values())
    rungs = _rungs(numel, bn_numel, codec, shards)

    def worst(b_up: int, b_down: int) -> Tuple[float, str]:
        return max((est_round_s(s, b_up, b_down), s.worker_id) for s in links)

    chosen = rungs[-1]
    t, slow = worst(chosen[2], chosen[2])
    fits = False
    for rung in rungs:
        t_r, slow_r = worst(rung[2], rung[2])
        if t_r <= budget_s:
            chosen, t, slow, fits = rung, t_r, slow_r, True
            break
    c_b, k, b_down = chosen
    # Per-worker upload codec: the least aggressive codec (never more
    # aggressive than the broadcast's) whose round still fits THIS link.
    if codec is not None:
        ups = {s.worker_id: c_b for s in links}
    else:
        cands = [c for c in CODEC_LADDER if round_bytes(c, numel, k, bn_numel)
                 >= round_bytes(c_b, numel, k, bn_numel)]
        ups = {}
        for s in links:
            ups[s.worker_id] = c_b
            for c in cands:
                if est_round_s(s, round_bytes(c, numel, k, bn_numel), b_down) <= budget_s:
                    ups[s.worker_id] = c
                    break
    return LinkPlan(c_b, k, ups, budget_s, t, slow, fits, sorted(links, key=lambda s: s.worker_id))


def tcp_rtt_s(sock: Optional[socket.socket]) -> Optional[float]:
    """Kernel smoothed RTT of a connected TCP socket (Linux TCP_INFO
    `tcpi_rtt`, microseconds at byte offset 68); None where unavailable."""
    opt = getattr(socket, "TCP_INFO", None)
    if sock is None or opt is None:
        return None
    try:
        raw = sock.getsockopt(socket.IPPROTO_TCP, opt, 104)
        if len(raw) < 72:
            return None
        (rtt_us,) = struct.unpack_from("I", raw, 68)
    except (OSError, struct.error):
        return None
    return float(rtt_us) / 1e6


def sample_from_fullsync(
    worker_id: str, nbytes: int, elapsed_s: float, rtt_s: Optional[float]
) -> LinkSample:
    """A FULL_SYNC moves `nbytes` and waits for one ack: goodput is the bytes
    over the elapsed time minus that RTT. Without an RTT the estimate stays
    conservative (the RTT and the worker's apply count as transfer time)."""
    rtt = rtt_s if rtt_s is not None and math.isfinite(rtt_s) and rtt_s >= 0 else 0.0
    xfer = max(elapsed_s - rtt, 1e-6)
    return LinkSample(worker_id, nbytes / xfer, rtt)
