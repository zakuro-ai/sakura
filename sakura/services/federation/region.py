"""Regional aggregator tier (SAK-0011): hierarchical per-region aggregation.

A ``RegionAggregator`` is a ``Leader`` for the workers of one region and a
worker toward the root ``Leader``. Intra-region rounds run at the regional
cadence (cheap links); the root leader runs inter-region rounds between the
aggregators at its own (self-paced) cadence. Only aggregator<->root traffic
crosses regions.

State held by an aggregator (all flat fp32, full parameter length):

* ``anchor`` / ``bn`` -- the REGIONAL anchor (inherited from ``Leader``).
  Region workers verify its consensus hash every regional round (level 1).
* ``g_anchor`` / ``g_bn`` -- the aggregator's copy of the ROOT anchor at the
  last committed root round ``g_round``. The root leader verifies its hash
  every inter-region round (level 2), exactly like a worker's anchor.
* ``pending`` -- inter-region correction not yet broadcast to the region.

The region's *intended* anchor is ``anchor + ef_bcast + pending``: the anchor
the workers would hold if every broadcast were lossless and every correction
already delivered. The drift uploaded to the root is always
``intended - g_anchor``, i.e. the sum of the regional updates since the last
global commit, so no compression residual is counted twice. After a root
round commits with update ``U`` for an uploaded drift ``D`` the aggregator
does what a worker does with its params (``params += U - D``):
``pending += U - D`` -- the intended anchor becomes ``g_anchor`` plus the
regional progress made since the upload snapshot, and the next regional
round folds ``pending`` into its broadcast (``Leader._fold_update``). Workers
need no change: they apply the broadcast bytes to their anchor and keep their
own local progress, the same as in any other round.

Persistence (``persist_path``, #144): one file holds BOTH tiers -- the
regional ``Leader`` state (round, anchor, bn, optimizer buffers, link plan)
and the root-side state (``g_anchor``/``g_bn``/``g_round``, ``pending``,
``ef_up``, root codecs, unreported samples). It is rewritten atomically,
under ``round_mutex``, after every state change: a committed regional round
(after ``pending`` was folded and zeroed), a committed root round, a root
FULL_SYNC or resend. So the file is always one consistent snapshot of the
last committed state at both levels, and a restarted aggregator resumes from
it: it presents ``g_round`` + the ``g_anchor`` hash to the root (nothing,
resend or FULL_SYNC, the SAK-0006 way) and its workers resume against the
restored regional anchor. Without a file it bootstraps from the root.
"""

from __future__ import annotations

import json
import logging
import os
import socket
import threading
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch

from .codec import anchor_hash16, apply_round_to_anchor, codec_encode, f32_from
from .leader import JoinPolicy, Leader
from .probe import LinkPlan
from .transport import (
    CHUNK_BYTES,
    CHUNK_HDR,
    FT_ACK_APPLIED,
    FT_BCAST_CHUNK,
    FT_BCAST_META,
    FT_FULLSYNC_CHUNK,
    FT_FULLSYNC_META,
    FT_HEARTBEAT,
    FT_HELLO,
    FT_HELLO_ACK,
    FT_ROUND_ABORT,
    FT_ROUND_BEGIN,
    FT_UPLOAD_CHUNK,
    FT_UPLOAD_META,
    IO_TIMEOUT,
    ROUND_HARD_TIMEOUT,
    ExThread,
    WireDead,
    connect,
    join_all,
    recv_blob,
    recv_ctl,
    recv_frame,
    send_frame,
    shutclose,
)

log = logging.getLogger("sakura.federation")

_STRIPES = (0, 1)


def round_wire_bytes(rec: Dict[str, Any]) -> int:
    """Payload bytes one committed round moved between a leader and its
    participants (uploads + one broadcast per participant). For the root
    leader of a hierarchy these are the inter-region (WAN) bytes; for a star
    leader whose workers sit in other regions, likewise."""
    return int(sum(rec["bytes_up"].values()) + len(rec["participants"]) * rec["bytes_down"])


def _lossy(codec: str) -> bool:
    return codec in ("int8", "int4") or codec.startswith("topk")


class RegionAggregator(Leader):
    """Leader for one region's workers; worker toward the root leader.

    ``expected`` lists the region's worker ids; ``allow_join`` /
    ``join_token`` admit regional workers outside it (``Leader``, SAK-0014).
    ``aggregator_id`` must be in the root leader's ``expected`` unless
    ``root_join_token`` matches the root's ``join_token``: it is presented in
    this aggregator's HELLO to the root. Regional rounds use ``codec``/``gap_s``/
    ``shards``/``outer_opt`` of this object; inter-region rounds use the root
    leader's codecs (announced in every root ROUND_BEGIN) and cadence.
    ``persist_path`` keeps the state of both tiers across a restart (see the
    module docstring); ``None`` (default) re-bootstraps from the root.
    """

    def __init__(
        self,
        model_fn: Callable[[], torch.nn.Module],
        port: int,
        expected: List[str],
        region: str,
        root_host: str,
        root_port: int,
        aggregator_id: Optional[str] = None,
        codec: str = "int8",
        gap_s: float = 5.0,
        seed: int = 1234,
        max_rounds: int = 10**9,
        shards: int = 1,
        link_duty: float = 0.0,
        outer_opt: str = "plain",
        outer_lr: float = 1.0,
        outer_momentum: float = 0.9,
        allow_join: Optional[JoinPolicy] = None,
        join_token: Optional[str] = None,
        root_join_token: Optional[str] = None,
        persist_path: Optional[str] = None,
    ):
        if root_join_token is not None and not root_join_token:
            raise ValueError("root_join_token must be a non-empty string")
        # The base Leader gets persist_path=None: its per-commit write
        # (Leader._persist) runs
        # inside the regional round, BEFORE pending is zeroed and without the
        # root-side state, so a crash right after it would fold pending twice.
        # The aggregator writes its own two-tier snapshot instead (_persist_tiers).
        super().__init__(
            model_fn,
            port,
            expected,
            codec=codec,
            gap_s=gap_s,
            persist_path=None,
            seed=seed,
            max_rounds=max_rounds,
            shards=shards,
            link_duty=link_duty,
            outer_opt=outer_opt,
            outer_lr=outer_lr,
            outer_momentum=outer_momentum,
            allow_join=allow_join,
            join_token=join_token,
        )
        # SAK-0014 toward the root: sent in the HELLO only, never logged.
        self._root_join_token = root_join_token
        self.region = region
        self.root = (root_host, root_port)
        self.aggregator_id = aggregator_id or f"region-{region}"
        n = self.fs.numel
        self.g_anchor = self.anchor.clone()
        self.g_bn = self.bn.clone()
        self.g_round = -1
        # root_codec decodes the root's broadcast; root_up_codec encodes our
        # upload. Both follow the root's ROUND_BEGIN (SAK-0013 link probe).
        self.root_codec: Optional[str] = None
        self.root_up_codec: Optional[str] = None
        self.ef_up: Optional[torch.Tensor] = None
        self.pending = torch.zeros(n)
        self._fold: Optional[Tuple[int, int]] = None
        # regional samples committed since the last committed root round:
        # the region's aggregation weight at the root (sample counts, not 1
        # per region -- see SAK-0011 Decision).
        self._samples = 0.0
        self.global_stats: List[Dict[str, Any]] = []
        # (regional round, root round) restored from persist_path, if any
        self.restored_from: Optional[Tuple[int, int]] = None
        self.ctl: Optional[socket.socket] = None
        self.data: Dict[int, socket.socket] = {}
        self.agg_persist_path = persist_path
        if persist_path and os.path.exists(persist_path):
            self._restore_tiers(persist_path)

    # ------------------------------------------------ persistence (#144)
    def _persist_tiers(self) -> None:
        """Atomic two-tier snapshot. Callers hold round_mutex (an RLock), so
        no regional round, root commit or resync is half-applied in it."""
        path = self.agg_persist_path
        if not path:
            return
        with self.round_mutex:
            snap: Dict[str, Any] = {
                # the regional Leader's keys, same names as Leader's file
                "round": self.round,
                "anchor": self.anchor,
                "bn": self.bn,
                **self._opt_state(),
                # root-side state of this aggregator
                "region": {
                    "aggregator_id": self.aggregator_id,
                    "g_anchor": self.g_anchor,
                    "g_bn": self.g_bn,
                    "g_round": self.g_round,
                    "pending": self.pending,
                    "ef_up": self.ef_up,
                    "root_codec": self.root_codec,
                    "root_up_codec": self.root_up_codec,
                    "samples": float(self._samples),
                },
            }
            if self.link_plan is not None:
                snap["link_plan"] = self.link_plan.to_state()
            tmp = path + ".tmp"
            torch.save(snap, tmp)
            os.replace(tmp, path)

    def _restore_tiers(self, path: str) -> None:
        """Resume from a ``_persist_tiers`` file. A file that does not fit this
        model (or is unreadable) is ignored: the aggregator then bootstraps
        from the root by FULL_SYNC, the no-file behaviour."""
        n = self.fs.numel
        try:
            st = torch.load(path, map_location="cpu")
            reg = st["region"]
            if st["anchor"].numel() != n or reg["g_anchor"].numel() != n:
                raise ValueError(f"persisted state has {st['anchor'].numel()} params, model {n}")
        except Exception as e:  # noqa: BLE001 — a bad file must not kill the region
            log.warning("%s ignoring persisted state %s (%r); bootstrapping from the root",
                        self.aggregator_id, path, e)
            return
        if st.get("link_plan"):  # SAK-0013: resume the plan (sets codec / K)
            plan = LinkPlan.from_state(st["link_plan"])
            self._restored_samples = {s.worker_id: s for s in plan.samples}
            self._apply_plan(plan)
        self.round = int(st["round"])
        self.anchor.copy_(st["anchor"])
        self.bn.copy_(st["bn"])
        self._restore_opt_state(st)  # Leader's additive restore (#143)
        self.g_anchor.copy_(reg["g_anchor"])
        self.g_bn.copy_(reg["g_bn"])
        self.g_round = int(reg["g_round"])
        self.pending.copy_(reg["pending"])
        self.root_codec = reg.get("root_codec")
        self.root_up_codec = reg.get("root_up_codec")
        ef_up = reg.get("ef_up")
        if self.root_up_codec is None or not _lossy(self.root_up_codec):
            self.ef_up = None
        elif ef_up is not None and ef_up.numel() == n:
            self.ef_up = ef_up.clone()
        else:
            self.ef_up = torch.zeros(n)
        self._samples = float(reg.get("samples", 0.0))
        self.restored_from = (self.round, self.g_round)
        log.info("%s restored from %s at regional round %d, root round %d",
                 self.aggregator_id, path, self.round, self.g_round)

    # ------------------------------------------------ regional (level 1) tier
    def _fold_update(self, update: torch.Tensor, lo: int, hi: int) -> torch.Tensor:
        # called under round_mutex from _run_round_locked; pending is zeroed
        # in run_round only once the round actually committed.
        self._fold = (lo, hi)
        return update + self.pending[lo:hi]

    def run_round(self) -> Optional[Dict[str, Any]]:
        with self.round_mutex:
            self._fold = None
            before = self.round
            try:
                rec = self._run_round_locked()
            finally:
                if self.round != before and self._fold is not None:
                    lo, hi = self._fold
                    self.pending[lo:hi].zero_()
            if rec is not None:
                self._samples += sum(float(v) for v in rec["n_samples"].values())
                # ties every regional round to the root round it builds on:
                # all regions must report the same global hash per g_round.
                rec["region"] = self.region
                rec["global_round"] = self.g_round
                rec["global_hash16"] = anchor_hash16(self.g_anchor, self.g_bn).hex()
            if self.round != before:
                # after the pending zeroing and the sample count: the file
                # must never hold a pending slice the committed regional
                # anchor already has, nor miss the round's samples
                self._persist_tiers()
            return rec

    def _intended(self, lo: int, hi: int) -> torch.Tensor:
        out = self.anchor[lo:hi] + self.pending[lo:hi]
        if self.ef_bcast is not None:
            out += self.ef_bcast[lo:hi]
        return out

    def _snapshot_drift(self, lo: int, hi: int) -> Tuple[torch.Tensor, torch.Tensor, float]:
        """Drift to upload for a root round: intended regional anchor minus
        the global anchor, plus the regional sample count. Taken under
        round_mutex so it never lands mid regional round."""
        with self.round_mutex:
            drift = self._intended(lo, hi) - self.g_anchor[lo:hi]
            bn_now = self.bn.clone()
            n, self._samples = self._samples, 0.0
        return drift, bn_now, n

    def _commit_root(
        self,
        rnd: int,
        payload: bytes,
        bn_bytes: bytes,
        drift: torch.Tensor,
        ef_new: Optional[torch.Tensor],
        lo: int,
        hi: int,
    ) -> None:
        """Apply a committed root round: g_anchor += U (same bytes, same
        function as the root), and the intended regional anchor moves by
        U - D -- global consensus plus the regional progress made since the
        snapshot (a worker's ``params += U - delta``)."""
        assert self.root_codec is not None
        with self.round_mutex:
            upd = apply_round_to_anchor(self.g_anchor, payload, self.root_codec, lo, hi)
            self.g_bn.copy_(f32_from(bn_bytes))
            self.pending[lo:hi].add_(upd - drift)
            if self.ef_up is not None and ef_new is not None:
                self.ef_up[lo:hi].copy_(ef_new)  # EF persists only on commit
            self.g_round = rnd
            self._persist_tiers()

    def _reset_to_global(self) -> None:
        """Drop the region's uncommitted progress: the intended anchor becomes
        the global anchor (a worker's FULL_SYNC / resend semantics)."""
        self.pending.copy_(self.g_anchor - self._intended(0, self.fs.numel) + self.pending)
        if self.ef_up is not None:
            self.ef_up.zero_()
        self._samples = 0.0

    # ------------------------------------------------ root-facing (level 2)
    def _hello(self) -> Dict[str, Any]:
        hello: Dict[str, Any] = {
            "worker_id": self.aggregator_id,
            "region": self.region,
            "role": "ctl",
            "torch": torch.__version__.split("+")[0],
            "last_committed": self.g_round,
            "anchor_hash16": anchor_hash16(self.g_anchor, self.g_bn).hex()
            if self.g_round >= 0
            else "00" * 16,
        }
        if self._root_join_token is not None:
            hello["join_token"] = self._root_join_token
        return hello

    def _connect_root(self) -> None:
        host, port = self.root
        hello = self._hello()
        self._close_root()
        self.data = {}
        for st in _STRIPES:  # data stripes first: the root's resync needs them
            s = connect(host, port)
            send_frame(s, FT_HELLO, 0, json.dumps({**hello, "role": "data", "stripe": st}).encode())
            ft, _, _ = recv_ctl(s, time.monotonic() + IO_TIMEOUT)
            if ft != FT_HELLO_ACK:
                raise WireDead(f"bad data hello-ack {ft}")
            self.data[st] = s
        self.ctl = connect(host, port)
        send_frame(self.ctl, FT_HELLO, 0, json.dumps(hello).encode())
        ft, _, payload = recv_ctl(self.ctl, time.monotonic() + IO_TIMEOUT)
        if ft != FT_HELLO_ACK:
            raise WireDead(f"bad ctl hello-ack {ft}")
        ack = json.loads(payload)
        if ack["numel"] != self.fs.numel:
            raise AssertionError("model shape mismatch with root leader")
        if self.root_codec is None:
            self.root_codec = self.root_up_codec = ack["codec"]
            if _lossy(self.root_codec):
                self.ef_up = torch.zeros(self.fs.numel)
        log.info("%s connected to root at committed=%d", self.aggregator_id, self.g_round)

    def _close_root(self) -> None:
        for s in [self.ctl, *self.data.values()]:
            shutclose(s)

    def _bootstrap_root(self) -> None:
        """Connect + receive the bootstrap FULL_SYNC (last_committed=-1), then
        seed the regional anchor from the global one before serving workers.
        A restored aggregator (``g_round >= 0``) only connects: its HELLO
        carries ``g_round`` + the ``g_anchor`` hash and the root resyncs it
        like any reconnecting worker (nothing, resend or FULL_SYNC), handled
        by the upstream loop with the region's progress kept or reset."""
        backoff = 1.0
        while not self.stop.is_set():
            try:
                self._connect_root()
                if self.g_round >= 0:
                    return
                assert self.ctl is not None
                ftype, rnd, payload = recv_ctl(self.ctl, time.monotonic() + 300.0)
                if ftype != FT_FULLSYNC_META:
                    raise WireDead(f"expected bootstrap FULL_SYNC, got {ftype}")
                self._handle_fullsync(rnd, payload, bootstrap=True)
                return
            except (WireDead, OSError, AssertionError) as e:
                log.warning("%s root bootstrap failed: %s (retry in %.0fs)",
                            self.aggregator_id, e, backoff)
                time.sleep(backoff)
                backoff = min(backoff * 2, 30.0)

    def _ack(self, rnd: int) -> None:
        assert self.ctl is not None
        send_frame(
            self.ctl,
            FT_ACK_APPLIED,
            rnd,
            json.dumps({"hash16": anchor_hash16(self.g_anchor, self.g_bn).hex()}).encode(),
        )

    def _handle_fullsync(self, rnd: int, payload: bytes, bootstrap: bool = False) -> None:
        assert self.ctl is not None
        meta = json.loads(payload)
        buf = bytearray(meta["plen"])
        recv_blob(self.data[0], FT_FULLSYNC_CHUNK, rnd, meta["plen"], buf)
        ftype, _, bn_payload = recv_ctl(self.ctl, time.monotonic() + IO_TIMEOUT)
        if ftype != FT_FULLSYNC_META:
            raise WireDead(f"bad fullsync bn sideband {ftype}")
        with self.round_mutex:
            self.g_anchor.copy_(torch.frombuffer(buf, dtype=torch.float32))
            self.g_bn.copy_(f32_from(bn_payload))
            self.g_round = rnd
            if bootstrap:
                # region not serving yet: the regional anchor IS the global one
                self.anchor.copy_(self.g_anchor)
                self.bn.copy_(self.g_bn)
                self.pending.zero_()
                if self.ef_bcast is not None:
                    self.ef_bcast.zero_()
                self._samples = 0.0
            else:
                self._reset_to_global()
            self._persist_tiers()
        self._ack(rnd)
        log.info("%s %s from root at round %d", self.aggregator_id,
                 "bootstrapped" if bootstrap else "FULL_SYNCed", rnd)

    def _handle_resend(self, rnd: int, payload: bytes) -> None:
        """Retained root round we never committed: apply it to g_anchor and
        reset the region's intended anchor to it (our upload for that round
        is already inside U; keeping our drift would count it twice)."""
        assert self.ctl is not None
        bmeta = json.loads(payload)
        buf = bytearray(bmeta["plen"])
        recv_blob(self.data[0], FT_BCAST_CHUNK, rnd, bmeta["plen"], buf)
        ftype, _, bn_bytes = recv_ctl(self.ctl, time.monotonic() + IO_TIMEOUT)
        if ftype != FT_BCAST_META:
            raise WireDead(f"bad resend bn sideband {ftype}")
        assert self.root_codec is not None
        with self.round_mutex:
            apply_round_to_anchor(self.g_anchor, bytes(buf),
                                  bmeta.get("codec", self.root_codec),
                                  bmeta.get("lo", 0), bmeta.get("hi"))
            self.g_bn.copy_(f32_from(bn_bytes))
            self.g_round = rnd
            self._reset_to_global()
            self._persist_tiers()
        self._ack(rnd)
        log.info("%s applied retained root round %d", self.aggregator_id, rnd)

    def _send_chunks(self, sock: socket.socket, rnd: int, part: bytes, base: int) -> None:
        view = memoryview(part)
        off = 0
        while off < len(view):
            chunk = view[off : off + CHUNK_BYTES]
            send_frame(sock, FT_UPLOAD_CHUNK, rnd, CHUNK_HDR.pack(base + off) + chunk.tobytes())
            off += len(chunk)

    def _root_round(self, rnd: int, meta: Dict[str, Any]) -> None:
        assert self.ctl is not None and self.root_codec is not None
        if meta["anchor_version"] != self.g_round:
            raise WireDead(f"root anchor version skew: {meta} vs {self.g_round}")
        lo = int(meta.get("lo", 0))
        hi = int(meta.get("hi", self.fs.numel))
        self.root_codec = meta.get("codec", self.root_codec)
        up: Optional[str] = meta.get("up_codec", self.root_up_codec)
        assert up is not None
        if up != self.root_up_codec:  # new basis: the old residual is stale
            log.info("%s root upload codec %s -> %s", self.aggregator_id,
                     self.root_up_codec, up)
            self.ef_up = torch.zeros(self.fs.numel) if _lossy(up) else None
            self.root_up_codec = up
        t0 = time.monotonic()
        drift, bn_now, n = self._snapshot_drift(lo, hi)
        committed = False
        try:
            payload, ef_new, _ = codec_encode(
                drift, self.ef_up[lo:hi] if self.ef_up is not None else None,
                self.root_up_codec,
            )
            # floor at 1: a region with no committed regional round since the
            # last root round (startup, all its workers down) must still be a
            # valid participant -- an all-zero weight sum aborts the root round.
            send_frame(self.ctl, FT_UPLOAD_META, rnd,
                       json.dumps({"plen": len(payload), "n_samples": max(n, 1.0),
                                   "codec": self.root_up_codec}).encode())
            view = memoryview(payload)
            half = len(payload) // 2
            ths = []
            for st, (b_lo, b_hi) in zip(_STRIPES, ((0, half), (half, len(payload)))):
                t = ExThread(self._send_chunks, (self.data[st], rnd, bytes(view[b_lo:b_hi]), b_lo),
                             name=f"up-{self.aggregator_id}-{st}")
                t.start()
                ths.append(t)
            join_all(ths, ROUND_HARD_TIMEOUT, "aggregator upload stripes")
            send_frame(self.ctl, FT_UPLOAD_META, rnd, bn_now.numpy().tobytes())
            ftype, frnd, pl = recv_ctl(self.ctl, time.monotonic() + ROUND_HARD_TIMEOUT)
            if ftype == FT_ROUND_ABORT:
                return
            if ftype != FT_BCAST_META or frnd != rnd:
                raise WireDead(f"bad root broadcast {ftype}/{frnd}")
            bmeta = json.loads(pl)
            buf = bytearray(bmeta["plen"])
            shares = [bmeta["plen"] // 2, bmeta["plen"] - bmeta["plen"] // 2]
            rxs = []
            for st, share in zip(_STRIPES, shares):
                t = ExThread(recv_blob, (self.data[st], FT_BCAST_CHUNK, rnd, share, buf),
                             name=f"bc-{self.aggregator_id}-{st}")
                t.start()
                rxs.append(t)
            join_all(rxs, ROUND_HARD_TIMEOUT, "aggregator broadcast stripes")
            ftype, _, bn_bytes = recv_ctl(self.ctl, time.monotonic() + IO_TIMEOUT)
            if ftype != FT_BCAST_META:
                raise WireDead(f"bad root bn sideband {ftype}")
            self._commit_root(rnd, bytes(buf), bn_bytes, drift, ef_new, lo, hi)
            committed = True
            self._ack(rnd)
            rec = {
                "global_round": rnd,
                "aggregator": self.aggregator_id,
                "region": self.region,
                "n_samples": n,
                "bytes_up": len(payload),
                "bytes_down": len(buf),
                "drift_norm": round(float(drift.norm()), 4),
                "t_total_s": round(time.monotonic() - t0, 3),
                "global_hash16": anchor_hash16(self.g_anchor, self.g_bn).hex(),
                "regional_round": self.round,
            }
            self.global_stats.append(rec)
            print("AGGREGATOR_ROUND " + json.dumps(rec), flush=True)
        finally:
            if not committed:
                with self.round_mutex:
                    self._samples += n  # the window belongs to the next round

    def _upstream_loop(self) -> None:
        while not self.stop.is_set():
            try:
                assert self.ctl is not None
                ftype, rnd, payload = recv_frame(self.ctl, time.monotonic() + IO_TIMEOUT + 60)
            except (WireDead, AssertionError) as e:
                if self.stop.is_set():
                    return
                log.warning("%s root link dead (%s); reconnecting", self.aggregator_id, e)
                self._reconnect_root()
                continue
            try:
                if ftype == FT_HEARTBEAT or ftype == FT_ROUND_ABORT:
                    continue
                if ftype == FT_ROUND_BEGIN:
                    self._root_round(rnd, json.loads(payload))
                elif ftype == FT_FULLSYNC_META:
                    self._handle_fullsync(rnd, payload)
                elif ftype == FT_BCAST_META:
                    self._handle_resend(rnd, payload)
            except Exception as e:  # noqa: BLE001 — upstream thread must never die
                if self.stop.is_set():
                    return
                log.warning("%s root frame %s round %d failed (%r); reconnecting",
                            self.aggregator_id, ftype, rnd, e)
                self._reconnect_root()

    def _reconnect_root(self) -> None:
        backoff = 1.0
        while not self.stop.is_set():
            try:
                self._connect_root()
                return
            except (WireDead, OSError, AssertionError) as e:
                log.warning("%s root reconnect failed: %s (retry in %.0fs)",
                            self.aggregator_id, e, backoff)
                time.sleep(backoff)
                backoff = min(backoff * 2, 30.0)

    # ------------------------------------------------ lifecycle
    def serve_forever(self, warmup_s: float = 5.0) -> None:
        """Bootstrap from the root, then serve the region. Blocks like
        ``Leader.serve_forever``; ``shutdown()`` (or ``stop``) ends it."""
        self._bootstrap_root()
        if self.stop.is_set():
            return
        threading.Thread(target=self._upstream_loop, daemon=True,
                         name=f"upstream-{self.aggregator_id}").start()
        super().serve_forever(warmup_s=warmup_s)

    def shutdown(self) -> None:
        self.stop.set()
        self._close_root()
        # shutdown() wakes the accept() blocked in Leader._serve, so the port
        # is free for a restarted aggregator right away.
        shutclose(getattr(self, "_srv", None))
        with self.lock:
            conns = list(self.workers.values())
        for wc in conns:
            self._drop(wc)
