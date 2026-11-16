"""Federation leader: owns the canonical anchor and drives consensus rounds."""

from __future__ import annotations

import hmac
import json
import logging
import os
import socket
import threading
import time
from typing import Any, Callable, Dict, Iterable, List, Optional, Set, Tuple, Union

import torch

from .codec import (
    f32_from,
    FlatState,
    anchor_hash16,
    apply_round_to_anchor,
    codec_decode,
    codec_encode,
    is_lossy,
    merge_bn_pooled,
    shard_bounds,
)
from .outer_opt import OUTER_KINDS, ef_decay, needs_momentum_buffer, outer_step
from .scope import format_advisory, scope_advisory
from .probe import (
    AUTO_CODEC,
    AUTO_SHARDS,
    LinkPlan,
    LinkSample,
    plan_links,
    round_budget_s,
    sample_from_fullsync,
    tcp_rtt_s,
)
from .transport import (
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
    POLL_SLICE,
    ROUND_HARD_TIMEOUT,
    ExThread,
    WireDead,
    _WorkerConn,
    _keepalive,
    join_all,
    recv_ctl,
    recv_blob,
    recv_frame,
    send_frame,
    shutclose,
)

log = logging.getLogger("sakura.federation")

# SAK-0014 join policy: an explicit set of worker_ids, or a predicate on the id.
JoinPolicy = Union[Iterable[str], Callable[[str], bool]]


class Leader:
    """Separate CPU-only process. Owns the canonical anchor.

    Threading model: one thread per accepted connection (handshake + resync),
    heartbeat thread, and the round loop. Rounds and resyncs are serialized by
    round_mutex; workers being resynced are excluded from rounds via wc.syncing;
    every cross-thread socket touch is generation-guarded."""

    def __init__(
        self,
        model_fn: Callable[[], torch.nn.Module],
        port: int,
        expected: List[str],
        codec: str = "int8",
        gap_s: float = 30.0,
        persist_path: Optional[str] = None,
        seed: int = 1234,
        max_rounds: int = 10**9,
        shards: int = 1,
        link_duty: float = 0.0,
        outer_opt: str = "plain",
        outer_lr: float = 1.0,
        outer_momentum: float = 0.9,
        dataset_size: Optional[int] = None,
        allow_join: Optional[JoinPolicy] = None,
        join_token: Optional[str] = None,
        link_probe: Optional[Callable[[], Dict[str, LinkSample]]] = None,
        outer_delay: int = 1,
    ):
        # link_duty > 0 enables self-pacing: the next gap is derived from the
        # measured transfer cost of the round just completed so that round
        # traffic occupies at most this fraction of the inter-region link.
        # Remote regions are the native deployment; nobody knows the path's
        # RTT/bandwidth in advance, so the framework measures instead of
        # being configured.
        self.link_duty = link_duty
        # codec="auto" / shards=0: chosen at bootstrap from the measured
        # links (SAK-0013); any explicit value is an operator setting and wins.
        if shards < 0:
            raise ValueError(f"shards must be >= 0 (0 = auto), got {shards}")
        self._auto_codec = codec == AUTO_CODEC
        self._auto_shards = shards == AUTO_SHARDS
        self.shards = 1 if self._auto_shards else shards
        self.link_probe = link_probe
        self._fullsync_samples: Dict[str, LinkSample] = {}
        # links measured before a restart (persisted plan): stand-ins for
        # workers that resume in sync, so no FULL_SYNC re-measures them
        self._restored_samples: Dict[str, LinkSample] = {}
        self.link_plan: Optional[LinkPlan] = None
        self.upload_codecs: Dict[str, str] = {}  # per-worker; default self.codec
        # every plan the leader adopted, with the first round it applies to
        self.plan_history: List[Dict[str, Any]] = []
        self._bootstrapped = False  # set once the bootstrap plan ran
        codec = "int8" if self._auto_codec else codec
        # SAKURA_OBSERVERS: worker_ids that ride along without participating in
        # rounds. Measured motivation (2x2080Ti + P620): a ~6x-slower worker in
        # the synchronous gather collapses the whole federation's cadence from
        # 3.55 to 0.94 rounds/s — merge-weight gating cannot fix a latency
        # problem. Observers never receive ROUND_BEGIN (zero latency impact)
        # and are kept consensus-fresh with a periodic FULLSYNC.
        self.observers = set(
            w for w in os.environ.get("SAKURA_OBSERVERS", "").split(",") if w
        )
        self.obs_sync_every = int(os.environ.get("SAKURA_OBSERVER_SYNC_EVERY", "40"))
        torch.manual_seed(seed)
        self.model = model_fn()
        self.fs = FlatState(self.model)
        self.anchor = torch.empty(self.fs.numel, dtype=torch.float32)
        self.fs.snapshot_params_to(self.anchor)
        self.bn = self.fs.bn_vec()
        self.var_mask = self.fs.bn_var_mask()
        self.mean_mask = ~self.var_mask
        self.codec = codec
        self.gap_s = gap_s
        self.port = port
        self.expected = list(expected)
        # SAK-0014: workers outside `expected` may join a RUNNING leader only
        # if the operator opted in — an allow-list / predicate on the id, or a
        # shared join token presented in the HELLO. Both None (the default)
        # keeps the roster fail-closed: unknown ids are rejected as before.
        if isinstance(allow_join, str):
            raise TypeError("allow_join must be a set of worker_ids or a predicate, not a str")
        self._join_pred: Optional[Callable[[str], bool]] = (
            allow_join if callable(allow_join) else None
        )
        self._join_allow: Set[str] = (
            set() if allow_join is None or callable(allow_join) else set(allow_join)
        )
        if join_token is not None and not join_token:
            raise ValueError("join_token must be a non-empty string")
        self._join_token = join_token
        self.persist_path = persist_path
        self.max_rounds = max_rounds
        self.round = 0
        _lossy = is_lossy(codec)
        self.ef_bcast = torch.zeros(self.fs.numel) if _lossy else None
        if outer_opt not in OUTER_KINDS:
            raise ValueError(f"outer_opt must be one of {OUTER_KINDS}, got {outer_opt!r}")
        self.outer_kind = outer_opt
        self.outer_lr = outer_lr
        self.outer_momentum = outer_momentum
        self.outer_m = (
            torch.zeros(self.fs.numel) if needs_momentum_buffer(outer_opt) else None
        )
        # delayed_nesterov: pseudo-gradient buffer + per-shard update count
        # (each streaming shard is its own N-update window).
        if outer_delay < 1:
            raise ValueError(f"outer_delay must be >= 1, got {outer_delay}")
        self.outer_delay = outer_delay
        self.outer_buf = (
            torch.zeros(self.fs.numel) if outer_opt == "delayed_nesterov" else None
        )
        self.outer_steps: Dict[int, int] = {}
        self.workers: Dict[str, _WorkerConn] = {w: _WorkerConn(w) for w in expected}
        # (round, payload, bn_bytes, lo, hi, pre_round_anchor_hash16)
        self.retained: Optional[Tuple[int, bytes, bytes, int, int, bytes]] = None
        self._retained_codec = self.codec  # a plan change must not relabel it
        self.lock = threading.Lock()  # workers registry
        self.round_mutex = threading.RLock()  # serializes rounds vs resyncs
        self.stop = threading.Event()
        self.stats: List[Dict[str, Any]] = []
        self.overruns = 0
        self._in_round = threading.Event()
        # SAK-0017 scope advisory: computed once, at the first committed
        # round, from the workers' measured throughput. Advisory only.
        self.dataset_size = dataset_size
        self.scope_advisory: Optional[Dict[str, Any]] = None
        # restarted leader must resume, not re-init from seed: a fresh random
        # anchor would FULL_SYNC over every trained worker on reconnect.
        if persist_path and os.path.exists(persist_path):
            st = torch.load(persist_path, map_location="cpu")
            self.round = st["round"]
            self.anchor.copy_(st["anchor"])
            self.bn.copy_(st["bn"])
            if st.get("link_plan"):  # SAK-0013: resume the plan, not int8/K=1
                plan = LinkPlan.from_state(st["link_plan"])
                self._restored_samples = {s.worker_id: s for s in plan.samples}
                self._apply_plan(plan)
            # after the plan: it sets codec/K (which buffers exist, and resets
            # the per-shard windows on a K change); the persisted optimizer
            # state was saved under that same plan.
            self._restore_opt_state(st)
            log.info("leader restored from %s at round %d", persist_path, self.round)

    # --- connection plumbing ---
    def _serve(self) -> None:
        # Workers dial in from other hosts/regions, so the leader must listen
        # on all interfaces; admission is gated by `expected` / the join policy.
        # nosemgrep: python.lang.security.audit.network.bind.avoid-bind-to-all-interfaces
        srv = socket.socket()
        srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        srv.bind(("0.0.0.0", self.port))
        srv.listen(16)
        srv.settimeout(1.0)
        self._srv = srv
        while not self.stop.is_set():
            try:
                s, addr = srv.accept()
            except socket.timeout:
                continue
            except OSError:
                break
            s.settimeout(POLL_SLICE)
            s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            _keepalive(s)
            # One thread per connection: a handshake blocked on a slow peer or
            # a resync waiting out an in-flight round must not stall every
            # other worker's reconnect behind it.
            threading.Thread(
                target=self._handshake, args=(s, addr), daemon=True
            ).start()

    def _handshake(self, s: socket.socket, addr: Any) -> None:
        try:
            ftype, _, payload = recv_frame(s, time.monotonic() + IO_TIMEOUT)
            hello = json.loads(payload)
            # never echo `hello` itself: it may carry the join token
            assert ftype == FT_HELLO, f"handshake mismatch (frame type {ftype})"
            # The torch base-version need NOT match the leader's. The wire is
            # byte-level (int8 codec + raw float32 buffers) and the parameter
            # layout is fixed by the model architecture, not the torch build —
            # so a heterogeneous fleet can federate, e.g. a Pascal GPU on cu118
            # torch alongside Ada/Turing GPUs on cu128. Warn so drift stays
            # visible without rejecting the worker.
            leader_torch = torch.__version__.split("+")[0]
            if hello.get("torch") != leader_torch:
                log.warning(
                    "worker %s torch %s != leader torch %s (allowed: wire is "
                    "version-agnostic)",
                    hello.get("worker_id"), hello.get("torch"), leader_torch,
                )
            wid, role, stripe = (
                hello["worker_id"],
                hello["role"],
                hello.get("stripe", 0),
            )
            wc = self._roster_entry(wid, hello.get("join_token"))
            ack = json.dumps(
                {
                    "round": self.round,
                    "codec": self.codec,
                    "numel": self.fs.numel,
                    "bn_numel": self.fs.bn_numel,
                    "shards": self.shards,
                }
            ).encode()
            if role == "ctl":
                with self.lock:
                    old = wc.ctl
                    wc.ctl = s
                    wc.region = hello.get("region", wc.region)
                    wc.last_committed = int(hello["last_committed"])
                    wc.alive = True
                    wc.syncing = True  # hidden from rounds until resync settles
                    # replacing a socket invalidates every thread blocked on the
                    # old one — without the bump, an in-flight round's gather
                    # sees WireDead on the OLD ctl, passes its gen guard, and
                    # _drop()s the brand-new sockets.
                    wc.gen += 1
                shutclose(old)
                wc.send(FT_HELLO_ACK, self.round, ack)  # under send_lock
                try:
                    # serialized vs rounds: a FULL_SYNC mid-round would
                    # interleave ROUND_BEGIN into the sync stream.
                    with self.round_mutex:
                        sample_before = self._fullsync_samples.get(wid)
                        self._resync_if_needed(
                            wc, bytes.fromhex(hello["anchor_hash16"])
                        )
                        # still under the mutex: the very next round sees
                        # the (re)joined worker as a participant (SAK-0014)
                        wc.syncing = False
                        if self._bootstrapped:
                            self._replan_on_join(wid, sample_before)
                except WireDead as e:
                    log.warning("resync of %s failed: %s", wid, e)
                    self._drop(wc)
            else:
                send_frame(
                    s, FT_HELLO_ACK, self.round, ack
                )  # not registered yet: no writer race
                with self.lock:
                    shutclose(wc.data.get(stripe))
                    wc.data[stripe] = s
                    wc.gen += 1
            log.info(
                "hello %s/%s stripe=%s last_committed=%s",
                wid,
                role,
                stripe,
                hello.get("last_committed"),
            )
        except (WireDead, AssertionError, KeyError, json.JSONDecodeError) as e:
            log.warning("handshake failed from %s: %s", addr, e)
            shutclose(s)

    # --- membership (SAK-0014) ---
    def _may_join(self, wid: str, token: Any) -> bool:
        if wid in self._join_allow:
            return True
        if self._join_pred is not None:
            try:
                if self._join_pred(wid):
                    return True
            except Exception as e:  # noqa: BLE001 — a faulty policy denies
                log.warning("allow_join predicate raised for %r: %r", wid, e)
        if self._join_token is not None and isinstance(token, str):
            return hmac.compare_digest(token.encode(), self._join_token.encode())
        return False

    def _roster_entry(self, wid: str, token: Any) -> _WorkerConn:
        """The worker's roster slot, registering an authorised joiner. The new
        slot starts with last_committed=-1, so the ctl handshake routes it
        through the ordinary FULL_SYNC path and it enters the next round."""
        with self.lock:
            wc = self.workers.get(wid)
            if wc is not None:
                return wc
        # the policy runs outside the registry lock (it is operator code)
        if not self._may_join(wid, token):
            raise AssertionError(
                f"unknown worker_id {wid!r} (expected {self.expected}; "
                "joining not authorised)"
            )
        with self.lock:
            wc = self.workers.get(wid)
            if wc is None:
                wc = self.workers[wid] = _WorkerConn(wid)
                log.info("worker %s joined the running fleet", wid)
            return wc

    def admit(self, worker_id: str) -> None:
        """Authorise `worker_id` to join this (running) leader."""
        with self.lock:
            self._join_allow.add(worker_id)

    def retire(self, worker_id: str) -> bool:
        """Remove `worker_id` from the roster: drop its connection between
        rounds and revoke its id-based authorisation, so a reconnect must be
        authorised again. Returns False if it was not in the roster."""
        with self.round_mutex:  # never mid-round: aggregation indexes workers
            wc = self.workers.get(worker_id)
            if wc is None:
                return False
            self._drop(wc)
            with self.lock:
                self.workers.pop(worker_id, None)
                self._join_allow.discard(worker_id)
                if worker_id in self.expected:
                    self.expected.remove(worker_id)
            log.info("worker %s retired from the fleet", worker_id)
            if self._bootstrapped:  # a slow leaver may free the fleet's plan
                self._plan_links(reason=f"retire:{worker_id}")
            return True

    def _resync_if_needed(self, wc: _WorkerConn, their_hash: bytes) -> None:
        ours = anchor_hash16(self.anchor, self.bn)
        if wc.last_committed == self.round and their_hash == ours:
            return
        # Retained-round resend is valid ONLY if the worker's anchor matches
        # the pre-round state. A worker that applied the round but died before
        # committing reports last_committed = round-1 with a POST-round anchor;
        # resending would apply the delta twice and fork consensus.
        if (
            wc.last_committed == self.round - 1
            and self.retained
            and self.retained[0] == self.round
            and their_hash == self.retained[5]
        ):
            rnd, payload, bn_bytes, rlo, rhi, _pre = self.retained
            log.info("resending retained round %d to %s", rnd, wc.worker_id)
            meta = {
                "plen": len(payload),
                "bn_len": len(bn_bytes),
                "resend": True,
                "lo": rlo,
                "hi": rhi,
                "codec": self._retained_codec,
            }
            wc.send(FT_BCAST_META, rnd, json.dumps(meta).encode())
            wc.send_data_blob(0, FT_BCAST_CHUNK, rnd, payload, 0)
            wc.send(FT_BCAST_META, rnd, bn_bytes)
            assert wc.ctl is not None  # wc.send above raised if ctl was gone
            ftype, frnd, pl = recv_ctl(wc.ctl, time.monotonic() + 120.0)
            if ftype != FT_ACK_APPLIED or frnd != rnd:
                raise WireDead(f"bad resend ack {ftype}/{frnd}")
            if bytes.fromhex(json.loads(pl)["hash16"]) != ours:
                raise WireDead(f"resend ack hash mismatch from {wc.worker_id}")
            wc.last_committed = rnd
            return
        log.info(
            "FULL_SYNC -> %s (their round %d, ours %d)",
            wc.worker_id,
            wc.last_committed,
            self.round,
        )
        blob = self.anchor.numpy().tobytes()
        bn_bytes = self.bn.numpy().tobytes()
        t_sync0 = time.monotonic()
        wc.send(
            FT_FULLSYNC_META,
            self.round,
            json.dumps({"plen": len(blob), "bn_len": len(bn_bytes)}).encode(),
        )
        wc.send_data_blob(0, FT_FULLSYNC_CHUNK, self.round, blob, 0)
        wc.send(FT_FULLSYNC_META, self.round, bn_bytes)
        assert wc.ctl is not None  # wc.send above raised if ctl was gone
        ftype, frnd, _ = recv_ctl(wc.ctl, time.monotonic() + 300.0)
        if ftype != FT_ACK_APPLIED or frnd != self.round:
            raise WireDead(f"bad fullsync ack {ftype}/{frnd}")
        wc.last_committed = self.round
        # passive link probe: the bootstrap FULL_SYNC is the bulk transfer
        self._fullsync_samples[wc.worker_id] = sample_from_fullsync(
            wc.worker_id, len(blob) + len(bn_bytes),
            time.monotonic() - t_sync0, tcp_rtt_s(wc.ctl),
        )

    # --- rounds ---
    def _gather_one(
        self, wc: _WorkerConn, gen: int, rnd: int, out: Dict[str, Any], deadline: float
    ) -> None:
        """Generation-guarded: if the worker was dropped/reconnected while we
        were blocked, we must neither touch its (new) sockets nor write into
        `uploads` for a stale stream."""
        try:
            ctl = wc.ctl
            if ctl is None or wc.gen != gen:
                return
            ftype, frnd, payload = recv_ctl(ctl, deadline)
            if ftype != FT_UPLOAD_META or frnd != rnd:
                raise WireDead(f"bad upload meta {ftype}/{frnd}")
            meta = json.loads(payload)
            plen = int(meta["plen"])
            float(meta["n_samples"])  # validate HERE: a bad key/type must
            if not 0 <= plen <= (1 << 31):  # drop THIS worker, not the round
                raise WireDead(f"absurd upload plen {plen}")
            buf = bytearray(plen)
            stripes = sorted(wc.data.keys())
            shares = [plen // len(stripes)] * len(stripes)
            shares[-1] += plen - sum(shares)
            threads = [
                ExThread(
                    recv_blob,
                    (wc.data[st], FT_UPLOAD_CHUNK, rnd, share, buf),
                    name=f"gather-{wc.worker_id}-{st}",
                )
                for st, share in zip(stripes, shares)
            ]
            for t in threads:
                t.start()
            join_all(
                threads,
                max(1.0, deadline - time.monotonic()),
                f"gather stripes {wc.worker_id}",
            )
            ftype, _, bn_payload = recv_ctl(ctl, time.monotonic() + IO_TIMEOUT)
            if ftype != FT_UPLOAD_META:
                raise WireDead(f"bad bn sideband {ftype}")
            if len(bn_payload) != self.fs.bn_numel * 4:
                raise WireDead(f"bn sideband size {len(bn_payload)}")
            if wc.gen == gen:
                out[wc.worker_id] = (meta, bytes(buf), bn_payload)
        except Exception as e:  # noqa: BLE001 — confine ANY per-worker fault
            log.warning("gather failed for %s: %s", wc.worker_id, e)
            if wc.gen == gen:
                self._drop(wc)

    def _drop(self, wc: _WorkerConn) -> None:
        with self.lock:
            wc.alive = False
            wc.gen += 1
            ctl, data = wc.ctl, wc.data
            wc.ctl = None
            wc.data = {}
            wc.data_locks = {}
        shutclose(ctl)
        for s in data.values():
            shutclose(s)

    def _connected(self) -> List[_WorkerConn]:
        with self.lock:
            return [
                w
                for w in self.workers.values()
                if w.alive and not w.syncing and w.ctl is not None
                and len(w.data) >= 1 and w.worker_id not in self.observers
            ]

    def _observer_sync(self) -> None:
        """Periodic consensus refresh for observer workers (daemon thread;
        serialized against rounds via round_mutex like handshake resyncs)."""
        for wid in self.observers:
            wc = self.workers.get(wid)
            if wc is None or not wc.alive or wc.ctl is None:
                continue
            with self.round_mutex:
                wc.syncing = True
                try:
                    self._resync_if_needed(wc, b"\x00" * 16)
                except WireDead:
                    self._drop(wc)
                except Exception as e:  # noqa: BLE001 — observers are best-effort
                    log.warning("observer sync %s: %r", wid, e)
                finally:
                    wc.syncing = False

    def run_round(self) -> Optional[Dict[str, Any]]:
        """Returns the committed round's stats record, or None on failure —
        the pacing loop must only ever pace off a FRESH measurement."""
        with self.round_mutex:
            return self._run_round_locked()

    def _run_round_locked(self) -> Optional[Dict[str, Any]]:
        rnd = self.round + 1
        parts = self._connected()
        if not parts:
            log.warning("round %d: no workers connected", rnd)
            return None
        lo, hi = shard_bounds(rnd, self.fs.numel, self.shards)
        gens = {wc.worker_id: wc.gen for wc in parts}
        up_codecs = {
            wc.worker_id: self.upload_codecs.get(wc.worker_id, self.codec)
            for wc in parts
        }
        t0 = time.monotonic()
        for wc in parts:
            try:
                wc.send(
                    FT_ROUND_BEGIN,
                    rnd,
                    json.dumps(
                        {
                            "anchor_version": self.round,
                            "lo": lo,
                            "hi": hi,
                            "codec": self.codec,
                            "up_codec": up_codecs[wc.worker_id],
                        }
                    ).encode(),
                )
            except WireDead:
                self._drop(wc)
        parts = [w for w in parts if w.alive]
        uploads: Dict[str, Any] = {}
        # gather deadline strictly below the join timeout so no zombie can
        # outlive the join and mutate `uploads` mid-aggregation.
        gather_deadline = time.monotonic() + ROUND_HARD_TIMEOUT - 10
        threads = [
            ExThread(
                self._gather_one,
                (w, gens[w.worker_id], rnd, uploads, gather_deadline),
                name=f"gather-{w.worker_id}",
            )
            for w in parts
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join(max(1.0, gather_deadline + 8 - time.monotonic()))
        for t, w in zip(threads, parts):
            if t.is_alive():
                log.warning("gather thread for %s still alive — dropping", w.worker_id)
                self._drop(w)  # gen bump neutralizes the zombie
        uploads = dict(uploads)  # snapshot: no further mutation visible
        if not uploads:
            self._abort(rnd, parts)
            return None
        t_gather = time.monotonic() - t0

        # aggregate — everything BEFORE the anchor apply may still abort the
        # round; a failure here must tell the workers (they'd otherwise hang
        # at their broadcast wait until the round ceiling).
        try:
            deltas, bns, ws, metas = [], [], [], []
            for wid, (meta, payload, bn_payload) in uploads.items():
                try:
                    up_codec = up_codecs[wid]
                    if meta.get("codec", up_codec) != up_codec:
                        raise ValueError(
                            f"upload codec {meta.get('codec')!r} != assigned {up_codec!r}"
                        )
                    d = codec_decode(payload, hi - lo, up_codec)
                    if not torch.isfinite(d).all():
                        raise FloatingPointError("non-finite delta")
                    n = float(meta["n_samples"])
                except Exception as e:  # noqa: BLE001 — confine to this worker
                    log.error("bad upload from %s (%s) — dropped", wid, e)
                    self._drop(self.workers[wid])
                    continue
                wc = self.workers[wid]
                wc.ema_n = n if wc.ema_n == 0 else 0.8 * wc.ema_n + 0.2 * n
                deltas.append(d)
                bns.append(f32_from(bn_payload))
                ws.append(wc.ema_n)
                metas.append((wid, meta))
            if not deltas:
                self._abort(rnd, parts)
                return None
            W = sum(ws)
            dbar = torch.zeros(hi - lo)
            for d, wgt in zip(deltas, ws):
                dbar.add_(d, alpha=wgt / W)
            # outer optimizer (G1): momentum on the aggregated pseudo-gradient.
            # "plain" == the original FedAvg behaviour (η_out=1, μ=0).
            m_slice = self.outer_m[lo:hi] if self.outer_m is not None else None
            # optional momentum warmup (SAKURA_MU_WARMUP=N rounds): at high
            # round frequency the first deltas otherwise replay through the
            # momentum buffer at ~1/(1-mu) gain before averaging stabilizes.
            mu = self.outer_momentum
            mu_warm = int(os.environ.get("SAKURA_MU_WARMUP", "0"))
            if mu_warm > 0:
                mu = mu * min(1.0, (self.round + 1) / mu_warm)
            buf_slice = self.outer_buf[lo:hi] if self.outer_buf is not None else None
            n_up = self.outer_steps.get(lo, 0) + 1
            self.outer_steps[lo] = n_up
            update = outer_step(
                dbar, m_slice, self.outer_kind, self.outer_lr, mu,
                buf=buf_slice, step=n_up, delay=self.outer_delay,
            )
            update = self._fold_update(update, lo, hi)
            # broadcast-side (dual-side) error feedback for any lossy codec.
            # outer "ef" (SparseLoCo): the residual is the momentum, decayed by μ.
            if self.ef_bcast is not None:
                beta = ef_decay(self.outer_kind, mu)
                ef_slice = self.ef_bcast[lo:hi]
                update = update + (ef_slice if beta == 1.0 else ef_slice * beta)
                payload, ef_new, contraction = codec_encode(update, None, self.codec)
                self.ef_bcast[lo:hi].copy_(ef_new)
            else:
                payload, _, contraction = codec_encode(update, None, self.codec)
            bn_merged = merge_bn_pooled(bns, ws, self.mean_mask, self.var_mask)
            bn_bytes = bn_merged.numpy().tobytes()
        except Exception:  # noqa: BLE001 — pre-commit failure: abort cleanly
            log.exception("round %d aggregation failed", rnd)
            self._abort(rnd, parts)
            return None

        adv = self._scope_advise(metas) if self.scope_advisory is None else None

        # leader applies the SAME bytes through the same function. The hash
        # of the PRE-round anchor travels with the retained payload: it is
        # the resend-eligibility check (see _resync_if_needed).
        pre_hash = anchor_hash16(self.anchor, self.bn)
        apply_round_to_anchor(self.anchor, payload, self.codec, lo, hi)
        self.bn.copy_(bn_merged)
        our_hash = anchor_hash16(self.anchor, self.bn)

        # broadcast + collect acks
        t_bc0 = time.monotonic()
        acked = []
        for wc in [w for w in parts if w.alive and w.worker_id in uploads]:
            if wc.gen != gens[wc.worker_id]:
                continue  # reconnected mid-round: resync path owns it now
            try:
                wc.send(
                    FT_BCAST_META,
                    rnd,
                    json.dumps(
                        {
                            "plen": len(payload),
                            "bn_len": len(bn_bytes),
                            **({"scope_advisory": adv} if adv else {}),
                        }
                    ).encode(),
                )
                stripes = sorted(wc.data.keys())
                view = memoryview(payload)
                shares = [len(payload) // len(stripes)] * len(stripes)
                shares[-1] += len(payload) - sum(shares)
                base = 0
                sthreads = []
                for st, share in zip(stripes, shares):
                    part = bytes(view[base : base + share])
                    tt = ExThread(
                        wc.send_data_blob,
                        (st, FT_BCAST_CHUNK, rnd, part, base),
                        name=f"bcast-{wc.worker_id}-{st}",
                    )
                    base += share
                    tt.start()
                    sthreads.append(tt)
                join_all(sthreads, ROUND_HARD_TIMEOUT, f"bcast stripes {wc.worker_id}")
                wc.send(FT_BCAST_META, rnd, bn_bytes)  # bn sideband
                acked.append(wc)
            except WireDead as e:
                log.warning("broadcast to %s failed: %s", wc.worker_id, e)
                self._drop(wc)
        t_bcast = time.monotonic() - t_bc0
        ok_hash = True
        for wc in acked:
            try:
                # capture once + gen-check: wc.ctl may have been replaced by a
                # reconnect since the broadcast; reading the NEW socket here
                # would steal the resync path's frames.
                ctl = wc.ctl
                if ctl is None or wc.gen != gens[wc.worker_id]:
                    continue
                ftype, frnd, pl = recv_ctl(ctl, time.monotonic() + ROUND_HARD_TIMEOUT)
                if ftype != FT_ACK_APPLIED or frnd != rnd:
                    raise WireDead(f"bad ack {ftype}/{frnd}")
                their = json.loads(pl)
                if bytes.fromhex(their["hash16"]) != our_hash:
                    log.critical(
                        "CONSENSUS HASH MISMATCH worker=%s round=%d", wc.worker_id, rnd
                    )
                    ok_hash = False
                    self._drop(wc)  # reconnect path triggers FULL_SYNC
                else:
                    wc.last_committed = rnd
            except (WireDead, AssertionError, json.JSONDecodeError) as e:
                log.warning("ack failed for %s: %s", wc.worker_id, e)
                self._drop(wc)

        self.round = rnd
        self.retained = (rnd, payload, bn_bytes, lo, hi, pre_hash)
        self._retained_codec = self.codec
        rec = {
            "round": rnd,
            "t_gather_s": round(t_gather, 3),
            "t_xfer_s": round(t_gather + t_bcast, 3),
            "t_total_s": round(time.monotonic() - t0, 3),
            "participants": [m[0] for m in metas],
            "n_samples": {m[0]: m[1]["n_samples"] for m in metas},
            "weights": {
                wid: round(w / W, 4) for wid, w in zip([m[0] for m in metas], ws)
            },
            "bytes_up": {m[0]: m[1]["plen"] for m in metas},
            "regions": sorted({self.workers[m[0]].region for m in metas}),
            "bytes_down": len(payload),
            "contraction": round(contraction, 4),
            "codec": self.codec,
            "shards": self.shards,
            "hash16": our_hash.hex(),
            "consensus_ok": ok_hash,
            "overruns": self.overruns,
        }
        if adv is not None:
            rec["scope_advisory"] = adv
            self.scope_advisory = adv
            (log.warning if adv["verdict"] == "warn" else log.info)(
                "%s", format_advisory(adv)
            )
        self.stats.append(rec)
        print("LEADER_ROUND " + json.dumps(rec), flush=True)
        if self.persist_path:
            self._persist(rnd, self.persist_path)
        return rec

    # --- persistence (SAK-0015) ---
    # Optimizer buffers persisted next to {round, anchor, bn}. All of them are
    # mutated only inside the aggregation of a round, so at the commit of
    # round r they are exactly the state as of r: saving them in the same
    # atomic write keeps "the file holds the last committed round" true for
    # them too (a later snapshot would double-apply on the re-run round).
    _OPT_TENSORS = ("ef_bcast", "outer_m", "outer_buf")

    def _opt_state(self) -> Dict[str, Any]:
        st: Dict[str, Any] = {
            k: getattr(self, k) for k in self._OPT_TENSORS if getattr(self, k) is not None
        }
        st["outer_steps"] = dict(self.outer_steps)
        return st

    def _persist(self, rnd: int, path: str) -> None:
        tmp = path + ".tmp"
        snap: Dict[str, Any] = {
            "round": rnd, "anchor": self.anchor, "bn": self.bn, **self._opt_state()
        }
        if self.link_plan is not None:  # additive key (SAK-0013)
            snap["link_plan"] = self.link_plan.to_state()
        torch.save(snap, tmp)
        os.replace(tmp, path)  # atomic per-commit persist

    def _restore_opt_state(self, st: Dict[str, Any]) -> None:
        """Additive restore: a file that predates a key (or a buffer this
        configuration does not keep) leaves that buffer at its fresh value,
        which is the pre-#143 behaviour."""
        for k in self._OPT_TENSORS:
            buf, saved = getattr(self, k), st.get(k)
            if buf is None or saved is None:
                if saved is not None:
                    log.warning("persisted %s ignored: not used by this configuration", k)
                continue
            if saved.numel() != buf.numel():
                log.warning("persisted %s has %d elements, model has %d; starting at zero",
                            k, saved.numel(), buf.numel())
                continue
            buf.copy_(saved)
        steps = st.get("outer_steps")
        if isinstance(steps, dict):
            self.outer_steps = {int(lo): int(n) for lo, n in steps.items()}

    def _scope_advise(
        self, metas: List[Tuple[str, Dict[str, Any]]]
    ) -> Optional[Dict[str, Any]]:
        """Per-worker samples/s = n_samples / window_s from the upload meta
        (workers that predate window_s are left out). The dataset size is the
        Leader's ``dataset_size`` or, failing that, the sum of every
        participant's reported ``shard_size`` (IID disjoint shards). Never
        raises: an advisory must not cost a round."""
        try:
            rates: Dict[str, float] = {}
            shards: List[int] = []
            for wid, meta in metas:
                win = float(meta.get("window_s") or 0.0)
                if win > 0:
                    rates[wid] = float(meta["n_samples"]) / win
                sz = meta.get("shard_size")
                if isinstance(sz, int) and sz > 0:
                    shards.append(sz)
            size = self.dataset_size
            if not size and shards and len(shards) == len(metas):
                size = sum(shards)
            return scope_advisory(rates, size)
        except Exception:  # noqa: BLE001 — advisory only
            log.exception("scope advisory failed")
            return None

    def _fold_update(self, update: torch.Tensor, lo: int, hi: int) -> torch.Tensor:
        """Hook between the outer step and the broadcast encode. The star
        leader commits the outer-step update as is; a regional aggregator
        (region.py, SAK-0011) folds the pending inter-region correction in."""
        return update

    def _abort(self, rnd: int, parts: List[_WorkerConn]) -> None:
        log.warning("round %d aborted", rnd)
        for wc in parts:
            if wc.alive and wc.ctl:
                try:
                    wc.send(FT_ROUND_ABORT, rnd)
                except WireDead:
                    self._drop(wc)

    def _heartbeat(self) -> None:
        # Heartbeats flow DURING rounds too: every recv path skips FT_HEARTBEAT
        # and send_lock prevents mid-frame interleave, while a long round with
        # a quiet ctl would otherwise lose its conntrack entry and give workers
        # nothing to detect a dead leader by.
        while not self.stop.is_set():
            time.sleep(5.0)
            # observers are excluded from rounds but must still be kept alive
            with self.lock:
                targets = [
                    w for w in self.workers.values()
                    if w.alive and not w.syncing and w.ctl is not None
                    and len(w.data) >= 1
                ]
            for wc in targets:
                try:
                    wc.send(FT_HEARTBEAT, self.round)
                except WireDead:
                    self._drop(wc)
                except Exception as e:  # noqa: BLE001 — heartbeat must never die
                    log.warning("heartbeat to %s: %r", wc.worker_id, e)

    def _plan_links(self, reason: str = "bootstrap") -> None:
        """Link probe (SAK-0013): log every measured link and, for
        codec="auto" / shards=0, pick the fleet codec, K and per-worker
        upload codecs from the slowest one. Runs at bootstrap and again when
        a worker joins or leaves; mid-run it is called between rounds (under
        round_mutex), so a new plan takes effect at the next ROUND_BEGIN."""
        try:
            if self.link_probe is not None:
                samples = self.link_probe()
            else:
                # latest measurement per worker; a restored plan's samples
                # stand in for workers that resumed without a FULL_SYNC
                samples = dict(self._restored_samples)
                samples.update({s.worker_id: s for s in
                                (self.link_plan.samples if self.link_plan else [])})
                samples.update(self._fullsync_samples)
        except Exception:  # noqa: BLE001 — a probe fault must not stop the fleet
            log.exception("link probe failed; keeping codec=%s shards=%d",
                          self.codec, self.shards)
            return
        samples = {w: s for w, s in samples.items()
                   if w in self.workers and w not in self.observers}
        plan = plan_links(
            samples, self.fs.numel, self.fs.bn_numel,
            round_budget_s(self.gap_s, self.link_duty),
            codec=None if self._auto_codec else self.codec,
            shards=None if self._auto_shards else self.shards,
        )
        if plan is None:
            if self.link_plan is not None:
                log.info("link probe (%s): no samples; keeping the %s plan",
                         reason, self.link_plan.codec)
            else:
                log.warning("link probe: no samples; keeping codec=%s shards=%d",
                            self.codec, self.shards)
            return
        if reason != "bootstrap" and plan.same_decision(self.link_plan):
            self.link_plan = plan  # fresher measurements, same bytes on the wire
            log.info("link probe (%s): plan unchanged (%s, K=%d)",
                     reason, plan.codec, plan.shards)
            return
        self._apply_plan(plan)
        if not plan.fits:
            log.warning("link probe: no rung fits the %.2fs round budget; "
                        "using the most aggressive (%s, K=%d)",
                        plan.budget_s, plan.codec, plan.shards)
        rec = {**plan.as_record(), "reason": reason, "from_round": self.round + 1}
        self.plan_history.append(rec)
        log.info("link probe: %s", json.dumps(rec))
        print("LEADER_LINK_PLAN " + json.dumps(rec), flush=True)

    def _replan_on_join(self, wid: str, sample_before: Optional[LinkSample]) -> None:
        """Re-probe after a (re)join: the FULL_SYNC just measured `wid`'s link,
        or (injected probe) `wid` is new to the plan. Called under round_mutex."""
        fresh = self._fullsync_samples.get(wid) is not sample_before
        known = self.link_plan is not None and any(
            s.worker_id == wid for s in self.link_plan.samples)
        if known and not fresh:
            return
        try:
            self._plan_links(reason=f"join:{wid}")
        except Exception:  # noqa: BLE001 — a re-plan must never cost the joiner
            log.exception("re-plan after %s joined failed", wid)

    def _apply_plan(self, plan: LinkPlan) -> None:
        """Adopt `plan` between rounds. Operator-set dimensions never move.
        The broadcast error-feedback residual is kept across a lossy->lossy
        codec change (it is un-sent update mass, codec-independent); a K
        change restarts the per-shard outer-step windows (their slices moved)."""
        self.link_plan = plan
        codec = plan.codec if self._auto_codec else self.codec
        shards = plan.shards if self._auto_shards else self.shards
        if shards != self.shards:
            self.outer_steps = {}
            if self.outer_buf is not None:
                self.outer_buf.zero_()
        self.codec, self.shards = codec, shards
        self.upload_codecs = dict(plan.upload_codecs) if self._auto_codec else {}
        if self.ef_bcast is None and is_lossy(self.codec):
            self.ef_bcast = torch.zeros(self.fs.numel)
        elif not is_lossy(self.codec):
            self.ef_bcast = None

    def serve_forever(self, warmup_s: float = 5.0) -> None:
        threading.Thread(target=self._serve, daemon=True).start()
        threading.Thread(target=self._heartbeat, daemon=True).start()
        # wait for ALL expected workers (observers included: their handshake
        # resync must land before rounds start monopolizing round_mutex)
        def _all_present() -> bool:
            with self.lock:
                live = [
                    w.worker_id for w in self.workers.values()
                    if w.alive and not w.syncing and w.ctl is not None
                    and len(w.data) >= 1
                ]
            return len(live) >= len(self.expected)
        while not _all_present() and not self.stop.is_set():
            time.sleep(0.5)
        log.info("all %d workers connected; warmup %.0fs", len(self.expected), warmup_s)
        self._plan_links()
        self._bootstrapped = True  # from here on joins / leaves re-plan
        time.sleep(warmup_s)
        paced_gap = self.gap_s
        while not self.stop.is_set() and self.round < self.max_rounds:
            t_round0 = time.monotonic()
            self._in_round.set()
            rec = None
            try:
                rec = self.run_round()
            except Exception:  # noqa: BLE001 — leader must survive any round
                log.exception("round crashed; continuing")
            finally:
                self._in_round.clear()
            if (rec is not None and self.observers
                    and self.round % self.obs_sync_every == 0):
                # synchronous: round_mutex is free right now — a background
                # thread would starve against back-to-back sub-second rounds
                self._observer_sync()
            gap = self.gap_s
            if self.link_duty > 0:
                # Pace on the TRANSFER legs of a FRESH, successful round.
                # t_total includes serial ack waits; a failed round must keep
                # the previous gap, not re-pace off a stale entry.
                if rec is not None:
                    t_x = rec["t_xfer_s"]
                    paced_gap = max(3.0, min(120.0, t_x * (1.0 / self.link_duty - 1.0)))
                    log.info(
                        "self-paced gap: %.1fs (xfer cost %.1fs, duty %.0f%%)",
                        paced_gap,
                        t_x,
                        100 * self.link_duty,
                    )
                gap = paced_gap
            # gap_s is a PERIOD (round-start to round-start), not idle time
            # appended after the round: at sub-second gaps the round's own
            # wall-time would otherwise dominate and quantize the cadence.
            t_next = t_round0 + gap
            now = time.monotonic()
            if now >= t_next:
                self.overruns += 1
                t_next = now
            while not self.stop.is_set():
                remaining = t_next - time.monotonic()
                if remaining <= 0:
                    break
                time.sleep(min(0.05, remaining))
