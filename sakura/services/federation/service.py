"""FederationService: sakura BaseService for cross-region local-SGD workers."""

from __future__ import annotations

import json
import logging
import os
import socket
import threading
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch

from sakura.service import BaseService

from .codec import (FlatState, anchor_hash16, apply_round_to_anchor,
                    codec_encode, f32_from, is_lossy)
from .rejoin import (DEFAULT_AVERAGE_DECAY, DEFAULT_REJOIN_POLICY,
                     apply_rejoin_policy, check_policy)
from .scope import format_advisory
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


class FederationService(BaseService):
    """sakura service: all parameter mutation happens on the training thread
    at step boundaries (on_train_step_begin); the background thread only
    talks to the leader and stages CPU buffers.

    Scope (SAK-0017): federation assumes IID shards and the same architecture
    on every worker. Non-IID shards add client drift that nothing here
    corrects or detects. It only beats the fleet's fastest GPU training alone
    when the job is compute-bound; at the first completed round the leader's
    scope advisory (``self.scope_advisory``) is logged here, as a warning
    when the fleet is likely to lose. Advisory only, never a refusal.
    ``shard_size`` (samples this worker sweeps per epoch; read from
    ``len(train_loader.sampler)`` when not given) feeds that advisory."""

    name = "federation"
    priority = 50

    def __init__(
        self,
        leader_host: str,
        leader_port: int,
        worker_id: str,
        batch_size: int,
        mode: str = "overlap",
        codec: str = "int8",
        region: str = "default",
        max_stale_steps: Optional[int] = None,
        shard_size: Optional[int] = None,
        join_token: Optional[str] = None,
        rejoin_momentum: str = DEFAULT_REJOIN_POLICY,
        rejoin_momentum_decay: float = DEFAULT_AVERAGE_DECAY,
        reconnect_backoff_max_s: float = 30.0,
        fallback_hosts: Optional[Sequence[Tuple[str, int]]] = None,
        failover_after: int = 3,
    ) -> None:
        super().__init__()
        assert mode in ("overlap", "block")
        check_policy(rejoin_momentum, rejoin_momentum_decay)
        assert max_stale_steps is None or max_stale_steps >= 0
        assert reconnect_backoff_max_s > 0
        # Ceiling of the exponential retry interval for the bootstrap and the
        # reconnect loops (they start at 2 s / 1 s and double). A LAN fleet or
        # a test whose leader restarts in about a second can lower it so a
        # worker does not sit out a long backoff after the leader is back.
        self.reconnect_backoff_max_s = reconnect_backoff_max_s
        self.leader = (leader_host, leader_port)
        # Failover (#144, SAK-0011): after `failover_after` consecutive failed
        # connection attempts to the current endpoint, try the next one of
        # [leader, *fallback_hosts] (cyclic, so a dead fallback leads back to
        # the primary). No failback while the current endpoint works. A worker
        # arriving at an endpoint whose anchor it does not hold presents
        # last_committed=-1, so that leader FULL_SYNCs it (its join policy --
        # `allow_join` / `join_token` -- must admit this worker_id).
        if failover_after < 1:
            raise ValueError(f"failover_after must be >= 1, got {failover_after}")
        self.endpoints: List[Tuple[str, int]] = [self.leader] + [
            (str(h), int(p)) for h, p in (fallback_hosts or [])
        ]
        self.failover_after = failover_after
        self._ep = 0  # index of self.leader in self.endpoints
        self._ep_failures = 0  # consecutive failed attempts at self._ep
        self._synced_ep: Optional[int] = None  # endpoint whose anchor we hold
        self.worker_id = worker_id
        self.region = region
        # SAK-0014: presented in the HELLO so a leader that does not list this
        # worker_id in `expected` can still admit it (leader `join_token=`).
        self.join_token = join_token
        # SAK-0014: what a FULL_SYNC does to the inner optimizer's momentum
        # ("noop" | "reset" | "average"; see rejoin.py). Applied on the
        # training thread together with the parameter hard-set.
        self.rejoin_momentum = rejoin_momentum
        self.rejoin_momentum_decay = rejoin_momentum_decay
        self._optimizer: Any = None
        self._fullsync_pending = False
        self.batch_size = batch_size
        self.mode = mode
        # codec = this worker's UPLOAD codec; bcast_codec decodes the leader's
        # broadcast. Both follow the leader's ROUND_BEGIN (SAK-0013), so the
        # constructor value only matters against a leader that omits them.
        self.codec = codec
        self.bcast_codec = codec
        # Streaming-DiLoCo staleness bound (overlap mode only): at most this
        # many optimizer steps may run between a round's snapshot and its
        # apply; the trainer waits at the step boundary once it is reached.
        # None = unbounded (one round of staleness, the E5 default).
        self.max_stale_steps = max_stale_steps
        self._steps_since_snap = 0
        self._n = 0
        self._n_lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        # training-thread <-> bg-thread handoff. _svc_lock serializes the
        # trainer's request servicing against the bg thread's request
        # lifecycle: after a timed-out request the bg thread acquires it to
        # withdraw the request, which guarantees no service is mid-flight
        # before it touches anchor/apply_vec again (a stale set-request
        # serviced during the NEXT round's anchor mutation tears parameters).
        self._svc_lock = threading.Lock()
        self._snap_req = threading.Event()
        self._snap_done = threading.Event()
        self._apply_req = threading.Event()
        self._apply_done = threading.Event()
        self._set_req = threading.Event()
        self._set_done = threading.Event()
        self._round_active = threading.Event()
        self.records: List[Dict[str, Any]] = []
        self._last_losses: List[float] = []
        self.fs: Optional[FlatState] = None
        # SAK-0017 scope advisory inputs: wall-clock training window per round
        # (throughput = samples / window) and this worker's shard size.
        self.shard_size = shard_size
        self._win_t0 = time.monotonic()
        self._snap_win: Optional[float] = None  # window start of the snapshot
        self._snap_window_s = 0.0
        self.scope_advisory: Optional[Dict[str, Any]] = None

    # ---------------- sakura event hooks (training thread) ----------------
    def on_train_begin(self, event: Any) -> None:
        model = event.model
        inner = getattr(model, "module", model)
        self.fs = FlatState(inner)
        self._optimizer = getattr(event, "optimizer", None)
        n = self.fs.numel
        self.anchor = torch.empty(n, dtype=torch.float32)
        # Worker-side error feedback for every lossy codec (int8, int4,
        # topk:*), matching the leader's broadcast-side ef_bcast.
        self.ef = torch.zeros(n) if is_lossy(self.codec) else None
        self.snap = (
            torch.empty(n, dtype=torch.float32).pin_memory()
            if self.fs.device.type == "cuda"
            else torch.empty(n, dtype=torch.float32)
        )
        self.delta_true = torch.empty(n, dtype=torch.float32)
        self.apply_vec = (
            torch.empty(n, dtype=torch.float32).pin_memory()
            if self.fs.device.type == "cuda"
            else torch.empty(n, dtype=torch.float32)
        )
        self._bn_pending: Optional[torch.Tensor] = None
        self._bn_snap = torch.empty(self.fs.bn_numel, dtype=torch.float32)
        self._snap_n = 0
        # Bootstrap retries like _reconnect does: a worker (re)starting while
        # a round is in flight waits out the round_mutex on the leader side;
        # one failed attempt must not kill the whole training process.
        backoff = min(2.0, self.reconnect_backoff_max_s)
        while not self._stop.is_set():
            try:
                self._connect_and_bootstrap()
                break
            except (WireDead, OSError, AssertionError) as e:
                for s in [
                    getattr(self, "ctl", None),
                    *getattr(self, "data", {}).values(),
                ]:
                    shutclose(s)
                log.warning(
                    "%s bootstrap failed: %s (retry in %.0fs)",
                    self.worker_id,
                    e,
                    backoff,
                )
                if self._connect_failed():
                    backoff = min(2.0, self.reconnect_backoff_max_s)
                    continue  # try the next endpoint right away
                # interruptible: on_train_end must not wait out a long backoff
                self._stop.wait(backoff)
                backoff = min(backoff * 2, self.reconnect_backoff_max_s)
        if self.shard_size is None:
            try:
                self.shard_size = len(event.train_loader.sampler)
            except Exception:  # noqa: BLE001 — unknown size skips one check
                self.shard_size = None
        with self._n_lock:
            self._win_t0 = time.monotonic()  # training starts now
        self._thread = threading.Thread(target=self._bg_loop, daemon=True)
        self._thread.start()

    def on_train_step_begin(self, event: Any) -> None:
        self._service_round_requests()
        if self.mode == "block" and self._round_active.is_set():
            t0 = time.monotonic()
            while self._round_active.is_set() and not self._stop.is_set():
                self._service_round_requests()
                time.sleep(0.005)
            self._rec("block_wait_ms", (time.monotonic() - t0) * 1e3)
        elif self._stale_bound_hit():
            t0 = time.monotonic()
            while self._stale_bound_hit() and not self._stop.is_set():
                self._service_round_requests()
                time.sleep(0.001)
            self._rec("stale_wait_ms", (time.monotonic() - t0) * 1e3)

    def _stale_bound_hit(self) -> bool:
        return (
            self.max_stale_steps is not None
            and self._round_active.is_set()
            and self._snap_done.is_set()
            and self._steps_since_snap >= self.max_stale_steps
        )

    def on_optimizer_step(self, event: Any) -> None:
        with self._n_lock:
            self._n += self.batch_size
        self._steps_since_snap += 1

    def on_epoch_end(self, event: Any) -> None:
        loss = event.metrics.get("loss")
        if loss is not None:
            self._last_losses.append(float(loss))
            del self._last_losses[:-5]

    def on_runtime_shutdown(self, runtime: Any) -> None:
        self._stop.set()
        self._round_active.clear()
        # Wake the bg thread NOW: it can be 80s deep in a ctl recv or waiting
        # on a snapshot the departed trainer will never service; a silent
        # join(10) expiry leaves it mutating `records` while the driver
        # serializes them.
        for s in [getattr(self, "ctl", None), *getattr(self, "data", {}).values()]:
            shutclose(s)
        if self._thread:
            self._thread.join(10.0)
            if self._thread.is_alive():
                log.warning("%s bg thread did not exit cleanly", self.worker_id)

    # ---------------- training-thread helpers ----------------
    def _service_round_requests(self) -> None:
        if not (
            self._snap_req.is_set()
            or self._apply_req.is_set()
            or self._set_req.is_set()
        ):
            return  # fast path: uncontended check, no lock per step
        assert self.fs is not None
        with self._svc_lock:
            if self._snap_req.is_set():
                t0 = time.monotonic()
                self.fs.snapshot_params_to(self.snap)
                self._bn_snap.copy_(self.fs.bn_vec())  # BN read belongs HERE:
                # the bg thread reading BN mid-step races the forward pass
                with self._n_lock:
                    self._snap_n, self._n = self._n, 0
                    now = time.monotonic()
                    self._snap_win, self._win_t0 = self._win_t0, now
                    self._snap_window_s = now - self._snap_win
                self._steps_since_snap = 0
                self._snap_req.clear()
                self._snap_done.set()
                self._rec("snapshot_stall_ms", (time.monotonic() - t0) * 1e3)
            if self._apply_req.is_set():
                t0 = time.monotonic()
                self.fs.add_params_from(self.apply_vec)
                if self._bn_pending is not None:
                    self.fs.set_bn_from(self._bn_pending)
                    self._bn_pending = None
                self._apply_req.clear()
                self._apply_done.set()
                self._rec("apply_stall_ms", (time.monotonic() - t0) * 1e3)
            if self._set_req.is_set():
                # resync: hard-set params/BN to the anchor (loses local
                # progress since the failed round's snapshot — correct by
                # protocol).
                self.fs.set_params_from(self.anchor)
                if self._bn_pending is not None:
                    self.fs.set_bn_from(self._bn_pending)
                    self._bn_pending = None
                if self._fullsync_pending:
                    self._fullsync_pending = False
                    n = apply_rejoin_policy(
                        self._optimizer,
                        self.rejoin_momentum,
                        self.rejoin_momentum_decay,
                    )
                    self._rec("rejoin_momentum_buffers", float(n))
                self._set_req.clear()
                self._set_done.set()

    def _post(
        self,
        ev_req: threading.Event,
        ev_done: threading.Event,
        timeout: float,
        what: str,
    ) -> None:
        """Post a request to the training thread and wait for completion.
        On timeout the request is WITHDRAWN under _svc_lock — acquiring it
        guarantees any in-flight service has finished."""
        ev_done.clear()
        ev_req.set()
        if ev_done.wait(timeout):
            return
        with self._svc_lock:
            ev_req.clear()
            if ev_done.is_set():  # completed in the timeout/lock window
                return
        raise WireDead(f"training thread never serviced {what}")

    def _rec(self, key: str, val: float) -> None:
        if self.records and "ts" in self.records[-1] and key not in self.records[-1]:
            self.records[-1][key] = round(val, 2)

    # ---------------- endpoints / failover ----------------
    def _connect_failed(self) -> bool:
        """Count one failed connection attempt at the current endpoint; after
        `failover_after` in a row move to the next endpoint. True when the
        endpoint changed (the caller retries at once, with a fresh backoff)."""
        self._ep_failures += 1
        if len(self.endpoints) < 2 or self._ep_failures < self.failover_after:
            return False
        old = self.leader
        self._ep = (self._ep + 1) % len(self.endpoints)
        self.leader = self.endpoints[self._ep]
        self._ep_failures = 0
        log.warning("%s failing over from %s:%d to %s:%d", self.worker_id,
                    old[0], old[1], self.leader[0], self.leader[1])
        return True

    def _hello(self) -> Dict[str, Any]:
        """ctl HELLO. Our committed state means something only to the endpoint
        we last synced with: anywhere else (first contact, failover) we ask for
        a FULL_SYNC with last_committed=-1."""
        synced = self._synced_ep == self._ep
        hello: Dict[str, Any] = {
            "worker_id": self.worker_id,
            "region": self.region,
            "role": "ctl",
            "torch": torch.__version__.split("+")[0],
            "last_committed": self.committed_round if synced else -1,
            "anchor_hash16": anchor_hash16(self.anchor, self._bn_anchor).hex()
            if synced else "00" * 16,
        }
        if self.join_token is not None:
            hello["join_token"] = self.join_token
        return hello

    # ---------------- background thread ----------------
    def _connect_and_bootstrap(self) -> None:
        assert self.fs is not None
        host, port = self.leader
        self._synced_ep = None  # bootstrap: always a FULL_SYNC
        hello = self._hello()
        # Data stripes FIRST: the leader's resync/full-sync fires on the ctl
        # HELLO and needs the data sockets already registered.
        self.data = {}
        for st in (0, 1):
            s = connect(host, port)
            send_frame(
                s,
                FT_HELLO,
                0,
                json.dumps({**hello, "role": "data", "stripe": st}).encode(),
            )
            recv_frame(s, time.monotonic() + IO_TIMEOUT)
            self.data[st] = s
        self.ctl = connect(host, port)
        send_frame(self.ctl, FT_HELLO, 0, json.dumps(hello).encode())
        ftype, _, payload = recv_ctl(self.ctl, time.monotonic() + IO_TIMEOUT)
        assert ftype == FT_HELLO_ACK, ftype
        ack = json.loads(payload)
        assert ack["numel"] == self.fs.numel, "model shape mismatch with leader"
        # bootstrap FULL_SYNC (leader sends because last_committed=-1)
        ftype, rnd, payload = recv_ctl(self.ctl, time.monotonic() + 300.0)
        assert ftype == FT_FULLSYNC_META, ftype
        meta = json.loads(payload)
        buf = bytearray(meta["plen"])
        recv_blob(self.data[0], FT_FULLSYNC_CHUNK, rnd, meta["plen"], buf)
        self.anchor.copy_(torch.frombuffer(buf, dtype=torch.float32))
        ftype, _, bn_payload = recv_ctl(self.ctl, time.monotonic() + IO_TIMEOUT)
        assert ftype == FT_FULLSYNC_META
        bn = f32_from(bn_payload).clone()
        self.fs.set_params_from(self.anchor)
        self.fs.set_bn_from(bn)
        self._bn_anchor = bn
        self.committed_round = rnd
        self._synced_ep = self._ep
        self._ep_failures = 0
        send_frame(
            self.ctl,
            FT_ACK_APPLIED,
            rnd,
            json.dumps({"hash16": anchor_hash16(self.anchor, bn).hex()}).encode(),
        )
        log.info("%s bootstrapped at round %d", self.worker_id, rnd)

    def _bg_loop(self) -> None:
        while not self._stop.is_set():
            try:
                ftype, rnd, payload = recv_frame(
                    self.ctl, time.monotonic() + IO_TIMEOUT + 60
                )
            except WireDead as e:
                if self._stop.is_set():
                    return
                log.warning("%s ctl dead (%s); reconnecting", self.worker_id, e)
                self._reconnect()
                continue
            try:
                if ftype == FT_HEARTBEAT:
                    continue
                if ftype == FT_ROUND_BEGIN:
                    self._do_round(rnd, json.loads(payload))
                elif ftype == FT_ROUND_ABORT:
                    self._round_active.clear()
                elif ftype == FT_FULLSYNC_META:
                    self._handle_fullsync(rnd, payload)
                elif ftype == FT_BCAST_META:
                    # retained-round resend after reconnect: our snapshot for
                    # that round is gone, so apply to the anchor then hard-set
                    # params to it (drop local progress; rare, correct).
                    self._handle_resend(rnd, payload)
            except Exception as e:  # noqa: BLE001 — bg thread must never die
                log.warning(
                    "%s frame %s round %d failed (%r); reconnecting",
                    self.worker_id,
                    ftype,
                    rnd,
                    e,
                )
                self._round_active.clear()
                self._reconnect()

    def _handle_fullsync(self, rnd: int, payload: bytes) -> None:
        meta = json.loads(payload)
        buf = bytearray(meta["plen"])
        recv_blob(self.data[0], FT_FULLSYNC_CHUNK, rnd, meta["plen"], buf)
        self.anchor.copy_(torch.frombuffer(buf, dtype=torch.float32))
        ftype, _, bn_payload = recv_ctl(self.ctl, time.monotonic() + IO_TIMEOUT)
        assert ftype == FT_FULLSYNC_META
        bn = f32_from(bn_payload).clone()
        self._bn_anchor = bn
        self._bn_pending = bn
        if self.ef is not None:
            self.ef.zero_()  # residual referenced the old trajectory
        self._fullsync_pending = True  # rejoin momentum policy (SAK-0014)
        try:
            self._post(self._set_req, self._set_done, 60.0, "resync set")
        finally:
            self._fullsync_pending = False
        self.committed_round = rnd
        self._synced_ep = self._ep  # this endpoint's anchor from now on
        send_frame(
            self.ctl,
            FT_ACK_APPLIED,
            rnd,
            json.dumps({"hash16": anchor_hash16(self.anchor, bn).hex()}).encode(),
        )
        log.info("%s resynced at round %d", self.worker_id, rnd)

    def _handle_resend(self, rnd: int, payload: bytes) -> None:
        bmeta = json.loads(payload)
        buf = bytearray(bmeta["plen"])
        recv_blob(self.data[0], FT_BCAST_CHUNK, rnd, bmeta["plen"], buf)
        ftype, _, bn_bytes = recv_ctl(self.ctl, time.monotonic() + IO_TIMEOUT)
        assert ftype == FT_BCAST_META
        apply_round_to_anchor(
            self.anchor,
            bytes(buf),
            bmeta.get("codec", self.bcast_codec),
            bmeta.get("lo", 0),
            bmeta.get("hi"),
        )
        bn = f32_from(bn_bytes).clone()
        self._bn_anchor = bn
        if self.ef is not None:
            self.ef.zero_()
        self._bn_pending = bn
        self._post(self._set_req, self._set_done, 60.0, "resend set")
        self.committed_round = rnd
        send_frame(
            self.ctl,
            FT_ACK_APPLIED,
            rnd,
            json.dumps(
                {"hash16": anchor_hash16(self.anchor, self._bn_anchor).hex()}
            ).encode(),
        )

    def _do_round(self, rnd: int, meta: Dict[str, Any]) -> None:
        assert (
            meta["anchor_version"] == self.committed_round
        ), f"anchor version skew: {meta} vs {self.committed_round}"
        committed = False
        try:
            committed = self._do_round_inner(rnd, meta)
        finally:
            if not committed:
                # round never applied: undo staged state.
                # EF is only written post-commit, so it needs no restore;
                # the window's samples belong to the next round.
                with self._n_lock:
                    self._n += self._snap_n
                    if self._snap_win is not None:  # window joins the next too
                        self._win_t0 = self._snap_win
            # zero UNCONDITIONALLY: a later failure before the next snapshot
            # is serviced must not re-add THIS round's count to _n.
            self._snap_n = 0
            self._snap_win = None

    def _follow_leader_codecs(self, rmeta: Dict[str, Any]) -> None:
        """Adopt the codecs the leader announced for this round. A new upload
        codec restarts error feedback: the old residual is in another basis."""
        assert self.fs is not None
        up = rmeta.get("up_codec", self.codec)
        self.bcast_codec = rmeta.get("codec", self.bcast_codec)
        if up != self.codec:
            log.info("%s upload codec %s -> %s (leader)", self.worker_id, self.codec, up)
            self.ef = torch.zeros(self.fs.numel) if is_lossy(up) else None
            self.codec = up

    def _do_round_inner(self, rnd: int, rmeta: Dict[str, Any]) -> bool:
        assert self.fs is not None
        self._follow_leader_codecs(rmeta)
        lo = rmeta.get("lo", 0)
        hi = rmeta.get("hi", self.fs.numel)
        t_round0 = time.monotonic()
        rec = {
            "ts": time.time(),
            "round": rnd,
            "worker": self.worker_id,
            "pre_loss": sum(self._last_losses[-5:])
            / max(len(self._last_losses[-5:]), 1),
        }
        self.records.append(rec)
        self._round_active.set()
        # 1. snapshot on training thread (params + BN, atomically vs steps)
        self._post(self._snap_req, self._snap_done, 60.0, "snapshot")
        # 2. delta + codec on the shard slice (bg thread, CPU flat ops)
        delta_slice = self.delta_true[lo:hi]
        torch.sub(self.snap[lo:hi], self.anchor[lo:hi], out=delta_slice)
        rec["drift_norm"] = round(float(delta_slice.norm()), 4)
        rec["shard"] = [lo, hi]
        payload, ef_new, contraction = codec_encode(
            delta_slice, self.ef[lo:hi] if self.ef is not None else None, self.codec
        )
        rec["contraction"] = round(contraction, 4)
        # BN was captured on the training thread with the param snapshot —
        # reading live BN buffers here would race the forward pass.
        bn_now = self._bn_snap
        t0 = time.monotonic()
        send_frame(
            self.ctl,
            FT_UPLOAD_META,
            rnd,
            json.dumps(
                {
                    "plen": len(payload),
                    "n_samples": self._snap_n,
                    "window_s": round(self._snap_window_s, 3),
                    "shard_size": self.shard_size,
                    "codec": self.codec,
                }
            ).encode(),
        )
        view = memoryview(payload)
        half = len(payload) // 2
        ths = []
        # NB: loop vars must NOT be named lo/hi — python leaks loop bindings
        # and lo/hi are the shard element-bounds used by the apply below.
        for st, (b_lo, b_hi) in zip((0, 1), ((0, half), (half, len(payload)))):
            # socket captured as an ARG: a zombie sender that outlives the
            # join must die on the OLD socket, not write stale chunks onto
            # whatever _reconnect put in self.data.
            t = ExThread(
                self._send_chunks,
                (self.data[st], rnd, bytes(view[b_lo:b_hi]), b_lo),
                name=f"up-{self.worker_id}-{st}",
            )
            t.start()
            ths.append(t)
        join_all(ths, ROUND_HARD_TIMEOUT, "upload stripes")
        send_frame(self.ctl, FT_UPLOAD_META, rnd, bn_now.numpy().tobytes())
        rec["upload_ms"] = round((time.monotonic() - t0) * 1e3, 1)
        rec["n_samples"] = self._snap_n
        rec["window_s"] = round(self._snap_window_s, 3)
        # 3. receive broadcast
        ftype, frnd, pl = recv_ctl(self.ctl, time.monotonic() + ROUND_HARD_TIMEOUT)
        if ftype == FT_ROUND_ABORT:
            self._round_active.clear()
            rec["aborted"] = True
            return False
        assert ftype == FT_BCAST_META and frnd == rnd, (ftype, frnd)
        bmeta = json.loads(pl)
        t0 = time.monotonic()
        buf = bytearray(bmeta["plen"])
        rxs = []
        shares = [bmeta["plen"] // 2, bmeta["plen"] - bmeta["plen"] // 2]
        for st, share in zip((0, 1), shares):
            t = ExThread(
                recv_blob,
                (self.data[st], FT_BCAST_CHUNK, rnd, share, buf),
                name=f"bc-{self.worker_id}-{st}",
            )
            t.start()
            rxs.append(t)
        join_all(rxs, ROUND_HARD_TIMEOUT, "broadcast stripes")
        ftype, _, bn_bytes = recv_ctl(self.ctl, time.monotonic() + IO_TIMEOUT)
        assert ftype == FT_BCAST_META
        rec["bcast_ms"] = round((time.monotonic() - t0) * 1e3, 1)
        # 4. anchor apply (CPU, shared function) + stage param apply
        upd = apply_round_to_anchor(self.anchor, bytes(buf), self.bcast_codec, lo, hi)
        bn_merged = f32_from(bn_bytes).clone()
        self._bn_anchor = bn_merged
        self.apply_vec.zero_()
        torch.sub(upd, delta_slice, out=self.apply_vec[lo:hi])
        # SAKURA_BN_EVERY=N: hard-set live BN stats from the merge only every
        # Nth round. At sub-second gaps the per-round hard-set rolls back the
        # current window's BN running-stat progress; the consensus hash uses
        # _bn_anchor (above, unconditional), so skipping the live set is safe.
        bn_every = int(os.environ.get("SAKURA_BN_EVERY", "1"))
        if bn_every <= 1 or rnd % bn_every == 0:
            self._bn_pending = bn_merged
        self._post(self._apply_req, self._apply_done, 60.0, "apply")
        if self.ef is not None:
            self.ef[lo:hi].copy_(ef_new)  # EF persists only on commit
        self.committed_round = rnd
        self._round_active.clear()
        h = anchor_hash16(self.anchor, bn_merged)
        send_frame(
            self.ctl, FT_ACK_APPLIED, rnd, json.dumps({"hash16": h.hex()}).encode()
        )
        rec["staleness_ms"] = round((time.monotonic() - t_round0) * 1e3, 1)
        adv = bmeta.get("scope_advisory")
        if isinstance(adv, dict) and self.scope_advisory is None:
            self.scope_advisory = adv
            rec["scope_advisory"] = adv.get("verdict")
            (log.warning if adv.get("verdict") == "warn" else log.info)(
                "%s %s", self.worker_id, format_advisory(adv)
            )
        print("WORKER_ROUND " + json.dumps(rec), flush=True)
        return True

    def _send_chunks(self, sock: socket.socket, rnd: int, part: bytes, base: int) -> None:
        view = memoryview(part)
        off = 0
        while off < len(view):
            chunk = view[off : off + CHUNK_BYTES]
            send_frame(
                sock, FT_UPLOAD_CHUNK, rnd, CHUNK_HDR.pack(base + off) + chunk.tobytes()
            )
            off += len(chunk)

    def _reconnect(self) -> None:
        # Withdraw any pending trainer request first (under _svc_lock, so an
        # in-flight service finishes before we proceed).
        with self._svc_lock:
            for ev in (self._snap_req, self._apply_req, self._set_req):
                ev.clear()
        for s in [getattr(self, "ctl", None), *getattr(self, "data", {}).values()]:
            shutclose(s)
        backoff = min(1.0, self.reconnect_backoff_max_s)
        while not self._stop.is_set():
            try:
                host, port = self.leader
                hello = self._hello()
                self.data = {}
                for st in (0, 1):  # data stripes first (resync needs them)
                    s = connect(host, port)
                    send_frame(
                        s,
                        FT_HELLO,
                        0,
                        json.dumps({**hello, "role": "data", "stripe": st}).encode(),
                    )
                    ft, _, _ = recv_ctl(s, time.monotonic() + IO_TIMEOUT)
                    if ft != FT_HELLO_ACK:
                        raise WireDead(f"bad data hello-ack {ft}")
                    self.data[st] = s
                self.ctl = connect(host, port)
                send_frame(self.ctl, FT_HELLO, 0, json.dumps(hello).encode())
                ft, _, _ = recv_ctl(self.ctl, time.monotonic() + IO_TIMEOUT)
                if ft != FT_HELLO_ACK:
                    raise WireDead(f"bad ctl hello-ack {ft}")
                self._ep_failures = 0
                log.info(
                    "%s reconnected to %s:%d at committed=%d",
                    self.worker_id,
                    host,
                    port,
                    hello["last_committed"],
                )
                return
            except (WireDead, OSError) as e:
                log.warning(
                    "%s reconnect failed: %s (retry in %.0fs)",
                    self.worker_id,
                    e,
                    backoff,
                )
                if self._connect_failed():
                    backoff = min(1.0, self.reconnect_backoff_max_s)
                    continue  # try the next endpoint right away
                # interruptible: on_train_end must not wait out a long backoff
                self._stop.wait(backoff)
                backoff = min(backoff * 2, self.reconnect_backoff_max_s)
