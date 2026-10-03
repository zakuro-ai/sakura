"""Federation leader: owns the canonical anchor and drives consensus rounds."""

from __future__ import annotations

import json
import logging
import os
import socket
import threading
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch

from .codec import (
    f32_from,
    FlatState,
    anchor_hash16,
    apply_round_to_anchor,
    codec_decode,
    codec_encode,
    merge_bn_pooled,
    shard_bounds,
)
from .outer_opt import OUTER_KINDS, outer_step
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
    ):
        # link_duty > 0 enables self-pacing: the next gap is derived from the
        # measured transfer cost of the round just completed so that round
        # traffic occupies at most this fraction of the inter-region link.
        # Remote regions are the native deployment; nobody knows the path's
        # RTT/bandwidth in advance, so the framework measures instead of
        # being configured.
        self.link_duty = link_duty
        self.shards = shards
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
        self.expected = expected
        self.persist_path = persist_path
        self.max_rounds = max_rounds
        self.round = 0
        _lossy = codec in ("int8", "int4") or codec.startswith("topk")
        self.ef_bcast = torch.zeros(self.fs.numel) if _lossy else None
        if outer_opt not in OUTER_KINDS:
            raise ValueError(f"outer_opt must be one of {OUTER_KINDS}, got {outer_opt!r}")
        self.outer_kind = outer_opt
        self.outer_lr = outer_lr
        self.outer_momentum = outer_momentum
        self.outer_m = torch.zeros(self.fs.numel) if outer_opt != "plain" else None
        self.workers: Dict[str, _WorkerConn] = {w: _WorkerConn(w) for w in expected}
        # (round, payload, bn_bytes, lo, hi, pre_round_anchor_hash16)
        self.retained: Optional[Tuple[int, bytes, bytes, int, int, bytes]] = None
        self.lock = threading.Lock()  # workers registry
        self.round_mutex = threading.RLock()  # serializes rounds vs resyncs
        self.stop = threading.Event()
        self.stats: List[Dict[str, Any]] = []
        self.overruns = 0
        self._in_round = threading.Event()
        # restarted leader must resume, not re-init from seed: a fresh random
        # anchor would FULL_SYNC over every trained worker on reconnect.
        if persist_path and os.path.exists(persist_path):
            st = torch.load(persist_path, map_location="cpu")
            self.round = st["round"]
            self.anchor.copy_(st["anchor"])
            self.bn.copy_(st["bn"])
            log.info("leader restored from %s at round %d", persist_path, self.round)

    # --- connection plumbing ---
    def _serve(self) -> None:
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
            assert ftype == FT_HELLO, f"handshake mismatch {hello}"
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
            if wid not in self.workers:
                raise AssertionError(
                    f"unknown worker_id {wid!r} (expected {self.expected})"
                )
            wc = self.workers[wid]
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
                        self._resync_if_needed(
                            wc, bytes.fromhex(hello["anchor_hash16"])
                        )
                    wc.syncing = False
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
        t0 = time.monotonic()
        for wc in parts:
            try:
                wc.send(
                    FT_ROUND_BEGIN,
                    rnd,
                    json.dumps(
                        {"anchor_version": self.round, "lo": lo, "hi": hi}
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
                    d = codec_decode(payload, hi - lo, self.codec)
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
            update = outer_step(
                dbar, m_slice, self.outer_kind, self.outer_lr, mu
            )
            # broadcast-side (dual-side) error feedback for any lossy codec.
            if self.ef_bcast is not None:
                update = update + self.ef_bcast[lo:hi]
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
                        {"plen": len(payload), "bn_len": len(bn_bytes)}
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
            "hash16": our_hash.hex(),
            "consensus_ok": ok_hash,
            "overruns": self.overruns,
        }
        self.stats.append(rec)
        print("LEADER_ROUND " + json.dumps(rec), flush=True)
        if self.persist_path:
            tmp = self.persist_path + ".tmp"
            torch.save({"round": rnd, "anchor": self.anchor, "bn": self.bn}, tmp)
            os.replace(tmp, self.persist_path)  # atomic per-commit persist
        return rec

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
