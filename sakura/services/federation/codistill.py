"""Co-distillation round mode (SAK-0012, design rung E8).

Weight averaging needs every worker to run the same architecture. Here the
workers exchange *predictions* instead: each one evaluates its own model on a
shared probe batch and uploads the logits; the leader averages them into an
ensemble target and sends it back; each worker adds a KL term toward that
target to its local loss. Nothing about the architectures crosses the wire,
only a ``rows x classes`` logit matrix, so a P620 can run a small CNN while a
4090 runs a ResNet and both still learn from each other.

Wire cost per worker per round (512 x 1000 probe, the ADR's sizing):

* up:   fp16 logits, ``rows*classes*2`` bytes  (1,024,000 B)
* down: ensemble target, per-row-scaled int8 by default
        (``rows*classes + rows*4 + 1`` bytes = 514,049 B), or fp16 on request.

plus three 16-byte frame headers (ROUND_BEGIN, LOGITS, TARGET): 1,538,097 B
per round with the int8 target, 2,048,049 B (< 2 MiB, > 2e6) with fp16. ``CodistillWorker.round_bytes`` and the
leader's ``CODISTILL_ROUND`` log line count what actually crossed the socket.

Decisions (recorded in docs/adr/0012-co-distillation-e8.md):

* The probe batch never travels. The leader hands out ``probe_seed`` and
  ``probe_rows`` at handshake; every worker derives the same probe locally
  (``make_probe`` for synthetic inputs, ``probe_indices`` to pick rows from a
  public pool each site already holds). Raw data and labels never leave a
  site; logits on *public or synthetic* inputs do.
* The target a worker receives is the leave-one-out mean of the *other*
  workers' logits: averaging a worker's own logits into its teacher only
  halves the useful gradient with two workers.
* Rounds are leader-timed (``gap_s``) and never block training: a worker
  publishes its latest probe logits whenever it has them, a background thread
  answers the round, and training reads the latest target. The leader keeps
  each worker's last logits for ``max_stale_rounds`` so a slow worker still
  teaches the fast one between its own refreshes.
* Separate port and frame-type space from the weight-delta Leader: this is a
  demonstrator beside it, not a replacement (SAK-0012 Proposal).
"""

from __future__ import annotations

import json
import logging
import select
import socket
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

from .transport import (
    HDR,
    IO_TIMEOUT,
    POLL_SLICE,
    ExThread,
    WireDead,
    _keepalive,
    connect,
    recv_frame,
    send_frame,
    shutclose,
)

log = logging.getLogger("sakura.federation")

# Own frame-type space (0x40+): a weight-delta worker pointed at this port, or
# a co-distillation worker pointed at a Leader, fails the handshake loudly
# instead of misparsing a payload.
(
    FT_CD_HELLO,
    FT_CD_HELLO_ACK,
    FT_CD_ROUND_BEGIN,
    FT_CD_LOGITS,
    FT_CD_TARGET,
    FT_CD_BYE,
) = range(0x40, 0x46)

# CD_TARGET payload: 1-byte tag, then the matrix.
TGT_NONE, TGT_FP16, TGT_INT8 = 0, 1, 2
TARGET_CODECS = {"fp16": TGT_FP16, "int8": TGT_INT8}


# ------------------------------------------------------------------ helpers

def make_probe(seed: int, rows: int, shape: Tuple[int, ...]) -> torch.Tensor:
    """Deterministic synthetic probe: identical on every host for the same
    (seed, rows, shape), independent of the global RNG."""
    g = torch.Generator().manual_seed(int(seed))
    return torch.randn((rows, *shape), generator=g)


def probe_indices(seed: int, rows: int, pool_size: int) -> torch.Tensor:
    """Deterministic choice of ``rows`` indices into a public pool of
    ``pool_size`` samples that every site holds (e.g. an unlabelled public
    split). Only the seed is shared; the pool never moves."""
    if rows > pool_size:
        raise ValueError(f"probe rows {rows} > public pool {pool_size}")
    g = torch.Generator().manual_seed(int(seed))
    return torch.randperm(pool_size, generator=g)[:rows]


def kd_loss(
    student_logits: torch.Tensor, target_logits: torch.Tensor, temperature: float = 2.0
) -> torch.Tensor:
    """KL(softmax(target/T) || softmax(student/T)) * T^2 (Hinton scaling, so
    the gradient magnitude does not shrink with T)."""
    t = float(temperature)
    return (
        F.kl_div(
            F.log_softmax(student_logits / t, dim=-1),
            F.softmax(target_logits.to(student_logits.dtype) / t, dim=-1),
            reduction="batchmean",
        )
        * t
        * t
    )


def encode_target(target: Optional[torch.Tensor], codec: str) -> bytes:
    if target is None:
        return bytes([TGT_NONE])
    t = target.detach().float().cpu().contiguous()
    if codec == "fp16":
        return bytes([TGT_FP16]) + t.half().numpy().tobytes()
    if codec == "int8":
        scale = t.abs().amax(dim=1).clamp_min(1e-12) / 127.0
        q = torch.round(t / scale[:, None]).clamp_(-127, 127).to(torch.int8)
        return bytes([TGT_INT8]) + scale.numpy().tobytes() + q.numpy().tobytes()
    raise ValueError(f"target codec must be one of {sorted(TARGET_CODECS)}, got {codec!r}")


def decode_target(payload: bytes, rows: int, classes: int) -> Optional[torch.Tensor]:
    if not payload:
        raise WireDead("empty CD_TARGET payload")
    tag, body = payload[0], payload[1:]
    if tag == TGT_NONE:
        return None
    if tag == TGT_FP16:
        if len(body) != rows * classes * 2:
            raise WireDead(f"fp16 target size {len(body)} != {rows * classes * 2}")
        return torch.frombuffer(bytearray(body), dtype=torch.float16).float().view(rows, classes)
    if tag == TGT_INT8:
        if len(body) != rows * 4 + rows * classes:
            raise WireDead(f"int8 target size {len(body)} != {rows * 4 + rows * classes}")
        scale = torch.frombuffer(bytearray(body[: rows * 4]), dtype=torch.float32)
        q = torch.frombuffer(bytearray(body[rows * 4 :]), dtype=torch.int8).view(rows, classes)
        return q.float() * scale[:, None]
    raise WireDead(f"unknown target tag {tag}")


def leave_one_out_targets(
    logits: Dict[str, torch.Tensor], recipients: List[str]
) -> Dict[str, Optional[torch.Tensor]]:
    """Ensemble target per recipient: mean of every *other* contributor's
    logits. A recipient with no peer contribution gets None (no KL term)."""
    out: Dict[str, Optional[torch.Tensor]] = {}
    if not logits:
        return {w: None for w in recipients}
    total = torch.stack([v.float() for v in logits.values()]).sum(0)
    n = len(logits)
    for w in recipients:
        if w in logits:
            out[w] = None if n < 2 else (total - logits[w].float()) / (n - 1)
        else:
            out[w] = total / n
    return out


def _frame_bytes(payload_len: int) -> int:
    return HDR.size + payload_len


# ------------------------------------------------------------------- leader

@dataclass
class _CDConn:
    worker_id: str
    sock: socket.socket
    arch: str = ""
    gen: int = 0
    lock: threading.Lock = field(default_factory=threading.Lock)


class CodistillLeader:
    """CPU-only round driver for co-distillation. Holds no model.

    Every ``gap_s`` it asks each connected worker for logits, builds the
    leave-one-out ensemble targets from the fresh plus not-too-stale logits it
    holds, and sends each worker its target. ``port=0`` binds an ephemeral
    port; read ``self.port`` after ``start()``."""

    def __init__(
        self,
        port: int,
        expected: List[str],
        probe_rows: int = 512,
        num_classes: int = 1000,
        probe_seed: int = 0,
        gap_s: float = 5.0,
        target_codec: str = "int8",
        max_stale_rounds: int = 4,
        round_timeout: float = 30.0,
        max_rounds: int = 10**9,
        host: str = "0.0.0.0",
    ):
        if target_codec not in TARGET_CODECS:
            raise ValueError(
                f"target_codec must be one of {sorted(TARGET_CODECS)}, got {target_codec!r}"
            )
        self.port = port
        self.host = host
        self.expected = list(expected)
        self.rows = probe_rows
        self.classes = num_classes
        self.probe_seed = probe_seed
        self.gap_s = gap_s
        self.target_codec = target_codec
        self.max_stale_rounds = max_stale_rounds
        self.round_timeout = round_timeout
        self.max_rounds = max_rounds
        self.round = 0
        self.conns: Dict[str, _CDConn] = {}
        self.latest: Dict[str, Tuple[int, torch.Tensor]] = {}  # wid -> (round, logits)
        self.stats: List[Dict[str, Any]] = []
        self.lock = threading.Lock()
        self.stop = threading.Event()
        self._srv: Optional[socket.socket] = None
        self._threads: List[threading.Thread] = []

    @property
    def logit_bytes(self) -> int:
        return self.rows * self.classes * 2

    def start(self) -> "CodistillLeader":
        srv = socket.socket()
        srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        srv.bind((self.host, self.port))
        srv.listen(16)
        srv.settimeout(0.5)
        self._srv = srv
        self.port = srv.getsockname()[1]
        for fn, name in ((self._serve, "cd-accept"), (self._loop, "cd-rounds")):
            t = threading.Thread(target=fn, name=name, daemon=True)
            t.start()
            self._threads.append(t)
        return self

    def close(self) -> None:
        self.stop.set()
        shutclose(self._srv)
        with self.lock:
            conns = list(self.conns.values())
            self.conns.clear()
        for c in conns:
            shutclose(c.sock)
        for t in self._threads:
            t.join(timeout=5.0)

    def _serve(self) -> None:
        assert self._srv is not None
        while not self.stop.is_set():
            try:
                s, _ = self._srv.accept()
            except socket.timeout:
                continue
            except OSError:
                break
            s.settimeout(POLL_SLICE)
            s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            _keepalive(s)
            threading.Thread(target=self._handshake, args=(s,), daemon=True).start()

    def _handshake(self, s: socket.socket) -> None:
        try:
            ftype, _, payload = recv_frame(s, time.monotonic() + IO_TIMEOUT)
            if ftype != FT_CD_HELLO:
                raise WireDead(f"expected CD_HELLO, got frame type {ftype}")
            hello = json.loads(payload)
            wid = hello["worker_id"]
            if wid not in self.expected:
                raise WireDead(f"unknown worker_id {wid!r} (expected {self.expected})")
            ack = {
                "round": self.round,
                "probe_seed": self.probe_seed,
                "probe_rows": self.rows,
                "num_classes": self.classes,
                "target_codec": self.target_codec,
            }
            send_frame(s, FT_CD_HELLO_ACK, self.round, json.dumps(ack).encode())
        except (WireDead, ValueError, KeyError) as e:
            log.warning("codistill handshake rejected: %s", e)
            shutclose(s)
            return
        with self.lock:
            old = self.conns.get(wid)
            self.conns[wid] = _CDConn(
                wid, s, str(hello.get("arch", "")), (old.gen + 1) if old else 0
            )
        if old is not None:
            shutclose(old.sock)  # reconnect replaces the stale connection
        log.info("codistill worker %s joined (arch=%s)", wid, hello.get("arch"))

    def _drop(self, c: _CDConn, why: str) -> None:
        with self.lock:
            if self.conns.get(c.worker_id) is c:
                del self.conns[c.worker_id]
        shutclose(c.sock)
        log.warning("codistill worker %s dropped: %s", c.worker_id, why)

    def _loop(self) -> None:
        next_t = time.monotonic() + self.gap_s
        while not self.stop.is_set() and self.round < self.max_rounds:
            if self.stop.wait(max(0.0, next_t - time.monotonic())):
                break
            next_t = time.monotonic() + self.gap_s
            with self.lock:
                conns = list(self.conns.values())
            if conns:
                self.run_round(conns)

    def run_round(self, conns: List[_CDConn]) -> Dict[str, Any]:
        self.round += 1
        rnd = self.round
        t0 = time.monotonic()
        deadline = t0 + self.round_timeout
        fresh: Dict[str, torch.Tensor] = {}
        up: Dict[str, int] = {}
        live: List[_CDConn] = []

        def gather(c: _CDConn) -> None:
            with c.lock:
                send_frame(c.sock, FT_CD_ROUND_BEGIN, rnd)
                while True:
                    ftype, frnd, payload = recv_frame(c.sock, deadline)
                    if ftype == FT_CD_LOGITS and frnd == rnd:
                        break
                    if ftype == FT_CD_BYE:
                        raise WireDead("worker said bye")
                    # a late answer to an earlier, timed-out round: skip it
            up[c.worker_id] = _frame_bytes(len(payload))
            if payload:
                if len(payload) != self.logit_bytes:
                    raise WireDead(f"logits size {len(payload)} != {self.logit_bytes}")
                fresh[c.worker_id] = (
                    torch.frombuffer(bytearray(payload), dtype=torch.float16)
                    .view(self.rows, self.classes)
                )

        threads = [ExThread(gather, (c,), name=f"cd-up-{c.worker_id}") for c in conns]
        for t in threads:
            t.start()
        for c, t in zip(conns, threads):
            t.join(max(0.1, deadline - time.monotonic()))
            if t.is_alive() or t.exc is not None:
                self._drop(c, repr(t.exc) if t.exc else "logit upload timed out")
            else:
                live.append(c)

        for wid, lg in fresh.items():
            self.latest[wid] = (rnd, lg)
        pool = {
            wid: lg
            for wid, (r, lg) in self.latest.items()
            if rnd - r <= self.max_stale_rounds
        }
        targets = leave_one_out_targets(pool, [c.worker_id for c in live])
        down: Dict[str, int] = {}
        for c in live:
            body = encode_target(targets[c.worker_id], self.target_codec)
            try:
                with c.lock:
                    send_frame(c.sock, FT_CD_TARGET, rnd, body)
                # ROUND_BEGIN header + target frame
                down[c.worker_id] = HDR.size + _frame_bytes(len(body))
            except WireDead as e:
                self._drop(c, repr(e))
        st = {
            "round": rnd,
            "participants": sorted(down),
            "fresh": sorted(fresh),
            "ensemble": sorted(pool),
            "bytes_up": up,
            "bytes_down": down,
            "wall_s": round(time.monotonic() - t0, 4),
        }
        self.stats.append(st)
        log.info("CODISTILL_ROUND %s", json.dumps(st))
        return st


# ------------------------------------------------------------------- worker

class CodistillWorker:
    """Worker side. Training never blocks on the WAN::

        cd = CodistillWorker("w0", host, port, arch="resnet18").start()
        probe = make_probe(cd.probe_seed, cd.probe_rows, (3, 32, 32))
        ...
        logits = model(probe)               # with grad
        tgt = cd.target()                   # latest ensemble, or None
        if tgt is not None:
            loss = loss + alpha * kd_loss(logits, tgt.to(logits.device))
        cd.publish(logits)                  # answered at the next round

    A background thread answers ROUND_BEGIN with the most recent published
    logits (an empty frame when nothing new was published since the last
    upload) and stores the returned target. It reconnects on wire failure.
    """

    def __init__(
        self,
        worker_id: str,
        host: str,
        port: int,
        arch: str = "",
        reconnect_s: float = 1.0,
    ):
        self.worker_id = worker_id
        self.host = host
        self.port = port
        self.arch = arch
        self.reconnect_s = reconnect_s
        self.probe_seed = 0
        self.probe_rows = 0
        self.num_classes = 0
        self.round_bytes: List[Dict[str, int]] = []
        self.targets_received = 0
        self._sock: Optional[socket.socket] = None
        self._pending: Optional[bytes] = None
        self._target: Optional[torch.Tensor] = None
        self._target_round = 0
        self._lock = threading.Lock()
        self._send_lock = threading.Lock()  # round thread vs close()'s BYE
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def start(self) -> "CodistillWorker":
        self._connect()
        self._thread = threading.Thread(
            target=self._run, name=f"cd-{self.worker_id}", daemon=True
        )
        self._thread.start()
        return self

    def _connect(self) -> None:
        s = connect(self.host, self.port)
        hello = {"worker_id": self.worker_id, "arch": self.arch}
        send_frame(s, FT_CD_HELLO, 0, json.dumps(hello).encode())
        ftype, _, payload = recv_frame(s, time.monotonic() + IO_TIMEOUT)
        if ftype != FT_CD_HELLO_ACK:
            shutclose(s)
            raise WireDead(f"expected CD_HELLO_ACK, got frame type {ftype}")
        ack = json.loads(payload)
        self.probe_seed = int(ack["probe_seed"])
        self.probe_rows = int(ack["probe_rows"])
        self.num_classes = int(ack["num_classes"])
        self._sock = s

    def publish(self, logits: torch.Tensor) -> None:
        """Hand over this worker's latest probe logits (rows x classes)."""
        if tuple(logits.shape) != (self.probe_rows, self.num_classes):
            raise ValueError(
                f"logits shape {tuple(logits.shape)} != probe "
                f"({self.probe_rows}, {self.num_classes})"
            )
        blob = logits.detach().to("cpu", torch.float16).contiguous().numpy().tobytes()
        with self._lock:
            self._pending = blob

    def target(self) -> Optional[torch.Tensor]:
        """Latest ensemble target (fp32, CPU) or None before the first one."""
        with self._lock:
            return self._target

    @property
    def target_round(self) -> int:
        with self._lock:
            return self._target_round

    def close(self) -> None:
        self._stop.set()
        s = self._sock
        if s is not None:
            try:
                with self._send_lock:
                    send_frame(s, FT_CD_BYE, 0)
            except WireDead:
                pass
        shutclose(s)
        if self._thread is not None:
            self._thread.join(timeout=5.0)

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                if self._sock is None:
                    self._connect()
                self._serve_rounds()
            except (WireDead, OSError, ValueError) as e:
                if self._stop.is_set():
                    return
                log.warning("codistill %s: %s; reconnecting", self.worker_id, e)
                shutclose(self._sock)
                self._sock = None
                self._stop.wait(self.reconnect_s)

    def _serve_rounds(self) -> None:
        s = self._sock
        assert s is not None
        while not self._stop.is_set():
            # Idle between rounds can exceed transport.STALL_LIMIT (gap_s is
            # the leader's choice): wait for readability first so recv_frame
            # only ever times a frame that has started arriving.
            try:
                readable, _, _ = select.select([s], [], [], POLL_SLICE)
            except (OSError, ValueError):
                if self._stop.is_set():
                    return
                raise
            if not readable:
                continue
            try:
                ftype, rnd, _ = recv_frame(s, time.monotonic() + IO_TIMEOUT)
            except WireDead:
                if self._stop.is_set():
                    return
                raise
            if ftype != FT_CD_ROUND_BEGIN:
                raise WireDead(f"expected CD_ROUND_BEGIN, got frame type {ftype}")
            with self._lock:
                blob, self._pending = self._pending, None
            body = blob or b""
            with self._send_lock:
                send_frame(s, FT_CD_LOGITS, rnd, body)
            ftype, frnd, payload = recv_frame(s, time.monotonic() + IO_TIMEOUT * 3)
            if ftype != FT_CD_TARGET or frnd != rnd:
                raise WireDead(f"expected CD_TARGET for round {rnd}, got {ftype}/{frnd}")
            tgt = decode_target(payload, self.probe_rows, self.num_classes)
            with self._lock:
                if tgt is not None:
                    self._target = tgt
                    self._target_round = rnd
                    self.targets_received += 1
                self.round_bytes.append(
                    {
                        "round": rnd,
                        "up": _frame_bytes(len(body)),
                        # ROUND_BEGIN header + target frame
                        "down": HDR.size + _frame_bytes(len(payload)),
                    }
                )
