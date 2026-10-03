"""Wire protocol, framing, and connection utilities for sakura federation."""

from __future__ import annotations

import socket
import struct
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

log = __import__("logging").getLogger("sakura.federation")

# ----------------------------------------------------------------- protocol

MAGIC = 0x5A4B  # "ZK"
# v2: HELLO carries `region`, ROUND_BEGIN/BCAST_META carry shard `lo`/`hi`.
# Mixed versions fail at the frame header — one stale binary must reject at
# handshake, not crash every round on a shard-sized payload it can't parse.
PROTO_VER = 2
HDR = struct.Struct("<HBBIQ")  # magic, ver, ftype, round, payload_len
CHUNK_HDR = struct.Struct("<Q")  # offset within the logical payload
CHUNK_BYTES = 1 << 20  # 1 MiB data chunks

(
    FT_HELLO,
    FT_HELLO_ACK,
    FT_ROUND_BEGIN,
    FT_UPLOAD_META,
    FT_UPLOAD_CHUNK,
    FT_BCAST_META,
    FT_BCAST_CHUNK,
    FT_ACK_APPLIED,
    FT_ROUND_ABORT,
    FT_FULLSYNC_META,
    FT_FULLSYNC_CHUNK,
    FT_HEARTBEAT,
    FT_BYE,
) = range(1, 14)

# Progress-based timeouts: sockets carry a short POLL slice; _recv_exact/
# _send_all loop on slice timeouts and enforce real deadlines themselves —
# the slice must never floor a longer wait.
POLL_SLICE = 5.0  # socket-level timeout slice; NOT a deadline
STALL_LIMIT = 45.0  # zero-progress ceiling: documented WAN stalls reach 30s+
IO_TIMEOUT = 20.0  # ctl ops deadline
DATA_IO_TIMEOUT = 40.0  # data-chunk per-frame deadline
ROUND_HARD_TIMEOUT = 180.0  # absolute per-round ceiling
HEARTBEAT_PERIOD = 5.0


class WireDead(Exception):
    """Socket-level failure. The connection must be closed and re-established."""


def _recv_exact(sock: socket.socket, n: int, deadline: Optional[float] = None) -> bytes:
    buf = bytearray(n)
    view = memoryview(buf)
    got = 0
    last_progress = time.monotonic()
    while got < n:
        now = time.monotonic()
        if deadline is not None and now > deadline:
            raise WireDead(f"recv deadline exceeded ({got}/{n} bytes)")
        if now - last_progress > STALL_LIMIT:
            raise WireDead(f"recv stall ({got}/{n} bytes)")
        try:
            k = sock.recv_into(view[got:], min(n - got, 1 << 20))
        except socket.timeout:
            continue  # poll slice; loop re-checks deadline/stall
        except OSError as e:
            raise WireDead(f"recv error: {e}") from e
        if k == 0:
            raise WireDead(f"peer closed ({got}/{n} bytes)")
        got += k
        last_progress = time.monotonic()
    return bytes(buf)


def _send_all(sock: socket.socket, data: bytes | bytearray | memoryview) -> None:
    view = memoryview(data)
    sent = 0
    last_progress = time.monotonic()
    while sent < len(view):
        if time.monotonic() - last_progress > STALL_LIMIT:
            raise WireDead(f"send stall ({sent}/{len(view)} bytes)")
        try:
            k = sock.send(view[sent : sent + (1 << 20)])
        except socket.timeout:
            continue  # poll slice; loop re-checks the stall budget
        except OSError as e:
            raise WireDead(f"send error: {e}") from e
        if k:
            sent += k
            last_progress = time.monotonic()


def send_frame(sock: socket.socket, ftype: int, rnd: int, payload: bytes = b"") -> None:
    _send_all(sock, HDR.pack(MAGIC, PROTO_VER, ftype, rnd, len(payload)) + payload)


def recv_frame(
    sock: socket.socket, deadline: Optional[float] = None
) -> Tuple[int, int, bytes]:
    hdr = _recv_exact(sock, HDR.size, deadline)
    magic, ver, ftype, rnd, plen = HDR.unpack(hdr)
    if magic != MAGIC or ver != PROTO_VER:
        raise WireDead(f"bad frame header magic={magic:#x} ver={ver}")
    if plen > (1 << 31):
        raise WireDead(f"absurd payload length {plen}")
    payload = _recv_exact(sock, plen, deadline) if plen else b""
    return ftype, rnd, payload


def send_blob(
    sock: socket.socket, ftype_chunk: int, rnd: int, blob: bytes | bytearray | memoryview
) -> None:
    """Stream a large logical payload as offset-tagged chunks on one socket."""
    view = memoryview(blob)
    off = 0
    while off < len(view):
        part = view[off : off + CHUNK_BYTES]
        send_frame(sock, ftype_chunk, rnd, CHUNK_HDR.pack(off) + part.tobytes())
        off += len(part)


def recv_blob(
    sock: socket.socket, ftype_chunk: int, rnd: int, total: int, out: bytearray
) -> None:
    """Receive `total` chunk bytes on this socket. Chunk offsets are ABSOLUTE
    within the logical payload; `out` is the full payload buffer (chunks from
    different stripes land in disjoint ranges, so concurrent writers are safe).

    Wire offsets are bounds-checked: a bytearray slice-assign past the end
    silently APPENDS in python, so a desynced/garbage offset must raise."""
    got = 0
    while got < total:
        ftype, frnd, payload = recv_frame(sock, time.monotonic() + DATA_IO_TIMEOUT)
        if ftype == FT_HEARTBEAT:
            continue
        if ftype != ftype_chunk or frnd != rnd:
            raise WireDead(
                f"unexpected frame ftype={ftype} rnd={frnd} (want {ftype_chunk}/{rnd})"
            )
        (off,) = CHUNK_HDR.unpack(payload[: CHUNK_HDR.size])
        data = payload[CHUNK_HDR.size :]
        if off + len(data) > len(out):
            raise WireDead(f"chunk offset {off}+{len(data)} exceeds payload {len(out)}")
        out[off : off + len(data)] = data
        got += len(data)
    if got != total:
        raise WireDead(f"blob over-receive: {got} != {total}")


def recv_ctl(
    sock: socket.socket, deadline: Optional[float] = None
) -> Tuple[int, int, bytes]:
    """recv_frame for ctl sockets: transparently skips heartbeats. EVERY ctl
    read outside recv_blob must use this — heartbeats land between any two
    ctl frames whenever a transfer outlasts HEARTBEAT_PERIOD."""
    while True:
        ftype, rnd, payload = recv_frame(sock, deadline)
        if ftype != FT_HEARTBEAT:
            return ftype, rnd, payload


class ExThread(threading.Thread):
    """Thread that captures its target's exception instead of dumping it to
    the default excepthook. join_all() turns silent stripe death into a
    visible WireDead."""

    def __init__(
        self,
        target: Callable[..., Any],
        args: Tuple[Any, ...] = (),
        name: Optional[str] = None,
    ) -> None:
        super().__init__(daemon=True, name=name)
        self._t, self._a = target, args
        self.exc: Optional[BaseException] = None

    def run(self) -> None:
        try:
            self._t(*self._a)
        except BaseException as e:  # noqa: BLE001
            self.exc = e


def join_all(threads: List[ExThread], timeout: float, what: str) -> None:
    deadline = time.monotonic() + timeout
    for t in threads:
        t.join(max(0.1, deadline - time.monotonic()))
    alive = [t for t in threads if t.is_alive()]
    failed = [t for t in threads if t.exc is not None]
    if alive or failed:
        raise WireDead(
            f"{what}: {len(alive)} stalled, " f"{[repr(t.exc) for t in failed]} failed"
        )


def _keepalive(s: socket.socket) -> None:
    """Dead-peer detection on idle sockets (conntrack evicts idle wg0 flows)."""
    s.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)
    try:
        s.setsockopt(socket.IPPROTO_TCP, socket.TCP_KEEPIDLE, 15)
        s.setsockopt(socket.IPPROTO_TCP, socket.TCP_KEEPINTVL, 5)
        s.setsockopt(socket.IPPROTO_TCP, socket.TCP_KEEPCNT, 3)
    except (AttributeError, OSError):
        pass  # darwin in local tests


def shutclose(s: Optional[socket.socket]) -> None:
    """shutdown-then-close: wakes any thread blocked in recv on this socket
    BEFORE the fd is released, so fd-number reuse can't hand the old reader
    a brand-new connection's bytes."""
    if s is None:
        return
    try:
        s.shutdown(socket.SHUT_RDWR)
    except OSError:
        pass
    try:
        s.close()
    except OSError:
        pass


def connect(host: str, port: int, timeout: float = 10.0) -> socket.socket:
    s = socket.create_connection((host, port), timeout=timeout)
    s.settimeout(POLL_SLICE)
    s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    _keepalive(s)
    # NEVER setsockopt SO_SNDBUF/SO_RCVBUF on this mesh — rmem_max caps at
    # 416 KB and kills autotune → 0.86 MB/s.
    return s


@dataclass
class _WorkerConn:
    worker_id: str
    region: str = "?"
    ctl: Optional[socket.socket] = None
    data: Dict[int, socket.socket] = field(default_factory=dict)
    last_committed: int = -1
    ema_n: float = 0.0
    alive: bool = True
    syncing: bool = False  # excluded from rounds while a resync is in flight
    gen: int = 0  # bumped on every drop; stale threads must not touch us
    # Serializes ctl-socket writes (heartbeat thread vs round loop vs the
    # accept thread's resync path). Data sockets get per-stripe locks for the
    # same reason (resync send_blob vs round broadcast).
    send_lock: threading.Lock = field(default_factory=threading.Lock)
    data_locks: Dict[int, threading.Lock] = field(default_factory=dict)

    def send(self, ftype: int, rnd: int, payload: bytes = b"") -> None:
        with self.send_lock:
            if self.ctl is None:
                raise WireDead(f"{self.worker_id}: ctl gone")
            send_frame(self.ctl, ftype, rnd, payload)

    def send_data_blob(
        self,
        stripe: int,
        ftype_chunk: int,
        rnd: int,
        part: bytes | bytearray | memoryview,
        base: int,
    ) -> None:
        lock = self.data_locks.setdefault(stripe, threading.Lock())
        with lock:
            sock = self.data.get(stripe)
            if sock is None:
                raise WireDead(f"{self.worker_id}: data[{stripe}] gone")
            view = memoryview(part)
            off = 0
            while off < len(view):
                chunk = view[off : off + CHUNK_BYTES]
                send_frame(
                    sock, ftype_chunk, rnd, CHUNK_HDR.pack(base + off) + chunk.tobytes()
                )
                off += len(chunk)
