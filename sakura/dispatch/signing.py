"""HMAC-signed envelope for cloudpickle `aux` payloads crossing the wire.

sakura#9 / zakuro#117: `cloudpickle.loads` on bytes that arrived from a
network peer (dispatcher <-> sakura-worker, over sakura_wire/QUIC) is a
remote-code-execution primitive if that peer is compromised or spoofed.
The full migration off cloudpickle onto the postcard/serde wire codec is
tracked separately; until it lands, every `aux` payload is wrapped in an
HMAC-SHA256-signed envelope on the sending side and the signature is
verified with `hmac.compare_digest` on the receiving side *before* the
payload is ever handed to `cloudpickle.loads`. A missing/mismatched key or
a tampered envelope raises `SignatureError` and cloudpickle is never
invoked on that payload.

The shared key is a symmetric HMAC key, not a keypair: both ends of a given
dispatcher<->worker connection must be configured with the same value.

- `LocalDispatcher` generates a random per-process key and injects it into
  the environment of the worker subprocess it spawns, so no operator
  configuration is required for the common (auto-spawned) case.
- `RemoteDispatcher` connecting to an already-running remote worker (or the
  worker process itself) reads the key from the `SAKURA_WIRE_HMAC_KEY`
  environment variable unless a key is passed explicitly.

Keys shorter than `MIN_KEY_BYTES` (16 bytes, after hex decoding) are refused,
so an empty or trivially short key cannot silently turn the HMAC into a
checksum anyone can forge. Verification fails closed: no key, a short key, a
short/tampered envelope or a wrong key all raise `SignatureError`.

`safe_loads` is the only sanctioned way to unpickle bytes that arrived over
the wire: it verifies the envelope first and only then calls
`cloudpickle.loads`. Dispatch *results* no longer go through it by default;
they use the typed codec in `sakura.dispatch.result_codec` (SAK-0016).

Replay protection (envelope v2, sakura#139)
-------------------------------------------
A valid signature alone does not stop a captured envelope from being sent
again. Every envelope therefore carries, under the MAC:

    magic "SKW\x02" (4) | kind (1) | issued_at_ns (8, big-endian) |
    nonce (16, random) | bound_to (16) | HMAC-SHA256 tag (32) | payload

- `kind` separates requests (`KIND_REQUEST`) from results (`KIND_RESULT`), so
  a request envelope reflected back (e.g. by the echo handler) is not
  accepted as a result, nor a result as a request.
- A result's `bound_to` is the nonce of the request it answers; the
  dispatcher checks it, so a result captured from one call cannot be replayed
  as the answer to another. Requests carry zeros there.
- The verifier rejects a timestamp older than `window_s` or further than
  `max_skew_s` in the future, and keeps a per-process, time-expiring,
  memory-bounded cache (`ReplayCache`) of nonces seen within the window: the
  same envelope verified twice raises `SignatureError`.

The v1 layout (tag || payload, no header) is refused with an explicit
"unsupported envelope version" error: dispatchers and workers already have
to roll together (#111, SAK-0016), and this change keeps that rule.
"""
from __future__ import annotations

import hashlib
import heapq
import hmac
import os
import secrets
import struct
import threading
import time
from dataclasses import dataclass
from typing import Any, Optional, Union

import cloudpickle  # type: ignore[import-untyped]

KEY_ENV_VAR = "SAKURA_WIRE_HMAC_KEY"
#: Replay window (seconds): envelopes issued longer ago than this are refused.
WINDOW_ENV_VAR = "SAKURA_WIRE_REPLAY_WINDOW_S"
#: Clock-skew tolerance (seconds) for envelopes stamped in the future.
SKEW_ENV_VAR = "SAKURA_WIRE_CLOCK_SKEW_S"
#: Minimum accepted HMAC key length in bytes (128 bits).
MIN_KEY_BYTES = 16
DEFAULT_WINDOW_S = 120.0
DEFAULT_MAX_SKEW_S = 30.0
#: Upper bound on remembered nonces per cache (about 10 MB worst case).
DEFAULT_MAX_ENTRIES = 65536

#: Envelope magic + version byte. v1 had no header at all (tag || payload).
MAGIC = b"SKW\x02"
_MAGIC_FAMILY = MAGIC[:3]
KIND_REQUEST = 0x01
KIND_RESULT = 0x02
NONCE_LEN = 16
_ZERO_BINDING = bytes(NONCE_LEN)
_TS = struct.Struct(">Q")
#: magic | kind | issued_at_ns | nonce | bound_to
HEADER_LEN = len(MAGIC) + 1 + _TS.size + NONCE_LEN + NONCE_LEN
_MAC_LEN = 32  # SHA-256 digest size
#: Offset of the HMAC tag inside an envelope (it follows the header).
MAC_OFFSET = HEADER_LEN
_MIN_ENVELOPE = HEADER_LEN + _MAC_LEN


class SignatureError(RuntimeError):
    """Raised when an aux envelope is missing, malformed, replayed, stale, or fails HMAC verification.

    Whenever this is raised, the enclosed payload has NOT been passed to
    cloudpickle.loads / pickle.loads.
    """


def _env_key() -> bytes:
    key_hex = os.environ.get(KEY_ENV_VAR)
    if not key_hex:
        raise SignatureError(
            f"{KEY_ENV_VAR} is not set and no explicit key was provided; "
            "refusing to sign/verify a cloudpickle aux payload. Configure "
            "the same HMAC key on both the dispatcher and the worker "
            "process."
        )
    try:
        return bytes.fromhex(key_hex)
    except ValueError:
        # Tolerate a raw (non-hex) passphrase rather than hard-failing; it
        # still has to pass the MIN_KEY_BYTES check in `_resolve_key`.
        return key_hex.encode("utf-8")


def _resolve_key(key: Optional[bytes]) -> bytes:
    k = bytes(key) if key is not None else _env_key()
    if len(k) < MIN_KEY_BYTES:
        raise SignatureError(
            f"HMAC key is {len(k)} bytes; at least {MIN_KEY_BYTES} bytes are "
            f"required (e.g. `{KEY_ENV_VAR}=$(openssl rand -hex 32)`)"
        )
    return k


def _env_seconds(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    try:
        value = float(raw)
    except ValueError:
        raise ValueError(f"{name}={raw!r} is not a number of seconds") from None
    if not value > 0:
        raise ValueError(f"{name} must be > 0 seconds, got {raw!r}")
    return value


class ReplayCache:
    """Per-process, time-expiring, memory-bounded record of verified nonces.

    `check_and_record` accepts an envelope's `(nonce, issued_at)` once: it
    refuses a timestamp older than `window_s` or more than `max_skew_s` in
    the future, and a nonce it has already seen. A nonce only needs to be
    remembered until its timestamp falls out of the window, after which the
    stale check rejects it anyway, so entries expire at `issued_at + window_s`.

    Memory is capped at `max_entries`. When the cache is full of unexpired
    nonces it evicts the one with the oldest timestamp and raises a floor:
    every later envelope stamped at or before that timestamp is refused. That
    keeps the bound without reopening a replay hole; the only cost is that,
    under a burst above `max_entries` per window, the oldest in-flight
    envelopes are refused rather than accepted twice. Thread-safe.
    """

    def __init__(
        self,
        *,
        window_s: float = DEFAULT_WINDOW_S,
        max_skew_s: float = DEFAULT_MAX_SKEW_S,
        max_entries: int = DEFAULT_MAX_ENTRIES,
    ) -> None:
        if not window_s > 0:
            raise ValueError("window_s must be > 0")
        if max_skew_s < 0:
            raise ValueError("max_skew_s must be >= 0")
        if max_entries < 1:
            raise ValueError("max_entries must be >= 1")
        self.window_s = float(window_s)
        self.max_skew_s = float(max_skew_s)
        self.max_entries = int(max_entries)
        self._seen: set[bytes] = set()
        self._heap: list[tuple[float, bytes]] = []  # (issued_at, nonce), oldest first
        self._floor = float("-inf")
        self._lock = threading.Lock()

    def __len__(self) -> int:
        with self._lock:
            return len(self._seen)

    def _expire(self, now: float) -> None:
        horizon = now - self.window_s
        while self._heap and self._heap[0][0] < horizon:
            _, nonce = heapq.heappop(self._heap)
            self._seen.discard(nonce)

    def check_and_record(self, nonce: bytes, issued_at: float, *, now: Optional[float] = None) -> None:
        """Record `nonce` or raise `SignatureError` (stale, future, replayed)."""
        t = time.time() if now is None else float(now)
        if issued_at < t - self.window_s:
            raise SignatureError(
                f"envelope is stale: issued {t - issued_at:.1f}s ago, replay window is "
                f"{self.window_s:g}s (check clocks or raise {WINDOW_ENV_VAR})"
            )
        if issued_at > t + self.max_skew_s:
            raise SignatureError(
                f"envelope is stamped {issued_at - t:.1f}s in the future, clock-skew "
                f"tolerance is {self.max_skew_s:g}s (check clocks or raise {SKEW_ENV_VAR})"
            )
        with self._lock:
            self._expire(t)
            if issued_at <= self._floor:
                raise SignatureError(
                    "envelope is older than the replay cache can vouch for (cache "
                    "overflowed under load); rejecting it"
                )
            if nonce in self._seen:
                raise SignatureError("envelope nonce already seen: replayed envelope rejected")
            while len(self._seen) >= self.max_entries:
                old_ts, old_nonce = heapq.heappop(self._heap)
                self._seen.discard(old_nonce)
                self._floor = max(self._floor, old_ts)
            if issued_at <= self._floor:
                raise SignatureError(
                    "envelope is older than the replay cache can vouch for (cache "
                    "overflowed under load); rejecting it"
                )
            self._seen.add(nonce)
            heapq.heappush(self._heap, (issued_at, nonce))


_default_cache: Optional[ReplayCache] = None
_default_cache_lock = threading.Lock()


def default_replay_cache() -> ReplayCache:
    """The process-wide cache used when `verify` is not given one.

    Built on first use from `SAKURA_WIRE_REPLAY_WINDOW_S` (default 120) and
    `SAKURA_WIRE_CLOCK_SKEW_S` (default 30).
    """
    global _default_cache
    with _default_cache_lock:
        if _default_cache is None:
            _default_cache = ReplayCache(
                window_s=_env_seconds(WINDOW_ENV_VAR, DEFAULT_WINDOW_S),
                max_skew_s=_env_seconds(SKEW_ENV_VAR, DEFAULT_MAX_SKEW_S),
            )
        return _default_cache


@dataclass(frozen=True)
class Verified:
    """An envelope that passed `verify_envelope` (MAC, kind, binding, freshness, replay).

    Only `verify_envelope` should build one; `safe_loads` accepts it so a
    caller that already verified an envelope does not verify it twice (which
    the replay cache would, correctly, reject).
    """

    payload: bytes
    nonce: bytes
    issued_at: float
    kind: int


def _check_kind(kind: int) -> None:
    if kind not in (KIND_REQUEST, KIND_RESULT):
        raise ValueError(f"unknown envelope kind {kind!r}")


def _binding(kind: int, bound_to: Optional[bytes], *, verifying: bool) -> bytes:
    if kind == KIND_REQUEST:
        if bound_to is not None:
            raise ValueError("request envelopes are not bound to another envelope")
        return _ZERO_BINDING
    if bound_to is None:
        if verifying:
            raise SignatureError(
                "verifying a result envelope requires the nonce of the request it answers"
            )
        raise ValueError("result envelopes must be bound to their request nonce")
    b = bytes(bound_to)
    if len(b) != NONCE_LEN:
        raise ValueError(f"bound_to must be a {NONCE_LEN}-byte request nonce")
    return b


def sign(
    payload: bytes,
    *,
    key: Optional[bytes] = None,
    kind: int = KIND_REQUEST,
    bound_to: Optional[bytes] = None,
    now: Optional[float] = None,
) -> bytes:
    """Wrap `payload` in a v2 envelope: header (fresh nonce + timestamp) || HMAC tag || payload.

    `kind=KIND_RESULT` requires `bound_to`, the nonce of the request being
    answered (`nonce_of(request_envelope)`). `now` overrides the issue time
    (seconds since the epoch), for tests.
    """
    _check_kind(kind)
    binding = _binding(kind, bound_to, verifying=False)
    k = _resolve_key(key)
    issued_ns = time.time_ns() if now is None else int(now * 1e9)
    header = MAGIC + bytes([kind]) + _TS.pack(issued_ns) + secrets.token_bytes(NONCE_LEN) + binding
    body = bytes(payload)
    mac = hmac.new(k, header + body, hashlib.sha256).digest()
    return header + mac + body


def _parse_header(envelope: bytes) -> tuple[bytes, int, int, bytes, bytes]:
    if not envelope.startswith(MAGIC):
        if envelope.startswith(_MAGIC_FAMILY) and len(envelope) > len(MAGIC):
            raise SignatureError(
                f"unsupported aux envelope version {envelope[3]}; this sakura speaks "
                f"version {MAGIC[3]} (dispatcher and worker must run the same sakura version)"
            )
        raise SignatureError(
            "aux envelope is not a v2 signed envelope (unsigned, garbage, or a "
            "pre-replay-protection v1 envelope); dispatcher and worker must run "
            "the same sakura version"
        )
    if len(envelope) < _MIN_ENVELOPE:
        raise SignatureError("aux envelope too short to contain its header and HMAC-SHA256 tag")
    header = envelope[:HEADER_LEN]
    kind = header[4]
    (issued_ns,) = _TS.unpack(header[5 : 5 + _TS.size])
    nonce = header[5 + _TS.size : 5 + _TS.size + NONCE_LEN]
    bound = header[5 + _TS.size + NONCE_LEN : HEADER_LEN]
    return header, kind, issued_ns, nonce, bound


def nonce_of(envelope: bytes) -> bytes:
    """Return the nonce in `envelope`'s header WITHOUT authenticating it.

    Use it on an envelope you signed yourself (the dispatcher, to know which
    nonce its result must be bound to), or on one that `verify` already
    accepted (the worker, to bind its result).
    """
    return _parse_header(bytes(envelope))[3]


def verify_envelope(
    envelope: bytes,
    *,
    key: Optional[bytes] = None,
    kind: int = KIND_REQUEST,
    bound_to: Optional[bytes] = None,
    replay_cache: Optional[ReplayCache] = None,
    now: Optional[float] = None,
) -> Verified:
    """Authenticate `envelope` and return it as a `Verified`.

    Checks, in order: version/length, HMAC tag (constant time), kind, request
    binding (constant time), timestamp window and the replay cache. Any
    failure raises `SignatureError`; the payload is never interpreted. The
    nonce is recorded only after everything else passed.
    """
    _check_kind(kind)
    envelope = bytes(envelope)
    header, got_kind, issued_ns, nonce, bound = _parse_header(envelope)
    k = _resolve_key(key)
    received_mac = envelope[MAC_OFFSET : MAC_OFFSET + _MAC_LEN]
    payload = envelope[_MIN_ENVELOPE:]
    expected_mac = hmac.new(k, header + payload, hashlib.sha256).digest()
    if not hmac.compare_digest(received_mac, expected_mac):
        raise SignatureError("aux envelope failed HMAC verification; rejecting payload")
    if got_kind != kind:
        raise SignatureError(
            f"aux envelope kind {got_kind} where {kind} was expected (a request "
            "reflected as a result, or the reverse); rejecting payload"
        )
    expected_binding = _binding(kind, bound_to, verifying=True)
    if not hmac.compare_digest(bound, expected_binding):
        raise SignatureError(
            "result envelope is bound to a different request; rejecting payload"
        )
    issued_at = issued_ns / 1e9
    cache = replay_cache if replay_cache is not None else default_replay_cache()
    cache.check_and_record(nonce, issued_at, now=now)
    return Verified(payload=payload, nonce=nonce, issued_at=issued_at, kind=got_kind)


def verify(
    envelope: bytes,
    *,
    key: Optional[bytes] = None,
    kind: int = KIND_REQUEST,
    bound_to: Optional[bytes] = None,
    replay_cache: Optional[ReplayCache] = None,
    now: Optional[float] = None,
) -> bytes:
    """Verify an envelope produced by `sign` and return the inner payload.

    Raises `SignatureError` — without ever treating the payload as anything
    but opaque bytes — if the envelope is short, an old version, unsigned,
    tampered, signed with another key, of the wrong kind, bound to another
    request, stale, too far in the future, or already seen. Callers MUST NOT
    call cloudpickle.loads / pickle.loads on `envelope` directly; only on the
    bytes this returns. See `verify_envelope` for the arguments.
    """
    return verify_envelope(
        envelope, key=key, kind=kind, bound_to=bound_to, replay_cache=replay_cache, now=now
    ).payload


def safe_loads(
    envelope: Union[bytes, Verified],
    *,
    key: Optional[bytes] = None,
    prefix: bytes = b"",
    kind: int = KIND_REQUEST,
    bound_to: Optional[bytes] = None,
    replay_cache: Optional[ReplayCache] = None,
    now: Optional[float] = None,
) -> Any:
    """Verify `envelope`, check its payload starts with `prefix`, then unpickle the rest.

    The single place where bytes received over the wire may reach
    `cloudpickle.loads` (the Semgrep rule `sakura.deserialization.
    cloudpickle-loads-untrusted` exempts only functions named `safe_loads`).
    Authentication happens strictly before deserialisation; any failure
    raises `SignatureError` and nothing is unpickled. A `Verified` (from
    `verify_envelope`) is accepted as already authenticated.
    """
    if isinstance(envelope, Verified):
        payload = envelope.payload
    else:
        payload = verify(
            envelope, key=key, kind=kind, bound_to=bound_to, replay_cache=replay_cache, now=now
        )
    if not payload.startswith(prefix):
        raise SignatureError("signed payload does not carry the expected format tag")
    return cloudpickle.loads(payload[len(prefix):])


__all__ = [
    "DEFAULT_MAX_ENTRIES",
    "DEFAULT_MAX_SKEW_S",
    "DEFAULT_WINDOW_S",
    "HEADER_LEN",
    "KEY_ENV_VAR",
    "KIND_REQUEST",
    "KIND_RESULT",
    "MAC_OFFSET",
    "MAGIC",
    "MIN_KEY_BYTES",
    "NONCE_LEN",
    "ReplayCache",
    "SKEW_ENV_VAR",
    "SignatureError",
    "Verified",
    "WINDOW_ENV_VAR",
    "default_replay_cache",
    "nonce_of",
    "safe_loads",
    "sign",
    "verify",
    "verify_envelope",
]
