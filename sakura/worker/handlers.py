"""Default handler implementations registered on every sakura-worker.

- HANDLER_ECHO (0xDEAD): bounces tensors and aux back unchanged.
- HANDLER_HEARTBEAT (0x0003): returns empty tensors + b"PONG" aux.
- HANDLER_EXEC_CLOUDPICKLED (0x0001): verifies the HMAC envelope on aux, then
  unpickles {"fn", "args", "kwargs"}, reconstructs tensor args from the tensors
  list as numpy arrays, invokes fn(*tensor_args, *args, **kwargs), and returns
  the result via aux in the typed result codec (SAK-0016), signed.

`default_registry()` returns a HandlerRegistry pre-populated with all three.
"""
from __future__ import annotations

from typing import Any, Tuple

import cloudpickle  # type: ignore[import-untyped]
import numpy as np

from sakura.dispatch import result_codec, signing
from sakura.worker.registry import HandlerRegistry

HANDLER_EXEC_CLOUDPICKLED = 0x0001
HANDLER_HEARTBEAT = 0x0003
HANDLER_ECHO = 0xDEAD

# dtype_id (matching the Dtype repr(u8) values in sakura-wire's codec) -> numpy dtype
_DTYPE_TABLE = {
    0: np.float32,
    1: np.float16,
    # 2 = BF16: numpy has no native bf16; fall back to uint16 view.
    2: np.uint16,
    3: np.uint8,   # F8E4M3 is a placeholder dtype for now; bytes-level handling.
    10: np.int64,
    11: np.int32,
    12: np.uint8,
    13: np.bool_,
}


def _tensors_to_arrays(tensors: list[dict[str, Any]]) -> list[np.ndarray[Any, np.dtype[Any]]]:
    out: list[np.ndarray[Any, np.dtype[Any]]] = []
    for t in tensors:
        dtype = _DTYPE_TABLE.get(t["dtype_id"], np.uint8)
        arr: np.ndarray[Any, np.dtype[Any]] = np.frombuffer(t["data"], dtype=dtype)
        if t["shape"]:
            arr = arr.reshape(tuple(t["shape"]))
        out.append(arr)
    return out


def handle_echo(tensors: list[dict[str, Any]], aux: bytes) -> Tuple[list[dict[str, Any]], bytes]:
    """Bounce tensors and aux back unchanged."""
    return (list(tensors), bytes(aux))


def handle_heartbeat(tensors: list[dict[str, Any]], aux: bytes) -> Tuple[list[dict[str, Any]], bytes]:
    """Liveness probe response."""
    return ([], b"PONG")


def _encode_result(out: Any) -> bytes:
    """Typed codec for the return value; signed cloudpickle only as a fallback."""
    if isinstance(out, BaseException):
        return result_codec.encode_error(out)
    try:
        return result_codec.encode_value(out)
    except result_codec.UnsupportedTypeError:
        # Not representable in the typed codec (custom class, torch tensor,
        # namedtuple...). Ship it as signed cloudpickle, tagged so the
        # dispatcher can refuse it (`allow_pickle_results=False`).
        return result_codec.PICKLE_MAGIC + bytes(cloudpickle.dumps(out))


def handle_exec_cloudpickled(tensors: list[dict[str, Any]], aux: bytes) -> Tuple[list[dict[str, Any]], bytes]:
    """Run a cloudpickled callable + args. Aux on the wire = signed, cloudpickled spec dict.

    Spec format:
      {"fn": <callable>, "args": (...), "kwargs": {...}}
    The decoded numpy arrays from `tensors` are passed as positional args BEFORE
    `args` so callers write `submit(fn, np_arr_1, np_arr_2, scalar=...)`.

    `aux` arrives over the wire from a network peer (the dispatcher). It is an
    HMAC-signed envelope (see `sakura.dispatch.signing`); `signing.safe_loads`
    verifies it BEFORE cloudpickle ever sees the bytes, so a spoofed caller
    without the key cannot get arbitrary code executed here by sending a
    crafted `aux`. A stale or replayed request is refused the same way. A
    `SignatureError` propagates (the request is dropped).

    The response aux is a signed result payload: the typed result codec
    (`sakura.dispatch.result_codec`) for values it can represent and for
    exceptions raised or returned by `fn`; signed cloudpickle, tagged
    `PICKLE_MAGIC`, only for other values. It is a `KIND_RESULT` envelope
    bound to the request's nonce.
    """
    spec = signing.safe_loads(aux)
    # `aux` is authenticated, fresh and not replayed at this point; bind the
    # result to its nonce so it cannot answer any other request (sakura#139).
    request_nonce = signing.nonce_of(aux)
    fn = spec["fn"]
    extra_args = spec.get("args", ())
    kwargs = spec.get("kwargs", {})
    arrays = _tensors_to_arrays(tensors)
    try:
        out = fn(*arrays, *extra_args, **kwargs)
    except Exception as exc:  # noqa: BLE001 - reported to the dispatcher
        out = exc
    return ([], signing.sign(_encode_result(out), kind=signing.KIND_RESULT, bound_to=request_nonce))


def default_registry() -> HandlerRegistry:
    r = HandlerRegistry()
    r.register(HANDLER_ECHO, handle_echo)
    r.register(HANDLER_HEARTBEAT, handle_heartbeat)
    r.register(HANDLER_EXEC_CLOUDPICKLED, handle_exec_cloudpickled)
    return r


__all__ = [
    "HANDLER_ECHO",
    "HANDLER_EXEC_CLOUDPICKLED",
    "HANDLER_HEARTBEAT",
    "default_registry",
    "handle_echo",
    "handle_exec_cloudpickled",
    "handle_heartbeat",
]
