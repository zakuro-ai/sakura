"""Default handler implementations registered on every sakura-worker.

- HANDLER_ECHO (0xDEAD): bounces tensors and aux back unchanged.
- HANDLER_HEARTBEAT (0x0003): returns empty tensors + b"PONG" aux.
- HANDLER_EXEC_CLOUDPICKLED (0x0001): cloudpickle.loads(aux) -> {"fn", "args",
  "kwargs"}, reconstructs tensor args from the tensors list as numpy arrays,
  invokes fn(*tensor_args, *args, **kwargs), and returns the cloudpickled
  return value via aux.

`default_registry()` returns a HandlerRegistry pre-populated with all three.
"""
from __future__ import annotations

from typing import Tuple

import cloudpickle
import numpy as np

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


def _tensors_to_arrays(tensors: list[dict]) -> list[np.ndarray]:
    out: list[np.ndarray] = []
    for t in tensors:
        dtype = _DTYPE_TABLE.get(t["dtype_id"], np.uint8)
        arr: np.ndarray = np.frombuffer(t["data"], dtype=dtype)
        if t["shape"]:
            arr = arr.reshape(tuple(t["shape"]))
        out.append(arr)
    return out


def handle_echo(tensors: list[dict], aux: bytes) -> Tuple[list[dict], bytes]:
    """Bounce tensors and aux back unchanged."""
    return (list(tensors), bytes(aux))


def handle_heartbeat(tensors: list[dict], aux: bytes) -> Tuple[list[dict], bytes]:
    """Liveness probe response."""
    return ([], b"PONG")


def handle_exec_cloudpickled(tensors: list[dict], aux: bytes) -> Tuple[list[dict], bytes]:
    """Run a cloudpickled callable + args. Aux on the wire = cloudpickled spec dict.

    Spec format:
      {"fn": <callable>, "args": (...), "kwargs": {...}}
    The decoded numpy arrays from `tensors` are passed as positional args BEFORE
    `args` so callers write `submit(fn, np_arr_1, np_arr_2, scalar=...)`.
    """
    spec = cloudpickle.loads(aux)
    fn = spec["fn"]
    extra_args = spec.get("args", ())
    kwargs = spec.get("kwargs", {})
    arrays = _tensors_to_arrays(tensors)
    out = fn(*arrays, *extra_args, **kwargs)
    return ([], cloudpickle.dumps(out))


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
