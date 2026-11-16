"""Typed (non-pickle) codec for dispatch *results* on the wire (SAK-0016).

The worker encodes the return value of a dispatched callable with this codec
and the dispatcher decodes it without ever running `cloudpickle.loads`, so a
compromised worker that still holds the fleet HMAC key cannot get code
executed on the dispatcher through a result.

Supported values (exact types; subclasses are *not* covered, see below):

- ``None``, ``bool``, ``int``, ``float``, ``complex``, ``str``, ``bytes``
- ``list``, ``tuple``, ``dict`` (keys and values of supported types)
- ``numpy.ndarray`` and numpy scalars of a bool / int / uint / float /
  complex dtype (no object, string, void or datetime dtypes)

An exception raised (or returned) by the callable is sent as an error record
(type name, message, formatted traceback) and re-raised on the dispatcher as
`RemoteExecutionError`; the original exception object is not reconstructed.

A value of any other type (custom classes, dataclasses, namedtuples, torch
tensors, ``OrderedDict``...) makes `encode_value` raise `UnsupportedTypeError`.
The worker then falls back to the HMAC-signed cloudpickle format, which the
dispatcher only accepts when ``allow_pickle_results`` is true (the default,
for backward compatibility); see `sakura.dispatch.remote`.

Wire layout of a result payload (the bytes inside the HMAC envelope)::

    TYPED_MAGIC (4 bytes) || status (b"V" value | b"X" error) || body

All lengths are big-endian unsigned; every length is bounds-checked against
the remaining buffer before use, and nesting depth is capped, so a malformed
payload raises `ResultCodecError` and never allocates more than it received.
"""
from __future__ import annotations

import re
import struct
import traceback
from typing import Any

import numpy as np

#: Format tag for a typed-codec result payload.
TYPED_MAGIC = b"SKR1"
#: Format tag for the signed-cloudpickle fallback result payload.
PICKLE_MAGIC = b"SKP1"

_STATUS_VALUE = b"V"
_STATUS_ERROR = b"X"

_MAX_DEPTH = 64
_MAX_NDIM = 32
# dtype.str of the numeric dtypes we accept, e.g. "<f4", "|b1", ">i8".
_DTYPE_RE = re.compile(r"^[<>|=][biufc][0-9]{1,2}$")


class ResultCodecError(ValueError):
    """Raised when a result payload is malformed or uses an unknown tag."""


class UnsupportedTypeError(TypeError):
    """Raised by `encode_value` for a value the typed codec cannot represent."""


class RemoteExecutionError(RuntimeError):
    """The dispatched callable raised (or returned) an exception on the worker.

    Attributes:
        remote_type: qualified name of the worker-side exception type.
        remote_message: ``str()`` of the worker-side exception.
        remote_traceback: formatted worker-side traceback (may be empty).
    """

    def __init__(self, remote_type: str, remote_message: str, remote_traceback: str = ""):
        super().__init__(f"{remote_type}: {remote_message}")
        self.remote_type = remote_type
        self.remote_message = remote_message
        self.remote_traceback = remote_traceback


# --------------------------------------------------------------------------
# encoding
# --------------------------------------------------------------------------

def _u32(n: int) -> bytes:
    return struct.pack(">I", n)


def _blob(tag: bytes, data: bytes) -> bytes:
    return tag + _u32(len(data)) + data


def _dtype_str(dtype: np.dtype[Any]) -> str:
    s = str(dtype.str)
    if not _DTYPE_RE.match(s):
        raise UnsupportedTypeError(f"numpy dtype {dtype!r} is not supported by the typed result codec")
    return s


def _encode(v: Any, out: list[bytes], depth: int) -> None:
    if depth > _MAX_DEPTH:
        raise UnsupportedTypeError(f"value nested deeper than {_MAX_DEPTH} levels")
    t = type(v)
    if v is None:
        out.append(b"N")
    elif t is bool:
        out.append(b"T" if v else b"F")
    elif t is int:
        n_bytes = (v.bit_length() + 8) // 8  # +1 bit for the sign
        out.append(_blob(b"i", v.to_bytes(n_bytes, "big", signed=True)))
    elif t is float:
        out.append(b"f" + struct.pack(">d", v))
    elif t is complex:
        out.append(b"c" + struct.pack(">dd", v.real, v.imag))
    elif t is str:
        out.append(_blob(b"s", v.encode("utf-8", "surrogatepass")))
    elif t is bytes:
        out.append(_blob(b"b", v))
    elif t is list or t is tuple:
        out.append((b"l" if t is list else b"t") + _u32(len(v)))
        for item in v:
            _encode(item, out, depth + 1)
    elif t is dict:
        out.append(b"d" + _u32(len(v)))
        for key, val in v.items():
            _encode(key, out, depth + 1)
            _encode(val, out, depth + 1)
    elif t is np.ndarray:
        ds = _dtype_str(v.dtype).encode("ascii")
        if v.ndim > _MAX_NDIM:
            raise UnsupportedTypeError(f"ndarray with {v.ndim} dims (max {_MAX_NDIM})")
        out.append(b"a" + bytes([len(ds)]) + ds + bytes([v.ndim]))
        out.append(b"".join(struct.pack(">Q", int(d)) for d in v.shape))
        data = np.ascontiguousarray(v).tobytes()
        out.append(struct.pack(">Q", len(data)) + data)
    elif isinstance(v, np.generic) and t.__module__ == "numpy":
        ds = _dtype_str(v.dtype).encode("ascii")
        out.append(b"n" + bytes([len(ds)]) + ds + _blob(b"", v.tobytes()))
    else:
        raise UnsupportedTypeError(
            f"{t.__module__}.{t.__qualname__} is not supported by the typed result codec"
        )


def encode_value(value: Any) -> bytes:
    """Encode a callable's return value. Raises `UnsupportedTypeError` if not representable."""
    out: list[bytes] = [TYPED_MAGIC, _STATUS_VALUE]
    _encode(value, out, 0)
    return b"".join(out)


def encode_error(exc: BaseException) -> bytes:
    """Encode an exception raised/returned by the callable as an error record."""
    t = type(exc)
    name = t.__qualname__ if t.__module__ == "builtins" else f"{t.__module__}.{t.__qualname__}"
    try:
        message = str(exc)
    except Exception:  # noqa: BLE001 - a broken __str__ must not hide the failure
        message = "<unprintable exception>"
    tb = "".join(traceback.format_exception(t, exc, exc.__traceback__))
    out: list[bytes] = [TYPED_MAGIC, _STATUS_ERROR]
    for s in (name, message, tb):
        out.append(_blob(b"s", s.encode("utf-8", "surrogatepass")))
    return b"".join(out)


# --------------------------------------------------------------------------
# decoding
# --------------------------------------------------------------------------

class _Reader:
    __slots__ = ("buf", "pos")

    def __init__(self, buf: bytes) -> None:
        self.buf = buf
        self.pos = 0

    def take(self, n: int) -> bytes:
        if n < 0 or self.pos + n > len(self.buf):
            raise ResultCodecError("truncated result payload")
        chunk = self.buf[self.pos:self.pos + n]
        self.pos += n
        return chunk

    def u8(self) -> int:
        return self.take(1)[0]

    def u32(self) -> int:
        n: int = struct.unpack(">I", self.take(4))[0]
        return n

    def u64(self) -> int:
        n: int = struct.unpack(">Q", self.take(8))[0]
        return n

    def remaining(self) -> int:
        return len(self.buf) - self.pos


def _read_dtype(r: _Reader) -> np.dtype[Any]:
    raw = r.take(r.u8())
    try:
        s = raw.decode("ascii")
    except UnicodeDecodeError as e:
        raise ResultCodecError("non-ascii dtype tag") from e
    if not _DTYPE_RE.match(s):
        raise ResultCodecError(f"dtype {s!r} not allowed in a result payload")
    try:
        return np.dtype(s)
    except TypeError as e:
        raise ResultCodecError(f"invalid dtype {s!r}") from e


def _read_str(r: _Reader) -> str:
    if r.take(1) != b"s":
        raise ResultCodecError("expected a string field")
    return r.take(r.u32()).decode("utf-8", "surrogatepass")


def _decode(r: _Reader, depth: int) -> Any:
    if depth > _MAX_DEPTH:
        raise ResultCodecError(f"result payload nested deeper than {_MAX_DEPTH} levels")
    tag = r.take(1)
    if tag == b"N":
        return None
    if tag == b"T":
        return True
    if tag == b"F":
        return False
    if tag == b"i":
        return int.from_bytes(r.take(r.u32()), "big", signed=True)
    if tag == b"f":
        return struct.unpack(">d", r.take(8))[0]
    if tag == b"c":
        re_, im = struct.unpack(">dd", r.take(16))
        return complex(re_, im)
    if tag == b"s":
        return r.take(r.u32()).decode("utf-8", "surrogatepass")
    if tag == b"b":
        return r.take(r.u32())
    if tag in (b"l", b"t"):
        n = r.u32()
        if n > r.remaining():  # every element takes at least one byte
            raise ResultCodecError("container length exceeds payload")
        items = [_decode(r, depth + 1) for _ in range(n)]
        return items if tag == b"l" else tuple(items)
    if tag == b"d":
        n = r.u32()
        if 2 * n > r.remaining():
            raise ResultCodecError("dict length exceeds payload")
        d: dict[Any, Any] = {}
        for _ in range(n):
            k = _decode(r, depth + 1)
            try:
                d[k] = _decode(r, depth + 1)
            except TypeError as e:  # unhashable key
                raise ResultCodecError(f"unhashable dict key of type {type(k).__name__}") from e
        return d
    if tag == b"a":
        dtype = _read_dtype(r)
        ndim = r.u8()
        if ndim > _MAX_NDIM:
            raise ResultCodecError(f"ndarray with {ndim} dims (max {_MAX_NDIM})")
        shape = tuple(r.u64() for _ in range(ndim))
        data = r.take(r.u64())
        count = 1
        for dim in shape:
            count *= dim
        if count * dtype.itemsize != len(data):
            raise ResultCodecError("ndarray data length does not match shape and dtype")
        return np.frombuffer(data, dtype=dtype).reshape(shape).copy()
    if tag == b"n":
        dtype = _read_dtype(r)
        data = r.take(r.u32())
        if len(data) != dtype.itemsize:
            raise ResultCodecError("numpy scalar data length does not match dtype")
        return np.frombuffer(data, dtype=dtype)[0]
    raise ResultCodecError(f"unknown value tag {tag!r}")


def decode(payload: bytes) -> Any:
    """Decode a typed result payload (starting with `TYPED_MAGIC`).

    Returns the value, or raises `RemoteExecutionError` for an error record.
    Raises `ResultCodecError` for any malformed payload. Never unpickles.
    """
    r = _Reader(bytes(payload))
    if r.take(len(TYPED_MAGIC)) != TYPED_MAGIC:
        raise ResultCodecError("not a typed result payload")
    status = r.take(1)
    if status == _STATUS_VALUE:
        value = _decode(r, 0)
    elif status == _STATUS_ERROR:
        name, message, tb = _read_str(r), _read_str(r), _read_str(r)
        if r.remaining():
            raise ResultCodecError("trailing bytes after error record")
        raise RemoteExecutionError(name, message, tb)
    else:
        raise ResultCodecError(f"unknown result status {status!r}")
    if r.remaining():
        raise ResultCodecError("trailing bytes after result value")
    return value


__all__ = [
    "PICKLE_MAGIC",
    "RemoteExecutionError",
    "ResultCodecError",
    "TYPED_MAGIC",
    "UnsupportedTypeError",
    "decode",
    "encode_error",
    "encode_value",
]
