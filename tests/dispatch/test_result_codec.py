"""SAK-0016: typed (non-pickle) codec for dispatch results.

Proves the codec round-trips the value types dispatch results actually use,
rejects what it cannot represent (so the worker can fall back explicitly),
and fails closed on malformed payloads without ever reaching cloudpickle.
"""
from __future__ import annotations

import collections
import dataclasses
import struct

import cloudpickle
import numpy as np
import pytest

from sakura.dispatch import result_codec as rc


@pytest.mark.parametrize(
    "value",
    [
        None,
        True,
        False,
        0,
        -1,
        127,
        128,
        -129,
        2**200,
        -(2**200),
        0.25,
        float("inf"),
        complex(1.5, -2.0),
        "",
        "héllo ✓",
        b"",
        b"\x00\xff",
        [],
        (),
        {},
        [1, "a", (2.0, None)],
        (1, [2, {"k": b"v"}]),
        {"loss": 0.25, "acc": 0.9, "epoch": 3, 7: "int key", (1, 2): "tuple key"},
    ],
)
def test_round_trips_builtin_values(value):
    out = rc.decode(rc.encode_value(value))
    assert out == value
    assert type(out) is type(value)


def test_round_trips_nan():
    out = rc.decode(rc.encode_value(float("nan")))
    assert out != out


@pytest.mark.parametrize(
    "dtype",
    [np.bool_, np.int8, np.int16, np.int32, np.int64, np.uint8, np.uint64,
     np.float16, np.float32, np.float64, np.complex64, np.complex128],
)
def test_round_trips_ndarrays(dtype):
    arr = (np.arange(24) % 3).astype(dtype).reshape(2, 3, 4)
    out = rc.decode(rc.encode_value(arr))
    assert isinstance(out, np.ndarray)
    assert out.dtype == arr.dtype
    assert out.shape == arr.shape
    np.testing.assert_array_equal(out, arr)
    assert out.flags.writeable


def test_round_trips_big_endian_non_contiguous_zero_d_and_empty_arrays():
    big = np.arange(6, dtype=">i4").reshape(2, 3)
    for arr in (big, big.T, np.array(3.5), np.zeros((0, 4), dtype=np.float32)):
        out = rc.decode(rc.encode_value(arr))
        assert out.dtype == arr.dtype and out.shape == arr.shape
        np.testing.assert_array_equal(out, arr)


def test_round_trips_numpy_scalars():
    for v in (np.float32(1.5), np.int64(-3), np.bool_(True), np.uint8(255), np.complex64(1 + 2j)):
        out = rc.decode(rc.encode_value({"m": v}))["m"]
        assert type(out) is type(v)
        assert out == v


@dataclasses.dataclass
class _Point:
    x: int


_NT = collections.namedtuple("_NT", "a b")


@pytest.mark.parametrize(
    "value",
    [
        _Point(1),
        _NT(1, 2),
        collections.OrderedDict(a=1),
        {1, 2},
        object(),
        np.array([object()], dtype=object),
        np.array(["a"]),
        np.str_("x"),
        [1, _Point(2)],
    ],
)
def test_unsupported_values_raise_unsupported_type(value):
    with pytest.raises(rc.UnsupportedTypeError):
        rc.encode_value(value)


def test_error_record_raises_remote_execution_error():
    try:
        raise ValueError("bad shape")
    except ValueError as exc:
        payload = rc.encode_error(exc)
    with pytest.raises(rc.RemoteExecutionError) as info:
        rc.decode(payload)
    assert info.value.remote_type == "ValueError"
    assert info.value.remote_message == "bad shape"
    assert "bad shape" in info.value.remote_traceback
    assert isinstance(info.value, RuntimeError)


def _typed(body: bytes) -> bytes:
    return rc.TYPED_MAGIC + b"V" + body


@pytest.mark.parametrize(
    "payload",
    [
        b"",
        b"SKR",
        rc.PICKLE_MAGIC + b"V" + b"N",
        rc.TYPED_MAGIC + b"Q" + b"N",
        _typed(b""),
        _typed(b"?"),
        _typed(b"NN"),  # trailing bytes
        _typed(b"s" + struct.pack(">I", 10) + b"abc"),  # truncated
        _typed(b"l" + struct.pack(">I", 2**31)),  # huge container, no body
        _typed(b"d" + struct.pack(">I", 1) + b"l" + struct.pack(">I", 0) + b"N"),  # unhashable key
        _typed(b"a" + bytes([3]) + b"|O8" + bytes([1]) + struct.pack(">QQ", 1, 8) + b"\0" * 8),
        _typed(b"a" + bytes([3]) + b"<f4" + bytes([1]) + struct.pack(">QQ", 4, 8) + b"\0" * 8),
        _typed(b"l" + struct.pack(">I", 1) * 1 + b"l\x00\x00\x00\x01" * 100 + b"N"),  # too deep
        rc.TYPED_MAGIC + b"X" + b"s" + struct.pack(">I", 1) + b"E",  # truncated error
    ],
)
def test_malformed_payloads_raise_codec_error(payload):
    with pytest.raises(rc.ResultCodecError):
        rc.decode(payload)


def test_decode_never_calls_cloudpickle(monkeypatch):
    def _spy(*args, **kwargs):
        raise AssertionError("the typed result codec must never unpickle")

    monkeypatch.setattr(cloudpickle, "loads", _spy)
    monkeypatch.setattr("pickle.loads", _spy)
    with pytest.raises(rc.ResultCodecError):
        rc.decode(_typed(cloudpickle.dumps({"x": 1})))
    assert rc.decode(rc.encode_value({"x": np.ones(3)}))["x"].sum() == 3
