"""HandlerRegistry: routes incoming RPCs to per-handler-id Python callables."""
from __future__ import annotations

import pytest

from sakura.worker.registry import HandlerRegistry


def _echo(tensors, aux):
    return (tensors, aux)


def _double_tensors(tensors, aux):
    out = []
    for t in tensors:
        out.append({**t, "data": bytes(b * 2 for b in t["data"])})
    return (out, aux)


class TestHandlerRegistry:
    def test_register_and_dispatch_known_handler(self):
        r = HandlerRegistry()
        r.register(0xDEAD, _echo)
        out_t, out_a = r.dispatch(0xDEAD, [{"shape": [1], "dtype_id": 12,
                                             "device_id": 0, "data": b"\x01"}], b"aux")
        assert out_t[0]["data"] == b"\x01"
        assert out_a == b"aux"

    def test_dispatch_unknown_handler_raises(self):
        r = HandlerRegistry()
        with pytest.raises(KeyError, match="0x1234"):
            r.dispatch(0x1234, [], b"")

    def test_double_register_replaces(self):
        r = HandlerRegistry()
        r.register(0xDEAD, _echo)
        r.register(0xDEAD, _double_tensors)
        out_t, _ = r.dispatch(0xDEAD,
                               [{"shape": [3], "dtype_id": 12, "device_id": 0, "data": b"\x01\x02\x03"}],
                               b"")
        assert out_t[0]["data"] == b"\x02\x04\x06"

    def test_handler_id_must_be_int(self):
        r = HandlerRegistry()
        with pytest.raises(TypeError, match="handler_id"):
            r.register("DEAD", _echo)

    def test_handler_must_be_callable(self):
        r = HandlerRegistry()
        with pytest.raises(TypeError, match="callable"):
            r.register(0xDEAD, "not a callable")

    def test_dispatch_returns_2_tuple_check(self):
        def _bad(tensors, aux):
            return tensors

        r = HandlerRegistry()
        r.register(0xBEEF, _bad)
        with pytest.raises(ValueError, match="must return.*2-tuple"):
            r.dispatch(0xBEEF, [], b"")
