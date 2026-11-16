"""Regression tests for sakura#9 at the actual worker call site.

`handle_exec_cloudpickled` (sakura/worker/handlers.py) is the receiving end
that used to call `cloudpickle.loads(aux)` directly on wire-supplied bytes.
These tests exercise the real function (no sakura_wire/native transport
needed — `aux` is just bytes) and prove:
  1. A legitimately signed spec dispatches and returns a signed, correct result
     (typed result codec, SAK-0016).
  2. A tampered / unsigned aux is rejected before cloudpickle.loads runs.
"""
from __future__ import annotations

import os

import cloudpickle
import pytest

from sakura.dispatch import signing
from sakura.worker.handlers import handle_exec_cloudpickled

KEY = b"\x33" * 32


def _add(x, y):
    return x + y


def test_handle_exec_cloudpickled_accepts_valid_signed_spec(monkeypatch):
    monkeypatch.setenv(signing.KEY_ENV_VAR, KEY.hex())
    spec = {"fn": _add, "args": (2, 3), "kwargs": {}}
    aux = signing.sign(cloudpickle.dumps(spec), key=KEY)

    tensors, out_aux = handle_exec_cloudpickled([], aux)

    assert tensors == []
    # SAK-0016: the response is a signed typed-codec payload, not a pickle.
    from sakura.dispatch import result_codec

    result = result_codec.decode(
        signing.verify(out_aux, key=KEY, kind=signing.KIND_RESULT, bound_to=signing.nonce_of(aux))
    )
    assert result == 5


def test_handle_exec_cloudpickled_rejects_tampered_aux(monkeypatch):
    monkeypatch.setenv(signing.KEY_ENV_VAR, KEY.hex())
    spec = {"fn": _add, "args": (2, 3), "kwargs": {}}
    aux = signing.sign(cloudpickle.dumps(spec), key=KEY)
    tampered = bytearray(aux)
    tampered[-1] ^= 0xFF  # corrupt the cloudpickled payload bytes

    with pytest.raises(signing.SignatureError):
        handle_exec_cloudpickled([], bytes(tampered))


def test_handle_exec_cloudpickled_rejects_unsigned_raw_cloudpickle(monkeypatch):
    """The exact old attack: an attacker sends raw cloudpickle bytes (no
    envelope) straight at the handler. It must be rejected, not executed."""
    monkeypatch.setenv(signing.KEY_ENV_VAR, KEY.hex())
    malicious_spec = {"fn": _add, "args": (2, 3), "kwargs": {}}
    raw_aux = cloudpickle.dumps(malicious_spec)  # no HMAC envelope

    with pytest.raises(signing.SignatureError):
        handle_exec_cloudpickled([], raw_aux)


def test_handle_exec_cloudpickled_never_calls_cloudpickle_loads_on_tampered_aux(monkeypatch):
    """Spy on cloudpickle.loads (as used inside sakura.worker.handlers) to
    prove the rejection happens strictly before deserialisation."""
    monkeypatch.setenv(signing.KEY_ENV_VAR, KEY.hex())
    spec = {"fn": _add, "args": (2, 3), "kwargs": {}}
    aux = signing.sign(cloudpickle.dumps(spec), key=KEY)
    tampered = bytearray(aux)
    tampered[signing.MAC_OFFSET] ^= 0xFF  # corrupt the MAC

    import sakura.worker.handlers as handlers_mod

    def _spy_loads(*args, **kwargs):
        raise AssertionError("cloudpickle.loads must not be called on an unverified aux")

    monkeypatch.setattr(handlers_mod.cloudpickle, "loads", _spy_loads)

    with pytest.raises(signing.SignatureError):
        handle_exec_cloudpickled([], bytes(tampered))


def test_handle_exec_cloudpickled_rejects_when_key_missing(monkeypatch):
    monkeypatch.delenv(signing.KEY_ENV_VAR, raising=False)
    spec = {"fn": _add, "args": (2, 3), "kwargs": {}}
    aux = signing.sign(cloudpickle.dumps(spec), key=KEY)  # signed with an explicit key

    # Handler has no explicit key wired in (mirrors the real call site,
    # which relies on the env var set by the dispatcher/supervisor); with
    # no env var set it must refuse rather than silently deserialize.
    with pytest.raises(signing.SignatureError):
        handle_exec_cloudpickled([], aux)


# --- SAK-0016: typed result codec on the response side ---------------------

from sakura.dispatch import result_codec  # noqa: E402


class _Opaque:
    def __init__(self, v):
        self.v = v


def _make_opaque(v):
    return _Opaque(v)


def _boom():
    raise KeyError("missing")


def _run(monkeypatch, fn, *args):
    monkeypatch.setenv(signing.KEY_ENV_VAR, KEY.hex())
    aux = signing.sign(cloudpickle.dumps({"fn": fn, "args": args, "kwargs": {}}), key=KEY)
    tensors, out_aux = handle_exec_cloudpickled([], aux)
    assert tensors == []
    return signing.verify(out_aux, key=KEY, kind=signing.KIND_RESULT, bound_to=signing.nonce_of(aux))


def test_result_uses_typed_codec(monkeypatch):
    payload = _run(monkeypatch, _add, 2, 3)
    assert payload.startswith(result_codec.TYPED_MAGIC)
    assert result_codec.decode(payload) == 5


def test_unsupported_result_falls_back_to_tagged_pickle(monkeypatch):
    payload = _run(monkeypatch, _make_opaque, 9)
    assert payload.startswith(result_codec.PICKLE_MAGIC)
    assert cloudpickle.loads(payload[len(result_codec.PICKLE_MAGIC):]).v == 9


def test_exception_in_fn_is_returned_as_error_record(monkeypatch):
    payload = _run(monkeypatch, _boom)
    with pytest.raises(result_codec.RemoteExecutionError) as info:
        result_codec.decode(payload)
    assert info.value.remote_type == "KeyError"


def test_handle_exec_cloudpickled_rejects_wrong_key(monkeypatch):
    monkeypatch.setenv(signing.KEY_ENV_VAR, KEY.hex())
    aux = signing.sign(cloudpickle.dumps({"fn": _add, "args": (1, 2), "kwargs": {}}), key=b"\x77" * 32)

    import sakura.dispatch.signing as signing_mod

    def _spy(*args, **kwargs):
        raise AssertionError("cloudpickle.loads must not run on a wrong-key aux")

    monkeypatch.setattr(signing_mod.cloudpickle, "loads", _spy)
    with pytest.raises(signing.SignatureError):
        handle_exec_cloudpickled([], aux)
