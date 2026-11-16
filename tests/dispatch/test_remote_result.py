"""SAK-0016: how RemoteDispatcher's future decodes a worker result.

`_WireFuture` is exercised with a fake native future, so these run without
the compiled sakura_wire extension. They prove the dispatcher authenticates
every result, decodes typed results without unpickling, and unpickles a
signed cloudpickle fallback only when `allow_pickle_results` is true.
"""
from __future__ import annotations

from dataclasses import dataclass

import cloudpickle
import numpy as np
import pytest

from sakura.dispatch import result_codec as rc
from sakura.dispatch import signing
from sakura.dispatch.remote import _WireFuture

KEY = b"\x44" * 32
OTHER_KEY = b"\x55" * 32
# Nonce of the (notional) request these results answer (sakura#139).
REQ_NONCE = b"\x66" * signing.NONCE_LEN


def _sign_result(payload: bytes, *, key=KEY, bound_to=REQ_NONCE) -> bytes:
    return signing.sign(payload, key=key, kind=signing.KIND_RESULT, bound_to=bound_to)


@dataclass
class _WireResult:
    aux: bytes
    elapsed_us: int = 7


class _FakeNativeFuture:
    def __init__(self, aux: bytes) -> None:
        self._aux = aux

    def result(self, timeout=None):
        return _WireResult(aux=self._aux)

    def done(self):
        return True

    def cancel(self):
        return False


class _Custom:
    def __init__(self, v):
        self.v = v


def _future(aux: bytes, *, key=KEY, allow_pickle_results=True, request_nonce=REQ_NONCE) -> _WireFuture:
    return _WireFuture(
        _FakeNativeFuture(aux),
        request_nonce=request_nonce,
        hmac_key=key,
        allow_pickle_results=allow_pickle_results,
    )


@pytest.fixture
def no_unpickle(monkeypatch):
    def _spy(*args, **kwargs):
        raise AssertionError("dispatcher must not unpickle this result")

    monkeypatch.setattr(signing.cloudpickle, "loads", _spy)


def test_typed_result_decodes_without_unpickling(no_unpickle):
    value = {"loss": 0.25, "logits": np.arange(4, dtype=np.float32)}
    aux = _sign_result(rc.encode_value(value))
    res = _future(aux, allow_pickle_results=False).result()
    assert res.value["loss"] == 0.25
    np.testing.assert_array_equal(res.value["logits"], value["logits"])
    assert res.elapsed_us == 7


def test_error_record_raises_remote_execution_error(no_unpickle):
    aux = _sign_result(rc.encode_error(ZeroDivisionError("division by zero")))
    with pytest.raises(rc.RemoteExecutionError, match="ZeroDivisionError: division by zero"):
        _future(aux).result()


@pytest.mark.parametrize("case", ["tampered", "unsigned", "wrong_key"])
@pytest.mark.parametrize("fmt", ["typed", "pickle"])
def test_unauthenticated_results_are_rejected(no_unpickle, case, fmt):
    payload = (
        rc.encode_value({"x": 1}) if fmt == "typed"
        else rc.PICKLE_MAGIC + cloudpickle.dumps(_Custom(1))
    )
    aux = _sign_result(payload)
    key = KEY
    if case == "tampered":
        aux = aux[:-1] + bytes([aux[-1] ^ 0x01])
    elif case == "unsigned":
        aux = payload
    elif case == "wrong_key":
        key = OTHER_KEY
    with pytest.raises(signing.SignatureError):
        _future(aux, key=key).result()


def test_signed_pickle_fallback_accepted_by_default():
    aux = _sign_result(rc.PICKLE_MAGIC + cloudpickle.dumps(_Custom(5)))
    assert _future(aux).result().value.v == 5


def test_signed_pickle_fallback_refused_when_disabled(no_unpickle):
    aux = _sign_result(rc.PICKLE_MAGIC + cloudpickle.dumps(_Custom(5)))
    with pytest.raises(rc.ResultCodecError, match="allow_pickle_results=False"):
        _future(aux, allow_pickle_results=False).result()


def test_untagged_legacy_pickle_result_is_refused(no_unpickle):
    """A pre-SAK-0016 worker signs raw cloudpickle with no format tag; the
    dispatcher must not guess and unpickle it."""
    aux = _sign_result(cloudpickle.dumps({"x": 1}))
    with pytest.raises(rc.ResultCodecError, match="unknown result format"):
        _future(aux).result()
