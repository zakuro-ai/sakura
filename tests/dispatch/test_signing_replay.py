"""Replay protection for signed dispatch envelopes (sakura#139).

Every envelope carries a random nonce and an issue timestamp under the MAC.
The verifier refuses stale or future-stamped envelopes and any nonce it has
already accepted within the window; a result is bound to the nonce of the
request it answers.
"""
from __future__ import annotations

import time
from dataclasses import dataclass

import cloudpickle
import pytest

from sakura.dispatch import result_codec as rc
from sakura.dispatch import signing
from sakura.dispatch.remote import _WireFuture
from sakura.worker.handlers import handle_exec_cloudpickled

KEY = b"\x5a" * 32
T0 = 1_800_000_000.0  # fixed "now" for deterministic clock tests


def _cache(**kw) -> signing.ReplayCache:
    kw.setdefault("window_s", 120.0)
    kw.setdefault("max_skew_s", 30.0)
    return signing.ReplayCache(**kw)


def _verify(env: bytes, cache: signing.ReplayCache, *, now: float = T0, **kw) -> bytes:
    return signing.verify(env, key=KEY, replay_cache=cache, now=now, **kw)


# --- envelope layout and versioning ----------------------------------------


def test_envelope_v2_layout_carries_fresh_nonce_and_timestamp():
    a = signing.sign(b"p", key=KEY)
    b = signing.sign(b"p", key=KEY)
    assert a.startswith(signing.MAGIC) and b.startswith(signing.MAGIC)
    assert len(a) == signing.HEADER_LEN + 32 + 1
    assert signing.nonce_of(a) != signing.nonce_of(b)
    assert len(signing.nonce_of(a)) == signing.NONCE_LEN
    issued = signing.verify_envelope(a, key=KEY, replay_cache=_cache()).issued_at
    assert abs(issued - time.time()) < 5


def test_v1_envelope_is_refused_clearly():
    import hashlib
    import hmac

    v1 = hmac.new(KEY, b"payload", hashlib.sha256).digest() + b"payload"
    with pytest.raises(signing.SignatureError, match="not a v2 signed envelope"):
        signing.verify(v1, key=KEY, replay_cache=_cache())


def test_unknown_future_version_is_refused_clearly():
    env = bytearray(signing.sign(b"p", key=KEY))
    env[3] = 0x09
    with pytest.raises(signing.SignatureError, match="unsupported aux envelope version 9"):
        signing.verify(bytes(env), key=KEY, replay_cache=_cache())


@pytest.mark.parametrize(
    "offset",
    [
        4,  # kind
        5,  # timestamp
        5 + 8,  # nonce
        5 + 8 + signing.NONCE_LEN,  # bound_to
        signing.MAC_OFFSET,  # tag
        -1,  # payload
    ],
)
def test_tampering_any_field_is_still_rejected(offset):
    env = bytearray(signing.sign(b"payload", key=KEY, now=T0))
    env[offset] ^= 0x01
    with pytest.raises(signing.SignatureError):
        _verify(bytes(env), _cache())


# --- replay, staleness, clock skew -----------------------------------------


def test_same_envelope_twice_is_rejected():
    cache = _cache()
    env = signing.sign(b"payload", key=KEY, now=T0)
    assert _verify(env, cache) == b"payload"
    with pytest.raises(signing.SignatureError, match="replayed"):
        _verify(env, cache, now=T0 + 1)


def test_same_envelope_twice_is_rejected_by_the_default_process_cache():
    env = signing.sign(b"payload", key=KEY)
    assert signing.verify(env, key=KEY) == b"payload"
    with pytest.raises(signing.SignatureError, match="replayed"):
        signing.verify(env, key=KEY)


def test_rejected_envelope_does_not_burn_its_nonce():
    """Only a fully verified envelope is recorded: a wrong-key attempt must not
    let an attacker pre-block a legitimate nonce."""
    cache = _cache()
    env = signing.sign(b"payload", key=KEY, now=T0)
    with pytest.raises(signing.SignatureError):
        signing.verify(env, key=b"\x01" * 32, replay_cache=cache, now=T0)
    assert _verify(env, cache) == b"payload"


def test_stale_timestamp_is_rejected():
    env = signing.sign(b"payload", key=KEY, now=T0 - 121)
    with pytest.raises(signing.SignatureError, match="stale"):
        _verify(env, _cache())


def test_far_future_timestamp_is_rejected():
    env = signing.sign(b"payload", key=KEY, now=T0 + 31)
    with pytest.raises(signing.SignatureError, match="future"):
        _verify(env, _cache())


@pytest.mark.parametrize("skew", [-119.0, -5.0, 5.0, 29.0])
def test_small_clock_skew_within_tolerance_is_accepted(skew):
    env = signing.sign(b"payload", key=KEY, now=T0 + skew)
    assert _verify(env, _cache()) == b"payload"


def test_window_and_skew_are_configurable_via_env(monkeypatch):
    monkeypatch.setattr(signing, "_default_cache", None)
    monkeypatch.setenv(signing.WINDOW_ENV_VAR, "10")
    monkeypatch.setenv(signing.SKEW_ENV_VAR, "2")
    cache = signing.default_replay_cache()
    assert (cache.window_s, cache.max_skew_s) == (10.0, 2.0)
    with pytest.raises(signing.SignatureError, match="stale"):
        signing.verify(signing.sign(b"p", key=KEY, now=time.time() - 11), key=KEY)
    with pytest.raises(signing.SignatureError, match="future"):
        signing.verify(signing.sign(b"p", key=KEY, now=time.time() + 5), key=KEY)


@pytest.mark.parametrize("bad", ["0", "-1", "soon"])
def test_bad_window_config_is_refused(monkeypatch, bad):
    monkeypatch.setattr(signing, "_default_cache", None)
    monkeypatch.setenv(signing.WINDOW_ENV_VAR, bad)
    with pytest.raises(ValueError):
        signing.default_replay_cache()


# --- request/result binding -------------------------------------------------


def test_result_for_a_different_request_is_rejected():
    req_a = signing.sign(b"req-a", key=KEY, now=T0)
    req_b = signing.sign(b"req-b", key=KEY, now=T0)
    res_a = signing.sign(b"res-a", key=KEY, kind=signing.KIND_RESULT,
                         bound_to=signing.nonce_of(req_a), now=T0)
    with pytest.raises(signing.SignatureError, match="different request"):
        _verify(res_a, _cache(), kind=signing.KIND_RESULT, bound_to=signing.nonce_of(req_b))
    assert _verify(res_a, _cache(), kind=signing.KIND_RESULT,
                   bound_to=signing.nonce_of(req_a)) == b"res-a"


def test_request_reflected_as_result_is_rejected():
    req = signing.sign(b"req", key=KEY, now=T0)
    with pytest.raises(signing.SignatureError, match="kind"):
        _verify(req, _cache(), kind=signing.KIND_RESULT, bound_to=signing.nonce_of(req))


def test_result_presented_as_request_is_rejected():
    res = signing.sign(b"res", key=KEY, kind=signing.KIND_RESULT,
                       bound_to=b"\x01" * signing.NONCE_LEN, now=T0)
    with pytest.raises(signing.SignatureError, match="kind"):
        _verify(res, _cache())


def test_result_verification_requires_the_request_nonce():
    res = signing.sign(b"res", key=KEY, kind=signing.KIND_RESULT,
                       bound_to=b"\x01" * signing.NONCE_LEN, now=T0)
    with pytest.raises(signing.SignatureError, match="requires the nonce"):
        _verify(res, _cache(), kind=signing.KIND_RESULT)


def test_signing_a_result_requires_a_binding():
    with pytest.raises(ValueError):
        signing.sign(b"res", key=KEY, kind=signing.KIND_RESULT)
    with pytest.raises(ValueError):
        signing.sign(b"res", key=KEY, kind=signing.KIND_RESULT, bound_to=b"short")


# --- cache bounds and eviction ----------------------------------------------


def test_expired_nonces_are_evicted():
    cache = _cache(window_s=10.0, max_skew_s=1.0)
    for i in range(5):
        _verify(signing.sign(b"p", key=KEY, now=T0 + i), cache, now=T0 + i)
    assert len(cache) == 5
    # At T0 + 12.5 the envelopes issued at T0..T0+2 have left the window.
    _verify(signing.sign(b"p", key=KEY, now=T0 + 12.5), cache, now=T0 + 12.5)
    assert len(cache) == 3


def test_cache_is_memory_bounded_without_reopening_replays():
    cache = _cache(max_entries=4)
    envs = [signing.sign(b"p", key=KEY, now=T0 + i * 0.001) for i in range(10)]
    for env in envs:
        _verify(env, cache)
    assert len(cache) == 4
    # Evicted (oldest) envelopes are refused, not accepted a second time.
    for env in envs:
        with pytest.raises(signing.SignatureError):
            _verify(env, cache)
    # A fresh envelope is still accepted.
    _verify(signing.sign(b"p", key=KEY, now=T0 + 1), cache)
    assert len(cache) == 4


# --- call sites -------------------------------------------------------------


def _add(x, y):
    return x + y


def test_worker_rejects_a_replayed_request(monkeypatch):
    monkeypatch.setenv(signing.KEY_ENV_VAR, KEY.hex())
    aux = signing.sign(cloudpickle.dumps({"fn": _add, "args": (2, 3), "kwargs": {}}), key=KEY)
    handle_exec_cloudpickled([], aux)

    import sakura.dispatch.signing as signing_mod

    def _spy(*args, **kwargs):
        raise AssertionError("a replayed request must not be unpickled")

    monkeypatch.setattr(signing_mod.cloudpickle, "loads", _spy)
    with pytest.raises(signing.SignatureError, match="replayed"):
        handle_exec_cloudpickled([], aux)


def test_worker_result_is_bound_to_its_request(monkeypatch):
    monkeypatch.setenv(signing.KEY_ENV_VAR, KEY.hex())
    aux = signing.sign(cloudpickle.dumps({"fn": _add, "args": (2, 3), "kwargs": {}}), key=KEY)
    _, out = handle_exec_cloudpickled([], aux)
    other = signing.sign(b"other request", key=KEY)
    with pytest.raises(signing.SignatureError, match="different request"):
        signing.verify(out, key=KEY, kind=signing.KIND_RESULT, bound_to=signing.nonce_of(other))


@dataclass
class _WireResult:
    aux: bytes
    elapsed_us: int = 1


class _FakeNativeFuture:
    def __init__(self, aux: bytes) -> None:
        self._aux = aux

    def result(self, timeout=None):
        return _WireResult(aux=self._aux)


def _result_for(req_nonce: bytes, value) -> bytes:
    return signing.sign(rc.encode_value(value), key=KEY, kind=signing.KIND_RESULT, bound_to=req_nonce)


def test_dispatcher_rejects_a_result_replayed_into_a_new_call():
    nonce_a = signing.nonce_of(signing.sign(b"a", key=KEY))
    nonce_b = signing.nonce_of(signing.sign(b"b", key=KEY))
    res_a = _result_for(nonce_a, 41)
    assert _WireFuture(_FakeNativeFuture(res_a), request_nonce=nonce_a, hmac_key=KEY).result().value == 41
    # The same captured result fed to a different call: wrong binding.
    with pytest.raises(signing.SignatureError, match="different request"):
        _WireFuture(_FakeNativeFuture(res_a), request_nonce=nonce_b, hmac_key=KEY).result()
    # ...and to a second future of the same call: already seen.
    with pytest.raises(signing.SignatureError, match="replayed"):
        _WireFuture(_FakeNativeFuture(res_a), request_nonce=nonce_a, hmac_key=KEY).result()


def test_future_result_can_be_read_twice():
    nonce = signing.nonce_of(signing.sign(b"a", key=KEY))
    fut = _WireFuture(_FakeNativeFuture(_result_for(nonce, {"x": 1})), request_nonce=nonce, hmac_key=KEY)
    assert fut.result().value == {"x": 1}
    assert fut.result().value == {"x": 1}


def test_future_error_is_raised_again_not_turned_into_a_replay():
    nonce = signing.nonce_of(signing.sign(b"a", key=KEY))
    aux = signing.sign(rc.encode_error(KeyError("k")), key=KEY, kind=signing.KIND_RESULT, bound_to=nonce)
    fut = _WireFuture(_FakeNativeFuture(aux), request_nonce=nonce, hmac_key=KEY)
    for _ in range(2):
        with pytest.raises(rc.RemoteExecutionError):
            fut.result()


def test_signed_pickle_result_is_unpickled_once_without_double_verification():
    nonce = signing.nonce_of(signing.sign(b"a", key=KEY))
    aux = signing.sign(rc.PICKLE_MAGIC + cloudpickle.dumps(complex(1, 2)), key=KEY,
                       kind=signing.KIND_RESULT, bound_to=nonce)
    assert _WireFuture(_FakeNativeFuture(aux), request_nonce=nonce, hmac_key=KEY).result().value == 1 + 2j
