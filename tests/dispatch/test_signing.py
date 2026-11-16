"""Regression tests for sakura#9: HMAC-signed aux envelopes.

Proves:
  1. A correctly-signed payload verifies and round-trips.
  2. A tampered / unsigned / wrong-key payload is REJECTED by `verify` and
     therefore never reaches `cloudpickle.loads` in the real call sites
     (`sakura/dispatch/remote.py`, `sakura/worker/handlers.py`).
"""
from __future__ import annotations

import os

import cloudpickle
import pytest

from sakura.dispatch import signing


KEY = b"\x11" * 32
OTHER_KEY = b"\x22" * 32


def test_sign_then_verify_round_trips():
    payload = cloudpickle.dumps({"hello": "world"})
    envelope = signing.sign(payload, key=KEY)
    recovered = signing.verify(envelope, key=KEY)
    assert recovered == payload
    assert cloudpickle.loads(recovered) == {"hello": "world"}


def test_verify_rejects_tampered_payload():
    payload = cloudpickle.dumps({"hello": "world"})
    envelope = signing.sign(payload, key=KEY)
    # Flip a byte inside the signed payload (simulating a MITM / spoofed peer).
    tampered = bytearray(envelope)
    tampered[-1] ^= 0xFF
    with pytest.raises(signing.SignatureError):
        signing.verify(bytes(tampered), key=KEY)


def test_verify_rejects_wrong_key():
    payload = cloudpickle.dumps({"hello": "world"})
    envelope = signing.sign(payload, key=KEY)
    with pytest.raises(signing.SignatureError):
        signing.verify(envelope, key=OTHER_KEY)


def test_verify_rejects_unsigned_raw_cloudpickle_bytes():
    """An attacker sending raw cloudpickle bytes with no envelope at all
    (i.e. exactly what the old, vulnerable call sites accepted) must be
    rejected, not silently treated as a valid (if short) envelope."""
    raw = cloudpickle.dumps({"attack": "payload"})
    with pytest.raises(signing.SignatureError):
        signing.verify(raw, key=KEY)


def test_verify_never_reaches_cloudpickle_loads_on_tampered_input(monkeypatch):
    """Directly proves the ordering: verification happens strictly before
    deserialisation. We patch cloudpickle.loads to fail the test if it is
    ever invoked, then feed `verify` a tampered envelope."""
    called = {"loads": False}

    def _spy_loads(*args, **kwargs):
        called["loads"] = True
        raise AssertionError("cloudpickle.loads must not be called on unverified bytes")

    monkeypatch.setattr(cloudpickle, "loads", _spy_loads)

    payload = cloudpickle.dumps({"hello": "world"})
    envelope = signing.sign(payload, key=KEY)
    tampered = bytearray(envelope)
    tampered[signing.MAC_OFFSET] ^= 0xFF  # corrupt the MAC itself

    with pytest.raises(signing.SignatureError):
        signing.verify(bytes(tampered), key=KEY)
        cloudpickle.loads(signing.verify(bytes(tampered), key=KEY))  # unreachable

    assert called["loads"] is False


def test_env_var_key_used_when_no_explicit_key(monkeypatch):
    monkeypatch.setenv(signing.KEY_ENV_VAR, KEY.hex())
    payload = b"some bytes"
    envelope = signing.sign(payload)
    assert signing.verify(envelope) == payload


def test_missing_key_raises_signature_error(monkeypatch):
    monkeypatch.delenv(signing.KEY_ENV_VAR, raising=False)
    with pytest.raises(signing.SignatureError):
        signing.sign(b"payload")


# --- SAK-0016: fail-closed key handling and the safe_loads choke point ------


@pytest.mark.parametrize("bad_key", [b"", b"\x01", b"x" * (signing.MIN_KEY_BYTES - 1)])
def test_short_or_empty_explicit_key_is_refused(bad_key):
    with pytest.raises(signing.SignatureError):
        signing.sign(b"payload", key=bad_key)
    # An envelope forged with an empty key must not verify under that key either.
    import hashlib
    import hmac as _hmac

    header = signing.sign(b"payload", key=KEY)[: signing.HEADER_LEN]
    forged = header + _hmac.new(bad_key, header + b"payload", hashlib.sha256).digest() + b"payload"
    with pytest.raises(signing.SignatureError):
        signing.verify(forged, key=bad_key)


@pytest.mark.parametrize("env_value", ["", "abcd", "short-pass"])
def test_short_env_key_is_refused(monkeypatch, env_value):
    monkeypatch.setenv(signing.KEY_ENV_VAR, env_value)
    with pytest.raises(signing.SignatureError):
        signing.sign(b"payload")


def test_long_passphrase_env_key_is_accepted(monkeypatch):
    monkeypatch.setenv(signing.KEY_ENV_VAR, "not hex but long enough passphrase")
    assert signing.verify(signing.sign(b"p")) == b"p"


def test_verify_fails_closed_without_any_key(monkeypatch):
    monkeypatch.delenv(signing.KEY_ENV_VAR, raising=False)
    envelope = signing.sign(b"payload", key=KEY)
    with pytest.raises(signing.SignatureError):
        signing.verify(envelope)


def _no_loads(monkeypatch):
    def _spy(*args, **kwargs):
        raise AssertionError("cloudpickle.loads must not run on unverified bytes")

    monkeypatch.setattr(signing.cloudpickle, "loads", _spy)


def test_safe_loads_round_trips_signed_payload():
    envelope = signing.sign(b"TAG!" + cloudpickle.dumps({"a": 1}), key=KEY)
    assert signing.safe_loads(envelope, key=KEY, prefix=b"TAG!") == {"a": 1}


@pytest.mark.parametrize("case", ["tampered", "unsigned", "wrong_key", "wrong_prefix", "no_key"])
def test_safe_loads_rejects_before_unpickling(monkeypatch, case):
    payload = b"TAG!" + cloudpickle.dumps({"a": 1})
    envelope = signing.sign(payload, key=KEY)
    key = KEY
    prefix = b"TAG!"
    if case == "tampered":
        envelope = envelope[:-1] + bytes([envelope[-1] ^ 0xFF])
    elif case == "unsigned":
        envelope = payload
    elif case == "wrong_key":
        key = OTHER_KEY
    elif case == "wrong_prefix":
        prefix = b"OTHR"
    elif case == "no_key":
        monkeypatch.delenv(signing.KEY_ENV_VAR, raising=False)
        key = None
    _no_loads(monkeypatch)
    with pytest.raises(signing.SignatureError):
        signing.safe_loads(envelope, key=key, prefix=prefix)
