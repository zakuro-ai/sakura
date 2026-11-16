"""Shared worker-spawn deadline helper (no native wire needed)."""
from __future__ import annotations

import pytest

from sakura.dispatch.spawn import (
    DEFAULT_STARTUP_TIMEOUT_S,
    STARTUP_TIMEOUT_ENV_VAR,
    resolve_startup_timeout_s,
    spawn_worker,
)


def test_default_is_generous(monkeypatch):
    monkeypatch.delenv(STARTUP_TIMEOUT_ENV_VAR, raising=False)
    assert resolve_startup_timeout_s() == DEFAULT_STARTUP_TIMEOUT_S >= 30.0


def test_env_override_and_explicit_precedence(monkeypatch):
    monkeypatch.setenv(STARTUP_TIMEOUT_ENV_VAR, "123")
    assert resolve_startup_timeout_s() == 123.0
    assert resolve_startup_timeout_s(7) == 7.0


def test_bad_env_value_is_rejected(monkeypatch):
    monkeypatch.setenv(STARTUP_TIMEOUT_ENV_VAR, "soon")
    with pytest.raises(ValueError, match=STARTUP_TIMEOUT_ENV_VAR):
        resolve_startup_timeout_s()


class _Sup:
    def __init__(self, exc=None):
        self.exc, self.kwargs = exc, None

    def spawn(self, **kwargs):
        self.kwargs = kwargs
        if self.exc:
            raise self.exc
        return "quic://127.0.0.1:1", b"cert"


def test_spawn_passes_resolved_timeout(monkeypatch):
    monkeypatch.delenv(STARTUP_TIMEOUT_ENV_VAR, raising=False)
    sup = _Sup()
    assert spawn_worker(sup, ["w"], {"A": "1"}) == ("quic://127.0.0.1:1", b"cert")
    assert sup.kwargs == {"cmd": ["w"], "env": {"A": "1"}, "startup_timeout_s": DEFAULT_STARTUP_TIMEOUT_S}


def test_timeout_error_gets_actionable_hint():
    sup = _Sup(RuntimeError("did not receive listen URI within 1s"))
    with pytest.raises(RuntimeError, match=STARTUP_TIMEOUT_ENV_VAR):
        spawn_worker(sup, ["w"], startup_timeout_s=1)


def test_other_errors_pass_through():
    sup = _Sup(RuntimeError("worker exited before reporting URI"))
    with pytest.raises(RuntimeError) as ei:
        spawn_worker(sup, ["w"])
    assert STARTUP_TIMEOUT_ENV_VAR not in str(ei.value)
