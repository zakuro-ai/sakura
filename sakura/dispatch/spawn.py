"""Shared worker-spawn helper: one place for the readiness deadline.

`python -m sakura.worker` imports the whole `sakura` package (torch,
lightning, ...) before it can bind its socket and print the
`SAKURA_WORKER_LISTENING` handshake line. On a loaded or cold host that
import alone can exceed a few seconds, so a tight deadline turns host load
into `did not receive listen URI within 10s`. The deadline is therefore
generous by default and overridable with `SAKURA_WORKER_STARTUP_TIMEOUT_S`.

A worker that dies early is not waited out: the supervisor reports
`worker exited before reporting URI` as soon as its stdout closes, and the
worker's stderr is inherited, so the traceback is visible.
"""
from __future__ import annotations

import os
from typing import Any, Mapping, Optional, Sequence

STARTUP_TIMEOUT_ENV_VAR = "SAKURA_WORKER_STARTUP_TIMEOUT_S"
DEFAULT_STARTUP_TIMEOUT_S = 60.0


def resolve_startup_timeout_s(explicit: Optional[float] = None) -> float:
    """Explicit value > `SAKURA_WORKER_STARTUP_TIMEOUT_S` > 60 s default."""
    if explicit is not None:
        return float(explicit)
    raw = os.environ.get(STARTUP_TIMEOUT_ENV_VAR)
    if raw:
        try:
            value = float(raw)
        except ValueError:
            raise ValueError(f"{STARTUP_TIMEOUT_ENV_VAR} must be a number, got {raw!r}") from None
        if value > 0:
            return value
    return DEFAULT_STARTUP_TIMEOUT_S


def spawn_worker(
    supervisor: Any,
    cmd: Sequence[str],
    env: Optional[Mapping[str, str]] = None,
    *,
    startup_timeout_s: Optional[float] = None,
) -> tuple[str, bytes]:
    """`supervisor.spawn(...)` with the shared startup deadline.

    Returns `(uri, cert_der)`. Raises `RuntimeError` on timeout or early exit.
    """
    timeout = resolve_startup_timeout_s(startup_timeout_s)
    try:
        uri, cert = supervisor.spawn(cmd=list(cmd), env=dict(env or {}), startup_timeout_s=timeout)
        return str(uri), bytes(cert)
    except RuntimeError as e:
        if "listen URI" in str(e):
            raise RuntimeError(
                f"{e} (worker cold start imports the full sakura package; raise "
                f"{STARTUP_TIMEOUT_ENV_VAR} on slow or loaded hosts)"
            ) from e
        raise
