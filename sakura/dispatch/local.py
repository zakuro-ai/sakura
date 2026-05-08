"""LocalDispatcher — auto-spawn a localhost sakura-worker, then dispatch via RemoteDispatcher.

Default dispatcher when SakuraRuntime is constructed with no `compute=`.
"""
from __future__ import annotations

import sys
from typing import Any, Callable, Optional

import sakura_wire as _native

from sakura.dispatch.base import Dispatcher, Future
from sakura.dispatch.remote import RemoteDispatcher


class LocalDispatcher(Dispatcher):
    """Auto-spawned localhost worker, accessed via QUIC over loopback."""

    def __init__(
        self,
        *,
        n_workers: int = 1,
        gpus: Optional[list[int]] = None,
        startup_timeout_s: float = 10.0,
        shutdown_timeout_s: float = 5.0,
    ):
        if n_workers != 1:
            raise NotImplementedError(
                "n_workers > 1 not supported in Plan 2; Plan 4+ adds pool support."
            )
        self._supervisor = _native.WorkerSupervisor(shutdown_timeout_s=shutdown_timeout_s)
        env = {}
        if gpus is not None:
            env["CUDA_VISIBLE_DEVICES"] = ",".join(str(g) for g in gpus)
        uri, cert = self._supervisor.spawn(
            cmd=[sys.executable, "-m", "sakura.worker", "--listen", "quic://127.0.0.1:0"],
            env=env if env else None,
            startup_timeout_s=startup_timeout_s,
        )
        self._uri = uri
        self._cert = cert
        self._inner = RemoteDispatcher(uri=uri, cert_der=cert, server_name="localhost")
        self._shutdown_called = False

    def submit(
        self,
        callable: Callable[..., Any],
        *args: Any,
        timeout_ms: Optional[int] = None,
        **kwargs: Any,
    ) -> Future:
        return self._inner.submit(callable, *args, timeout_ms=timeout_ms, **kwargs)

    def shutdown(self, *, timeout_s: float = 30.0) -> None:
        if self._shutdown_called:
            return
        self._shutdown_called = True
        try:
            self._supervisor.shutdown()
        except Exception:
            pass

    def stats(self) -> dict:
        return {"kind": "local", "uri": self._uri}


__all__ = ["LocalDispatcher"]
