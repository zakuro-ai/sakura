"""LocalDispatcher — auto-spawn a localhost sakura-worker, then dispatch via RemoteDispatcher.

Default dispatcher when SakuraRuntime is constructed with no `compute=`.
"""
from __future__ import annotations

import secrets
import sys
from typing import TYPE_CHECKING, Any, Callable, Optional

if TYPE_CHECKING:
    import sakura_wire as _native

from sakura.dispatch import signing
from sakura.dispatch.base import Dispatcher, Future
from sakura.dispatch.remote import RemoteDispatcher
from sakura.dispatch.spawn import spawn_worker


def _worker_flow_args(flow_control: Optional["_native.FlowControl"]) -> list[str]:
    """`sakura-worker` CLI flags reproducing `flow_control` on the worker end.

    Empty for None or `FlowControl()` (quinn defaults), so the default spawn
    command is unchanged.
    """
    if flow_control is None or flow_control.rtt_ms is None:
        return []
    return [
        "--rtt-ms", repr(float(flow_control.rtt_ms)),
        "--bandwidth-bytes-per-s", str(int(flow_control.bandwidth_bytes_per_s)),
        "--min-window", str(int(flow_control.min_window)),
        "--max-window", str(int(flow_control.max_window)),
    ]


class LocalDispatcher(Dispatcher):
    """Auto-spawned localhost worker, accessed via QUIC over loopback."""

    def __init__(
        self,
        *,
        n_workers: int = 1,
        gpus: Optional[list[int]] = None,
        startup_timeout_s: Optional[float] = None,
        shutdown_timeout_s: float = 5.0,
        allow_pickle_results: bool = True,
        flow_control: Optional["_native.FlowControl"] = None,
    ):
        """`flow_control` (optional `sakura.wire.FlowControl`, SAK-0010) sizes
        the QUIC windows on BOTH ends: the client connection and the spawned
        worker (forwarded as `--rtt-ms --bandwidth-bytes-per-s --min-window
        --max-window`), since windows are per endpoint. None keeps quinn's
        defaults on both.

        `startup_timeout_s=None` uses the shared spawn deadline (60 s, or
        `SAKURA_WORKER_STARTUP_TIMEOUT_S`); see `sakura.dispatch.spawn`."""
        if n_workers != 1:
            raise NotImplementedError(
                "n_workers > 1 not supported in Plan 2; Plan 4+ adds pool support."
            )
        # Lazy import: the native wire is only needed once a LocalDispatcher
        # is actually constructed, so `import sakura` works on installs
        # without the Rust extension (e.g. the no-Rust CI job, pure-Python
        # federation deployments).
        import sakura_wire as _native
        self._supervisor = _native.WorkerSupervisor(shutdown_timeout_s=shutdown_timeout_s)
        # Random per-instance HMAC key, shared with the spawned worker only
        # via its subprocess environment, so the aux envelope on this
        # loopback connection is signed/verified without any operator
        # configuration (see sakura.dispatch.signing).
        hmac_key = secrets.token_bytes(32)
        env = {signing.KEY_ENV_VAR: hmac_key.hex()}
        if gpus is not None:
            env["CUDA_VISIBLE_DEVICES"] = ",".join(str(g) for g in gpus)
        uri, cert = spawn_worker(
            self._supervisor,
            cmd=[sys.executable, "-m", "sakura.worker", "--listen", "quic://127.0.0.1:0"]
            + _worker_flow_args(flow_control),
            env=env,
            startup_timeout_s=startup_timeout_s,
        )
        self._uri = uri
        self._cert = cert
        self._inner = RemoteDispatcher(
            uri=uri,
            cert_der=cert,
            server_name="localhost",
            hmac_key=hmac_key,
            allow_pickle_results=allow_pickle_results,
            flow_control=flow_control,
        )
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

    def stats(self) -> dict[str, Any]:
        return {"kind": "local", "uri": self._uri}


__all__ = ["LocalDispatcher"]
