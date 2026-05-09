"""RemoteDispatcher: connects to an existing sakura-worker via quic://."""
from __future__ import annotations

import sys

import pytest

sakura_wire = pytest.importorskip("sakura_wire")

from sakura.dispatch.remote import RemoteDispatcher


def _square(x):
    return int(x.sum()) ** 2


def test_remote_dispatcher_round_trips_a_callable():
    """Spin up a worker via the supervisor, point a RemoteDispatcher at it, run a callable."""
    import numpy as np

    sup = sakura_wire.WorkerSupervisor(shutdown_timeout_s=5.0)
    try:
        uri, cert = sup.spawn(
            cmd=[sys.executable, "-m", "sakura.worker", "--listen", "quic://127.0.0.1:0"],
            startup_timeout_s=10.0,
        )
        d = RemoteDispatcher(uri=uri, cert_der=cert, server_name="localhost")
        try:
            fut = d.submit(_square, np.array([1, 2, 3, 4], dtype=np.int64))
            result = fut.result(timeout=5.0)
            assert result.value == (1 + 2 + 3 + 4) ** 2
        finally:
            d.shutdown()
    finally:
        sup.shutdown()


def test_remote_dispatcher_rejects_non_quic_uri():
    with pytest.raises(ValueError, match="quic://"):
        RemoteDispatcher(uri="http://localhost:8080", cert_der=b"", server_name="localhost")
