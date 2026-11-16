"""RemoteDispatcher: connects to an existing sakura-worker via quic://."""
from __future__ import annotations

import sys

import pytest

sakura_wire = pytest.importorskip("sakura_wire")

from sakura.dispatch.remote import RemoteDispatcher
from sakura.dispatch.spawn import spawn_worker


def _square(x):
    return int(x.sum()) ** 2


def test_remote_dispatcher_round_trips_a_callable():
    """Spin up a worker via the supervisor, point a RemoteDispatcher at it, run a callable."""
    import numpy as np

    from sakura.dispatch.signing import KEY_ENV_VAR

    hmac_key = b"\x01" * 32
    sup = sakura_wire.WorkerSupervisor(shutdown_timeout_s=5.0)
    try:
        uri, cert = spawn_worker(
            sup,
            cmd=[sys.executable, "-m", "sakura.worker", "--listen", "quic://127.0.0.1:0"],
            env={KEY_ENV_VAR: hmac_key.hex()},
        )
        d = RemoteDispatcher(uri=uri, cert_der=cert, server_name="localhost", hmac_key=hmac_key)
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


def _raise_value_error(x):
    raise ValueError(f"bad input {int(x.sum())}")


class _Opaque:
    def __init__(self, v):
        self.v = v


def _make_opaque():
    return _Opaque(3)


def test_remote_dispatcher_typed_results_errors_and_pickle_opt_out():
    """SAK-0016 end to end over the real QUIC worker: typed results decode,
    a raised exception comes back as RemoteExecutionError (instead of the
    connection dropping), and allow_pickle_results=False refuses the signed
    cloudpickle fallback while the default accepts it."""
    import numpy as np

    from sakura.dispatch.result_codec import RemoteExecutionError, ResultCodecError
    from sakura.dispatch.signing import KEY_ENV_VAR

    hmac_key = b"\x02" * 32
    sup = sakura_wire.WorkerSupervisor(shutdown_timeout_s=5.0)
    try:
        uri, cert = spawn_worker(
            sup,
            cmd=[sys.executable, "-m", "sakura.worker", "--listen", "quic://127.0.0.1:0"],
            env={KEY_ENV_VAR: hmac_key.hex()},
        )
        strict = RemoteDispatcher(uri=uri, cert_der=cert, hmac_key=hmac_key,
                                  allow_pickle_results=False)
        arr = np.array([1.0, 2.0], dtype=np.float32)
        value = strict.submit(lambda a: {"sum": float(a.sum()), "x2": a * 2}, arr).result(timeout=5.0).value
        assert value["sum"] == 3.0
        np.testing.assert_array_equal(value["x2"], arr * 2)

        with pytest.raises(RemoteExecutionError, match="ValueError: bad input 3"):
            strict.submit(_raise_value_error, arr).result(timeout=5.0)

        with pytest.raises(ResultCodecError, match="allow_pickle_results=False"):
            strict.submit(_make_opaque).result(timeout=5.0)

        lenient = RemoteDispatcher(uri=uri, cert_der=cert, hmac_key=hmac_key)
        assert lenient.submit(_make_opaque).result(timeout=5.0).value.v == 3
    finally:
        sup.shutdown()
