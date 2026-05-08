"""End-to-end Plan 1 acceptance test.

Spawns a sakura-worker subprocess via WorkerSupervisor, opens a Dispatcher
against its URI with the cert it printed, submits an RPC at HANDLER_ECHO,
and verifies the response tensors are byte-identical.
"""
from __future__ import annotations

import sys

import pytest

sakura_wire = pytest.importorskip("sakura_wire")

from sakura.wire import Dispatcher, TlsConfig, WorkerSupervisor

HANDLER_ECHO = 0xDEAD
DTYPE_F32 = 0
DEVICE_CPU = 0


def _f32_bytes(values: list[float]) -> bytes:
    import struct
    return b"".join(struct.pack("<f", v) for v in values)


def test_echo_round_trip_through_worker():
    sup = WorkerSupervisor(shutdown_timeout_s=5.0)
    try:
        uri, cert = sup.spawn(
            cmd=[sys.executable, "-m", "sakura.worker", "--listen", "quic://127.0.0.1:0"],
            startup_timeout_s=10.0,
        )
        assert uri.startswith("quic://127.0.0.1:")
        assert isinstance(cert, bytes) and len(cert) > 100  # rough self-signed cert size

        tls = TlsConfig(cert, "localhost")
        d = Dispatcher(uri, tls)

        payload_a = _f32_bytes([1.0, 2.0, 3.0, 4.0])
        payload_b = _f32_bytes([10.0, 20.0])
        aux = b"hello-aux"

        fut = d.submit(
            HANDLER_ECHO,
            [
                {"shape": [4], "dtype_id": DTYPE_F32, "device_id": DEVICE_CPU, "data": payload_a},
                {"shape": [2], "dtype_id": DTYPE_F32, "device_id": DEVICE_CPU, "data": payload_b},
            ],
            aux,
            timeout_ms=5000,
        )
        result = fut.result(timeout=5.0)
        assert result.aux == aux
        tensors = result.tensors()
        assert len(tensors) == 2
        assert tensors[0] == payload_a
        assert tensors[1] == payload_b
    finally:
        sup.shutdown()


def test_dispatcher_rejects_non_quic_uri():
    tls = TlsConfig(b"\x00" * 256, "localhost")
    with pytest.raises(ValueError):
        Dispatcher("http://localhost:8080", tls)
