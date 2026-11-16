"""SAK-0010: QUIC flow-control windows exposed to Python.

`FlowControl()` keeps quinn's defaults; `FlowControl(rtt_ms, bandwidth_bytes_per_s)`
sizes the windows from the bandwidth-delay product. The e2e test runs a worker and
a Dispatcher that both use non-default windows and moves a multi-window payload.
"""
from __future__ import annotations

import sys

import pytest

sakura_wire = pytest.importorskip("sakura_wire")

from sakura.wire import Dispatcher, FlowControl, TlsConfig, WorkerSupervisor
from sakura.dispatch.spawn import resolve_startup_timeout_s

pytestmark = pytest.mark.wire

HANDLER_ECHO = 0xDEAD
DTYPE_U8 = 12
DEVICE_CPU = 0


def test_default_keeps_quinn_windows():
    assert FlowControl().windows() is None


def test_bdp_scaled_windows():
    w = FlowControl(rtt_ms=248, bandwidth_bytes_per_s=12_500_000).windows()
    # 248 ms x 12.5 MB/s = 3.1 MB BDP, x2 headroom; connection windows = 8 streams.
    assert w == {
        "stream_receive_window": 6_200_000,
        "receive_window": 49_600_000,
        "send_window": 49_600_000,
    }


def test_clamps():
    lan = FlowControl(rtt_ms=0.2, bandwidth_bytes_per_s=1_000_000).windows()
    assert lan["stream_receive_window"] == 1_250_000  # never below quinn's default
    capped = FlowControl(
        rtt_ms=248, bandwidth_bytes_per_s=12_500_000, max_window=4_000_000
    ).windows()
    assert capped["stream_receive_window"] == 4_000_000


@pytest.mark.parametrize(
    "kwargs",
    [
        {"rtt_ms": 100},
        {"bandwidth_bytes_per_s": 1},
        {"rtt_ms": 0, "bandwidth_bytes_per_s": 1},
        {"rtt_ms": float("nan"), "bandwidth_bytes_per_s": 1},
        {"rtt_ms": 10, "bandwidth_bytes_per_s": 0},
        {"rtt_ms": 10, "bandwidth_bytes_per_s": 1, "min_window": 10, "max_window": 5},
    ],
)
def test_invalid_inputs_raise(kwargs):
    with pytest.raises(ValueError):
        FlowControl(**kwargs)


def test_echo_with_bdp_windows_through_worker():
    sup = WorkerSupervisor(shutdown_timeout_s=5.0)
    try:
        uri, cert = sup.spawn(
            cmd=[
                sys.executable, "-m", "sakura.worker",
                "--listen", "quic://127.0.0.1:0",
                "--echo-only",
                "--rtt-ms", "248",
                "--bandwidth-bytes-per-s", "12500000",
            ],
            startup_timeout_s=resolve_startup_timeout_s(),
        )
        fc = FlowControl(rtt_ms=248, bandwidth_bytes_per_s=12_500_000)
        d = Dispatcher(uri, TlsConfig(cert, "localhost"), flow_control=fc)
        payload = bytes(range(256)) * (8 * 1024 * 1024 // 256)  # 8 MiB > 6.2 MB window
        fut = d.submit(
            HANDLER_ECHO,
            [{"shape": [len(payload)], "dtype_id": DTYPE_U8, "device_id": DEVICE_CPU,
              "data": payload}],
            b"fc",
            timeout_ms=30000,
        )
        result = fut.result(timeout=30.0)
        assert result.aux == b"fc"
        assert result.tensors()[0] == payload
    finally:
        sup.shutdown()


def test_worker_rejects_half_link_spec():
    from sakura.worker.__main__ import main

    with pytest.raises(SystemExit):
        main(["--rtt-ms", "248"])
