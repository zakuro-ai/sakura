"""#141: RemoteDispatcher / LocalDispatcher forward `flow_control=` (SAK-0010).

The spy tests replace the native `sakura_wire.Dispatcher` / `WorkerSupervisor`
to check that the exact `FlowControl` object reaches the native constructor and
that LocalDispatcher hands the same windows to the worker it spawns. The e2e
tests push a payload larger than the BDP stream window through real endpoints
configured on both ends.
"""
from __future__ import annotations

import sys

import pytest

sakura_wire = pytest.importorskip("sakura_wire")

from sakura.dispatch.local import LocalDispatcher, _worker_flow_args
from sakura.dispatch.remote import RemoteDispatcher
from sakura.dispatch.spawn import spawn_worker
from sakura.wire import FlowControl

RTT_MS = 248
BW = 12_500_000  # bytes/s; 2 x BDP = 6.2 MB stream window


class _SpyDispatcher:
    calls: list[tuple[tuple, dict]] = []

    def __init__(self, *args, **kwargs):
        _SpyDispatcher.calls.append((args, kwargs))


class _SpySupervisor:
    cmds: list[list[str]] = []

    def __init__(self, **_kwargs):
        pass

    def spawn(self, *, cmd, env, startup_timeout_s):
        _SpySupervisor.cmds.append(list(cmd))
        return "quic://127.0.0.1:1", b"cert"

    def shutdown(self):
        pass


@pytest.fixture
def spies(monkeypatch):
    _SpyDispatcher.calls = []
    _SpySupervisor.cmds = []
    monkeypatch.setattr(sakura_wire, "Dispatcher", _SpyDispatcher)
    monkeypatch.setattr(sakura_wire, "WorkerSupervisor", _SpySupervisor)
    return _SpyDispatcher, _SpySupervisor


def test_flow_control_getters():
    fc = FlowControl(rtt_ms=RTT_MS, bandwidth_bytes_per_s=BW, max_window=50_000_000)
    assert fc.rtt_ms == pytest.approx(RTT_MS)
    assert fc.bandwidth_bytes_per_s == BW
    assert fc.max_window == 50_000_000
    assert fc.min_window > 0
    default = FlowControl()
    assert default.rtt_ms is None and default.bandwidth_bytes_per_s is None


def test_remote_dispatcher_passes_flow_control_to_native(spies):
    disp, _ = spies
    fc = FlowControl(rtt_ms=RTT_MS, bandwidth_bytes_per_s=BW)
    RemoteDispatcher(uri="quic://127.0.0.1:1", cert_der=b"cert", flow_control=fc)
    (_args, kwargs), = disp.calls
    assert kwargs["flow_control"] is fc


def test_remote_dispatcher_default_is_none(spies):
    disp, _ = spies
    RemoteDispatcher(uri="quic://127.0.0.1:1", cert_der=b"cert")
    (_args, kwargs), = disp.calls
    assert kwargs["flow_control"] is None


def test_local_dispatcher_configures_client_and_spawned_worker(spies):
    disp, sup = spies
    fc = FlowControl(rtt_ms=RTT_MS, bandwidth_bytes_per_s=BW, min_window=2_000_000)
    d = LocalDispatcher(flow_control=fc)
    d.shutdown()
    (_args, kwargs), = disp.calls
    assert kwargs["flow_control"] is fc
    (cmd,) = sup.cmds
    assert cmd[cmd.index("--rtt-ms") + 1] == repr(float(RTT_MS))
    assert cmd[cmd.index("--bandwidth-bytes-per-s") + 1] == str(BW)
    assert cmd[cmd.index("--min-window") + 1] == "2000000"
    assert cmd[cmd.index("--max-window") + 1] == str(fc.max_window)


def test_local_dispatcher_default_spawn_command_unchanged(spies):
    disp, sup = spies
    LocalDispatcher().shutdown()
    assert sup.cmds == [
        [sys.executable, "-m", "sakura.worker", "--listen", "quic://127.0.0.1:0"]
    ]
    (_args, kwargs), = disp.calls
    assert kwargs["flow_control"] is None


def test_quinn_default_flow_control_adds_no_worker_flags():
    assert _worker_flow_args(None) == []
    assert _worker_flow_args(FlowControl()) == []


def test_worker_flags_reproduce_the_same_windows(monkeypatch):
    """The flags LocalDispatcher forwards rebuild identical windows in the worker."""
    import sakura.worker.__main__ as worker_main

    fc = FlowControl(rtt_ms=37.5, bandwidth_bytes_per_s=BW, min_window=1_000_000,
                     max_window=3_000_000)
    seen = {}

    def fake_run_server(*, addr, callback, print_handshake, flow_control):
        seen["fc"] = flow_control

    monkeypatch.setattr(sakura_wire, "run_server", fake_run_server)
    argv = ["--listen", "quic://127.0.0.1:0"] + _worker_flow_args(fc)
    assert worker_main.main(argv) == 0
    assert seen["fc"].windows() == fc.windows()


def test_worker_rejects_clamps_without_link():
    from sakura.worker.__main__ import main

    with pytest.raises(SystemExit):
        main(["--max-window", "1000"])


def _nbytes(x):
    return int(x.nbytes)


def test_local_dispatcher_e2e_with_bdp_windows():
    import numpy as np

    fc = FlowControl(rtt_ms=RTT_MS, bandwidth_bytes_per_s=BW)
    d = LocalDispatcher(flow_control=fc)
    try:
        arr = np.zeros(8 * 1024 * 1024, dtype=np.uint8)  # 8 MiB > 6.2 MB window
        assert d.submit(_nbytes, arr).result(timeout=30.0).value == arr.nbytes
    finally:
        d.shutdown()


def test_remote_dispatcher_e2e_with_bdp_windows_on_both_ends():
    import numpy as np

    from sakura.dispatch.signing import KEY_ENV_VAR

    hmac_key = b"\x02" * 32
    fc = FlowControl(rtt_ms=RTT_MS, bandwidth_bytes_per_s=BW)
    sup = sakura_wire.WorkerSupervisor(shutdown_timeout_s=5.0)
    try:
        uri, cert = spawn_worker(
            sup,
            cmd=[sys.executable, "-m", "sakura.worker", "--listen", "quic://127.0.0.1:0"]
            + _worker_flow_args(fc),
            env={KEY_ENV_VAR: hmac_key.hex()},
        )
        d = RemoteDispatcher(uri=uri, cert_der=cert, hmac_key=hmac_key, flow_control=fc)
        arr = np.zeros(8 * 1024 * 1024, dtype=np.uint8)
        assert d.submit(_nbytes, arr).result(timeout=30.0).value == arr.nbytes
    finally:
        sup.shutdown()
