"""Thin Python wrapper around the maturin-built `sakura_wire` cdylib.

Plan 1 exposes:
    - sakura.wire.Dispatcher, Future, Result, TlsConfig (re-exports)
    - sakura.wire.FlowControl (re-export): QUIC flow-control windows (SAK-0010)
    - sakura.wire.WorkerSupervisor (re-export)
    - sakura.wire.serve_echo() — used by sakura-worker daemon

Plan 2 layers SakuraRuntime, Service, and the dispatcher abstractions
on top of these primitives.
"""
from __future__ import annotations

from typing import Any

import sakura_wire as _native

Dispatcher = _native.Dispatcher
Future = _native.Future
Result = _native.Result
TlsConfig = _native.TlsConfig
FlowControl = _native.FlowControl
WorkerSupervisor = _native.WorkerSupervisor

__version__ = _native.__version__


def serve_echo(
    *, listen: str, print_handshake: bool = True, flow_control: Any = None
) -> None:
    """Run a blocking QUIC server with the echo handler registered.

    Prints a single line to stdout when ready (so the supervisor can pick up the
    URI + cert), then serves forever. Plan 1 only — Plan 2 replaces this with
    a real handler-registry server.

    `flow_control` is an optional `FlowControl` (SAK-0010); None keeps quinn's
    default windows.
    """
    if not listen.startswith("quic://"):
        raise ValueError(f"--listen must be a quic:// URI, got: {listen}")
    addr = listen[len("quic://"):]
    _native.run_echo_server(
        addr=addr, print_handshake=print_handshake, flow_control=flow_control
    )
