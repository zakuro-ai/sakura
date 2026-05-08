"""Thin Python wrapper around the maturin-built `sakura_wire` cdylib.

Plan 1 exposes:
    - sakura.wire.Dispatcher, Future, Result, TlsConfig (re-exports)
    - sakura.wire.WorkerSupervisor (re-export)
    - sakura.wire.serve_echo() — used by sakura-worker daemon

Plan 2 layers SakuraRuntime, Service, and the dispatcher abstractions
on top of these primitives.
"""
from __future__ import annotations

import sakura_wire as _native

Dispatcher = _native.Dispatcher
Future = _native.Future
Result = _native.Result
TlsConfig = _native.TlsConfig
WorkerSupervisor = _native.WorkerSupervisor

__version__ = _native.__version__


def serve_echo(*, listen: str, print_handshake: bool = True) -> None:
    """Run a blocking QUIC server with the echo handler registered.

    Prints a single line to stdout when ready (so the supervisor can pick up the
    URI + cert), then serves forever. Plan 1 only — Plan 2 replaces this with
    a real handler-registry server.
    """
    if not listen.startswith("quic://"):
        raise ValueError(f"--listen must be a quic:// URI, got: {listen}")
    addr = listen[len("quic://"):]
    _native.run_echo_server(addr=addr, print_handshake=print_handshake)
