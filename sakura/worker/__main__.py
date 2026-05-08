"""`sakura-worker` daemon — QUIC server that dispatches via HandlerRegistry.

Plan 1 used `run_echo_server` (only HANDLER_ECHO). Plan 2 swaps to
`run_server` with the full default registry (echo + heartbeat +
exec_cloudpickled), so cloudpickled callables can be dispatched
end-to-end.

Run via:
    sakura-worker --listen quic://127.0.0.1:0
"""
from __future__ import annotations

import argparse


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="sakura-worker")
    parser.add_argument(
        "--listen",
        default="quic://127.0.0.1:0",
        help="Bind address (use port :0 for an ephemeral port; default).",
    )
    parser.add_argument(
        "--no-handshake",
        action="store_true",
        default=False,
        help="Suppress the SAKURA_WORKER_LISTENING handshake line on stdout.",
    )
    parser.add_argument(
        "--echo-only",
        action="store_true",
        default=False,
        help="Run only the echo handler (Plan 1 mode). Default: full registry.",
    )
    args = parser.parse_args(argv)

    if not args.listen.startswith("quic://"):
        raise ValueError(f"--listen must be a quic:// URI, got: {args.listen}")
    addr = args.listen[len("quic://"):]
    print_handshake = not args.no_handshake

    import sakura_wire as _native

    if args.echo_only:
        _native.run_echo_server(addr=addr, print_handshake=print_handshake)
    else:
        from sakura.worker.handlers import default_registry
        registry = default_registry()
        _native.run_server(addr=addr, callback=registry.dispatch, print_handshake=print_handshake)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
