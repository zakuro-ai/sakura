"""`sakura-worker` daemon — QUIC server that dispatches via HandlerRegistry.

Plan 1 used `run_echo_server` (only HANDLER_ECHO). Plan 2 swaps to
`run_server` with the full default registry (echo + heartbeat +
exec_cloudpickled), so cloudpickled callables can be dispatched
end-to-end.

Run via:
    sakura-worker --listen quic://127.0.0.1:0

On a long-RTT path, size the QUIC flow-control windows from the link
(SAK-0010): ``--rtt-ms 248 --bandwidth-bytes-per-s 12500000``. Without
both flags the server keeps quinn's default windows. ``--min-window`` /
``--max-window`` (bytes) override the window clamps.
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
    parser.add_argument(
        "--rtt-ms",
        type=float,
        default=None,
        help="Link round-trip time in ms; with --bandwidth-bytes-per-s, sizes the "
        "QUIC flow-control windows from the bandwidth-delay product (SAK-0010).",
    )
    parser.add_argument(
        "--bandwidth-bytes-per-s",
        type=int,
        default=None,
        help="Link bottleneck bandwidth in bytes/s (not bits); see --rtt-ms.",
    )
    parser.add_argument(
        "--min-window",
        type=int,
        default=None,
        help="Lower clamp on the per-stream receive window in bytes (needs --rtt-ms).",
    )
    parser.add_argument(
        "--max-window",
        type=int,
        default=None,
        help="Upper clamp on the per-stream receive window in bytes (needs --rtt-ms).",
    )
    args = parser.parse_args(argv)
    if (args.rtt_ms is None) != (args.bandwidth_bytes_per_s is None):
        parser.error("--rtt-ms and --bandwidth-bytes-per-s must be given together")
    if args.rtt_ms is None and (args.min_window is not None or args.max_window is not None):
        parser.error("--min-window / --max-window need --rtt-ms and --bandwidth-bytes-per-s")

    if not args.listen.startswith("quic://"):
        raise ValueError(f"--listen must be a quic:// URI, got: {args.listen}")
    addr = args.listen[len("quic://"):]
    print_handshake = not args.no_handshake

    import sakura_wire as _native

    flow_control = None
    if args.rtt_ms is not None:
        flow_control = _native.FlowControl(
            rtt_ms=args.rtt_ms,
            bandwidth_bytes_per_s=args.bandwidth_bytes_per_s,
            min_window=args.min_window,
            max_window=args.max_window,
        )

    if args.echo_only:
        _native.run_echo_server(
            addr=addr, print_handshake=print_handshake, flow_control=flow_control
        )
    else:
        from sakura.worker.handlers import default_registry
        registry = default_registry()
        _native.run_server(
            addr=addr,
            callback=registry.dispatch,
            print_handshake=print_handshake,
            flow_control=flow_control,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
