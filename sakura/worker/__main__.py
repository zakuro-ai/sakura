"""`sakura-worker` — minimal QUIC server for Plan 1.

This worker speaks the sakura-wire protocol on top of QUIC, registers a
single echo handler at HANDLER_ECHO (0xDEAD), and prints a single line to
stdout once it is listening:

    SAKURA_WORKER_LISTENING <uri> <cert_der_hex>

The supervisor parses that line to learn the dynamic port and the
self-signed cert it must trust. After printing, the worker serves until
killed.
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
        "--print-cert-hex",
        action="store_true",
        default=True,
        help="Print the self-signed cert (hex) on the listening line. Required for the supervisor handshake.",
    )
    args = parser.parse_args(argv)

    from sakura.wire import serve_echo  # imports sakura_wire native module
    serve_echo(listen=args.listen, print_handshake=args.print_cert_hex)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
