"""sakura.worker — daemon entry point used by the WorkerSupervisor.

Plan 2 uses a full handler registry (echo + heartbeat + exec_cloudpickled).
Plan 1's echo-only mode is still available via `sakura-worker --echo-only`.
"""
from sakura.worker.handlers import default_registry
from sakura.worker.registry import HandlerRegistry

__all__ = ["HandlerRegistry", "default_registry", "main"]


def main(argv: list[str] | None = None) -> int:
    """Re-exported `main` so existing entry-points keep resolving cleanly."""
    from sakura.worker.__main__ import main as _main
    return _main(argv)
