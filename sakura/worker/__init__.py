"""sakura.worker — daemon entry point used by the WorkerSupervisor.

In Plan 1 the worker only registers an echo handler (HANDLER_ECHO = 0xDEAD)
which bounces all input tensors back unchanged. Plan 2 adds the real
HANDLER_EXEC_CLOUDPICKLED handler that runs user-supplied callables.
"""
__all__ = ["main"]

from sakura.worker.__main__ import main
