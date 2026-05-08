"""Dispatcher / Future / Result abstractions.

Concrete dispatchers (Local, Remote, InThread) implement Dispatcher.submit;
returned Futures resolve to Results. The high-level Python surface accepts
a Python callable + tensor args, cloudpickles the callable, and dispatches
HANDLER_EXEC_CLOUDPICKLED.
"""
from __future__ import annotations

import abc
from dataclasses import dataclass
from typing import Any, Callable, Optional


@dataclass
class Result:
    """The decoded result of an RPC.

    `value` is the return value of the dispatched callable.
    `elapsed_us` is wall time (worker-side for remote, local for in_thread).
    """
    value: Any
    elapsed_us: int = 0


class Future(abc.ABC):
    """Promise of a Result. Backed by a sakura_wire.Future or in-process oneshot."""

    @abc.abstractmethod
    def result(self, timeout: Optional[float] = None) -> Result:
        """Block until the result is ready or timeout (in seconds)."""

    @abc.abstractmethod
    def done(self) -> bool:
        """True iff the future has resolved (or was cancelled)."""

    @abc.abstractmethod
    def cancel(self) -> bool:
        """Best-effort cancellation. Returns True if it was newly cancelled."""


class Dispatcher(abc.ABC):
    """Submit cloudpickled callables to a worker (local, remote, or in-thread)."""

    @abc.abstractmethod
    def submit(
        self,
        callable: Callable[..., Any],
        *args: Any,
        timeout_ms: Optional[int] = None,
        **kwargs: Any,
    ) -> Future:
        """Dispatch `callable(*args, **kwargs)` to the worker. Returns a Future."""

    def shutdown(self, *, timeout_s: float = 30.0) -> None:
        """Default: no-op. Subclasses with subprocesses or sockets override."""

    def stats(self) -> dict:
        """Default: empty stats."""
        return {}


__all__ = ["Dispatcher", "Future", "Result"]
