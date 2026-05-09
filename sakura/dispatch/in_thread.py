"""InThreadDispatcher — synchronous in-process execution for tests/debug."""
from __future__ import annotations

import time
from typing import Any, Callable, Optional

from sakura.dispatch.base import Dispatcher, Future, Result


class _ResolvedFuture(Future):
    """A Future that's already resolved (or already errored)."""

    def __init__(self, value: Any = None, exc: Optional[BaseException] = None,
                 elapsed_us: int = 0):
        self._value = value
        self._exc = exc
        self._elapsed_us = elapsed_us

    def result(self, timeout: Optional[float] = None) -> Result:
        if self._exc is not None:
            raise self._exc
        return Result(value=self._value, elapsed_us=self._elapsed_us)

    def done(self) -> bool:
        return True

    def cancel(self) -> bool:
        return False


class InThreadDispatcher(Dispatcher):
    """Run the callable synchronously in the caller's thread."""

    def submit(
        self,
        callable: Callable[..., Any],
        *args: Any,
        timeout_ms: Optional[int] = None,
        **kwargs: Any,
    ) -> Future:
        t0 = time.perf_counter_ns()
        try:
            value = callable(*args, **kwargs)
        except BaseException as exc:  # noqa: BLE001
            return _ResolvedFuture(exc=exc, elapsed_us=(time.perf_counter_ns() - t0) // 1000)
        return _ResolvedFuture(value=value, elapsed_us=(time.perf_counter_ns() - t0) // 1000)

    def shutdown(self, *, timeout_s: float = 30.0) -> None:
        pass

    def stats(self) -> dict:
        return {"kind": "in_thread"}


__all__ = ["InThreadDispatcher"]
