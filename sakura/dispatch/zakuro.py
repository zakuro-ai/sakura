"""ZakuroDispatcher — wraps zakuro.Compute for users with existing Zakuro infra.

Loses the sakura-wire codec wins (Zakuro's wire format is its own); kept for
backward compatibility and for users who want to leverage Zakuro's worker
allocation features.
"""
from __future__ import annotations

import time
from typing import Any, Callable, Optional

from sakura.dispatch.base import Dispatcher, Future, Result


class _ZkFuture(Future):
    def __init__(self, value, exc, elapsed_us):
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


class ZakuroDispatcher(Dispatcher):
    """Wraps zakuro.Compute and dispatches via @zk.fn."""

    def __init__(self, zk_compute: Any):
        self._zk_compute = zk_compute

    def submit(
        self,
        callable: Callable[..., Any],
        *args: Any,
        timeout_ms: Optional[int] = None,
        **kwargs: Any,
    ) -> Future:
        import zakuro as zk

        # Wrap user callable with @zk.fn so Zakuro can ship it.
        @zk.fn
        def _wrapped(*a, **kw):
            return callable(*a, **kw)

        t0 = time.perf_counter_ns()
        try:
            value = _wrapped.to(self._zk_compute)(*args, **kwargs)
        except Exception as exc:
            return _ZkFuture(value=None, exc=exc,
                              elapsed_us=(time.perf_counter_ns() - t0) // 1000)
        return _ZkFuture(value=value, exc=None,
                          elapsed_us=(time.perf_counter_ns() - t0) // 1000)

    def stats(self) -> dict:
        return {"kind": "zakuro"}


__all__ = ["ZakuroDispatcher"]
