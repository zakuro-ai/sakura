"""ThreadDispatcher — run callables on a Python thread for in-process parallelism.

Trades subprocess+QUIC overhead (LocalDispatcher) for the GIL: works because
torch's C++ kernels release the GIL during compute, so a thread-side eval
runs in true parallel with the main training thread for tensor ops. Pure
Python work between kernels still serializes through the GIL.

When to use:
- AsyncEval / AsyncCheckpoint where the overlap target is a tensor-heavy
  function (forward, backward, save). Thread overlap captures the win
  without a 50ms+ subprocess round-trip per epoch.
- In-process testing where InThreadDispatcher is too synchronous to
  exercise concurrency code paths.

When NOT to use:
- Pure-Python overlap targets (string munging, etc.) — GIL serializes
  these; LocalDispatcher's subprocess is the only true parallelism path.
- Hard isolation (eval crashes shouldn't take down training) —
  LocalDispatcher's subprocess is what isolates.

`max_workers` controls in-flight concurrency; submissions beyond that
queue. Backpressure semantics match the other dispatchers: oldest
in-flight blocks at submit-time when the pool is saturated.
"""
from __future__ import annotations

import threading
import time
from concurrent.futures import Future as _StdFuture, ThreadPoolExecutor
from typing import Any, Callable, Optional

from sakura.dispatch.base import Dispatcher, Future, Result


class _ThreadFuture(Future):
    """Sakura Future wrapping a stdlib concurrent.futures.Future."""

    def __init__(self, std_future: _StdFuture, t_submit: int):
        self._fut = std_future
        self._t_submit = t_submit

    def result(self, timeout: Optional[float] = None) -> Result:
        value = self._fut.result(timeout=timeout)
        elapsed_us = (time.perf_counter_ns() - self._t_submit) // 1000
        return Result(value=value, elapsed_us=elapsed_us)

    def done(self) -> bool:
        return self._fut.done()

    def cancel(self) -> bool:
        return self._fut.cancel()


class ThreadDispatcher(Dispatcher):
    """Dispatch to a Python thread pool.

    Real parallelism for tensor-heavy callables (torch releases the GIL in
    C++ kernels). The default pool has `max_workers=1` because the typical
    use case (AsyncEval) has at most one in-flight eval per epoch — bigger
    pools just hold extra state without speeding anything up.
    """

    def __init__(self, *, max_workers: int = 1):
        self._pool = ThreadPoolExecutor(max_workers=max(1, int(max_workers)),
                                         thread_name_prefix="sakura-thread")
        self._closed = False
        self._lock = threading.Lock()

    def submit(
        self,
        callable: Callable[..., Any],
        *args: Any,
        timeout_ms: Optional[int] = None,
        **kwargs: Any,
    ) -> Future:
        with self._lock:
            if self._closed:
                raise RuntimeError("ThreadDispatcher is shutdown; cannot submit")
            t0 = time.perf_counter_ns()
            std_fut = self._pool.submit(callable, *args, **kwargs)
        return _ThreadFuture(std_fut, t0)

    def shutdown(self, *, timeout_s: float = 30.0) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
        # `wait=True` blocks for in-flight tasks; honor timeout via cancel_futures
        # on Python 3.9+ where the pool may release pending tasks early.
        self._pool.shutdown(wait=True, cancel_futures=True)

    def stats(self) -> dict:
        return {"kind": "thread"}


__all__ = ["ThreadDispatcher"]
