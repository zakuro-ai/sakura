"""ProcessDispatcher — run callables in a persistent spawned worker process.

Why not ``ThreadDispatcher``? Threads share the training loop's GIL. A training loop
that is launch-bound (many small kernels, eager Python) is slowed down by *any* Python
work running beside it: measured, a thread writing a 146 MB checkpoint made the
concurrent training epochs 2.4x slower, cancelling the overlap win. A separate process
has its own GIL, so the work really runs in parallel.

Why not ``LocalDispatcher``? That one cloudpickles every argument and ships it over QUIC;
for multi-hundred-MB state dicts the serialisation cost lands on the caller. Here tensors
travel through ``torch.multiprocessing`` shared memory: the caller pays one memcpy (done
by the queue's feeder thread, not the training thread) and the worker maps the same pages.

Usage::

    disp = ProcessDispatcher(initializer=build_eval_state, initargs=(cfg,), nice=5)
    fut = disp.submit(write_checkpoint, cpu_state_dict, "ckpt/epoch_3.pt")
    fut.result().value

Callables are cloudpickled (lambdas / closures are fine). Heavy, reusable objects (a model
replica, a data loader) should be built once by ``initializer`` and read inside callables via
:func:`worker_state`. Entry points that launch a ``ProcessDispatcher`` must be importable
under the ``spawn`` start method (guard scripts with ``if __name__ == "__main__":``).
"""
from __future__ import annotations

import itertools
import os
import threading
import time
import traceback
from typing import Any, Callable, Optional

import cloudpickle  # type: ignore[import-untyped]
import torch.multiprocessing as tmp

from sakura.dispatch.base import Dispatcher, Future, Result

_STATE: dict[str, Any] = {}

# --- tensor packing -------------------------------------------------------------------
# Sending N tensors through torch.multiprocessing's default (file_descriptor) strategy
# costs N open file descriptors per message; a state dict has hundreds of tensors and a
# few epochs exhaust the default ulimit ("Too many open files"). Instead every message
# carries its tensors in ONE flat shared uint8 buffer plus a small spec of views.
_ALIGN = 64


class _TensorRef:
    __slots__ = ("shape", "dtype", "offset", "nbytes")

    def __init__(self, shape: tuple[int, ...], dtype: Any, offset: int, nbytes: int) -> None:
        self.shape, self.dtype, self.offset, self.nbytes = shape, dtype, offset, nbytes


def _pack(obj: Any) -> tuple[Any, Any]:
    """Replace every tensor in ``obj`` by a ref into one shared flat buffer.

    Returns ``(skeleton, flat)``; ``flat`` is None when ``obj`` holds no tensors."""
    import torch

    tensors: list[Any] = []
    refs: list[_TensorRef] = []
    cursor = 0

    def walk(x: Any) -> Any:
        nonlocal cursor
        if isinstance(x, torch.Tensor):
            t = x.detach().cpu().contiguous()
            nbytes = t.numel() * t.element_size()
            ref = _TensorRef(tuple(t.shape), t.dtype, cursor, nbytes)
            cursor += (nbytes + _ALIGN - 1) // _ALIGN * _ALIGN
            tensors.append(t)
            refs.append(ref)
            return ref
        if isinstance(x, dict):
            return {k: walk(v) for k, v in x.items()}
        if isinstance(x, (list, tuple)) and not hasattr(x, "_fields"):
            return type(x)(walk(v) for v in x)
        return x

    skeleton = walk(obj)
    if not tensors:
        return skeleton, None
    flat = torch.empty(max(cursor, 1), dtype=torch.uint8)
    flat.share_memory_()  # type: ignore[no-untyped-call]
    for t, ref in zip(tensors, refs):
        if ref.nbytes:
            flat[ref.offset : ref.offset + ref.nbytes].copy_(t.reshape(-1).view(torch.uint8))
    return skeleton, flat


def _unpack(skeleton: Any, flat: Any) -> Any:
    """Inverse of :func:`_pack`. Tensors are independent copies (one memcpy, done in the
    receiving process): views into the flat uint8 buffer would alias each other's storage,
    which e.g. ``torch.save`` rejects, and would let a callable scribble on shared memory."""
    import torch

    def walk(x: Any) -> Any:
        if isinstance(x, _TensorRef):
            if x.nbytes == 0:
                return torch.empty(x.shape, dtype=x.dtype)
            return flat[x.offset : x.offset + x.nbytes].view(x.dtype).reshape(x.shape).clone()
        if isinstance(x, dict):
            return {k: walk(v) for k, v in x.items()}
        if isinstance(x, (list, tuple)) and not hasattr(x, "_fields"):
            return type(x)(walk(v) for v in x)
        return x

    return walk(skeleton)


def worker_state() -> dict[str, Any]:
    """Inside a ProcessDispatcher worker: the dict holding the ``initializer`` result
    under the ``"init"`` key (empty in the parent process)."""
    return _STATE


def _pack_exception(exc: BaseException) -> tuple[Optional[bytes], str]:
    tb = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
    try:
        return cloudpickle.dumps(exc), tb
    except Exception:  # noqa: BLE001 — unpicklable exception: fall back to text
        return None, tb


def _worker_main(req_q: Any, res_q: Any, nice: int, torch_threads: Optional[int]) -> None:
    if nice:
        try:
            os.nice(nice)
        except OSError:
            pass
    if torch_threads is not None:
        import torch

        torch.set_num_threads(torch_threads)
    try:
        # The initializer arrives as the first queue message (not as a spawn argument):
        # ``Process.start()`` blocks the parent until the child has re-imported ``__main__``
        # whenever the spawn payload exceeds the pipe buffer, which would stall the
        # training loop for seconds; the queue is fed by a background thread.
        init_blob = req_q.get()
        if init_blob is not None:
            fn, args, kwargs = cloudpickle.loads(init_blob)
            _STATE["init"] = fn(*args, **kwargs)
        res_q.put((-1, True, None))
    except BaseException as exc:  # noqa: BLE001
        res_q.put((-1, False, _pack_exception(exc)))
        return
    while True:
        msg = req_q.get()
        if msg is None:
            return
        rid, fn_blob, skeleton, flat = msg
        t0 = time.perf_counter_ns()
        try:
            args, kwargs = _unpack(skeleton, flat)
            value = cloudpickle.loads(fn_blob)(*args, **kwargs)
            del args, kwargs, flat, msg
            res_q.put((rid, True, (_pack(value), (time.perf_counter_ns() - t0) // 1000)))
        except BaseException as exc:  # noqa: BLE001
            res_q.put((rid, False, _pack_exception(exc)))


class _ProcessFuture(Future):
    def __init__(self) -> None:
        self._event = threading.Event()
        self._ok = True
        self._payload: Any = None
        self._cancelled = False

    def _set(self, ok: bool, payload: Any) -> None:
        if not self._event.is_set():
            self._ok, self._payload = ok, payload
            self._event.set()

    def result(self, timeout: Optional[float] = None) -> Result:
        if not self._event.wait(timeout):
            raise TimeoutError("ProcessDispatcher future not ready")
        if self._cancelled:
            raise RuntimeError("future was cancelled")
        if self._ok:
            (skeleton, flat), elapsed_us = self._payload
            return Result(value=_unpack(skeleton, flat), elapsed_us=elapsed_us)
        blob, tb = self._payload
        if blob is not None:
            try:
                exc = cloudpickle.loads(blob)
            except Exception:  # noqa: BLE001
                exc = None
            if isinstance(exc, BaseException):
                raise exc
        raise RuntimeError(f"ProcessDispatcher task failed in worker:\n{tb}")

    def done(self) -> bool:
        return self._event.is_set()

    def cancel(self) -> bool:
        if self._event.is_set():
            return False
        self._cancelled = True
        self._event.set()
        return True


class ProcessDispatcher(Dispatcher):
    """Persistent worker process fed through shared-memory queues.

    Args:
        initializer, initargs: run once in the worker before the first task; its return
            value is available through :func:`worker_state` (``["init"]``).
        nice: ``os.nice`` increment applied in the worker so background work yields to
            the training process (POSIX).
        torch_threads: ``torch.set_num_threads`` in the worker (default: torch's choice).
        startup_timeout_s: how long to wait for the worker (spawn + imports + initializer)
            before the first ``result()`` raises. Submissions made earlier are queued.
    """

    def __init__(
        self,
        *,
        initializer: Optional[Callable[..., Any]] = None,
        initargs: tuple[Any, ...] = (),
        initkwargs: Optional[dict[str, Any]] = None,
        nice: int = 0,
        torch_threads: Optional[int] = None,
        startup_timeout_s: float = 120.0,
    ) -> None:
        ctx = tmp.get_context("spawn")
        self._req = ctx.Queue()
        self._res = ctx.Queue()
        self._proc = ctx.Process(
            target=_worker_main,
            args=(self._req, self._res, nice, torch_threads),
            daemon=True,
            name="sakura-process-worker",
        )
        self._proc.start()
        self._req.put(
            cloudpickle.dumps((initializer, initargs, initkwargs or {}))
            if initializer is not None
            else None
        )
        self._ids = itertools.count()
        self._pending: dict[int, _ProcessFuture] = {}
        self._lock = threading.Lock()
        self._closed = False
        self._ready = threading.Event()
        self._init_error: Optional[tuple[Optional[bytes], str]] = None
        self._startup_timeout_s = startup_timeout_s
        self._n_done = 0
        self._collector = threading.Thread(target=self._collect, name="sakura-process-collector", daemon=True)
        self._collector.start()

    # ------------------------------------------------------------------ public

    def submit(
        self,
        callable: Callable[..., Any],
        *args: Any,
        timeout_ms: Optional[int] = None,
        **kwargs: Any,
    ) -> Future:
        fut = _ProcessFuture()
        with self._lock:
            if self._closed:
                raise RuntimeError("ProcessDispatcher is shutdown; cannot submit")
            rid = next(self._ids)
            self._pending[rid] = fut
        skeleton, flat = _pack((args, kwargs))
        self._req.put((rid, cloudpickle.dumps(callable), skeleton, flat))
        return fut

    def wait_ready(self, timeout: Optional[float] = None) -> None:
        """Block until the worker finished spawning and running the initializer."""
        if not self._ready.wait(timeout if timeout is not None else self._startup_timeout_s):
            raise TimeoutError("ProcessDispatcher worker did not start in time")
        if self._init_error is not None:
            blob, tb = self._init_error
            raise RuntimeError(f"ProcessDispatcher initializer failed:\n{tb}")

    def shutdown(self, *, timeout_s: float = 30.0) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
        try:
            self._req.put(None)
            self._proc.join(timeout_s)
        finally:
            if self._proc.is_alive():
                self._proc.terminate()
                self._proc.join(5.0)
            self._fail_pending("dispatcher shut down")
            self._collector.join(2.0)

    def stats(self) -> dict[str, Any]:
        return {"kind": "process", "pid": self._proc.pid, "alive": self._proc.is_alive(),
                "completed": self._n_done}

    # ---------------------------------------------------------------- internal

    def _collect(self) -> None:
        import queue

        while True:
            try:
                rid, ok, payload = self._res.get(timeout=0.25)
            except queue.Empty:
                if not self._proc.is_alive():
                    if not self._ready.is_set():
                        self._init_error = (None, "worker process died before it was ready")
                    self._fail_pending("worker process died")
                    self._ready.set()
                    return
                if self._closed and not self._pending:
                    return
                continue
            except (EOFError, OSError):
                self._fail_pending("result channel closed")
                return
            if rid == -1:
                if not ok:
                    self._init_error = payload
                    self._fail_pending("worker initializer failed")
                self._ready.set()
                continue
            with self._lock:
                fut = self._pending.pop(rid, None)
            self._n_done += 1
            if fut is not None:
                fut._set(ok, payload)

    def _fail_pending(self, why: str) -> None:
        with self._lock:
            pending, self._pending = self._pending, {}
        for fut in pending.values():
            fut._set(False, (None, why))


__all__ = ["ProcessDispatcher", "worker_state"]
