"""Sakura + PyTorch DDP — rank-0 async eval dispatch.

In DistributedDataParallel training every rank executes the same forward +
backward, gradients are all-reduced, and then each rank steps the optimizer.
Only rank 0 needs to ship a checkpoint to the eval worker; ranks 1..N-1
can ignore the eval hook entirely.

``DDPAsyncEvalCallback`` implements this:

- Rank 0 snapshots the model's state_dict (using the same async CUDA-stream
  trick as ``SakuraHFCallback`` when running on CUDA), cloudpickles it, and
  dispatches ``_remote_evaluate`` through Zakuro.
- Ranks 1..N-1 simply ``torch.distributed.barrier`` at the sync points —
  they never touch the callback's thread pool.

Drop it in wherever you already call ``torch.distributed.barrier`` at
epoch boundaries (most hand-written DDP loops).

Example::

    import torch.distributed as dist
    import zakuro as zk
    from sakura.ddp import DDPAsyncEvalCallback

    cb = DDPAsyncEvalCallback(
        model_factory=lambda: MyModel(),
        eval_fn=my_eval_fn,
        eval_payload=(val_tensors, 32),
        val_compute=zk.Compute(uri="quic://eval-worker:4433"),
        world_size=dist.get_world_size(),
        rank=dist.get_rank(),
    )

    for epoch in range(num_epochs):
        train_one_epoch(model, train_loader)
        dist.barrier()
        cb.on_epoch_end(epoch, model)      # rank 0 dispatches; others no-op
    cb.on_train_end()                       # rank 0 drains; others no-op

The callback is framework-agnostic (no Lightning / HF / Keras coupling).
Train with raw PyTorch DDP, with `accelerate`, or with anything that
exposes rank + world_size and hands you a model at epoch boundaries.
"""

from __future__ import annotations

import time
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any, Callable, Literal, Optional

import cloudpickle

import zakuro as zk

# Re-use the remote evaluation primitive from the HF integration so the
# worker-side cache (keyed by ``cache_key``) can be shared across both
# callbacks in the same worker process.
from sakura.huggingface import (
    _align_dtype_in_place,
    _get_or_build_model,
    _WORKER_MODEL_CACHE,
)


@zk.fn
def _ddp_remote_evaluate(
    state_bytes: bytes,
    model_factory_bytes: bytes,
    eval_fn_bytes: bytes,
    eval_payload_bytes: bytes,
    cache_key: Optional[str] = None,
    fp16: bool = False,
) -> dict:
    """Remote eval entry point for DDP. Mirrors SakuraHFCallback's remote_evaluate
    so the worker-side cache in ``_WORKER_MODEL_CACHE`` is shared."""
    import io as _io

    import cloudpickle as _cp
    import torch as _torch

    state_dict = _torch.load(_io.BytesIO(state_bytes), map_location="cpu")
    eval_fn = _cp.loads(eval_fn_bytes)
    eval_payload = _cp.loads(eval_payload_bytes)

    model = _get_or_build_model(cache_key, model_factory_bytes)
    if fp16:
        state_dict = _align_dtype_in_place(model, state_dict)
    model.load_state_dict(state_dict)
    model.eval()
    return eval_fn(model, eval_payload)


class DDPAsyncEvalCallback:
    """Rank-0 async evaluation dispatch for PyTorch DDP training loops.

    Parameters
    ----------
    model_factory:
        Callable returning a fresh model on the eval worker. Cloudpickle-serialisable.
    eval_fn:
        ``(model, payload) -> dict[str, float]`` — runs on the eval worker.
    eval_payload:
        Opaque payload passed to ``eval_fn``. Shipped via cloudpickle.
    val_compute:
        Zakuro compute target. ``None`` → standalone fallback (rank-0 in-process).
    rank:
        The current process's rank (0..world_size-1).
    world_size:
        Total number of DDP ranks.
    cache_key:
        Worker-side cache key to reuse the model architecture across epochs.
    fp16_state_dict:
        Ship weights in half precision.
    max_pending:
        Cap on in-flight evaluations.
    on_backpressure:
        See ``SakuraHFCallback`` — applies only on rank 0.
    async_copy:
        CUDA stream overlap for the GPU→CPU snapshot (rank 0, CUDA only).
    verbose:
        Rank 0 prints metrics as they arrive.
    """

    def __init__(
        self,
        *,
        model_factory: Callable[[], Any],
        eval_fn: Callable[[Any, Any], dict],
        eval_payload: Any,
        rank: int,
        world_size: int,
        val_compute: Optional[Any] = None,
        cache_key: Optional[str] = "ddp-default",
        fp16_state_dict: bool = False,
        max_pending: int = 4,
        on_backpressure: Literal["skip", "queue", "block"] = "skip",
        async_copy: bool = True,
        verbose: bool = True,
    ) -> None:
        self._rank = int(rank)
        self._world_size = int(world_size)
        self._is_driver = self._rank == 0
        self._verbose = verbose and self._is_driver

        if not self._is_driver:
            # Non-driver ranks do nothing. Keep attributes as no-ops so
            # downstream code can treat every rank uniformly.
            self._compute = None
            self._pool = None
            self._pending = []
            self._history = []
            return

        self._compute = val_compute if val_compute is not None else zk.Compute()
        self._model_factory_bytes = cloudpickle.dumps(model_factory)
        self._eval_fn_bytes = cloudpickle.dumps(eval_fn)
        self._eval_payload_bytes = cloudpickle.dumps(eval_payload)
        self._cache_key = cache_key
        self._fp16 = fp16_state_dict
        self._max_pending = max(1, int(max_pending))
        self._backpressure_policy = on_backpressure
        self._async_copy = bool(async_copy)

        self._pool = ThreadPoolExecutor(
            max_workers=self._max_pending,
            thread_name_prefix="sakura-ddp-eval",
        )
        self._pending: list[tuple[int, Future]] = []
        self._history: list[dict] = []

    # ................................................................. API

    @property
    def history(self) -> list[dict]:
        return list(self._history) if self._is_driver else []

    @property
    def is_driver(self) -> bool:
        """True on rank 0 (the only rank that dispatches)."""
        return self._is_driver

    def on_epoch_end(self, epoch: int, model: Any) -> None:
        """Call at every epoch boundary on every rank.

        Non-driver ranks return immediately. Driver rank snapshots weights
        and submits the eval to the pool.
        """
        if not self._is_driver:
            return
        self._collect_done()

        # Respect max_pending.
        if len(self._pending) >= self._max_pending:
            self._drain_oldest()

        # Adaptive backpressure.
        if (
            hasattr(self._compute, "is_backpressured")
            and self._compute.is_backpressured()
        ):
            if self._backpressure_policy == "skip":
                self._history.append({"epoch": epoch, "skipped": True})
                if self._verbose:
                    print(f"[Sakura-DDP] epoch={epoch} SKIPPED (backpressured)")
                return
            if self._backpressure_policy == "block" and self._pending:
                self._drain_oldest()

        # Snapshot (async CUDA stream if available).
        copy_event = None
        try:
            import torch as _torch

            first_param = next(model.parameters(), None)
            use_async = (
                self._async_copy
                and first_param is not None
                and first_param.device.type == "cuda"
                and _torch.cuda.is_available()
            )
        except Exception:
            use_async = False

        if use_async:
            import torch as _torch

            copy_stream = _torch.cuda.Stream()
            with _torch.cuda.stream(copy_stream):
                state_dict_snapshot = {
                    k: v.detach().to("cpu", non_blocking=True)
                    for k, v in _dict_for_ddp(model).items()
                }
            copy_event = _torch.cuda.Event()
            copy_event.record(copy_stream)
        else:
            state_dict_snapshot = {
                k: v.detach().cpu() for k, v in _dict_for_ddp(model).items()
            }

        fut = self._pool.submit(
            self._dispatch, state_dict_snapshot, epoch, copy_event
        )
        self._pending.append((epoch, fut))

    def on_train_end(self) -> None:
        """Drain all pending futures (rank 0 only)."""
        if not self._is_driver:
            return
        while self._pending:
            self._drain_oldest()
        self._pool.shutdown(wait=True)

    # ........................................................... internals

    def _dispatch(
        self, state_dict: dict, epoch: int, copy_event: Any = None,
    ) -> dict:
        if copy_event is not None:
            copy_event.synchronize()

        if self._fp16:
            import torch as _torch

            state_dict = {
                k: (v.to(_torch.float16) if v.dtype == _torch.float32 else v)
                for k, v in state_dict.items()
            }

        # In-process fast path: skip torch.save when compute resolves to standalone.
        if getattr(self._compute, "uri", None) is None and getattr(self._compute, "host", None) is None:
            started = time.perf_counter()
            model = _get_or_build_model(self._cache_key, self._model_factory_bytes)
            if self._fp16:
                state_dict = _align_dtype_in_place(model, state_dict)
            model.load_state_dict(state_dict)
            model.eval()
            eval_fn = cloudpickle.loads(self._eval_fn_bytes)
            payload = cloudpickle.loads(self._eval_payload_bytes)
            metrics = eval_fn(model, payload)
            if isinstance(metrics, dict):
                metrics = dict(metrics)
                metrics["epoch"] = epoch
                metrics["elapsed_secs"] = time.perf_counter() - started
            return metrics

        # Remote path.
        import io as _io

        import torch as _torch

        buf = _io.BytesIO()
        _torch.save(state_dict, buf)
        state_bytes = buf.getvalue()

        started = time.perf_counter()
        metrics = _ddp_remote_evaluate.to(self._compute)(
            state_bytes,
            self._model_factory_bytes,
            self._eval_fn_bytes,
            self._eval_payload_bytes,
            self._cache_key,
            self._fp16,
        )
        if isinstance(metrics, dict):
            metrics = dict(metrics)
            metrics["epoch"] = epoch
            metrics["elapsed_secs"] = time.perf_counter() - started
        return metrics

    def _collect_done(self) -> None:
        still_pending: list[tuple[int, Future]] = []
        for epoch, fut in self._pending:
            if fut.done():
                self._record(fut.result())
            else:
                still_pending.append((epoch, fut))
        self._pending = still_pending

    def _drain_oldest(self) -> None:
        if not self._pending:
            return
        _, fut = self._pending.pop(0)
        self._record(fut.result())

    def _record(self, result: Any) -> None:
        if not isinstance(result, dict):
            return
        self._history.append(result)
        if self._verbose:
            fmt = " ".join(
                f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
                for k, v in result.items()
            )
            print(f"[Sakura-DDP] {fmt}")


def _dict_for_ddp(model: Any) -> dict:
    """Return a state_dict for either a plain module or a DDP-wrapped one.

    DDP wraps the real model in ``.module``; calling ``.state_dict()`` on
    the wrapper gives prefix keys like ``module.layer.weight``. Strip the
    prefix so the eval side (plain model) can ``load_state_dict`` cleanly.
    """
    inner = getattr(model, "module", model)
    return inner.state_dict()


__all__ = ["DDPAsyncEvalCallback"]
