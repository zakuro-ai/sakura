"""Sakura + HuggingFace ``Trainer`` — async evaluation over Zakuro.

The standard ``transformers.Trainer`` blocks training on every evaluation
phase. ``SakuraHFCallback`` turns evaluation into a non-blocking Zakuro
dispatch: at the end of each training epoch the current model state is
cloudpickled and submitted to a ``ThreadPoolExecutor``; the future is
reaped when convenient (configurable) so evaluation overlaps with
subsequent training passes.

Key performance knobs:

- ``drain="lazy"`` (default): at ``on_epoch_end`` only reap futures that
  are *already done*; never block on pending ones. The final drain happens
  in ``on_train_end``. Training wall time stays ≈ N × T even when eval is
  slower than training.
- ``drain="strict"``: reap each future at the *next* ``on_epoch_end`` (the
  original pre-v0.2 behaviour). Useful if downstream code needs metrics
  in epoch order and is OK paying the wait.
- ``cache_key``: when set, the worker caches the model architecture in a
  module-level dict and reuses it across epochs. Eliminates the per-epoch
  ``model_from_config`` + ``load_state_dict_into_new_module`` cost.
- ``fp16_state_dict=True``: cast weights to ``fp16`` before transfer (cast
  back on the worker). Halves network bytes for fp32 models.
"""

from __future__ import annotations

import time
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any, Callable, Literal, Optional

import cloudpickle

try:
    from transformers import TrainerCallback
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "sakura.huggingface requires `transformers`. "
        "Install with `pip install transformers`."
    ) from exc

import zakuro as zk


# ---------------------------------------------------------------------------
# Remote evaluation primitive — with persistent-validator cache
# ---------------------------------------------------------------------------

# Worker-side cache keyed by ``cache_key`` (per ModelFactory shape). Lives in
# the worker process's module globals, so subsequent calls reuse the model
# without re-instantiating the architecture.
_WORKER_MODEL_CACHE: dict[str, Any] = {}


@zk.fn
def _remote_evaluate(
    state_bytes: bytes,
    model_factory_bytes: bytes,
    eval_fn_bytes: bytes,
    eval_payload_bytes: bytes,
    cache_key: Optional[str] = None,
    fp16: bool = False,
) -> dict:
    """Run the user-supplied eval function on the worker.

    When ``cache_key`` is provided the module-level ``_WORKER_MODEL_CACHE``
    stores the instantiated model keyed by that string. First call for a key
    runs the factory; subsequent calls reuse the cached architecture and only
    ``load_state_dict`` is repeated.

    When ``fp16`` is ``True`` the state_dict arrives in half precision; the
    worker upcasts each tensor to the model's native dtype on ``load``.

    ``state_bytes`` is a ``torch.save`` blob, not a cloudpickle dump — the
    serialisation is ~1.7× faster and releases the GIL much more on the
    producer side, which matters while training is happening concurrently.
    """
    import io as _io

    import cloudpickle as _cp
    import torch as _torch

    # torch.load handles the state_dict blob; cloudpickle still handles
    # the function/payload objects (closures aren't torch-picklable).
    state_dict = _torch.load(_io.BytesIO(state_bytes), map_location="cpu")
    eval_fn = _cp.loads(eval_fn_bytes)
    eval_payload = _cp.loads(eval_payload_bytes)

    if cache_key is not None and cache_key in _WORKER_MODEL_CACHE:
        model = _WORKER_MODEL_CACHE[cache_key]
    else:
        model_factory = _cp.loads(model_factory_bytes)
        model = model_factory()
        if cache_key is not None:
            _WORKER_MODEL_CACHE[cache_key] = model

    if fp16:
        # Upcast to the model's existing parameter dtype on load.
        target_dtype = next(model.parameters()).dtype
        state_dict = {
            k: (v.to(target_dtype) if hasattr(v, "to") else v)
            for k, v in state_dict.items()
        }

    model.load_state_dict(state_dict)
    model.eval()

    return eval_fn(model, eval_payload)


# ---------------------------------------------------------------------------
# Callback
# ---------------------------------------------------------------------------


class SakuraHFCallback(TrainerCallback):
    """HuggingFace ``TrainerCallback`` that offloads evaluation to Zakuro.

    Parameters
    ----------
    model_factory:
        Callable returning a fresh model instance on the worker. Must be
        cloudpickle-serialisable (module-level functions and classes are fine;
        closures also work via cloudpickle).
    eval_fn:
        Callable ``(model, eval_payload) -> dict[str, float]`` that runs the
        evaluation pass on the worker and returns metrics.
    eval_payload:
        Opaque object passed to ``eval_fn``. Shipped via cloudpickle.
    val_compute:
        Zakuro compute target. ``None`` → standalone (in-process) fallback.
    drain:
        ``"lazy"`` (default) — only reap already-done futures at each
        ``on_epoch_end``; never block. ``"strict"`` — block at each
        ``on_epoch_end`` until the previous future is ready.
    cache_key:
        Identifier for the persistent model cache on the worker. When set,
        the validator architecture is reused across epochs. Default
        ``"default"`` (per-callback instance). Set to ``None`` to disable
        caching.
    fp16_state_dict:
        Ship model weights in half precision. Halves network bytes for
        fp32 models; the worker upcasts on ``load``.
    max_pending:
        Cap on the number of in-flight evaluations. When the cap is reached
        ``on_epoch_end`` blocks on the oldest future (acts like strict for
        that one call) to keep memory bounded.
    on_backpressure:
        Behaviour when ``val_compute`` is an ``AdaptiveCompute`` and its
        ``is_backpressured()`` returns ``True`` at dispatch time. One of:

        - ``"skip"`` (default): do not dispatch this epoch's eval. Training
          wall time stays capped at ≈ ``N × T``. A row with
          ``{"skipped": True}`` is appended to ``history`` so downstream
          code can see the miss.
        - ``"queue"``: dispatch regardless (the original behaviour). Use
          when you want every epoch's metrics at the cost of wall time.
        - ``"block"``: drain the oldest pending future first, freeing a
          slot, then dispatch. Balances throughput against memory.
    async_copy:
        When ``True`` and the model sits on CUDA, the on-epoch-end
        snapshot is scheduled on a dedicated CUDA stream and handed off
        to the worker thread via an event. The main training thread no
        longer waits for the PCIe transfer. Falls back to a blocking
        ``.cpu()`` on CPU/MPS/unknown devices.
    verbose:
        Print each epoch's metrics when ready (default True).
    """

    def __init__(
        self,
        *,
        model_factory: Callable[[], Any],
        eval_fn: Callable[[Any, Any], dict],
        eval_payload: Any,
        val_compute: Optional[Any] = None,  # Compute or AdaptiveCompute
        drain: Literal["lazy", "strict"] = "lazy",
        cache_key: Optional[str] = "default",
        fp16_state_dict: bool = False,
        max_pending: int = 4,
        on_backpressure: Literal["skip", "queue", "block"] = "skip",
        async_copy: bool = True,
        verbose: bool = True,
    ) -> None:
        self._compute = val_compute if val_compute is not None else zk.Compute()
        self._model_factory_bytes = cloudpickle.dumps(model_factory)
        self._eval_fn_bytes = cloudpickle.dumps(eval_fn)
        self._eval_payload_bytes = cloudpickle.dumps(eval_payload)
        self._drain_mode = drain
        self._cache_key = cache_key
        self._fp16 = fp16_state_dict
        self._max_pending = max(1, int(max_pending))
        self._backpressure_policy = on_backpressure
        # When the model lives on CUDA, schedule the GPU→CPU copy on a
        # dedicated stream so the main training stream stays unblocked. The
        # worker thread synchronises on the recorded event before pickling.
        self._async_copy = bool(async_copy)
        self._verbose = verbose

        self._pool = ThreadPoolExecutor(
            max_workers=self._max_pending,
            thread_name_prefix="sakura-hf-eval",
        )
        self._pending: list[tuple[int, Future]] = []
        self._history: list[dict] = []

    # .................................................................. API

    @property
    def history(self) -> list[dict]:
        return list(self._history)

    # ........................................................ HF hooks

    def on_epoch_end(self, args, state, control, **kwargs):  # noqa: ARG002
        self._collect_done(state)

        # Respect max_pending: if we've already got that many in flight,
        # block on the oldest one to keep memory bounded.
        if len(self._pending) >= self._max_pending:
            self._drain_oldest(state)
        # Strict mode always blocks on oldest at every epoch.
        elif self._drain_mode == "strict" and self._pending:
            self._drain_oldest(state)

        model = kwargs.get("model")
        if model is None:
            return

        epoch = int(state.epoch) if state.epoch is not None else len(self._history)

        # Adaptive backpressure: if the allocator tells us every worker is
        # saturated, honour the configured policy before paying the
        # cloudpickle cost of packaging the state_dict.
        if (
            hasattr(self._compute, "is_backpressured")
            and self._compute.is_backpressured()
        ):
            if self._backpressure_policy == "skip":
                skipped = {"epoch": epoch, "skipped": True}
                self._record(skipped, state)
                return
            if self._backpressure_policy == "block" and self._pending:
                self._drain_oldest(state)
            # "queue" falls through to the normal dispatch below.

        # Snapshot the weights now (next epoch's training would otherwise
        # mutate them in place). For CUDA models we issue the PCIe copy on
        # a dedicated stream and hand the event to the worker thread, so
        # the main thread returns in O(ms) instead of waiting on the
        # ~150–200 ms transfer. For CPU/MPS/unknown devices we fall back
        # to the blocking copy.
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
                # non_blocking=True returns a fresh CPU tensor whose data
                # is still being DMA'd from the GPU. We do *not* call
                # .clone() — that would force a wait for the DMA to finish.
                # The worker thread synchronises on copy_event before
                # reading these tensors.
                state_dict_snapshot = {
                    k: v.detach().to("cpu", non_blocking=True)
                    for k, v in model.state_dict().items()
                }
            copy_event = _torch.cuda.Event()
            copy_event.record(copy_stream)
        else:
            state_dict_snapshot = {
                k: v.detach().cpu() for k, v in model.state_dict().items()
            }

        fut = self._pool.submit(
            self._dispatch_with_snapshot, state_dict_snapshot, epoch, copy_event
        )
        self._pending.append((epoch, fut))

    def on_train_end(self, args, state, control, **kwargs):  # noqa: ARG002
        # Drain everything still outstanding.
        while self._pending:
            self._drain_oldest(state)
        self._pool.shutdown(wait=True)

    # ............................................................. internals

    def _dispatch_with_snapshot(
        self, state_dict_snapshot: dict, epoch: int, copy_event: Any = None,
    ) -> dict:
        """Run on the worker thread: fp16 cast + torch.save + remote call.

        Keeps the main thread free to keep training while we package the
        (large) state dict.

        When ``copy_event`` is supplied, the snapshot is still mid-transfer
        on a separate CUDA stream. We synchronise on the event here (in the
        pool thread) so the main thread never paid the wait cost.

        Uses ``torch.save`` instead of cloudpickle for the state_dict —
        measured 1.7× faster and releases the GIL ~2× more often on the
        producer side (~480 ms → ~280 ms for 268 MB, concurrent-thread
        CPU share jumps from 39 % to 72 % of baseline).
        """
        if copy_event is not None:
            copy_event.synchronize()

        if self._fp16:
            import torch as _torch

            state_dict_snapshot = {
                k: (v.to(_torch.float16) if v.dtype == _torch.float32 else v)
                for k, v in state_dict_snapshot.items()
            }
        import io as _io

        import torch as _torch

        buf = _io.BytesIO()
        _torch.save(state_dict_snapshot, buf)
        state_bytes = buf.getvalue()
        return self._dispatch(state_bytes, epoch)

    def _dispatch(self, state_bytes: bytes, epoch: int) -> dict:
        started = time.perf_counter()
        metrics = _remote_evaluate.to(self._compute)(
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

    def _collect_done(self, state) -> None:
        """Reap any futures that happen to be ready; don't block."""
        still_pending: list[tuple[int, Future]] = []
        for epoch, fut in self._pending:
            if fut.done():
                self._record(fut.result(), state)
            else:
                still_pending.append((epoch, fut))
        self._pending = still_pending

    def _drain_oldest(self, state) -> None:
        if not self._pending:
            return
        _, fut = self._pending.pop(0)
        self._record(fut.result(), state)

    def _record(self, result: Any, state) -> None:
        if not isinstance(result, dict):
            return
        self._history.append(result)
        if hasattr(state, "log_history") and state.log_history is not None:
            state.log_history.append(dict(result))
        if self._verbose:
            fmt = " ".join(
                f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
                for k, v in result.items()
            )
            print(f"[Sakura-HF] {fmt}")


__all__ = ["SakuraHFCallback"]
