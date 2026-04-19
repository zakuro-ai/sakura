"""Sakura + HuggingFace ``Trainer`` — async evaluation over Zakuro.

The standard ``transformers.Trainer`` blocks training on every evaluation
phase: one epoch trains, then the same process runs the full eval loop
before the next epoch starts. For long eval passes (large validation sets,
metric computation with ``evaluate``) this can eat 20–50% of end-to-end
wall time.

``SakuraHFCallback`` turns evaluation into a non-blocking Zakuro dispatch:
at the end of each training epoch the current model state is cloudpickled
and submitted to a ``ThreadPoolExecutor``; the future is reaped at the
start of the next epoch so eval overlaps with the next training pass.

Usage::

    from transformers import Trainer, TrainingArguments
    from sakura.huggingface import SakuraHFCallback
    import zakuro as zk

    def model_factory():
        return AutoModelForSequenceClassification.from_pretrained("bert-base-uncased")

    def eval_fn(model, eval_dataset_bytes):
        # Rebuild the dataloader on the worker side, run model.eval(), return metrics.
        ...

    trainer = Trainer(
        model=model,
        args=TrainingArguments(..., eval_strategy="no"),  # disable built-in eval
        train_dataset=train_ds,
        callbacks=[
            SakuraHFCallback(
                val_compute=zk.Compute(uri="quic://worker:4433"),
                model_factory=model_factory,
                eval_fn=eval_fn,
                eval_dataset=val_ds,
            )
        ],
    )
    trainer.train()
"""

from __future__ import annotations

import time
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any, Callable, Optional

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
# Remote evaluation primitive
# ---------------------------------------------------------------------------


@zk.fn
def _remote_evaluate(
    state_bytes: bytes,
    model_factory_bytes: bytes,
    eval_fn_bytes: bytes,
    eval_payload_bytes: bytes,
) -> dict:
    """Run the user-supplied eval function on the worker.

    The caller cloudpickles:
      - ``state_bytes``         — serialized ``model.state_dict()``
      - ``model_factory_bytes`` — ``() -> nn.Module`` rebuilding the architecture
      - ``eval_fn_bytes``       — ``(model, eval_payload) -> dict`` returning metrics
      - ``eval_payload_bytes``  — whatever the eval_fn needs (dataset, tokenizer, …)

    The ``eval_fn`` signature is intentionally opaque so callers can pass their
    own evaluation pipeline (e.g. HuggingFace ``evaluate.load("accuracy")``,
    custom metrics, subword-aligned F1, etc.) without Sakura pinning an API.
    """
    import cloudpickle as _cp

    state_dict = _cp.loads(state_bytes)
    model_factory = _cp.loads(model_factory_bytes)
    eval_fn = _cp.loads(eval_fn_bytes)
    eval_payload = _cp.loads(eval_payload_bytes)

    model = model_factory()
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
        evaluation pass on the worker and returns metrics. Free to use
        HuggingFace datasets, torch dataloaders, or anything else installed on
        the worker.
    eval_payload:
        Opaque object passed to ``eval_fn`` as the second argument. Typically
        the validation dataset (serialised via cloudpickle). Keeping it opaque
        lets the caller pass whatever shape their ``eval_fn`` expects without
        Sakura making assumptions.
    val_compute:
        Zakuro compute target. ``None`` → Zakuro's standalone (in-process)
        fallback, useful for local benchmarks.
    verbose:
        Print each epoch's metrics when ready (default True).
    """

    def __init__(
        self,
        *,
        model_factory: Callable[[], Any],
        eval_fn: Callable[[Any, Any], dict],
        eval_payload: Any,
        val_compute: Optional[zk.Compute] = None,
        verbose: bool = True,
    ) -> None:
        self._compute = val_compute if val_compute is not None else zk.Compute()
        self._model_factory_bytes = cloudpickle.dumps(model_factory)
        self._eval_fn_bytes = cloudpickle.dumps(eval_fn)
        self._eval_payload_bytes = cloudpickle.dumps(eval_payload)
        self._verbose = verbose

        self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="sakura-hf-eval")
        self._pending: Optional[Future] = None
        self._history: list[dict] = []

    # .................................................................. API

    @property
    def history(self) -> list[dict]:
        return list(self._history)

    # ........................................................ HF hooks

    def on_epoch_end(self, args, state, control, **kwargs):  # noqa: ARG002
        # Drain previous future before dispatching the new one so logs stay
        # in order and pending state doesn't grow.
        self._drain(state)

        model = kwargs.get("model")
        if model is None:
            return
        state_bytes = cloudpickle.dumps(
            {k: v.detach().cpu() for k, v in model.state_dict().items()}
        )
        epoch = int(state.epoch) if state.epoch is not None else len(self._history)
        self._pending = self._pool.submit(self._dispatch, state_bytes, epoch)

    def on_train_end(self, args, state, control, **kwargs):  # noqa: ARG002
        self._drain(state)
        self._pool.shutdown(wait=True)

    # ............................................................. internals

    def _dispatch(self, state_bytes: bytes, epoch: int) -> dict:
        started = time.perf_counter()
        metrics = _remote_evaluate.to(self._compute)(
            state_bytes,
            self._model_factory_bytes,
            self._eval_fn_bytes,
            self._eval_payload_bytes,
        )
        if isinstance(metrics, dict):
            metrics = dict(metrics)
            metrics["epoch"] = epoch
            metrics["elapsed_secs"] = time.perf_counter() - started
        return metrics

    def _drain(self, state) -> None:
        if self._pending is None:
            return
        result = self._pending.result()
        self._pending = None
        if isinstance(result, dict):
            self._history.append(result)
            # Publish the metrics into HF's log history so downstream tooling
            # (TrainerState, WandB, MLflow, …) sees them.
            if hasattr(state, "log_history") and state.log_history is not None:
                state.log_history.append(dict(result))
            if self._verbose:
                fmt = " ".join(
                    f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
                    for k, v in result.items()
                )
                print(f"[Sakura-HF] {fmt}")


__all__ = ["SakuraHFCallback"]
