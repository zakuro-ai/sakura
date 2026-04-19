"""Sakura + TensorFlow / Keras — async validation over Zakuro.

``SakuraKerasCallback`` is the Keras analogue of the Lightning callback.
At ``on_epoch_end`` the current model weights are shipped to a Zakuro
worker via ``@zk.fn``; the future is reaped at the start of the next epoch
so validation overlaps with the next training pass.

Usage::

    import tensorflow as tf
    from sakura.tensorflow import SakuraKerasCallback
    import zakuro as zk

    def model_factory():
        return tf.keras.Sequential([...])  # architecture only, no weights

    def val_fn(model, val_payload):
        # Rebuild / reuse the validation pipeline on the worker, return a dict.
        x, y = val_payload
        loss, acc = model.evaluate(x, y, verbose=0)
        return {"val_loss": float(loss), "val_acc": float(acc)}

    model = model_factory()
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    model.fit(
        x_train, y_train, epochs=10,
        callbacks=[
            SakuraKerasCallback(
                model_factory=model_factory,
                val_fn=val_fn,
                val_payload=(x_val, y_val),
                val_compute=zk.Compute(uri="quic://worker:4433"),
            )
        ],
    )

The weights are transferred via ``get_weights()`` / ``set_weights()`` on
numpy arrays so they pickle cleanly without pulling full TF graph state.
"""

from __future__ import annotations

import time
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any, Callable, Optional

import cloudpickle

try:
    from tensorflow.keras.callbacks import Callback
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "sakura.tensorflow requires `tensorflow`. "
        "Install with `pip install tensorflow`."
    ) from exc

import zakuro as zk


# ---------------------------------------------------------------------------
# Remote validation primitive
# ---------------------------------------------------------------------------


@zk.fn
def _remote_keras_validate(
    weights_bytes: bytes,
    model_factory_bytes: bytes,
    val_fn_bytes: bytes,
    val_payload_bytes: bytes,
) -> dict:
    """Rebuild the model + weights on the worker, run ``val_fn``, return metrics."""
    import cloudpickle as _cp

    weights = _cp.loads(weights_bytes)
    model_factory = _cp.loads(model_factory_bytes)
    val_fn = _cp.loads(val_fn_bytes)
    val_payload = _cp.loads(val_payload_bytes)

    model = model_factory()
    model.set_weights(weights)
    return val_fn(model, val_payload)


# ---------------------------------------------------------------------------
# Callback
# ---------------------------------------------------------------------------


class SakuraKerasCallback(Callback):
    """Keras callback that offloads validation to a Zakuro worker.

    Parameters
    ----------
    model_factory:
        Callable returning a fresh ``tf.keras.Model`` with matching
        architecture but no trained weights. Cloudpickle-serialisable.
    val_fn:
        Callable ``(model, val_payload) -> dict[str, float]`` that runs on the
        worker and returns metrics. ``model`` has the current epoch's weights
        loaded when the callback invokes it.
    val_payload:
        Opaque payload for ``val_fn`` (typically ``(x_val, y_val)`` or a
        ``tf.data.Dataset``). Shipped via cloudpickle.
    val_compute:
        Zakuro compute target; ``None`` → standalone fallback.
    verbose:
        Print each epoch's metrics when ready (default True).
    """

    def __init__(
        self,
        *,
        model_factory: Callable[[], Any],
        val_fn: Callable[[Any, Any], dict],
        val_payload: Any,
        val_compute: Optional[zk.Compute] = None,
        verbose: bool = True,
    ) -> None:
        super().__init__()
        self._compute = val_compute if val_compute is not None else zk.Compute()
        self._model_factory_bytes = cloudpickle.dumps(model_factory)
        self._val_fn_bytes = cloudpickle.dumps(val_fn)
        self._val_payload_bytes = cloudpickle.dumps(val_payload)
        self._verbose = verbose

        self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="sakura-keras-val")
        self._pending: Optional[Future] = None
        self._history: list[dict] = []

    # ...................................................................

    @property
    def history(self) -> list[dict]:
        return list(self._history)

    def on_epoch_end(self, epoch, logs=None):  # noqa: ARG002
        # Reap previous future before dispatching the next one.
        self._drain()

        if self.model is None:
            return
        weights = self.model.get_weights()
        weights_bytes = cloudpickle.dumps(weights)
        self._pending = self._pool.submit(self._dispatch, weights_bytes, epoch)

    def on_train_end(self, logs=None):  # noqa: ARG002
        self._drain()
        self._pool.shutdown(wait=True)

    # ........................................................ internals

    def _dispatch(self, weights_bytes: bytes, epoch: int) -> dict:
        started = time.perf_counter()
        metrics = _remote_keras_validate.to(self._compute)(
            weights_bytes,
            self._model_factory_bytes,
            self._val_fn_bytes,
            self._val_payload_bytes,
        )
        if isinstance(metrics, dict):
            metrics = dict(metrics)
            metrics["epoch"] = epoch
            metrics["elapsed_secs"] = time.perf_counter() - started
        return metrics

    def _drain(self) -> None:
        if self._pending is None:
            return
        result = self._pending.result()
        self._pending = None
        if isinstance(result, dict):
            self._history.append(result)
            if self._verbose:
                fmt = " ".join(
                    f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
                    for k, v in result.items()
                )
                print(f"[Sakura-Keras] {fmt}")


__all__ = ["SakuraKerasCallback"]
