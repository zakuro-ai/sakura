"""Generic async trainer, now backed by Zakuro dispatch instead of MPI."""

from __future__ import annotations

import logging
import cloudpickle
import time
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any, Callable, Optional

import zakuro as zk


@zk.fn
def _remote_test(
    state_bytes: bytes,
    model_factory_bytes: bytes,
    test_fn_bytes: bytes,
) -> dict:
    """Run the test phase on a Zakuro worker.

    The caller cloudpickles (via ``@zk.fn``):
      - ``state_bytes``   — the trainer's current serialized state dict
      - ``model_factory`` — rebuilds the model in the worker process
      - ``test_fn``       — ``def test(model) -> dict`` returning metrics

    Returning a plain dict keeps the protocol framework-agnostic.
    """
    import cloudpickle as _pickle

    state_dict = _pickle.loads(state_bytes)
    model_factory = _pickle.loads(model_factory_bytes)
    test_fn = _pickle.loads(test_fn_bytes)

    model = model_factory()
    model.load_state_dict(state_dict)
    return test_fn(model)


class AsyncTrainer:
    """Overlaps test phases with the next training epoch via Zakuro dispatch.

    Historically this class forked into two MPI ranks; rank 0 trained while
    rank 1 validated on exchanged state dicts. The same semantics now use a
    single process: the test phase is shipped to a Zakuro worker and its
    future is reaped at the start of the next epoch, so the two phases run
    concurrently without MPI.

    Parameters
    ----------
    trainer:
        Any object exposing ``train(train_loader)``, ``serialized_state_dict()``,
        ``_epochs`` (iterable), ``_epoch``, and ``_metrics`` with a writable
        ``test`` attribute. The original ``Trainer`` class satisfies this.
    model_factory:
        Callable returning a fresh untrained model instance. Must be
        cloudpickle-serialisable.
    test_fn:
        Callable ``(model) -> dict`` that the worker runs to produce metrics.
    val_compute:
        Zakuro compute target for the test phase. ``None`` → Zakuro's
        standalone (in-process) fallback.
    """

    def __init__(
        self,
        trainer: Any,
        model_factory: Callable[[], Any],
        test_fn: Callable[[Any], dict],
        val_compute: Optional[zk.Compute] = None,
    ) -> None:
        self._trainer = trainer
        self._compute = val_compute if val_compute is not None else zk.Compute()
        self._model_factory_bytes = cloudpickle.dumps(model_factory)
        self._test_fn_bytes = cloudpickle.dumps(test_fn)
        self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="sakura-test")
        self._pending: Optional[Future] = None

    def run(self, train_loader: Any = None, test_loader: Any = None) -> None:
        """Drive the training loop, overlapping test with next epoch's train."""
        _ = test_loader  # unused locally; the worker rebuilds the loader itself
        try:
            for self._trainer._epoch in self._trainer._epochs:
                # Reap previous epoch's test result before the next train step.
                if self._pending is not None:
                    metrics = self._pending.result()
                    self._pending = None
                    try:
                        self._trainer._metrics.test = metrics
                    except Exception as exc:  # pragma: no cover — user metrics type
                        logging.warning("failed to attach test metrics: %r", exc)

                self._trainer.train(train_loader=train_loader)

                # Dispatch new test asynchronously.
                state_bytes = self._trainer.serialized_state_dict()
                # Some Trainer implementations return already-pickled bytes;
                # others return a dict. Accept either and normalize to bytes.
                if not isinstance(state_bytes, (bytes, bytearray)):
                    state_bytes = cloudpickle.dumps(state_bytes)
                self._pending = self._pool.submit(
                    self._dispatch,
                    bytes(state_bytes),
                )
        finally:
            if self._pending is not None:
                try:
                    metrics = self._pending.result(timeout=300)
                    try:
                        self._trainer._metrics.test = metrics
                    except Exception:
                        pass
                finally:
                    self._pending = None
            self._pool.shutdown(wait=False)

    def _dispatch(self, state_bytes: bytes) -> dict:
        started = time.perf_counter()
        metrics = _remote_test.to(self._compute)(
            state_bytes,
            self._model_factory_bytes,
            self._test_fn_bytes,
        )
        if isinstance(metrics, dict):
            metrics.setdefault("elapsed_secs", time.perf_counter() - started)
        return metrics


__all__ = ["AsyncTrainer"]
