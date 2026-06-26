"""AsyncEval — dispatch eval_fn at epoch end, gather result, record to history.

Plan 3 implementation is framework-agnostic: eval_fn signature is
`fn(epoch: int, payload: Any) -> dict`. Plan 4 framework adapters wrap this
with model_factory/state_dict for real model evaluation.
"""
from __future__ import annotations

from typing import Any, Callable, Literal

from sakura.dispatch.base import Dispatcher, Future
from sakura.events import OnEpochEnd, OnTrainEnd
from sakura.service import BaseService


class BackpressureSaturatedError(Exception):
    """Raised by a dispatcher when its in-flight queue is full."""


class AsyncEval(BaseService):
    name = "async_eval"
    priority = 80

    def __init__(
        self,
        *,
        eval_fn: Callable[[int, Any], dict],
        eval_payload: Any,
        dispatcher: Dispatcher,
        max_pending: int = 4,
        on_backpressure: Literal["skip", "queue", "block"] = "skip",
        every: int = 1,
    ):
        super().__init__()
        self._eval_fn = eval_fn
        self._eval_payload = eval_payload
        self._dispatcher = dispatcher
        self._max_pending = max(1, int(max_pending))
        self._on_backpressure = on_backpressure
        self._every = max(1, int(every))
        self._pending: list[tuple[int, Future]] = []
        self._history: list[dict] = []

    @property
    def history(self) -> list[dict]:
        return list(self._history)

    def on_epoch_end(self, event: OnEpochEnd) -> None:
        if event.rank != 0:
            return  # rank-0 only
        if event.epoch % self._every != 0:
            return  # gating

        # Reap any done futures (lazy drain).
        self._collect_done()

        # Backpressure check.
        if len(self._pending) >= self._max_pending:
            self._block_oldest()

        try:
            fut = self._dispatcher.submit(self._eval_fn, event.epoch, self._eval_payload)
            self._pending.append((event.epoch, fut))
        except BackpressureSaturatedError:
            if self._on_backpressure == "skip":
                self._history.append({"epoch": event.epoch, "skipped": True,
                                       "reason": "backpressure"})
                return
            if self._on_backpressure == "block":
                self._block_oldest()
                fut = self._dispatcher.submit(self._eval_fn, event.epoch, self._eval_payload)
                self._pending.append((event.epoch, fut))
                return
            # "queue": re-raise so the caller sees it.
            raise

    def on_train_end(self, event: OnTrainEnd) -> None:
        # Drain pending futures.
        while self._pending:
            self._block_oldest()

    def _collect_done(self) -> None:
        still_pending: list[tuple[int, Future]] = []
        for epoch, fut in self._pending:
            if fut.done():
                self._record(epoch, fut)
            else:
                still_pending.append((epoch, fut))
        self._pending = still_pending

    def _block_oldest(self) -> None:
        if not self._pending:
            return
        epoch, fut = self._pending.pop(0)
        self._record(epoch, fut)

    def _record(self, epoch: int, fut: Future) -> None:
        try:
            r = fut.result()
            v = r.value if hasattr(r, "value") else r
            if isinstance(v, dict):
                rec = dict(v)
                rec.setdefault("epoch", epoch)
                self._history.append(rec)
        except Exception as exc:  # noqa: BLE001
            self._history.append({"epoch": epoch, "skipped": True,
                                   "reason": type(exc).__name__})


__all__ = ["AsyncEval", "BackpressureSaturatedError"]
