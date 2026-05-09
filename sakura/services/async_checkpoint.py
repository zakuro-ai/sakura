"""AsyncCheckpoint — dispatch state-dict writes off the training thread.

Trigger modes:
- every="epoch" — write every epoch
- every=N (int) — write every N epochs
- every="best" — write when `metric` improves (requires `metric` and `mode`)

The actual write logic is the user-supplied `writer(state, path) -> dict`
callable, dispatched via the configured Dispatcher so disk I/O doesn't
block training. Default writer (when not supplied) uses torch.save.
"""
from __future__ import annotations

import os
from typing import Any, Callable, Literal, Optional, Union

from sakura.dispatch.base import Dispatcher, Future
from sakura.events import OnEpochEnd, OnTrainEnd
from sakura.service import BaseService


def _torch_save_writer(state, path):
    import torch
    torch.save(state, path)
    return {"path": str(path)}


class AsyncCheckpoint(BaseService):
    name = "async_checkpoint"
    priority = 85

    def __init__(
        self,
        *,
        dir: str,
        dispatcher: Dispatcher,
        state_provider: Callable[[], Any],
        every: Union[Literal["epoch", "best"], int] = "epoch",
        metric: Optional[str] = None,
        mode: Literal["min", "max"] = "min",
        format: Literal["torch", "safetensors"] = "torch",
        writer: Optional[Callable[[Any, str], dict]] = None,
        keep: Optional[int] = 3,
    ):
        super().__init__()
        self._dir = dir
        self._dispatcher = dispatcher
        self._state_provider = state_provider
        self._every = every
        self._metric = metric
        self._mode = mode
        self._format = format
        self._writer = writer if writer is not None else _torch_save_writer
        self._keep = keep
        self._history: list[dict] = []
        self._pending: list[Future] = []
        self._best_metric: Optional[float] = None
        os.makedirs(self._dir, exist_ok=True)
        if every == "best" and metric is None:
            raise ValueError("every='best' requires a `metric` name")
        self.requires = ()  # explicit (already default; left here for clarity)

    @property
    def history(self) -> list[dict]:
        return list(self._history)

    def on_epoch_end(self, event: OnEpochEnd) -> None:
        if event.rank != 0:
            return
        should_write = self._should_write(event)
        if not should_write:
            return
        path = os.path.join(self._dir, f"epoch_{event.epoch:04d}.{self._ext()}")
        state = self._state_provider()
        try:
            fut = self._dispatcher.submit(self._writer, state, path)
            self._pending.append(fut)
            # Reap done.
            self._reap_done()
        except BaseException as exc:  # noqa: BLE001
            self._history.append({"epoch": event.epoch, "skipped": True,
                                   "reason": type(exc).__name__})

    def on_train_end(self, event: OnTrainEnd) -> None:
        # Drain.
        for fut in self._pending:
            try:
                r = fut.result()
                v = r.value if hasattr(r, "value") else r
                if isinstance(v, dict):
                    self._history.append(v)
            except BaseException:
                pass
        self._pending.clear()

    def _ext(self) -> str:
        return "pt" if self._format == "torch" else "safetensors"

    def _should_write(self, event: OnEpochEnd) -> bool:
        if self._every == "epoch":
            return True
        if isinstance(self._every, int):
            return (event.epoch % self._every) == 0
        if self._every == "best":
            v = event.metrics.get(self._metric) if self._metric else None
            if not isinstance(v, (int, float)):
                return False
            if self._best_metric is None:
                self._best_metric = float(v)
                return True
            improved = (
                v < self._best_metric if self._mode == "min" else v > self._best_metric
            )
            if improved:
                self._best_metric = float(v)
                return True
            return False
        return False

    def _reap_done(self) -> None:
        still: list[Future] = []
        for fut in self._pending:
            if fut.done():
                try:
                    r = fut.result()
                    v = r.value if hasattr(r, "value") else r
                    if isinstance(v, dict):
                        self._history.append(v)
                except BaseException:
                    pass
            else:
                still.append(fut)
        self._pending = still


__all__ = ["AsyncCheckpoint"]
