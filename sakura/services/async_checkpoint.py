"""AsyncCheckpoint — dispatch state-dict writes off the training thread.

Trigger modes:
- every="epoch" — write every epoch
- every=N (int) — write every N epochs
- every="best" — write when `metric` improves (requires `metric` and `mode`)
- every_seconds=T — write at the first epoch end at least T seconds after the previous
  write. Checkpoint cost is per *write*, not per epoch, so on fast epochs a time budget
  bounds the overhead (and the work lost on a crash) independently of epoch length.

`keep=N` retains only the N newest files this service wrote (older ones are deleted once
their successors are fully written).

`max_pending=K` (default 1) bounds in-flight writes: when K writes have not finished, a new
trigger is *skipped* (recorded in ``history`` as ``reason="backpressure"``) instead of
queueing another multi-hundred-MB state behind a slow disk. Periodic checkpoints are
redundant by construction, so dropping one is safe; an unbounded queue is not (memory grows
with every skipped-but-queued state). ``every="best"`` writes are never skipped.

The actual write logic is the user-supplied `writer(state, path) -> dict`
callable, dispatched via the configured Dispatcher so disk I/O doesn't
block training. Default writer (when not supplied) uses torch.save.
"""
from __future__ import annotations

import os
import time
from typing import Any, Callable, Literal, Optional, Union

from sakura.dispatch.base import Dispatcher, Future
from sakura.events import OnEpochEnd, OnTrainEnd
from sakura.service import BaseService


def _torch_save_writer(state: Any, path: str) -> dict[str, Any]:
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
        writer: Optional[Callable[[Any, str], dict[str, Any]]] = None,
        keep: Optional[int] = 3,
        every_seconds: Optional[float] = None,
        max_pending: int = 1,
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
        if every_seconds is not None and every_seconds <= 0:
            raise ValueError("every_seconds must be > 0")
        self._every_seconds = every_seconds
        if max_pending < 1:
            raise ValueError("max_pending must be >= 1")
        self._max_pending = int(max_pending)
        self._last_write_t: Optional[float] = None
        self._submitted: list[tuple[Future, str]] = []
        self._written: list[str] = []
        self._history: list[dict[str, Any]] = []
        self._pending: list[Future] = []
        self._best_metric: Optional[float] = None
        os.makedirs(self._dir, exist_ok=True)
        if every == "best" and metric is None:
            raise ValueError("every='best' requires a `metric` name")
        self.requires = ()  # explicit (already default; left here for clarity)

    @property
    def history(self) -> list[dict[str, Any]]:
        return list(self._history)

    def on_epoch_end(self, event: OnEpochEnd) -> None:
        if event.rank != 0:
            return
        should_write = self._should_write(event)
        if not should_write:
            return
        self._reap_done()
        if self._every != "best" and len(self._pending) >= self._max_pending:
            self._history.append({"epoch": event.epoch, "skipped": True, "reason": "backpressure"})
            return
        path = os.path.join(self._dir, f"epoch_{event.epoch:04d}.{self._ext()}")
        state = self._state_provider()
        try:
            fut = self._dispatcher.submit(self._writer, state, path)
            self._last_write_t = time.monotonic()
            self._pending.append(fut)
            self._submitted.append((fut, path))
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
        self._rotate()

    def _ext(self) -> str:
        return "pt" if self._format == "torch" else "safetensors"

    def _should_write(self, event: OnEpochEnd) -> bool:
        if self._every_seconds is not None:
            return (
                self._last_write_t is None
                or time.monotonic() - self._last_write_t >= self._every_seconds
            )
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
        self._rotate()

    def _rotate(self) -> None:
        """Delete this service's oldest *completed* files beyond ``keep``."""
        still: list[tuple[Future, str]] = []
        for fut, path in self._submitted:
            if fut.done():
                self._written.append(path)
            else:
                still.append((fut, path))
        self._submitted = still
        if self._keep is None:
            return
        while len(self._written) > self._keep:
            old = self._written.pop(0)
            try:
                os.remove(old)
            except OSError:
                pass


__all__ = ["AsyncCheckpoint"]
