"""Telemetry — JSON-line event sink (one record per dispatched event).

Output sink:
- callable[[dict], None]: invoked with each record (e.g., list.append for tests)
- str: a filesystem path; opened on first write, flushed on shutdown
- IO[str]: any text-mode file-like object
- None: stderr (default)
"""
from __future__ import annotations

import dataclasses
import json
import sys
import time
from typing import Any, Callable, IO, Optional, Union

from sakura.events import Event
from sakura.service import BaseService


_PRIMITIVE = (str, int, float, bool, type(None))


def _safe_payload(event: Event) -> dict:
    """Extract serializable fields from an event payload, skipping non-JSON values."""
    out: dict[str, Any] = {}
    for f in dataclasses.fields(event):
        if f.name in ("rank", "world_size"):
            continue
        v = getattr(event, f.name)
        out[f.name] = _safe_value(v)
    return out


def _safe_value(v: Any) -> Any:
    if isinstance(v, _PRIMITIVE):
        return v
    if isinstance(v, dict):
        return {k: _safe_value(val) for k, val in v.items() if isinstance(k, str)}
    if isinstance(v, (list, tuple)):
        return [_safe_value(i) for i in v]
    if isinstance(v, BaseException):
        return {"exc_type": type(v).__name__, "msg": str(v)}
    # Skip arbitrary objects (model, optimizer, dataloader): not JSON-serializable.
    return None


class Telemetry(BaseService):
    name = "telemetry"
    priority = 0

    def __init__(
        self,
        *,
        output: Union[Callable[[dict], None], IO[str], str, None] = None,
    ):
        super().__init__()
        self._output = output if output is not None else sys.stderr
        self._opened: Optional[IO[str]] = None

    def _emit(self, record: dict) -> None:
        out = self._output
        if callable(out):
            try:
                out(record)
            except Exception:
                pass  # best-effort: telemetry failures must not interrupt the training loop
            return
        if isinstance(out, str):
            if self._opened is None:
                self._opened = open(out, "a", encoding="utf-8")
            f = self._opened
        else:
            f = out
        try:
            f.write(json.dumps(record, default=str) + "\n")
            f.flush()
        except Exception:
            pass  # best-effort: telemetry failures must not interrupt the training loop

    def on_runtime_shutdown(self, runtime: Any) -> None:
        if self._opened is not None:
            try:
                self._opened.close()
            except Exception:
                pass  # best-effort: telemetry failures must not interrupt the training loop
            self._opened = None

    def on_event(self, event: Event) -> None:
        record = {
            "ts": time.time(),
            "event": type(event).__name__,
            "service": self.name,
            "rank": event.rank,
            "world_size": event.world_size,
            "payload": _safe_payload(event),
        }
        self._emit(record)


__all__ = ["Telemetry"]
