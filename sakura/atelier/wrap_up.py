"""Graceful wrap-up: the run's cooperative "stop learning, keep what you have" signal.

The signal is a file, so any process can raise it (the zak-forge runner, a person with ``touch``):
``OUT/wrap-up-requested``, or the path in env ``ATELIER_WRAP_UP_FILE`` when set. Its body is
optional JSON ``{"reason": str}``. Each training backend asks a :class:`WrapUpWatch` at its next
step/batch boundary (epoch boundary only where the trainer has no step hook); once the answer is
yes the backend stops learning, keeps its latest (or best) checkpoint, and returns normally, so
export, evaluation and packaging run as usual. ``__main__`` then writes
``stopped_early: {reason: "wrap_up", epoch, step}`` into ``report.json`` (CONTRACTS §2.1).

The same step boundary enforces the spec's ``limits`` (SAK-0019): a watch built with ``max_steps`` also
says stop once that many optimizer steps are done (``stopped_early.reason`` is then ``max_steps``), and
``limits.time_budget_s`` is a :class:`TimeBudget`, a timer that raises the wrap-up signal itself.

Nothing here imports a training stack.
"""

from __future__ import annotations

import json
import os
import threading
from pathlib import Path
from typing import Any

ENV_VAR = "ATELIER_WRAP_UP_FILE"
FILE_NAME = "wrap-up-requested"
REASON = "wrap_up"  # ``stopped_early.reason`` in report.json
REASON_MAX_STEPS = "max_steps"  # ... when ``limits.max_steps`` ended training instead
TIME_BUDGET = "time_budget"  # the ``reason`` in the signal file a TimeBudget writes


def signal_path(out_root: Path | str | None = None) -> Path | None:
    """Env ``ATELIER_WRAP_UP_FILE`` if set, else ``OUT/wrap-up-requested``, else None."""
    env = os.environ.get(ENV_VAR)
    if env:
        return Path(env)
    return None if out_root is None else Path(out_root) / FILE_NAME


def requested(out_root: Path | str | None = None) -> bool:
    """One ``stat``: cheap enough for every training step."""
    path = signal_path(out_root)
    return path is not None and path.exists()


def read_reason(out_root: Path | str | None = None) -> str | None:
    """The ``reason`` in the signal file's JSON body; None when absent, empty or not JSON."""
    path = signal_path(out_root)
    try:
        body = json.loads(path.read_text(encoding="utf-8")) if path else None
    except (OSError, ValueError):
        return None
    reason = body.get("reason") if isinstance(body, dict) else None
    return reason if isinstance(reason, str) and reason else None


class WrapUpWatch:
    """What one training loop holds: ``if watch.stop(epoch, step): break``.

    Sticky: once the signal was seen it stays seen (the file may be removed later), and the
    position of the first sighting is the ``stopped_early`` the report carries.
    """

    def __init__(self, out_root: Path | str | None = None, max_steps: int | None = None) -> None:
        self._out_root = out_root
        #: ``limits.max_steps``: stop once ``step`` (optimizer steps done so far) reaches it. None = no cap.
        self.max_steps = max_steps
        #: The ``step`` of the latest :meth:`stop` call that gave one: how far the loop has got.
        self.last_step: int | None = None
        self.stopped_early: dict[str, Any] | None = None

    @property
    def triggered(self) -> bool:
        return self.stopped_early is not None

    def stop(self, epoch: float | None = None, step: int | None = None) -> bool:
        if step is not None:
            self.last_step = int(step)
        if self.stopped_early is None:
            if requested(self._out_root):
                reason: str | None = REASON
            elif self.max_steps is not None and step is not None and int(step) >= self.max_steps:
                reason = REASON_MAX_STEPS
            else:
                reason = None
            if reason is not None:
                self.stopped_early = {"reason": reason, "epoch": None if epoch is None else float(epoch),
                                      "step": None if step is None else int(step)}
        return self.stopped_early is not None

    def reason(self) -> str | None:
        return read_reason(self._out_root)


def max_steps_of(spec: Any) -> int | None:
    """``limits.max_steps`` of a ``ResolvedSpec`` (or an ``AtelierSpec``); None when the spec sets none."""
    limits = getattr(getattr(spec, "spec", spec), "limits", None)
    return getattr(limits, "max_steps", None)


class TimeBudget:
    """``limits.time_budget_s``: raise the wrap-up signal ``seconds`` after ``with`` is entered.

    The backend then stops at its next step boundary exactly as for a runner's request, so there is one
    stop mechanism, not two. A signal somebody else already raised is left alone. On exit the timer is
    cancelled and a signal file this budget wrote is removed (a later run into the same directory must
    not start already stopped); ``fired`` says whether this budget raised the signal."""

    def __init__(self, out_root: Path | str | None, seconds: float | None) -> None:
        self._path = signal_path(out_root)
        self._seconds = seconds
        self._timer: threading.Timer | None = None
        self.fired = False

    def _raise(self) -> None:
        if self._path is not None and not self._path.exists():
            self._path.parent.mkdir(parents=True, exist_ok=True)
            self._path.write_text(json.dumps({"reason": TIME_BUDGET}), encoding="utf-8")
            self.fired = True

    def __enter__(self) -> TimeBudget:
        if self._seconds is not None and self._path is not None:
            self._timer = threading.Timer(self._seconds, self._raise)
            self._timer.daemon = True
            self._timer.start()
        return self

    def __exit__(self, *exc: object) -> None:
        if self._timer is not None:
            self._timer.cancel()
            self._timer.join()
        if self.fired and self._path is not None:
            self._path.unlink(missing_ok=True)
