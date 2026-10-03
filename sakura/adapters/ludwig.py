"""LudwigAdapter -- ludwig.callbacks.Callback that bridges Ludwig's training
lifecycle to the atelier MetricsSink, and optionally to Sakura runtime events
(same pattern as sakura/adapters/huggingface.py: translate framework hooks
into Sakura events).

The MetricsSink bridge is the part the atelier engine actually depends on:
CONTRACTS §2.1's metrics.jsonl is more specific than the generic Sakura event
stream (per-split, per-metric-name, line-flushed JSON the runner tails live),
so it gets a direct path here rather than going through sakura.events.
"""

from __future__ import annotations

from typing import Any

try:
    from ludwig.callbacks import Callback
except ImportError:  # pragma: no cover -- exercised only where ludwig is installed
    class Callback:  # type: ignore[no-redef]
        pass

from sakura.adapters.base import Adapter
from sakura.atelier.types import MetricsSink
from sakura.events import OnEpochEnd, OnTrainBegin, OnTrainEnd
from sakura.runtime import SakuraRuntime

#: Ludwig's own metric ids -> the atelier contract's metric vocabulary
#: (CONTRACTS §2.1 lists loss, accuracy, f1, auc, map50, map50_95, rmse, wer,
#: perplexity; Ludwig spells RMSE out in full).
METRIC_ALIASES = {"root_mean_squared_error": "rmse"}


class LudwigAdapter(Callback, Adapter):  # type: ignore[misc]
    """Forwards every point Ludwig's ProgressTracker records to a MetricsSink,
    one metrics.jsonl line per (split, metric, checkpoint)."""

    def __init__(self, metrics: MetricsSink, output_feature: str,
                 runtime: SakuraRuntime | None = None) -> None:
        Callback.__init__(self)
        Adapter.__init__(self, runtime)  # type: ignore[arg-type]
        self._metrics = metrics
        self._output_feature = output_feature

    def emit(self, event: Any) -> None:
        if self.runtime is not None:
            super().emit(event)

    def _log_split(self, split: str, metrics_by_feature: dict[str, Any], epoch: int) -> None:
        for name, points in metrics_by_feature.get(self._output_feature, {}).items():
            if not points:
                continue
            last = points[-1]  # TrainerMetric(epoch, step, value): most recent checkpoint
            step = int(getattr(last, "step", last[1] if isinstance(last, tuple) else 0))
            value: Any = getattr(last, "value", last[-1] if isinstance(last, tuple) else last)
            if value is None:
                continue
            try:
                self._metrics.log(METRIC_ALIASES.get(name, name), float(value),
                                   step=step, split=split, epoch=float(epoch))
            except ValueError:
                # NaN point: MetricsSink refuses it rather than draw a hole in
                # the curve (its own contract) -- here that means skip, not crash.
                continue

    def on_train_start(self, model: Any, config: dict[str, Any], config_fp: str | None) -> None:
        self.emit(OnTrainBegin(model=model, optimizer=None, train_loader=None, val_loader=None,
                                rank=0, world_size=1))

    def on_epoch_end(self, trainer: Any, progress_tracker: Any, save_path: str) -> None:
        self._log_split("train", progress_tracker.train_metrics, progress_tracker.epoch)
        self.emit(OnEpochEnd(epoch=progress_tracker.epoch, model=None, optimizer=None,
                              metrics={}, rank=0, world_size=1))

    def on_eval_end(self, trainer: Any, progress_tracker: Any, save_path: str) -> None:
        self._log_split("validation", progress_tracker.validation_metrics, progress_tracker.epoch)

    def on_train_end(self, output_directory: str) -> None:
        self.emit(OnTrainEnd(model=None, history=[], rank=0, world_size=1))


__all__ = ["LudwigAdapter", "METRIC_ALIASES"]
