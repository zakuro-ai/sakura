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

import subprocess
import sys
import time
from typing import Any, TextIO

try:
    from ludwig.callbacks import Callback
except ImportError:  # pragma: no cover -- exercised only where ludwig is installed
    class Callback:  # type: ignore[no-redef]
        pass

from sakura.adapters.base import Adapter
from sakura.atelier.types import MetricsSink
from sakura.atelier.wrap_up import WrapUpWatch
from sakura.events import OnEpochEnd, OnTrainBegin, OnTrainEnd
from sakura.runtime import SakuraRuntime

#: Ludwig's own metric ids -> the atelier contract's metric vocabulary
#: (CONTRACTS §2.1 lists loss, accuracy, f1, auc, map50, map50_95, rmse, wer,
#: perplexity; Ludwig spells RMSE out in full).
METRIC_ALIASES = {"root_mean_squared_error": "rmse"}

#: Points this adapter adds beside Ludwig's own metrics, once per epoch, under
#: split ``train``: the learning rate the optimizer used, and the GPU's busy
#: share (0-100) when one can be read. They are what a live view needs that
#: Ludwig's ProgressTracker does not hold as a metric.
SYSTEM_METRICS = ("lr", "gpu_util")


def gpu_utilization() -> float | None:
    """The current GPU's busy share in percent, or None when none can be read.

    `torch.cuda.utilization()` needs pynvml; `nvidia-smi` is the fallback the
    runner image always has. Best effort by design: a missing reading never
    costs a training run anything but this one number."""
    try:
        import torch

        if torch.cuda.is_available():
            return float(torch.cuda.utilization())
    except Exception:  # noqa: BLE001 -- pynvml absent, driver mismatch, no GPU
        pass
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5, check=False)
    except (OSError, subprocess.SubprocessError):
        return None
    for line in (out.stdout or "").splitlines():
        try:
            return float(line.strip())
        except ValueError:
            continue
    return None


def _latest(metrics_by_feature: dict[str, Any], feature: str, name: str) -> float | None:
    points = (metrics_by_feature or {}).get(feature, {}).get(name) or []
    if not points:
        return None
    last = points[-1]
    value: Any = getattr(last, "value", last[-1] if isinstance(last, tuple) else last)
    if value is None:
        return None
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    return v if v == v else None


def progress_line(epoch: int, epochs_max: int | None, *, train_loss: float | None,
                  val_loss: float | None, val_acc: float | None, lr: float | None,
                  epoch_seconds: float | None) -> str:
    """One plain line per epoch for the job log: the runner forwards the
    engine's stdout line by line, and Ludwig's own console output is a
    progress bar the log cannot show."""
    head = f"epoch {epoch}/{epochs_max}" if epochs_max else f"epoch {epoch}"
    parts = [head]
    if train_loss is not None:
        parts.append(f"train loss {train_loss:.4f}")
    if val_loss is not None:
        parts.append(f"val loss {val_loss:.4f}")
    if val_acc is not None:
        parts.append(f"val accuracy {val_acc:.1%}")
    if lr is not None:
        parts.append(f"lr {lr:.2g}")
    if epoch_seconds is not None:
        parts.append(f"{epoch_seconds:.1f}s")
    return "training: " + " · ".join(parts)


class LudwigAdapter(Callback, Adapter):  # type: ignore[misc]
    """Forwards every point Ludwig's ProgressTracker records to a MetricsSink,
    one metrics.jsonl line per (split, metric, checkpoint)."""

    def __init__(self, metrics: MetricsSink, output_feature: str,
                 runtime: SakuraRuntime | None = None, wrap_up: WrapUpWatch | None = None, *,
                 epochs_max: int | None = None, stream: TextIO | None = None) -> None:
        Callback.__init__(self)
        Adapter.__init__(self, runtime)  # type: ignore[arg-type]
        self._metrics = metrics
        self._output_feature = output_feature
        self._epochs_max = epochs_max
        #: Where the per-epoch progress line goes; stdout, which the runner
        #: streams into the job log as it is written.
        self._stream = stream
        self._epoch_started = time.monotonic()
        #: Graceful wrap-up (sakura.atelier.wrap_up): None = this adapter never stops training.
        self.wrap_up = wrap_up
        self._checkpointed = False
        self._armed_config: Any = None  # the trainer config whose skip_all_evaluation a wrap-up flipped

    def on_save_best_checkpoint(self, trainer: Any, progress_tracker: Any, save_path: str) -> None:
        """Ludwig has now written the checkpoint ``model.train`` loads when it returns."""
        self._checkpointed = True

    def _can_stop_now(self, trainer: Any, progress_tracker: Any, save_path: str) -> bool:
        """Ludwig returns its best checkpoint and raises "No checkpoint was saved" when there is none, and
        an evaluation round (an epoch apart by default) is the only thing that writes one. A wrap-up seen
        before the first round must not wait for it: it flips Ludwig's own ``skip_all_evaluation`` switch
        (0.17.9 ``trainer.py``), whose teardown saves ``best.ckpt`` at the step training stops on, and
        ``api.py`` then skips the training report that would need an evaluation point. This batch is the
        last one, so the switch costs nothing. When a best checkpoint already exists it is kept (the better
        model; stopping is just the next step), and nothing is flipped."""
        if self._checkpointed or bool(getattr(trainer, "skip_all_evaluation", False)):
            return True
        self._armed_config = getattr(trainer, "config", None)
        trainer.skip_all_evaluation = True
        if self._armed_config is not None:
            self._armed_config.skip_all_evaluation = True
        self._checkpointed = True
        return True

    def on_train_end(self, output_directory: str) -> None:
        # Before Ludwig saves the model, and its config with it: a delivered model must not carry the switch
        # (a later run that continues from it would otherwise never evaluate).
        if self._armed_config is not None:
            self._armed_config.skip_all_evaluation = False
        self.emit(OnTrainEnd(model=None, history=[], rank=0, world_size=1))

    def on_batch_end(self, trainer: Any, progress_tracker: Any, save_path: str, sync_step: bool = True) -> None:
        """Step boundary. Ludwig's own SIGINT handling ends training by capping ``total_steps`` at the
        current step (its batch and epoch loops both test it); wrap-up does the same, after making sure a
        checkpoint will exist (see :meth:`_can_stop_now`), so the trainer winds down normally and
        ``model.train`` returns with it. The request is recorded when first seen."""
        requested = self.wrap_up is not None and self.wrap_up.stop(
            getattr(progress_tracker, "epoch", None), getattr(progress_tracker, "steps", None))
        if requested and self._can_stop_now(trainer, progress_tracker, save_path):
            trainer.total_steps = min(int(getattr(trainer, "total_steps", 0) or 0),
                                      int(getattr(progress_tracker, "steps", 0)))

    def should_early_stop(self, trainer: Any, progress_tracker: Any, is_coordinator: bool) -> bool:
        """Checked after every evaluation round, once its checkpoint is saved. Normally the batch hook has
        stopped the run already; this ends it when that hook could not write a checkpoint."""
        return self.wrap_up is not None and self.wrap_up.stop(
            getattr(progress_tracker, "epoch", None), getattr(progress_tracker, "steps", None))

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

    def on_epoch_start(self, trainer: Any, progress_tracker: Any, save_path: str) -> None:
        self._epoch_started = time.monotonic()

    def on_epoch_end(self, trainer: Any, progress_tracker: Any, save_path: str) -> None:
        epoch = progress_tracker.epoch
        self._log_split("train", progress_tracker.train_metrics, epoch)
        self._log_system(progress_tracker, epoch)
        self.emit(OnEpochEnd(epoch=epoch, model=None, optimizer=None,
                              metrics={}, rank=0, world_size=1))

    def _log_system(self, progress_tracker: Any, epoch: int) -> None:
        """`lr` and `gpu_util` as metric points, and the epoch's progress line.

        Never raises: this is reporting, and a training run must not die
        because a GPU counter could not be read."""
        step = int(getattr(progress_tracker, "steps", 0) or 0)
        lr: float | None = None
        try:
            raw = getattr(progress_tracker, "learning_rate", None)
            lr = None if raw is None else float(raw)
        except (TypeError, ValueError):
            lr = None
        gpu = gpu_utilization()
        for name, value in (("lr", lr), ("gpu_util", gpu)):
            if value is None:
                continue
            try:
                self._metrics.log(name, value, step=step, split="train", epoch=float(epoch))
            except ValueError:
                continue
        feature = self._output_feature
        line = progress_line(
            int(epoch), self._epochs_max,
            train_loss=_latest(progress_tracker.train_metrics, feature, "loss"),
            val_loss=_latest(getattr(progress_tracker, "validation_metrics", {}), feature, "loss"),
            val_acc=_latest(getattr(progress_tracker, "validation_metrics", {}), feature, "accuracy"),
            lr=lr, epoch_seconds=time.monotonic() - self._epoch_started)
        stream = self._stream if self._stream is not None else sys.stdout
        try:
            print(line, file=stream, flush=True)
        except (OSError, ValueError):
            pass

    def on_eval_end(self, trainer: Any, progress_tracker: Any, save_path: str) -> None:
        self._log_split("validation", progress_tracker.validation_metrics, progress_tracker.epoch)


__all__ = ["LudwigAdapter", "METRIC_ALIASES", "SYSTEM_METRICS", "gpu_utilization", "progress_line"]
