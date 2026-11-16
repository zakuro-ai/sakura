"""The Ludwig adapter's per-epoch reporting (zak-marketplace BE-6): besides
Ludwig's own metrics it writes `lr` and `gpu_util` points and prints one plain
progress line per epoch, which the runner streams into the job log. A live
run on staging (hub job a870d75e) showed nothing for two minutes: Ludwig's own
console output is a progress bar the log cannot show. Needs neither Ludwig nor
a GPU: the progress tracker is a stand-in with Ludwig's attribute names."""
from __future__ import annotations

import io
import json
import types

from sakura.adapters import ludwig as adapter
from sakura.atelier.types import MetricsSink


def _tracker(epoch=3):
    point = types.SimpleNamespace
    return types.SimpleNamespace(
        epoch=epoch, steps=21, learning_rate=1e-4,
        train_metrics={"label": {"loss": [point(epoch=epoch, step=21, value=0.25)],
                                 "accuracy": [point(epoch=epoch, step=21, value=0.9)]}},
        validation_metrics={"label": {"loss": [point(epoch=epoch, step=21, value=1.5)],
                                      "accuracy": [point(epoch=epoch, step=21, value=0.42)]}})


def test_epoch_end_logs_lr_gpu_and_prints_a_progress_line(tmp_path, monkeypatch):
    monkeypatch.setattr(adapter, "gpu_utilization", lambda: 87.0)
    out = io.StringIO()
    with MetricsSink(tmp_path / "metrics.jsonl") as sink:
        cb = adapter.LudwigAdapter(sink, output_feature="label", epochs_max=30, stream=out)
        cb.on_epoch_start(None, _tracker(), "")
        cb.on_epoch_end(None, _tracker(), "")
    points = [json.loads(line) for line in (tmp_path / "metrics.jsonl").read_text().splitlines()]
    system = {p["name"]: p for p in points if p["name"] in adapter.SYSTEM_METRICS}
    assert system["lr"]["value"] == 1e-4 and system["lr"]["epoch"] == 3.0
    assert system["gpu_util"]["value"] == 87.0 and system["gpu_util"]["split"] == "train"
    line = out.getvalue().strip()
    assert line.startswith("training: epoch 3/30")
    assert "train loss 0.2500" in line and "val loss 1.5000" in line and "val accuracy 42.0%" in line
    assert "lr 0.0001" in line


def test_no_gpu_reading_writes_no_gpu_point(tmp_path, monkeypatch):
    monkeypatch.setattr(adapter, "gpu_utilization", lambda: None)
    out = io.StringIO()
    tracker = _tracker()
    tracker.learning_rate = None
    with MetricsSink(tmp_path / "metrics.jsonl") as sink:
        adapter.LudwigAdapter(sink, output_feature="label", stream=out).on_epoch_end(None, tracker, "")
    names = [json.loads(line)["name"] for line in (tmp_path / "metrics.jsonl").read_text().splitlines()]
    assert "gpu_util" not in names and "lr" not in names
    assert out.getvalue().startswith("training: epoch 3 ·")


def test_gpu_utilization_never_raises(monkeypatch):
    def boom(*a, **k):
        raise OSError("no nvidia-smi")
    monkeypatch.setattr(adapter.subprocess, "run", boom)
    value = adapter.gpu_utilization()
    assert value is None or isinstance(value, float)


def test_progress_line_without_numbers_is_just_the_epoch():
    assert adapter.progress_line(1, None, train_loss=None, val_loss=None, val_acc=None,
                                 lr=None, epoch_seconds=None) == "training: epoch 1"
