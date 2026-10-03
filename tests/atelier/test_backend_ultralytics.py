"""Unit tests for sakura/atelier/backends/ultralytics.py.

Runnable in CI without ultralytics/torch/onnxruntime installed: heavy
imports are faked via sys.modules injection (same pattern as
tests/conftest.py's gnutools mock), so what is actually under test is this
module's own logic -- the metrics-CSV replay, the imgsz/conf/iou/batch/rect
preprocess.json plumbing between export() and runtime_eval(), the real
val-image counting (``_count_val_images``, pure Python + pyyaml, no
ultralytics needed), and that a bad export format is refused before anything
heavy is imported.

A real, end-to-end GPU run (train -> export -> onnxruntime runtime_eval) is
exercised by a manual GPU smoke run, not here; it is not asserted in CI.
"""

from __future__ import annotations

import csv
import json
import sys
import types
from unittest.mock import MagicMock

import pytest

from sakura.atelier.backends import ultralytics as backend_mod
from sakura.atelier.types import Artifact, MetricsSink, RunDir, read_metrics


def test_backend_identity():
    b = backend_mod.BACKEND
    assert b.name == "ultralytics"
    assert b.tasks == frozenset({"object_detection"})
    assert b.licence == "AGPL-3.0"


def test_export_refuses_an_unsupported_format_without_importing_ultralytics(tmp_path):
    out = RunDir(tmp_path / "run").ensure()
    with pytest.raises(ValueError, match="only exports 'onnx'"):
        backend_mod.BACKEND.export(result=MagicMock(), fmt="torchscript", out=out)
    assert "ultralytics" not in sys.modules


def test_csv_replay_logs_one_point_per_epoch_per_metric(tmp_path):
    out = RunDir(tmp_path / "run").ensure()
    csv_path = tmp_path / "results.csv"
    with open(csv_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["epoch", "train/box_loss", "metrics/mAP50(B)", "metrics/mAP50-95(B)"])
        w.writeheader()
        w.writerow({"epoch": "0", "train/box_loss": "1.5", "metrics/mAP50(B)": "0.1", "metrics/mAP50-95(B)": "0.05"})
        w.writerow({"epoch": "1", "train/box_loss": "1.1", "metrics/mAP50(B)": "0.3", "metrics/mAP50-95(B)": "0.15"})

    with MetricsSink(out.metrics_path) as sink:
        backend_mod._replay_csv_into_sink(csv_path, sink, epochs=2)

    points, _ = read_metrics(out.metrics_path)
    by_name = {(p["name"], p["split"]): p["value"] for p in points if p["epoch"] == 1.0}
    assert by_name[("loss", "train")] == 1.1
    assert by_name[("map50", "validation")] == 0.3
    assert by_name[("map50_95", "validation")] == 0.15


def test_count_val_images_reads_a_list_file(tmp_path):
    val_txt = tmp_path / "val_holdout.txt"
    val_txt.write_text("./images/a.jpg\n./images/b.jpg\n./images/c.jpg\n")
    data_yaml = tmp_path / "data.yaml"
    data_yaml.write_text(f"train: train.txt\nval: {val_txt}\nnames:\n  0: x\n")
    assert backend_mod._count_val_images(str(data_yaml)) == 3


def test_count_val_images_reads_an_image_directory(tmp_path):
    val_dir = tmp_path / "images" / "val"
    val_dir.mkdir(parents=True)
    (val_dir / "a.jpg").write_bytes(b"x")
    (val_dir / "b.png").write_bytes(b"x")
    (val_dir / "readme.txt").write_bytes(b"x")  # not an image, must not be counted
    data_yaml = tmp_path / "data.yaml"
    data_yaml.write_text(f"train: train/\nval: {val_dir}\nnames:\n  0: x\n")
    assert backend_mod._count_val_images(str(data_yaml)) == 2


def _fake_ultralytics_module(val_result_dict: dict, onnx_export_path, val_calls: list):
    """A fake ultralytics module exposing just enough of YOLO for
    runtime_eval/export to run against (recording .val()'s kwargs in
    val_calls), without the real 400MB+ dependency in CI."""
    fake = types.ModuleType("ultralytics")

    class FakeValResult:
        results_dict = val_result_dict

    class FakeYOLO:
        def __init__(self, path, task=None):
            self.path = path

        def val(self, **kwargs):
            val_calls.append(kwargs)
            return FakeValResult()

        def export(self, **kwargs):
            return str(onnx_export_path)

    fake.YOLO = FakeYOLO
    return fake


def _make_data_yaml(tmp_path, n_val_images: int = 5):
    val_txt = tmp_path / "val_holdout.txt"
    val_txt.write_text("\n".join(f"./images/{i}.jpg" for i in range(n_val_images)) + "\n")
    data_yaml = tmp_path / "data.yaml"
    data_yaml.write_text(f"train: train.txt\nval: {val_txt}\nnames:\n  0: x\n")
    return data_yaml


def test_runtime_eval_reports_the_measured_value_and_leaves_matches_training_to_the_engine(tmp_path, monkeypatch):
    # Backend.runtime_eval(artifact, data) per CONTRACTS §2.2 has no
    # TrainResult to compare against, so matches_training is the engine's
    # call (Track E1's `run` loop), not guessed here via a side-channel.
    out = RunDir(tmp_path / "run").ensure()
    artifact = Artifact(path=out.artifacts / "model.onnx", format="onnx", sha256="x", bytes=1)
    artifact.path.write_bytes(b"fake -- the stubbed YOLO below never reads it")
    (out.artifacts / "preprocess.json").write_text(
        json.dumps({"imgsz": 320, "conf": 0.001, "iou": 0.7, "batch": 1, "rect": False})
    )
    data_yaml = _make_data_yaml(tmp_path, n_val_images=7)

    val_calls: list = []
    monkeypatch.setitem(sys.modules, "ultralytics", _fake_ultralytics_module(
        {"metrics/mAP50(B)": 0.50}, artifact.path, val_calls,
    ))
    data = MagicMock()
    data.extra = {"yolo_data_yaml": str(data_yaml)}

    ev = backend_mod.BACKEND.runtime_eval(artifact, data)
    assert ev.value == 0.50
    assert ev.n == 7
    assert ev.matches_training is None
    assert ev.reason is None


def test_runtime_eval_reuses_the_exact_imgsz_conf_iou_export_recorded(tmp_path, monkeypatch):
    # The whole point of the preprocess.json sidecar: a runtime_eval/training
    # mismatch must be export drift, not this re-score using different
    # settings than training's own validation did.
    out = RunDir(tmp_path / "run").ensure()
    artifact = Artifact(path=out.artifacts / "model.onnx", format="onnx", sha256="x", bytes=1)
    artifact.path.write_bytes(b"fake")
    (out.artifacts / "preprocess.json").write_text(
        json.dumps({"imgsz": 512, "conf": 0.25, "iou": 0.5, "batch": 1, "rect": True})
    )
    data_yaml = _make_data_yaml(tmp_path)

    val_calls: list = []
    monkeypatch.setitem(sys.modules, "ultralytics", _fake_ultralytics_module(
        {"metrics/mAP50(B)": 0.1}, artifact.path, val_calls,
    ))
    data = MagicMock()
    data.extra = {"yolo_data_yaml": str(data_yaml)}

    backend_mod.BACKEND.runtime_eval(artifact, data)
    assert val_calls == [{
        "data": str(data_yaml), "device": "cpu", "verbose": False, "plots": False,
        "split": "val", "imgsz": 512, "conf": 0.25, "iou": 0.5, "half": False,
        "batch": 1, "rect": True,
    }]


def test_runtime_eval_refuses_a_preprocess_sidecar_that_claims_a_non_unit_batch(tmp_path, monkeypatch):
    # A static ONNX export (the only kind this backend produces) cannot run
    # any batch but 1 -- a sidecar claiming otherwise is a bug upstream, not
    # something to silently paper over.
    out = RunDir(tmp_path / "run").ensure()
    artifact = Artifact(path=out.artifacts / "model.onnx", format="onnx", sha256="x", bytes=1)
    artifact.path.write_bytes(b"fake")
    (out.artifacts / "preprocess.json").write_text(
        json.dumps({"imgsz": 320, "conf": 0.001, "iou": 0.7, "batch": 8, "rect": True})
    )
    monkeypatch.setitem(sys.modules, "ultralytics", _fake_ultralytics_module({}, artifact.path, []))
    data = MagicMock()
    data.extra = {"yolo_data_yaml": str(_make_data_yaml(tmp_path))}

    with pytest.raises(ValueError, match="batch=8"):
        backend_mod.BACKEND.runtime_eval(artifact, data)


def test_runtime_eval_falls_back_to_val_mode_defaults_without_a_preprocess_sidecar(tmp_path, monkeypatch):
    out = RunDir(tmp_path / "run").ensure()
    artifact = Artifact(path=out.artifacts / "model.onnx", format="onnx", sha256="x", bytes=1)
    artifact.path.write_bytes(b"fake")  # no preprocess.json next to it
    data_yaml = _make_data_yaml(tmp_path)

    val_calls: list = []
    monkeypatch.setitem(sys.modules, "ultralytics", _fake_ultralytics_module(
        {"metrics/mAP50(B)": 0.1}, artifact.path, val_calls,
    ))
    data = MagicMock()
    data.extra = {"yolo_data_yaml": str(data_yaml)}

    backend_mod.BACKEND.runtime_eval(artifact, data)
    assert val_calls[0]["imgsz"] == 640 and val_calls[0]["conf"] == 0.001 and val_calls[0]["iou"] == 0.7
    assert val_calls[0]["batch"] == 1 and val_calls[0]["rect"] is False
