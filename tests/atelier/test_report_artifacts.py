"""Every delivered artifact says whether it was tested (BE-9).

`report.json`'s `artifacts[]` rows carry `{format, size_bytes, validated,
eval_rows, eval_acc}`: a TorchScript file that never had a row through it
must not read like the ONNX file that scored 42%."""

from __future__ import annotations

from pathlib import Path

from sakura.atelier.__main__ import _artifact_entry
from sakura.atelier.types import Artifact, RunDir


def _art(out: RunDir, fmt: str, name: str) -> Artifact:
    return Artifact(path=out.artifacts / name, format=fmt, sha256="s", bytes=1234)


def test_a_rescored_export_is_validated_with_its_rows_and_accuracy(tmp_path: Path) -> None:
    out = RunDir(tmp_path / "run").ensure()
    ev = {"format": "onnx", "metric": "accuracy", "value": 0.42, "n": 100, "matches_training": True,
          "reason": None}
    entry = _artifact_entry(_art(out, "onnx", "model.onnx"), out, ev)
    assert entry["format"] == "onnx" and entry["size_bytes"] == 1234 and entry["bytes"] == 1234
    assert entry["validated"] is True and entry["eval_rows"] == 100 and entry["eval_acc"] == 0.42
    assert entry["matches_training"] is True and "validation_reason" not in entry


def test_an_export_with_no_rows_scored_is_not_validated(tmp_path: Path) -> None:
    out = RunDir(tmp_path / "run").ensure()
    ev = {"format": "torchscript", "metric": "accuracy", "value": None, "n": 0, "matches_training": None,
          "reason": "no runtime scorer implemented for torchscript"}
    entry = _artifact_entry(_art(out, "torchscript", "model.torchscript.pt"), out, ev)
    assert entry["validated"] is False and entry["eval_rows"] == 0 and entry["eval_acc"] is None
    assert entry["validation_reason"] == "no runtime scorer implemented for torchscript"


def test_a_regression_export_has_a_value_but_no_accuracy(tmp_path: Path) -> None:
    out = RunDir(tmp_path / "run").ensure()
    ev = {"format": "onnx", "metric": "rmse", "value": 1.5, "n": 20, "matches_training": None, "reason": None}
    entry = _artifact_entry(_art(out, "onnx", "model.onnx"), out, ev)
    assert entry["validated"] is True and entry["eval_acc"] is None
    assert entry["eval_metric"] == "rmse" and entry["eval_value"] == 1.5


def test_rows_without_a_number_are_not_a_validation(tmp_path: Path) -> None:
    out = RunDir(tmp_path / "run").ensure()
    ev = {"format": "onnx", "metric": "accuracy", "value": None, "n": 50, "reason": "session load failed"}
    entry = _artifact_entry(_art(out, "onnx", "model.onnx"), out, ev)
    assert entry["validated"] is False and entry["eval_rows"] == 0
