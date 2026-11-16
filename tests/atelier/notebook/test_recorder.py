import json

import pandas as pd

from sakura.atelier.notebook import recorder
from sakura.atelier.types import MaterialisedData, ResolvedSpec, RunDir


def _data(tmp_path, train, validation, labels=None):
    return MaterialisedData(root=tmp_path, train=train, validation=validation,
                            manifest_sha256="ab" * 32, labels=labels)


def _point(step, split="train", name="loss", value=1.0):
    return {"t": 1, "step": step, "epoch": None, "split": split, "name": name, "value": value}


def test_frame_summary_of_a_dataframe_has_shape_dtypes_nulls_and_balance():
    df = pd.DataFrame({"x": [1, 2, None], "y": ["a", "b", "a"]})
    text = recorder.frame_summary(df, "train")
    assert "train: 3 rows x 2 columns" in text and "nulls" in text and "y" in text
    assert "y balance: a=2, b=1" in text
    assert "x: mean" in text


def test_frame_summary_of_a_list_of_dicts_and_of_none_does_not_raise():
    assert "2 examples" in recorder.frame_summary([{"prompt": "a", "completion": "b"}] * 2, "train")
    assert "no validation split" in recorder.frame_summary(None, "validation")


def test_frame_summary_of_a_dict_an_empty_list_and_a_scalar_does_not_raise():
    assert "images" in recorder.frame_summary({"images": "a/", "labels": "b/"}, "train")
    assert "0 examples" in recorder.frame_summary([], "train")
    assert "int" in recorder.frame_summary(3, "train")


def test_curves_render_one_png_per_metric(tmp_path):
    p = tmp_path / "metrics.jsonl"
    rows = [_point(s, sp, n, 1.0 / (s + 1))
            for s in range(5) for sp in ("train", "validation") for n in ("loss", "accuracy")]
    p.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    outs = recorder.NotebookRecorder().curves(p)
    pngs = [o for o in outs if o["output_type"] == "display_data"]
    assert len(pngs) == 2 and all(o["data"]["image/png"].startswith("iVBOR") for o in pngs)


def test_curves_with_one_point_or_no_file_say_so(tmp_path):
    p = tmp_path / "metrics.jsonl"
    p.write_text(json.dumps(_point(0)) + "\n")
    outs = recorder.NotebookRecorder().curves(p)
    assert outs[0]["output_type"] == "stream" and "1 point" in outs[0]["text"]
    assert "no metrics.jsonl" in recorder.NotebookRecorder().curves(tmp_path / "missing.jsonl")[0]["text"]


def test_curves_with_an_empty_file_or_a_torn_last_line_do_not_raise(tmp_path):
    p = tmp_path / "metrics.jsonl"
    p.write_text("")
    outs = recorder.NotebookRecorder().curves(p)
    assert outs[0]["output_type"] == "stream" and "empty" in outs[0]["text"]
    # A sink flushes per line; a tail read may still see a partial last record.
    p.write_text(json.dumps(_point(0)) + "\n" + json.dumps(_point(1)) + "\n" + '{"t": 1, "ste')
    assert recorder.read_curves(p) == {"loss": {"train": [(0, 1.0), (1, 1.0)]}}
    assert recorder.NotebookRecorder().curves(p)[0]["output_type"] == "display_data"


def test_curves_with_a_corrupt_file_return_a_note(tmp_path):
    p = tmp_path / "metrics.jsonl"
    p.write_text("not json\n")
    outs = recorder.NotebookRecorder().curves(p)
    assert outs[0]["output_type"] == "stream" and "unreadable" in outs[0]["text"]


def test_eval_and_artifact_tables_are_html_with_every_row():
    report = {"runtime_eval": [{"format": "onnx", "runtime": "onnxruntime", "metric": "accuracy",
                                "value": 0.9, "n": 10, "matches_training": True, "reason": None}],
              "artifacts": [{"path": "artifacts/model.onnx", "format": "onnx",
                             "sha256": "ff" * 32, "bytes": 123}],
              "metrics": {"accuracy": 0.91}, "train_seconds": 12.5}
    r = recorder.NotebookRecorder()
    assert "onnxruntime" in r.eval_table(report)[0]["data"]["text/html"]
    assert ("ff" * 32) in r.artifact_table(report)[0]["data"]["text/html"]
    assert "12.5" in r.training_summary(report)[0]["data"]["text/plain"]


def test_tables_escape_html_and_tolerate_a_bare_report():
    r = recorder.NotebookRecorder()
    report = {"artifacts": [{"path": "<b>x</b>", "format": "onnx", "sha256": "", "bytes": 1}]}
    assert "&lt;b&gt;" in r.artifact_table(report)[0]["data"]["text/html"]
    assert r.eval_table({})[0]["output_type"] == "stream"
    assert "train_seconds" in r.training_summary({})[0]["data"]["text/plain"]
    assert r.artifact_table({})[0]["output_type"] == "display_data"


def test_hyperparameters_table_lists_every_resolved_value():
    resolved = ResolvedSpec(spec=None, hyperparameters={"lr": 0.001, "layers": [64, 32]}, backend="x")  # type: ignore[arg-type]
    html = recorder.NotebookRecorder().hyperparameters(resolved)[0]["data"]["text/html"]
    assert "lr" in html and "0.001" in html and "[64, 32]" in html


def test_data_summary_covers_both_splits_labels_and_manifest(tmp_path):
    data = _data(tmp_path, pd.DataFrame({"x": [1, 2]}), None, labels=["cat", "dog"])
    outs = recorder.NotebookRecorder().data_summary(data, spec=None)
    text = outs[0]["text"]
    assert "train: 2 rows x 1 columns" in text and "no validation split" in text
    assert "cat" in text and ("ab" * 32) in text


def test_manifest_view_reads_labels_and_preprocess(tmp_path):
    rd = RunDir(tmp_path).ensure()
    (rd.artifacts / "labels.json").write_text('["cat", "dog"]')
    outs = recorder.NotebookRecorder().manifest_view(rd)
    assert any("cat" in o.get("text", "") for o in outs)
    assert any("preprocess.json: not produced" in o.get("text", "") for o in outs)


_EVALUATION = {
    "splits": {"train": {"n": 8, "accuracy": 0.9}, "validation": {"n": 4, "accuracy": 0.75, "loss": 0.5123456}},
    "classes": ["a", "b"],
    "label_distribution": {"validation": {"a": 2, "b": 2}},
    "majority_baseline": {"validation": 0.5},
    "confusion_matrix": {"split": "validation", "labels": ["a", "b"], "matrix": [[1, 1], [0, 2]]},
    "per_class": [{"label": "a", "precision": 1.0, "recall": 0.5, "f1": 0.6667, "support": 2}],
    "misclassified": [{"inputs": {"text": "<hi>"}, "true": "a", "predicted": "b", "confidence": 0.7}],
    "warnings": ["validation split has only 4 rows"],
}


def test_evaluation_view_shows_tables_warnings_and_a_confusion_figure():
    outs = recorder.NotebookRecorder().evaluation_view({"evaluation": _EVALUATION})
    html_text = "".join(o["data"]["text/html"] for o in outs if o["output_type"] == "display_data" and "text/html" in o["data"])
    stream = "".join(o["text"] for o in outs if o["output_type"] == "stream")
    assert "f1_macro" not in html_text and "0.5123" in html_text and "&lt;hi&gt;" in html_text
    assert "WARNING: validation split has only 4 rows" in stream and "baseline" in stream
    assert any("image/png" in o.get("data", {}) for o in outs)


def test_evaluation_view_tolerates_a_missing_or_partial_block():
    r = recorder.NotebookRecorder()
    assert "no evaluation block" in r.evaluation_view({})[0]["text"]
    outs = r.evaluation_view({"evaluation": {"splits": {"validation": {"n": 3, "rmse": 1.5}}}})
    assert "rmse" in outs[0]["data"]["text/html"] and len(outs) == 1


def test_eval_table_prints_the_onnx_provider_and_warns_on_cpu():
    report = {"onnx_provider": "CPUExecutionProvider",
              "runtime_eval": [{"format": "onnx", "runtime": "onnxruntime", "provider": "CPUExecutionProvider",
                                "metric": "accuracy", "value": 0.9, "n": 10, "matches_training": True, "reason": None}]}
    outs = recorder.NotebookRecorder().eval_table(report)
    text = "".join(o.get("text", "") for o in outs)
    assert "CPUExecutionProvider" in text and "falls back to CPU" in text
