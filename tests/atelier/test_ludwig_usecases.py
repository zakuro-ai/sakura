"""Ludwig use cases beyond classification: time-series forecasting and text classification.

Spec, registry, data and config compilation are checked everywhere (they import no Ludwig); the engine runs need
Ludwig >= 0.17 and onnxruntime (the backend's own venv) and are skipped without them."""
from __future__ import annotations

import json
from importlib import metadata

import numpy as np
import pandas as pd
import pytest
import yaml

from sakura.atelier.backends import backend_name
from sakura.atelier.backends.ludwig import compile_config
from sakura.atelier.data import materialise
from sakura.atelier.registry import list_presets, load_models, resolve
from sakura.atelier.spec import load_spec


def _series_csv(path, n=1000):
    t = np.arange(n, dtype=np.float32)
    y = np.sin(2 * np.pi * t / 24) + 0.3 * np.sin(2 * np.pi * t / 7) + 0.05 * np.random.default_rng(0).standard_normal(n)
    pd.DataFrame({"y": y.astype("float32")}).to_csv(path, index=False)
    return path


def _ts_spec(csv, **over):
    spec = {
        "atelier": 1, "name": "ts", "task": "time_series_forecasting",
        "data": {"uri": str(csv), "format": "csv", "split": {"validation_fraction": 0.2}},
        "features": {"inputs": [{"name": "y", "type": "timeseries"}], "output": {"name": "y", "type": "timeseries"}},
        "model": "ludwig_forecaster", "hp": "time_series_forecasting/ecd@1",
        "overrides": {"window_size": 24, "horizon": 6, "epochs": 6, "batch_size": 64}, "seed": 3, "export": ["onnx"],
    }
    spec.update(over)
    return yaml.safe_dump(spec)


def _text_csv(path, n=240):
    rng = np.random.default_rng(1)
    filler = ["the", "a", "movie", "was", "really", "film", "story", "plot", "acting", "and"]
    rows = []
    for i in range(n):
        pos = i % 2 == 0
        words = list(rng.choice(filler, size=6)) + [rng.choice(["great", "good", "love"] if pos else ["awful", "bad", "hate"])]
        rng.shuffle(words)
        rows.append({"review": " ".join(words), "sentiment": "pos" if pos else "neg"})
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def _text_spec(csv, **over):
    spec = {
        "atelier": 1, "name": "txt", "task": "text_classification",
        "data": {"uri": str(csv), "format": "csv", "split": {"validation_fraction": 0.25}},
        "features": {"inputs": [{"name": "review", "type": "text"}], "output": {"name": "sentiment", "type": "category"}},
        "model": "ludwig_text", "hp": "text_classification/ecd@1", "overrides": {"epochs": 8, "batch_size": 32},
        "seed": 5, "export": ["onnx"],
    }
    spec.update(over)
    return yaml.safe_dump(spec)


# ------------------------------------------------------------------ registry / spec

def test_new_tasks_route_to_ludwig_and_have_presets():
    assert backend_name("time_series_forecasting", "ludwig_forecaster") == "ludwig"
    assert backend_name("text_classification", "ludwig_text") == "ludwig"
    models = load_models()
    assert models["ludwig_forecaster"]["exports"] == ["onnx"] and models["ludwig_text"]["exports"] == ["onnx"]
    assert "time_series_forecasting/ecd@1" in list_presets("time_series_forecasting")
    assert "text_classification/ecd@1" in list_presets("text_classification")


def test_forecasting_spec_resolves(tmp_path):
    resolved = resolve(load_spec(_ts_spec(_series_csv(tmp_path / "s.csv"))))
    assert resolved.backend == "ludwig"
    assert resolved.hyperparameters["window_size"] == 24 and resolved.hyperparameters["encoder"] == "rnn"


def test_a_forecasting_model_cannot_serve_another_task(tmp_path):
    with pytest.raises(ValueError, match="does not support task"):
        resolve(load_spec(_ts_spec(_series_csv(tmp_path / "s.csv"), task="tabular_regression",
                                    hp="tabular/ecd@1")))


# ------------------------------------------------------------------------ compilation

def test_forecasting_compiles_to_windowed_timeseries_features(tmp_path):
    cfg = compile_config(resolve(load_spec(_ts_spec(_series_csv(tmp_path / "s.csv")))))
    (i,), (o,) = cfg["input_features"], cfg["output_features"]
    assert (i["type"], i["column"], i["preprocessing"]["window_size"]) == ("timeseries", "y", 24)
    assert (o["type"], o["column"], o["preprocessing"]["horizon"], o["horizon"]) == ("timeseries", "y", 6, 6)
    assert i["name"] != o["name"]  # Ludwig needs unique feature names; both read the one column


def test_forecasting_encoder_and_backend_config_are_honoured(tmp_path):
    spec = yaml.safe_load(_ts_spec(_series_csv(tmp_path / "s.csv")))
    spec["overrides"]["encoder"] = "transformer"
    spec["backend_config"] = {"trainer": {"epochs": 2}}
    cfg = compile_config(resolve(load_spec(yaml.safe_dump(spec))))
    assert cfg["input_features"][0]["encoder"] == {"type": "transformer"}
    assert cfg["trainer"]["epochs"] == 2  # backend_config is merged last


@pytest.mark.parametrize("encoder", ["patchtst", "nbeats", "stacked_cnn", "nonsense"])
def test_encoders_that_cannot_be_exported_are_rejected_before_training(tmp_path, encoder):
    spec = yaml.safe_load(_ts_spec(_series_csv(tmp_path / "s.csv")))
    spec["overrides"]["encoder"] = encoder
    with pytest.raises(ValueError, match="not supported"):
        compile_config(resolve(load_spec(yaml.safe_dump(spec))))


def test_forecasting_takes_exactly_one_series(tmp_path):
    spec = yaml.safe_load(_ts_spec(_series_csv(tmp_path / "s.csv")))
    spec["features"]["inputs"].append({"name": "other", "type": "number"})
    with pytest.raises(ValueError, match="exactly one series"):
        compile_config(resolve(load_spec(yaml.safe_dump(spec))))


def test_text_classification_compiles_with_and_without_a_pretrained_encoder(tmp_path):
    csv = _text_csv(tmp_path / "t.csv")
    plain = compile_config(resolve(load_spec(_text_spec(csv))))
    assert plain["input_features"][0] == {"name": "review", "type": "text", "encoder": {"type": "parallel_cnn"}}
    spec = yaml.safe_load(_text_spec(csv))
    spec["overrides"]["pretrained_model"] = "distilbert-base-uncased"
    enc = compile_config(resolve(load_spec(yaml.safe_dump(spec))))["input_features"][0]["encoder"]
    assert enc["type"] == "auto_transformer" and enc["pretrained_model_name_or_path"] == "distilbert-base-uncased"


# ------------------------------------------------------------------------------ data

def test_forecasting_validation_fraction_is_chronological_not_random(tmp_path):
    csv = _series_csv(tmp_path / "s.csv", n=500)
    data = materialise(load_spec(_ts_spec(csv)), tmp_path / "d")
    full = pd.read_csv(csv)
    assert len(data.train) == 400 and len(data.validation) == 100
    assert data.train["y"].tolist() == full["y"].iloc[:400].tolist()  # a prefix, in order
    assert data.validation["y"].tolist() == full["y"].iloc[400:].tolist()  # the tail


def test_forecasting_explicit_row_ranges(tmp_path):
    csv = _series_csv(tmp_path / "s.csv", n=500)
    spec = yaml.safe_load(_ts_spec(csv))
    spec["data"]["split"] = {"train": "0:300", "validation": "300:500"}
    data = materialise(load_spec(yaml.safe_dump(spec)), tmp_path / "d")
    assert (len(data.train), len(data.validation)) == (300, 200)


def test_the_series_column_is_selected_once(tmp_path):
    data = materialise(load_spec(_ts_spec(_series_csv(tmp_path / "s.csv"))), tmp_path / "d")
    assert list(data.train.columns) == ["y"]  # one column although the spec names it as input AND output


# ----------------------------------------------------------------------- engine runs

def _ludwig_ready(min_version=(0, 17)):
    pytest.importorskip("ludwig")
    pytest.importorskip("onnxruntime")
    version = tuple(int(x) for x in metadata.version("ludwig").split(".")[:2])
    if version < min_version:
        pytest.skip(f"needs ludwig >= {'.'.join(map(str, min_version))}, found {metadata.version('ludwig')}")


def _run(tmp_path, text):
    """One engine run in its OWN interpreter, like production: several trainings sharing one pytest process can
    crash in the native libraries (ONNX export / onnxruntime / cuDNN), which says nothing about a real run."""
    import subprocess
    import sys

    spec = tmp_path / "spec.yaml"
    spec.write_text(text, encoding="utf-8")
    out = tmp_path / "run"
    proc = subprocess.run([sys.executable, "-W", "ignore", "-m", "sakura.atelier", "run", str(spec), "--out", str(out),
                           "--device", "cpu"], capture_output=True, text=True)
    report_path = out / "report.json"
    report = json.loads(report_path.read_text()) if report_path.exists() else {}
    assert proc.returncode >= 0, f"the engine process was killed by signal {-proc.returncode}: {proc.stderr[-400:]}"
    return proc.returncode, out, report


def test_forecasting_run_trains_exports_onnx_and_rescores(tmp_path):
    _ludwig_ready()
    rc, out, report = _run(tmp_path, _ts_spec(_series_csv(tmp_path / "s.csv")))
    assert rc == 0 and report["status"] == "done" and report["backend"] == "ludwig"
    assert report["metrics"]["rmse"] > 0
    ev = report["runtime_eval"][0]
    assert ev["format"] == "onnx" and ev["metric"] == "rmse" and ev["value"] is not None
    assert ev["matches_training"] is True, ev  # the ONNX graph agrees with Ludwig's own evaluation
    names = {m["name"] for m in map(json.loads, (out / "metrics.jsonl").read_text().splitlines())}
    assert "rmse" in names


def test_forecasting_predict_returns_the_horizon(tmp_path):
    _ludwig_ready()
    rc, out, report = _run(tmp_path, _ts_spec(_series_csv(tmp_path / "s.csv")))
    assert rc == 0
    from sakura.atelier.backends import load_backend
    from sakura.atelier.types import Artifact

    path = out / "artifacts" / "model.onnx"
    art = Artifact(path=path, format="onnx", sha256="", bytes=path.stat().st_size)
    recent = pd.read_csv(tmp_path / "s.csv")["y"].tolist()[-60:]
    res = load_backend("ludwig").predict(art, {"series": recent, "horizon": 6})
    assert res["horizon"] == 6 and len(res["forecast"]) == 6 and all(np.isfinite(res["forecast"]))
    with pytest.raises(ValueError, match="series"):
        load_backend("ludwig").predict(art, {"horizon": 6})


def test_text_classification_run_learns_and_exports_onnx(tmp_path):
    _ludwig_ready(min_version=(0, 11))
    rc, out, report = _run(tmp_path, _text_spec(_text_csv(tmp_path / "t.csv")))
    assert rc == 0 and report["status"] == "done"
    assert report["metrics"]["accuracy"] >= 0.8
    ev = report["runtime_eval"][0]
    assert ev["metric"] == "accuracy" and ev["value"] is not None and ev["matches_training"] is True, ev


def test_text_classification_run_reports_the_evaluation_block_and_the_onnx_provider(tmp_path):
    _ludwig_ready()
    rc, out, report = _run(tmp_path, _text_spec(_text_csv(tmp_path / "t.csv")))
    assert rc == 0 and report["status"] == "done"
    assert report["onnx_provider"] == report["runtime_eval"][0]["provider"]
    assert report["onnx_provider"] in {"CPUExecutionProvider", "CUDAExecutionProvider"}
    ev = report["evaluation"]
    assert set(ev["classes"]) == {"pos", "neg"}
    assert ev["splits"]["train"]["n"] + ev["splits"]["validation"]["n"] == 240
    assert {"accuracy", "f1_macro", "roc_auc", "loss"} <= set(ev["splits"]["validation"])
    assert sum(sum(row) for row in ev["confusion_matrix"]["matrix"]) == ev["splits"]["validation"]["n"]
    assert [c["label"] for c in ev["per_class"]] == ev["classes"]
    assert all(set(m) == {"inputs", "true", "predicted", "confidence"} for m in ev["misclassified"])
    assert any("validation_fraction" in w for w in ev["warnings"])
