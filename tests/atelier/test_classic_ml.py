"""classic_ml backend (plan Task 5): LightGBM, XGBoost, CatBoost and two scikit-learn estimators behind one
ColumnTransformer; the exported ONNX takes the raw columns and re-scores like the trained model."""

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("sklearn")
pytest.importorskip("skl2onnx")
pytest.importorskip("onnxruntime")

from sakura.atelier.backends import backend_name, load_backend  # noqa: E402
from sakura.atelier.registry import resolve  # noqa: E402
from sakura.atelier.spec import load_spec  # noqa: E402
from sakura.atelier.types import MaterialisedData, MetricsSink, RunDir  # noqa: E402

IDS = ["lightgbm", "xgboost", "catboost", "random_forest", "hist_gradient_boosting"]

SPEC = """
atelier: 1
name: t
task: {task}
data: {{uri: "file:///tmp/none", format: csv, split: {{validation_fraction: 0.25}}}}
features:
  inputs: [{{name: size, type: number}}, {{name: rooms, type: number}}, {{name: city, type: category}}]
  output: {{name: y, type: {otype}}}
model: {model}
hp: tabular/{model}@1
export: [onnx, joblib]
"""


def _table(task, n=200, seed=0):
    rng = np.random.default_rng(seed)
    size = rng.uniform(20, 200, n)
    rooms = rng.integers(1, 6, n).astype(float)
    city = rng.choice(["paris", "lyon", "nice"], n)
    size[::17] = np.nan  # missing values are imputed inside the exported graph
    value = np.nan_to_num(size, nan=100.0) * 3 + rooms * 10 + np.where(city == "paris", 200, 0)
    y = value if task == "tabular_regression" else np.where(value > np.median(value), "high", "low")
    return pd.DataFrame({"size": size, "rooms": rooms, "city": city, "y": y})


def _run(model, task, tmp_path):
    otype = "number" if task == "tabular_regression" else "category"
    r = resolve(load_spec(SPEC.format(task=task, otype=otype, model=model)))
    backend = load_backend(r.backend)
    df = _table(task)
    labels = None if task == "tabular_regression" else ["high", "low"]
    data = MaterialisedData(root=tmp_path, train=df.iloc[:150], validation=df.iloc[150:], manifest_sha256="x",
                            labels=labels)
    out = RunDir(tmp_path / "out").ensure()
    with MetricsSink(out.metrics_path) as m:
        result = backend.train(r, data, out, m, None)
    return backend, result, data, out


@pytest.mark.parametrize("model", IDS)
def test_each_id_routes_to_classic_ml(model):
    assert backend_name("tabular_classification", model) == "classic_ml"
    assert backend_name("tabular_regression", model) == "classic_ml"


@pytest.mark.parametrize("model", IDS)
@pytest.mark.parametrize("task", ["tabular_classification", "tabular_regression"])
def test_each_estimator_trains_and_onnx_matches(model, task, tmp_path):
    pytest.importorskip({"lightgbm": "lightgbm", "xgboost": "xgboost", "catboost": "catboost"}.get(model, "sklearn"))
    backend, result, data, out = _run(model, task, tmp_path)
    metric = "r2" if task == "tabular_regression" else "accuracy"
    assert metric in result.metrics
    art = backend.export(result, "onnx", out)
    ev = backend.runtime_eval(art, data)
    assert (ev.runtime, ev.metric, ev.n) == ("onnxruntime", metric, 50)
    assert ev.value == pytest.approx(result.metrics[metric], abs=1e-3)  # the graph scores like the model
    pred = backend.predict(art, {"size": 80.0, "rooms": 3.0, "city": "paris"})
    assert len(pred) == 1 and ("label" in pred[0] if task == "tabular_classification" else "value" in pred[0])


def test_joblib_export_and_predictions_hook(tmp_path):
    from sakura.atelier.backends import classic_ml

    backend, result, data, out = _run("random_forest", "tabular_classification", tmp_path)
    art = backend.export(result, "joblib", out)
    assert art.format == "joblib" and art.path.exists()
    assert backend.predict(art, [{"size": 80.0, "rooms": 3.0, "city": "nice"}])[0]["label"] in {"high", "low"}
    p = classic_ml.predictions(result, None, data)
    assert p.kind == "classification" and p.classes == ["high", "low"]
    assert len(p.splits["validation"].y_true) == 50 and len(p.splits["validation"].proba[0]) == 2
    # a misclassified row's `validation:<n>` is the source row (CONTRACTS §2.6), which curation.exclude_samples reads
    assert p.splits["validation"].row_ids == list(range(150, 200))


def test_unknown_category_at_predict_time_does_not_crash(tmp_path):
    backend, result, data, out = _run("hist_gradient_boosting", "tabular_classification", tmp_path)
    art = backend.export(result, "onnx", out)
    assert backend.predict(art, {"size": 80.0, "rooms": 3.0, "city": "marseille"})[0]["label"] in {"high", "low"}


def test_catboost_trains_on_three_classes(tmp_path):
    """CatBoost's multiclass `predict` returns one column per row (shape (n, 1)); the scoring code must
    flatten it like every other estimator's (SAK-0019, open questions)."""
    pytest.importorskip("catboost")
    r = resolve(load_spec(SPEC.format(task="tabular_classification", otype="category", model="catboost")))
    backend = load_backend(r.backend)
    df = _table("tabular_regression")
    df["y"] = pd.qcut(df["y"], 3, labels=["low", "mid", "high"]).astype(str)
    data = MaterialisedData(root=tmp_path, train=df.iloc[:150], validation=df.iloc[150:], manifest_sha256="x",
                            labels=["high", "low", "mid"])
    out = RunDir(tmp_path / "out").ensure()
    with MetricsSink(out.metrics_path) as m:
        result = backend.train(r, data, out, m, None)
    assert result.metrics["accuracy"] > 0.8  # three balanced classes: chance is 1/3
    art = backend.export(result, "onnx", out)
    ev = backend.runtime_eval(art, data)
    assert (ev.n, ev.metric) == (50, "accuracy") and ev.value == pytest.approx(result.metrics["accuracy"], abs=1e-3)
    pred = backend.predict(art, {"size": 80.0, "rooms": 3.0, "city": "paris"})
    assert set(pred[0]["probs"]) == {"high", "low", "mid"}
    from sakura.atelier.backends.classic_ml import predictions
    assert set(predictions(result, r, data).splits["validation"].y_pred) <= {"high", "low", "mid"}
