"""classic_ml: LightGBM, XGBoost, CatBoost and scikit-learn's random forest / histogram gradient boosting
on tables (tabular_classification, tabular_regression).

One preprocessing step for every estimator: a ``ColumnTransformer`` that median-imputes the numeric columns
and one-hot encodes the categorical ones (unknown categories at predict time encode as all zeros). The
exported ONNX is that transformer's graph merged in front of the estimator's, so it takes the RAW columns --
one input per feature column, float for numbers, string for categories -- and nothing outside the file is
needed to score a row. Classes are encoded 0..K-1 in ``data.labels`` order and shipped as ``classes.json``.

``joblib`` is the fitted scikit-learn pipeline, pickled: load it only from a run you trust (it is this run's
own output). The training stack is imported lazily (CONTRACTS §2: importing the engine imports none of it).
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any

from sakura.atelier.types import Artifact, Checkpoint, MaterialisedData, MetricsSink, ResolvedSpec, RunDir, RuntimeEval, TrainResult
from sakura.atelier.wrap_up import max_steps_of

LICENSES = {
    "lightgbm": "MIT", "xgboost": "Apache-2.0", "catboost": "Apache-2.0",
    "random_forest": "BSD-3-Clause", "hist_gradient_boosting": "BSD-3-Clause",
}
MODELS = tuple(LICENSES)
_NUMERIC = frozenset({"number", "binary"})
_CATEGORICAL = frozenset({"category"})
_EVAL_CAP = 100_000
_ONNX_OPSET = {"": 17, "ai.onnx.ml": 3}
_ML_TOOLS_OPSET = 15  # onnxmltools 1.16's highest
_LOADED: dict[str, Any] = {}


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _columns(resolved: ResolvedSpec) -> tuple[list[str], list[str], str]:
    features = resolved.spec.features
    if features is None:
        raise ValueError("classic_ml: the spec names no features (input and output columns)")
    num: list[str] = []
    cat: list[str] = []
    for f in features.inputs:
        if f.type in _NUMERIC:
            num.append(f.name)
        elif f.type in _CATEGORICAL:
            cat.append(f.name)
        else:
            raise ValueError(f"classic_ml takes number, binary and category inputs; {f.name!r} is {f.type!r}")
    return num, cat, features.output.name


def _frame(rows: Any, num: list[str], cat: list[str]) -> Any:
    """The model's input frame: numbers as float (missing = NaN, imputed in the graph), categories as str
    (missing = "")."""
    import pandas as pd

    df = pd.DataFrame(rows if not isinstance(rows, dict) else [rows])
    out = pd.DataFrame(index=range(len(df)))
    for c in num:
        out[c] = pd.to_numeric(df[c], errors="coerce").astype("float32").to_numpy() if c in df else float("nan")
    for c in cat:
        out[c] = df[c].astype(object).where(df[c].notna(), "").astype(str).to_numpy() if c in df else ""
    return out


def _preprocessor(num: list[str], cat: list[str]) -> Any:
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import OneHotEncoder

    parts = []
    if num:
        parts.append(("num", SimpleImputer(strategy="median"), num))
    if cat:
        parts.append(("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat))
    return ColumnTransformer(parts, sparse_threshold=0.0)


def _estimator(model: str, classification: bool, hp: dict[str, Any], seed: int) -> Any:
    n = int(hp.get("n_estimators", 300))
    lr = float(hp.get("learning_rate", 0.05))
    extra = dict(hp.get("params") or {})
    # hp.threads, else OMP_NUM_THREADS, else the library's own default (all cores): an unset thread count on a
    # shared, loaded host makes LightGBM's OpenMP pool spin (40x slower measured on 16 cores at load 200+).
    threads = hp.get("threads") or int(os.environ.get("OMP_NUM_THREADS") or 0) or None
    if threads is not None:
        threads = int(threads)
    if model == "lightgbm":
        import lightgbm as lgb

        cls = lgb.LGBMClassifier if classification else lgb.LGBMRegressor
        return cls(n_estimators=n, learning_rate=lr, num_leaves=int(hp.get("num_leaves", 31)),
                   random_state=seed, verbose=-1, n_jobs=threads if threads else -1, **extra)
    if model == "xgboost":
        import xgboost as xgb

        cls = xgb.XGBClassifier if classification else xgb.XGBRegressor
        return cls(n_estimators=n, learning_rate=lr, max_depth=int(hp.get("max_depth", 6)),
                   random_state=seed, tree_method="hist", n_jobs=threads, **extra)
    if model == "catboost":
        import catboost as cb

        cls = cb.CatBoostClassifier if classification else cb.CatBoostRegressor
        return cls(iterations=n, learning_rate=lr, depth=int(hp.get("max_depth", 6)), random_seed=seed,
                   verbose=0, allow_writing_files=False, thread_count=threads or -1, **extra)
    if model == "random_forest":
        from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

        cls = RandomForestClassifier if classification else RandomForestRegressor
        depth = hp.get("max_depth")
        return cls(n_estimators=n, max_depth=None if depth is None else int(depth), random_state=seed,
                   n_jobs=threads, **extra)
    if model == "hist_gradient_boosting":
        from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor

        cls = HistGradientBoostingClassifier if classification else HistGradientBoostingRegressor
        return cls(max_iter=n, learning_rate=lr, random_state=seed, **extra)
    raise ValueError(f"classic_ml has no model {model!r} (one of {', '.join(MODELS)})")


def _score(classification: bool, y_true: Any, y_pred: Any) -> float:
    from sklearn.metrics import accuracy_score, r2_score

    return float(accuracy_score(y_true, y_pred) if classification else r2_score(y_true, y_pred))


class _Fitted:
    """What train() hands export(): the pipeline and what it was trained on."""

    def __init__(self, model: str, pipeline: Any, num: list[str], cat: list[str], target: str,
                 classes: list[str] | None, width: int) -> None:
        self.model, self.pipeline, self.num, self.cat = model, pipeline, num, cat
        self.target, self.classes, self.width = target, classes, width

    @property
    def classification(self) -> bool:
        return self.classes is not None

    def meta(self) -> dict[str, Any]:
        return {"model": self.model, "numeric": self.num, "categorical": self.cat, "target": self.target,
                "classes": self.classes}


class ClassicMLBackend:
    name = "classic_ml"
    tasks = frozenset({"tabular_classification", "tabular_regression"})

    def train(self, spec: ResolvedSpec, data: MaterialisedData, out: RunDir, metrics: MetricsSink,
              resume: Checkpoint | None) -> TrainResult:
        import numpy as np
        from sklearn.pipeline import Pipeline

        if resume is not None:
            raise ValueError("classic_ml models train in one pass; there is no checkpoint to resume from")
        out.ensure()
        model = spec.spec.model
        classification = spec.spec.task == "tabular_classification"
        num, cat, target = _columns(spec)
        X_train = _frame(data.train, num, cat)
        classes = None
        if classification:
            classes = [str(c) for c in (data.labels or sorted({str(v) for v in data.train[target]}))]
            index = {c: i for i, c in enumerate(classes)}
            y_train = np.array([index[str(v)] for v in data.train[target]])
        else:
            y_train = data.train[target].astype("float64").to_numpy()
        # limits.max_steps: these estimators have no step loop, their length is the number of boosting
        # rounds / trees (n_estimators, CatBoost's iterations, HistGradientBoosting's max_iter).
        hp, cap, limits_applied = spec.hyperparameters, max_steps_of(spec), None
        if cap is not None and cap < int(hp.get("n_estimators", 300)):
            hp, limits_applied = {**hp, "n_estimators": cap}, {"max_steps": cap}
        pipeline = Pipeline([("pre", _preprocessor(num, cat)),
                             ("est", _estimator(model, classification, hp, spec.spec.seed))])
        t0 = time.time()
        pipeline.fit(X_train, y_train)
        train_seconds = time.time() - t0
        width = int(pipeline.named_steps["pre"].transform(X_train.iloc[:1]).shape[1])  # the estimator's input
        fitted = _Fitted(model, pipeline, num, cat, target, classes, width)

        name = "accuracy" if classification else "r2"
        scores: dict[str, float] = {}
        for split, frame in (("train", data.train), ("validation", data.validation)):
            if frame is None or len(frame) == 0:
                continue
            y_true, y_pred = _truth_and_prediction(fitted, frame)
            scores[split] = _score(classification, y_true, y_pred)
            metrics.log(name, scores[split], step=1, split=split)
        final = {name: scores.get("validation", scores.get("train", float("nan")))}

        ckpt = out.checkpoint
        ckpt.mkdir(parents=True, exist_ok=True)
        (ckpt / "classic_ml.json").write_text(json.dumps(fitted.meta()))
        return TrainResult(model=fitted, metrics=final, train_seconds=train_seconds,
                           checkpoint=Checkpoint(path=ckpt, model=model, task=spec.spec.task),
                           provenance={"estimator": model, "license": LICENSES[model]},
                           limits_applied=limits_applied,
                           epoch_metrics=[{"epoch": 1, **{f"{s}_{name}": v for s, v in scores.items()}}])

    def export(self, result: TrainResult, fmt: str, out: RunDir) -> Artifact:
        fitted: _Fitted = result.model
        out.artifacts.mkdir(parents=True, exist_ok=True)
        (out.artifacts / "classic_ml.json").write_text(json.dumps(fitted.meta()))
        if fmt == "joblib":
            import joblib

            path = out.artifacts / "model.joblib"
            joblib.dump(fitted.pipeline, path)
        elif fmt == "onnx":
            import onnx

            path = out.artifacts / "model.onnx"
            onnx.save(_to_onnx(fitted), str(path))
        else:
            raise ValueError(f"classic_ml exports onnx and joblib, not {fmt!r}")
        return Artifact(path=path, format=fmt, sha256=_sha256(path), bytes=path.stat().st_size)

    def runtime_eval(self, artifact: Artifact, data: MaterialisedData) -> RuntimeEval:
        meta = json.loads((artifact.path.parent / "classic_ml.json").read_text())
        classification = meta["classes"] is not None
        name = "accuracy" if classification else "r2"
        frame = data.validation if data.validation is not None and len(data.validation) else data.train
        frame = frame.iloc[:_EVAL_CAP]
        if artifact.format != "onnx":
            return RuntimeEval(format=artifact.format, runtime="none", metric=name, value=None, n=0,
                               matches_training=None, reason=f"{artifact.format} is the training object itself")
        got = _run(artifact, meta, frame)
        if classification:
            index = {c: i for i, c in enumerate(meta["classes"])}
            y_true = [index.get(str(v), -1) for v in frame[meta["target"]]]
            y_pred = [int(i) for i in got["index"]]
        else:
            y_true = frame[meta["target"]].astype("float64").tolist()
            y_pred = got["value"]
        return RuntimeEval(format="onnx", runtime="onnxruntime", metric=name, value=_score(classification, y_true, y_pred),
                           n=len(frame), matches_training=None, reason=None)

    def predict(self, artifact: Artifact, inputs: Any) -> Any:
        meta = json.loads((artifact.path.parent / "classic_ml.json").read_text())
        rows = inputs if isinstance(inputs, list) else [inputs]
        if artifact.format == "joblib":
            import joblib

            pipeline = _LOADED.setdefault(str(artifact.path), joblib.load(artifact.path))
            X = _frame(rows, meta["numeric"], meta["categorical"])
            if meta["classes"] is None:
                return [{"value": float(v)} for v in pipeline.predict(X)]
            proba = pipeline.predict_proba(X)
            return [_answer(meta["classes"], p) for p in proba]
        got = _run(artifact, meta, rows)
        if meta["classes"] is None:
            return [{"value": float(v)} for v in got["value"]]
        return [_answer(meta["classes"], p) for p in got["proba"]]


def _answer(classes: list[str], proba: Any) -> dict[str, Any]:
    probs = {c: float(p) for c, p in zip(classes, proba)}
    return {"label": max(probs, key=lambda c: probs[c]), "probs": probs}


def _truth_and_prediction(fitted: _Fitted, frame: Any) -> tuple[list[Any], list[Any]]:
    X = _frame(frame, fitted.num, fitted.cat)
    import numpy as np

    # one value per row: CatBoost's multiclass predict returns a column (shape (n, 1)), every other estimator a vector
    pred = np.asarray(fitted.pipeline.predict(X)).reshape(-1)
    if fitted.classification:
        index = {c: i for i, c in enumerate(fitted.classes or [])}
        return [index.get(str(v), -1) for v in frame[fitted.target]], [int(p) for p in pred]
    return frame[fitted.target].astype("float64").tolist(), [float(p) for p in pred]


# ---------------------------------------------------------------------------------------------------- onnx

def _estimator_onnx(fitted: _Fitted, width: int) -> Any:
    """The estimator alone, on one float input ``features`` of ``width`` columns."""
    import onnx
    from skl2onnx.common.data_types import FloatTensorType

    est = fitted.pipeline.named_steps["est"]
    types = [("features", FloatTensorType([None, width]))]
    if fitted.model in ("lightgbm", "xgboost"):
        import onnxmltools
        from onnxmltools.convert.common.data_types import FloatTensorType as MLFloat

        # onnxmltools takes its own tensor types and converts up to opset 15 (the preprocessor follows it)
        ml_types = [("features", MLFloat([None, width]))]
        if fitted.model == "lightgbm":
            return onnxmltools.convert_lightgbm(est, initial_types=ml_types, target_opset=_ML_TOOLS_OPSET, zipmap=False)
        return onnxmltools.convert_xgboost(est, initial_types=ml_types, target_opset=_ML_TOOLS_OPSET)
    if fitted.model == "catboost":
        import tempfile

        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "m.onnx"
            est.save_model(str(p), format="onnx")
            return onnx.load(str(p))
    from skl2onnx import to_onnx

    options = {id(est): {"zipmap": False}} if fitted.classification else None
    if fitted.model != "hist_gradient_boosting":
        return to_onnx(est, initial_types=types, target_opset=dict(_ONNX_OPSET), options=options)
    # skl2onnx 1.20 passes a Python bool for a leaf's nodes_missing_value_tracks_true, which onnx >= 1.17
    # refuses in an INTS attribute; coerce it to int for this one conversion and put the function back.
    from skl2onnx.common import tree_ensemble

    original = tree_ensemble.add_node

    def add_node(*args: Any, **kwargs: Any) -> Any:
        if "nodes_missing_value_tracks_true" in kwargs:
            kwargs["nodes_missing_value_tracks_true"] = int(kwargs["nodes_missing_value_tracks_true"])
        return original(*args, **kwargs)

    tree_ensemble.add_node = add_node
    try:
        return to_onnx(est, initial_types=types, target_opset=dict(_ONNX_OPSET), options=options)
    finally:
        tree_ensemble.add_node = original


def _to_onnx(fitted: _Fitted) -> Any:
    """Preprocessor graph (raw columns -> float matrix) merged in front of the estimator's graph."""
    import onnx
    from onnx import compose, helper
    from skl2onnx import to_onnx
    from skl2onnx.common.data_types import FloatTensorType, StringTensorType

    pre = fitted.pipeline.named_steps["pre"]
    est_model = _estimator_onnx(fitted, fitted.width)
    opsets = {o.domain: o.version for o in est_model.opset_import}
    opsets.setdefault("", _ONNX_OPSET[""])
    opsets.setdefault("ai.onnx.ml", _ONNX_OPSET["ai.onnx.ml"])
    types = [(c, FloatTensorType([None, 1])) for c in fitted.num] + [(c, StringTensorType([None, 1])) for c in fitted.cat]
    pre_model = to_onnx(pre, initial_types=types, target_opset={"": opsets[""], "ai.onnx.ml": opsets["ai.onnx.ml"]})
    # Both graphs import exactly the same opsets, or merge_models refuses them.
    for m in (pre_model, est_model):
        have = {o.domain for o in m.opset_import}
        for domain, version in opsets.items():
            if domain not in have:
                m.opset_import.append(helper.make_opsetid(domain, version))
        for o in m.opset_import:
            o.version = opsets[o.domain]
    est_model.ir_version = pre_model.ir_version = max(pre_model.ir_version, est_model.ir_version)
    est_in = est_model.graph.input[0].name
    merged = compose.merge_models(pre_model, est_model, io_map=[(pre_model.graph.output[0].name, est_in)],
                                  prefix2="est_")
    onnx.checker.check_model(merged)
    return merged


def _session(artifact: Artifact) -> Any:
    from sakura.atelier.ort import inference_session

    key = str(artifact.path)
    if key not in _LOADED:
        _LOADED[key] = inference_session(artifact.path, providers=["CPUExecutionProvider"])
    return _LOADED[key]


def _run(artifact: Artifact, meta: dict[str, Any], rows: Any) -> dict[str, Any]:
    """Score raw rows through the ONNX graph: classification -> index + proba, regression -> value."""
    import numpy as np

    session = _session(artifact)
    X = _frame(rows, meta["numeric"], meta["categorical"])
    feed = {c: X[c].to_numpy(dtype=np.float32).reshape(-1, 1) for c in meta["numeric"]}
    feed |= {c: X[c].to_numpy(dtype=object).astype(str).reshape(-1, 1) for c in meta["categorical"]}
    names = [o.name for o in session.get_outputs()]
    outs = dict(zip(names, session.run(names, feed)))
    if meta["classes"] is None:
        (value,) = [v for v in outs.values()]
        return {"value": [float(v) for v in np.asarray(value).reshape(-1)]}
    proba = next(v for k, v in outs.items() if "prob" in k.lower())
    if isinstance(proba, list):  # a ZipMap output (CatBoost): one {class_index: p} per row
        proba = [[row[k] for k in sorted(row)] for row in proba]
    proba = np.asarray(proba, dtype=np.float64)
    return {"proba": proba.tolist(), "index": proba.argmax(axis=1).tolist()}


# ------------------------------------------------------------------------------------- evaluation block

def predictions(result: TrainResult, resolved: ResolvedSpec | None, data: MaterialisedData) -> Any:
    """``evaluation.build_evaluation``'s hook: each split scored by the trained pipeline."""
    from sakura.atelier.evaluation import ModelPredictions, SplitPredictions

    fitted: _Fitted = result.model
    splits = {}
    for split, frame in (("train", data.train), ("validation", data.validation)):
        if frame is None or len(frame) == 0:
            continue
        X = _frame(frame, fitted.num, fitted.cat)
        inputs = frame[fitted.num + fitted.cat].to_dict("records")
        if fitted.classification:
            classes = fitted.classes or []
            proba = fitted.pipeline.predict_proba(X)
            splits[split] = SplitPredictions(
                y_true=[str(v) for v in frame[fitted.target]],
                y_pred=[classes[int(i)] for i in proba.argmax(axis=1)],
                proba=proba.tolist(), inputs=inputs, row_ids=list(frame.index))
        else:
            splits[split] = SplitPredictions(y_true=frame[fitted.target].astype("float64").tolist(),
                                             y_pred=[float(v) for v in fitted.pipeline.predict(X)], inputs=inputs,
                                             row_ids=list(frame.index))
    return ModelPredictions(kind="classification" if fitted.classification else "regression", splits=splits,
                            classes=fitted.classes)


BACKEND = ClassicMLBackend()
