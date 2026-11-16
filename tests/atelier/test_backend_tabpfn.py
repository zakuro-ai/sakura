"""Unit tests for sakura/atelier/backends/tabpfn.py.

Runnable in CI without tabpfn installed: the TabPFN teacher is replaced by a
small deterministic stand-in exposing the same sklearn surface
(``predict_proba`` / ``predict`` / ``classes_``), so the distillation and the
student's ONNX export run FOR REAL (torch training, torch.onnx.export,
onnxruntime re-scoring) rather than against mocks. The ONNX round trip needs
``onnx`` + ``onnxruntime``, which the CI venv does not install (they live in
the backend's own env, docs/atelier/envs/tabpfn.txt), so those tests skip
there and run in that env.

The measured teacher-vs-student numbers on real TabPFN (breast-cancer) are in
docs/atelier/tabpfn-export.md, produced by scripts/atelier/smoke_e2.py and
scripts/atelier/tabpfn_export_probe.py on the GPU host.
"""

from __future__ import annotations

import json
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from sakura.atelier.backends import tabpfn as backend_mod
from sakura.atelier.types import Artifact, RunDir

FAST = {"distill_epochs": 60, "distill_augment": 10, "distill_hidden": 32}


@pytest.fixture(autouse=True)
def _one_torch_thread():
    """Distillation trains a real MLP: CI pods are CPU-quota-limited and an
    oversubscribed torch thread pool stalls the job (#134). One thread, restored after."""
    import torch

    before = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(before)


class _Teacher:
    """Deterministic stand-in for a fitted TabPFNClassifier/Regressor: a
    fixed nonlinear function of the features, string class labels."""

    classes_ = np.array(["benign", "malignant"])

    @staticmethod
    def _score(X):
        X = np.asarray(X, dtype=np.float64)
        return 2.0 * X[:, 0] - 1.5 * X[:, 1] + np.sin(2 * X[:, 2])

    def predict_proba(self, X):
        p = 1.0 / (1.0 + np.exp(-3.0 * self._score(X)))
        return np.stack([1.0 - p, p], axis=1)

    def predict(self, X):
        return self._score(X) * 10.0 + 100.0


def _data(n=240, d=4, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d))
    y_cls = np.where(_Teacher._score(X) > 0, "malignant", "benign")
    y_reg = _Teacher().predict(X)
    return X, y_cls, y_reg


def _fitted(task="tabular_classification", X=None, y=None, seed=42, hp=FAST):
    X_, y_cls, y_reg = _data()
    X = X_ if X is None else X
    if y is None:
        y = y_cls if task == "tabular_classification" else y_reg
    classes = ["benign", "malignant"] if task == "tabular_classification" else None
    return backend_mod._Fitted(estimator=_Teacher(), task=task, X_train=X, y_train=y,
                               classes=classes, seed=seed, hyperparameters=dict(hp))


def test_backend_identity():
    b = backend_mod.BACKEND
    assert b.name == "tabpfn"
    assert b.tasks == frozenset({"tabular_classification", "tabular_regression"})


def test_export_refuses_an_unsupported_format(tmp_path):
    out = RunDir(tmp_path / "run").ensure()
    with pytest.raises(ValueError, match="only exports 'onnx'"):
        backend_mod.BACKEND.export(result=SimpleNamespace(model=None), fmt="torchscript", out=out)


def test_transfer_set_keeps_the_training_rows_and_grows_to_the_requested_size():
    X, _, _ = _data(n=50)
    Xt = backend_mod._transfer_set(X, 500, swap=0.3, noise=0.05, rng=np.random.default_rng(0))
    assert Xt.shape == (500, X.shape[1])
    np.testing.assert_array_equal(Xt[:50], X)
    # synthetic rows stay on the data's support (no blow-up from the jitter)
    assert np.all(np.abs(Xt[50:]) < np.abs(X).max() * 1.5 + 1.0)


def test_distilled_classifier_student_matches_its_teacher():
    X_val, y_val, _ = _data(n=200, seed=1)
    student = backend_mod._distil(_fitted(), X_val, y_val)
    assert student.teacher_agreement >= 0.93, student.teacher_agreement
    assert student.classes == ["benign", "malignant"]
    assert student.features == ["f0", "f1", "f2", "f3"]
    assert backend_mod._student_scores(student, X_val, y_val) >= 0.9


def test_distillation_is_deterministic_for_a_seed():
    import torch

    X_val, _, _ = _data(n=50, seed=2)
    a = backend_mod._distil(_fitted(seed=7))
    b = backend_mod._distil(_fitted(seed=7))
    with torch.no_grad():
        xa = a.module(torch.as_tensor(X_val, dtype=torch.float32))
        xb = b.module(torch.as_tensor(X_val, dtype=torch.float32))
    assert torch.equal(xa, xb)


def test_a_step_cap_ends_the_distillation_early_and_says_so():
    """limits.max_steps (SAK-0019): TabPFN has no loop of its own, the student's distillation is what a
    step cap bounds."""
    import torch

    X_val, _, _ = _data(n=50, seed=2)
    full = backend_mod._distil(_fitted(seed=7))
    capped_fit = _fitted(seed=7)
    capped_fit.max_steps = 3
    capped = backend_mod._distil(capped_fit)
    assert capped.steps_capped and not full.steps_capped
    with torch.no_grad():
        x = torch.as_tensor(X_val, dtype=torch.float32)
        assert not torch.equal(full.module(x), capped.module(x))  # 3 steps is not 60 epochs
    roomy = _fitted(seed=7)
    roomy.max_steps = 10**9
    uncapped = backend_mod._distil(roomy)
    assert not uncapped.steps_capped
    with torch.no_grad():
        assert torch.equal(full.module(x), uncapped.module(x))  # a cap that is never reached changes nothing


def test_distilled_regression_student_tracks_its_teacher():
    X_val, _, y_val = _data(n=200, seed=3)
    student = backend_mod._distil(_fitted("tabular_regression"), X_val, y_val)
    spread = float(np.std(y_val))
    # RMSE to the teacher well under the target's own spread
    assert student.teacher_agreement < 0.25 * spread, (student.teacher_agreement, spread)


def test_student_imputes_missing_values_inside_the_graph():
    import torch

    student = backend_mod._distil(_fitted())
    x = torch.tensor([[np.nan, 0.0, 0.0, 0.0]], dtype=torch.float32)
    with torch.no_grad():
        out = student.module(x)
    assert torch.isfinite(out).all()


def test_unsupported_feature_dtypes_are_a_precise_export_error_not_a_crash():
    X, y, _ = _data(n=40)
    df = pd.DataFrame(X, columns=["a", "b", "c", "d"]).assign(seen=pd.date_range("2026-01-01", periods=40))
    with pytest.raises(backend_mod.TabPFNExportError, match=r"numeric and categorical.*unsupported columns.*seen"):
        backend_mod._distil(_fitted(X=df, y=y))


# ------------------------------------------------------------ categoricals (#145)

_COLOUR = {"red": 1.5, "blue": -1.5, "green": 0.3}
_SHAPE = {"circle": 0.8, "square": -0.8}


class _CatTeacher:
    """Stand-in teacher over a mixed frame: two numeric columns plus a
    string ``colour`` (object) and a pandas-category ``shape``; an unseen or
    missing category contributes 0. Records every frame it is asked about."""

    classes_ = np.array(["no", "yes"])

    def __init__(self):
        self.seen = []

    @staticmethod
    def _score(X):
        colour = X["colour"].astype(object).map(lambda v: _COLOUR.get(v, 0.0)).to_numpy(dtype=np.float64)
        shape = X["shape"].astype(object).map(lambda v: _SHAPE.get(v, 0.0)).to_numpy(dtype=np.float64)
        return 2.0 * X["a"].to_numpy(dtype=np.float64) - 1.5 * X["b"].to_numpy(dtype=np.float64) + 2.0 * colour + shape

    def predict_proba(self, X):
        assert isinstance(X, pd.DataFrame), type(X)
        self.seen.append(X)
        p = 1.0 / (1.0 + np.exp(-3.0 * self._score(X)))
        return np.stack([1.0 - p, p], axis=1)

    def predict(self, X):
        assert isinstance(X, pd.DataFrame), type(X)
        self.seen.append(X)
        return self._score(X) * 10.0 + 100.0


def _cat_frame(n=300, seed=0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "a": rng.normal(size=n),
        "colour": rng.choice(list(_COLOUR), size=n).astype(object),
        "b": rng.normal(size=n),
        "shape": pd.Categorical(rng.choice(list(_SHAPE), size=n), categories=list(_SHAPE)),
    })


def _cat_fitted(task="tabular_classification", n=300, seed=42, hp=FAST, frame=None):
    X = _cat_frame(n) if frame is None else frame
    teacher = _CatTeacher()
    s = teacher._score(X)
    y = np.where(s > 0, "yes", "no") if task == "tabular_classification" else s * 10.0 + 100.0
    classes = ["no", "yes"] if task == "tabular_classification" else None
    return backend_mod._Fitted(estimator=teacher, task=task, X_train=X, y_train=y,
                               classes=classes, seed=seed, hyperparameters=dict(hp))


def _cat_val(n=200, seed=1):
    X = _cat_frame(n, seed)
    return X, np.where(_CatTeacher._score(X) > 0, "yes", "no")


def test_encoder_splits_columns_and_buckets_unknown_and_missing_to_zero():
    X = _cat_frame(60)
    enc = backend_mod._Encoder.fit(X, max_categories=64)
    assert enc.features == ["a", "colour", "b", "shape"]
    assert enc.numeric_names == ["a", "b"] and enc.categorical_names == ["colour", "shape"]
    assert enc.vocab == [["blue", "green", "red"], ["circle", "square"]]
    q = pd.DataFrame({"a": [0.0, 1.0, 2.0, 3.0], "colour": ["red", "purple", None, ""],
                      "b": [0.0] * 4, "shape": ["square", "circle", np.nan, "circle"]})
    np.testing.assert_array_equal(enc.codes(q), [[3, 2], [0, 1], [0, 0], [0, 1]])
    assert enc.tokens(q)[2].tolist() == [backend_mod.MISSING_TOKEN, backend_mod.MISSING_TOKEN]
    # an object column holding only numbers is numeric, not categorical
    assert backend_mod._Encoder.fit(pd.DataFrame({"n": np.array([1, 2.5, None], dtype=object),
                                                  "s": ["x", "y", "x"]}), 64).numeric_names == ["n"]


def test_encoder_keeps_the_most_frequent_categories_up_to_the_cap():
    X = pd.DataFrame({"c": ["a"] * 5 + ["b"] * 3 + ["c"] * 2 + ["d"], "x": np.arange(11.0)})
    enc = backend_mod._Encoder.fit(X, max_categories=2)
    assert enc.vocab == [["a", "b"]]
    np.testing.assert_array_equal(enc.codes(pd.DataFrame({"c": ["c", "b"], "x": [0.0, 0.0]}))[:, 0], [0, 2])


def test_transfer_set_asks_the_teacher_about_original_categorical_values():
    fitted = _cat_fitted()
    backend_mod._distil(fitted)
    frame = pd.concat(fitted.estimator.seen, ignore_index=True)  # asked in 2000-row chunks
    assert list(frame.columns) == ["a", "colour", "b", "shape"]
    assert len(frame) == 300 * FAST["distill_augment"]
    assert set(frame["colour"]) <= set(_COLOUR)  # raw strings, never codes
    assert isinstance(frame["shape"].dtype, pd.CategoricalDtype)  # the dtype TabPFN was fitted on
    pd.testing.assert_frame_equal(frame.iloc[:300].reset_index(drop=True),
                                  fitted.X_train.reset_index(drop=True), check_dtype=False)


def test_distilled_student_with_categorical_features_matches_its_teacher():
    X_val, y_val = _cat_val()
    student = backend_mod._distil(_cat_fitted(), X_val, y_val)
    assert student.features == ["a", "colour", "b", "shape"]
    assert student.teacher_agreement >= 0.93, student.teacher_agreement
    assert backend_mod._student_scores(student, X_val, y_val) >= 0.9


def test_categorical_regression_student_tracks_its_teacher():
    X_val = _cat_frame(200, seed=3)
    y_val = _CatTeacher._score(X_val) * 10.0 + 100.0
    student = backend_mod._distil(_cat_fitted("tabular_regression"), X_val, y_val)
    assert student.teacher_agreement < 0.25 * float(np.std(y_val)), student.teacher_agreement


def _ort_logits(artifact, X):
    sess, meta = backend_mod._session(artifact.path)
    return sess.run(None, backend_mod._feed(sess, meta, X))[0]


def test_onnx_round_trip_with_categorical_features(tmp_path):
    pytest.importorskip("onnx")
    pytest.importorskip("onnxruntime")
    import torch

    out = RunDir(tmp_path / "run").ensure()
    X_val, y_val = _cat_val(150, seed=4)
    fitted = _cat_fitted()
    fitted.student = backend_mod._distil(fitted, X_val, y_val)
    artifact = backend_mod.BACKEND.export(SimpleNamespace(model=fitted), "onnx", out)

    sess, meta = backend_mod._session(artifact.path)
    assert meta["format"] == backend_mod.STUDENT_FORMAT_CATEGORICAL
    assert [(i.name, i.type) for i in sess.get_inputs()] == [("X", "tensor(float)"), ("X_cat", "tensor(string)")]
    assert json.loads(meta["inputs"]) == {"X": ["a", "b"], "X_cat": ["colour", "shape"]}
    assert json.loads(meta["categorical"]) == {"colour": ["blue", "green", "red"], "shape": ["circle", "square"]}

    # the in-graph LabelEncoder + one-hot computes what the torch student computes,
    # unseen and missing categories included
    q = pd.concat([X_val.head(5), pd.DataFrame({"a": [0.5, 0.5, np.nan], "colour": ["purple", None, "red"],
                                                "b": [0.1, 0.1, 0.1], "shape": ["hexagon", None, "circle"]})],
                  ignore_index=True)
    enc = fitted.student.encoder
    with torch.no_grad():
        ref = fitted.student.module(*backend_mod._student_inputs(enc, enc.numeric_matrix(q), enc.codes(q))).numpy()
    got = _ort_logits(artifact, q)
    assert np.isfinite(got).all()
    np.testing.assert_allclose(got, ref, atol=1e-5)

    ev = backend_mod.BACKEND.runtime_eval(artifact, SimpleNamespace(validation=(X_val, y_val)))
    assert ev.reason is None and ev.metric == "accuracy"
    assert ev.value == pytest.approx(backend_mod._student_scores(fitted.student, X_val, y_val), abs=1e-9)

    one = backend_mod.BACKEND.predict(artifact, {"a": 1.0, "colour": "red", "b": -1.0, "shape": "circle"})
    assert one["prediction"] == "yes"
    assert sum(one["proba"].values()) == pytest.approx(1.0, abs=1e-5)
    unseen = backend_mod.BACKEND.predict(artifact, {"a": 0.0, "colour": "mauve", "b": 0.0, "shape": None})
    assert set(unseen["proba"]) == {"no", "yes"}
    many = backend_mod.BACKEND.predict(artifact, [[1.0, "red", -1.0, "circle"], [-1.0, "blue", 1.0, "square"]])
    assert [r["prediction"] for r in many["predictions"]] == ["yes", "no"]


def test_onnx_round_trip_with_categorical_features_only(tmp_path):
    pytest.importorskip("onnx")
    pytest.importorskip("onnxruntime")
    out = RunDir(tmp_path / "run").ensure()
    rng = np.random.default_rng(0)
    X = pd.DataFrame({"colour": rng.choice(list(_COLOUR), size=200).astype(object), "a": 0.0, "b": 0.0,
                      "shape": rng.choice(list(_SHAPE), size=200).astype(object)})

    class OnlyCats(_CatTeacher):
        @staticmethod
        def _score(F):
            return _CatTeacher._score(F.assign(a=0.0, b=0.0))

    fitted = _cat_fitted(frame=X)
    fitted.estimator = OnlyCats()
    fitted.X_train = X[["colour", "shape"]]
    fitted.student = backend_mod._distil(fitted)
    artifact = backend_mod.BACKEND.export(SimpleNamespace(model=fitted), "onnx", out)
    sess, _ = backend_mod._session(artifact.path)
    assert [i.name for i in sess.get_inputs()] == ["X_cat"]
    pred = backend_mod.BACKEND.predict(artifact, {"colour": "red", "shape": "circle"})
    assert pred["prediction"] == "yes"


def test_dataframe_feature_names_are_recorded_for_predict():
    X, y, _ = _data(n=60)
    df = pd.DataFrame(X, columns=["radius", "texture", "smooth", "area"])
    assert backend_mod._distil(_fitted(X=df, y=y)).features == ["radius", "texture", "smooth", "area"]


def test_export_without_the_onnx_package_fails_loudly_and_writes_nothing(tmp_path, monkeypatch):
    out = RunDir(tmp_path / "run").ensure()
    monkeypatch.setitem(sys.modules, "onnx", None)  # import onnx -> ImportError
    result = SimpleNamespace(model=_fitted())
    with pytest.raises(backend_mod.TabPFNExportError, match="'onnx' package is required"):
        backend_mod.BACKEND.export(result, "onnx", out)
    assert not (out.artifacts / "model.onnx").exists()


def test_onnx_round_trip_rescoring_and_predict_shape(tmp_path):
    pytest.importorskip("onnx")
    pytest.importorskip("onnxruntime")
    out = RunDir(tmp_path / "run").ensure()
    X_val, y_val, _ = _data(n=150, seed=4)
    fitted = _fitted(X=pd.DataFrame(_data()[0], columns=["a", "b", "c", "d"]))
    fitted.student = backend_mod._distil(fitted, X_val, y_val)
    artifact = backend_mod.BACKEND.export(SimpleNamespace(model=fitted), "onnx", out)
    assert artifact.path.exists() and artifact.bytes == artifact.path.stat().st_size

    _, meta = backend_mod._session(artifact.path)
    assert meta["format"] == backend_mod.STUDENT_FORMAT
    assert meta["contains_teacher_weights"] == "false"
    assert "licence" in meta["licence_gate"] and "owner's call" in meta["licence_gate"]

    data = SimpleNamespace(validation=(pd.DataFrame(X_val, columns=["a", "b", "c", "d"]), y_val))
    ev = backend_mod.BACKEND.runtime_eval(artifact, data)
    assert ev.reason is None and ev.metric == "accuracy" and ev.matches_training is None
    # onnxruntime scores what the torch student scores (fp32, same graph)
    assert ev.value == pytest.approx(backend_mod._student_scores(fitted.student, X_val, y_val), abs=1e-9)

    one = backend_mod.BACKEND.predict(artifact, {"a": 1.0, "b": -1.0, "c": 0.0, "d": 0.0})
    assert one["prediction"] == "malignant"
    assert set(one["proba"]) == {"benign", "malignant"}
    assert sum(one["proba"].values()) == pytest.approx(1.0, abs=1e-5)
    many = backend_mod.BACKEND.predict(artifact, X_val[:3].tolist())
    assert len(many["predictions"]) == 3
    with pytest.raises(ValueError, match="missing required features"):
        backend_mod.BACKEND.predict(artifact, {"a": 1.0})


def test_onnx_round_trip_regression(tmp_path):
    pytest.importorskip("onnx")
    pytest.importorskip("onnxruntime")
    out = RunDir(tmp_path / "run").ensure()
    X_val, _, y_clean = _data(n=100, seed=5)
    y_val = y_clean + np.random.default_rng(6).normal(scale=5.0, size=len(y_clean))
    teacher_rmse = float(np.sqrt(np.mean((_Teacher().predict(X_val) - y_val) ** 2)))
    fitted = _fitted("tabular_regression")
    fitted.student = backend_mod._distil(fitted, X_val, y_val)
    fitted.student.teacher_metrics = {"rmse": teacher_rmse}
    artifact = backend_mod.BACKEND.export(SimpleNamespace(model=fitted), "onnx", out)
    ev = backend_mod.BACKEND.runtime_eval(artifact, SimpleNamespace(validation=(X_val, y_val)))
    assert ev.metric == "rmse" and ev.reason is None
    assert ev.value < 0.5 * float(np.std(y_val))
    # regression is judged scale-free against the teacher's stamped RMSE (±2%),
    # not by the engine's absolute ±0.02
    assert ev.matches_training is (abs(ev.value - teacher_rmse) <= 0.02 * teacher_rmse)
    pred = backend_mod.BACKEND.predict(artifact, X_val[0].tolist())
    assert pred["proba"] is None and isinstance(pred["prediction"], float)


def test_runtime_eval_reports_a_precise_reason_when_onnxruntime_cannot_load_the_file(tmp_path):
    out = RunDir(tmp_path / "run").ensure()
    artifact = Artifact(path=out.artifacts / "model.onnx", format="onnx", sha256="x", bytes=4)
    artifact.path.write_bytes(b"not an onnx file")
    data = SimpleNamespace(validation=(np.zeros((2, 2), dtype=np.float32), np.array([0, 1])))

    ev = backend_mod.BACKEND.runtime_eval(artifact, data)
    assert ev.value is None
    assert ev.matches_training is None
    assert "onnxruntime could not re-score" in ev.reason


def test_runtime_eval_refuses_an_onnx_file_that_is_not_a_student(tmp_path, monkeypatch):
    # An ONNX graph without the sakura.* metadata (e.g. a raw TabPFN core
    # export) cannot be decoded into labels: say so instead of guessing.
    import types

    out = RunDir(tmp_path / "run").ensure()
    artifact = Artifact(path=out.artifacts / "model.onnx", format="onnx", sha256="x", bytes=4)
    artifact.path.write_bytes(b"fake -- the stubbed onnxruntime below never reads it")

    fake_ort = types.ModuleType("onnxruntime")

    class FakeSession:
        def __init__(self, *a, **kw):
            pass

        def get_modelmeta(self):
            return SimpleNamespace(custom_metadata_map={})

    fake_ort.InferenceSession = FakeSession
    fake_ort.SessionOptions = SimpleNamespace  # sakura.atelier.ort sets the thread counts on it
    monkeypatch.setitem(sys.modules, "onnxruntime", fake_ort)
    monkeypatch.setattr(backend_mod, "_SESSIONS", {})

    data = SimpleNamespace(validation=(np.zeros((3, 2), dtype=np.float32), np.array([1, 1, 0])))
    ev = backend_mod.BACKEND.runtime_eval(artifact, data)
    assert ev.value is None
    assert "is not a tabpfn-distilled-mlp/1 / tabpfn-distilled-mlp/2 artefact" in ev.reason


def test_runtime_eval_leaves_matches_training_to_the_engine_on_success(tmp_path, monkeypatch):
    # Backend.runtime_eval(artifact, data) per CONTRACTS §2.2 has no
    # TrainResult to compare against, so matches_training is the engine's
    # call (it holds TabPFN's own accuracy and applies CONTRACTS §2.3).
    import json
    import types

    out = RunDir(tmp_path / "run").ensure()
    artifact = Artifact(path=out.artifacts / "model.onnx", format="onnx", sha256="x", bytes=4)
    artifact.path.write_bytes(b"fake -- the stubbed onnxruntime below never reads it")

    fake_ort = types.ModuleType("onnxruntime")

    class FakeSession:
        def __init__(self, *a, **kw):
            pass

        def get_modelmeta(self):
            return SimpleNamespace(custom_metadata_map={
                "sakura.format": backend_mod.STUDENT_FORMAT,
                "sakura.task": "tabular_classification",
                "sakura.classes": json.dumps(["0", "1"]),
                "sakura.features": json.dumps(["a", "b"]),
            })

        def get_inputs(self):
            return [SimpleNamespace(name="X")]

        def run(self, out_names, feed):
            n = feed["X"].shape[0]
            return [np.tile(np.array([0.0, 1.0]), (n, 1))]  # always predicts class "1"

    fake_ort.InferenceSession = FakeSession
    fake_ort.SessionOptions = SimpleNamespace  # sakura.atelier.ort sets the thread counts on it
    monkeypatch.setitem(sys.modules, "onnxruntime", fake_ort)
    monkeypatch.setattr(backend_mod, "_SESSIONS", {})

    data = SimpleNamespace(validation=(np.zeros((3, 2), dtype=np.float32), np.array([1, 1, 0])))
    ev = backend_mod.BACKEND.runtime_eval(artifact, data)
    assert ev.value == pytest.approx(2 / 3)
    assert ev.matches_training is None
    assert ev.reason is None
