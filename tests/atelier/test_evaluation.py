"""The `evaluation` block (CONTRACTS §2.6): metric helpers on plain lists, the
warnings, the shaping of a backend's predictions, the generic fallback and the
JSON round trip. CPU only; scikit-learn is the one dependency."""

from __future__ import annotations

import json
import types

import pytest

pytest.importorskip("sklearn")

from sakura.atelier import evaluation as ev  # noqa: E402

Y_TRUE = ["a", "a", "b", "b", "c", "c"]
Y_PRED = ["a", "b", "b", "b", "c", "a"]
PROBA = [[0.7, 0.2, 0.1], [0.3, 0.6, 0.1], [0.1, 0.8, 0.1], [0.2, 0.7, 0.1], [0.1, 0.1, 0.8], [0.6, 0.1, 0.3]]


def test_classification_metrics_from_labels_and_probabilities():
    m = ev.classification_metrics(Y_TRUE, Y_PRED, PROBA, ["a", "b", "c"])
    assert m["n"] == 6 and m["accuracy"] == pytest.approx(4 / 6, abs=1e-6)
    assert m["f1_macro"] == pytest.approx((0.5 + 0.8 + 2 / 3) / 3, abs=1e-5)
    assert m["precision_macro"] == pytest.approx((0.5 + 2 / 3 + 1.0) / 3, abs=1e-5)
    assert m["recall_macro"] == pytest.approx((0.5 + 1.0 + 0.5) / 3, abs=1e-5)
    assert 0 < m["f1_weighted"] <= 1 and m["loss"] > 0 and 0.5 < m["roc_auc"] <= 1


def test_binary_roc_auc_uses_the_second_class_probability():
    m = ev.classification_metrics(["n", "n", "p", "p"], ["n", "p", "p", "p"],
                                  [[0.9, 0.1], [0.4, 0.6], [0.3, 0.7], [0.1, 0.9]], ["n", "p"])
    assert m["roc_auc"] == 1.0 and m["accuracy"] == 0.75


def test_metrics_without_probabilities_or_with_one_class_skip_loss_and_auc():
    m = ev.classification_metrics(["a", "a"], ["a", "a"], None, ["a", "b"])
    assert "roc_auc" not in m and "loss" not in m and m["accuracy"] == 1.0
    one = ev.classification_metrics(["a", "a"], ["a", "a"], [[0.9, 0.1], [0.8, 0.2]], ["a", "b"])
    assert "roc_auc" not in one and one["loss"] > 0


def test_multiclass_auc_survives_a_class_absent_from_the_split():
    m = ev.classification_metrics(["a", "b", "a", "b"], ["a", "b", "b", "b"],
                                  [[0.8, 0.1, 0.1], [0.1, 0.8, 0.1], [0.2, 0.7, 0.1], [0.1, 0.8, 0.1]], ["a", "b", "c"])
    assert "roc_auc" in m


def test_empty_split_is_just_n():
    assert ev.classification_metrics([], [], None, ["a"]) == {"n": 0}
    assert ev.regression_metrics([], []) == {"n": 0}


def test_regression_metrics():
    m = ev.regression_metrics([1.0, 2.0, 3.0], [1.0, 3.0, 5.0])
    assert m == {"n": 3, "loss": pytest.approx(5 / 3, abs=1e-5), "rmse": pytest.approx((5 / 3) ** 0.5, abs=1e-5),
                 "mae": pytest.approx(1.0)}


def test_distribution_baseline_confusion_and_per_class():
    d = ev.label_distribution(Y_TRUE, ["a", "b", "c", "d"])
    assert d == {"a": 2, "b": 2, "c": 2, "d": 0}
    assert ev.label_distribution(["x", "y", "y"]) == {"y": 2, "x": 1}
    assert ev.majority_baseline({"y": 2, "x": 1}) == pytest.approx(2 / 3, abs=1e-5)
    assert ev.majority_baseline({}) is None
    cm = ev.confusion(Y_TRUE, Y_PRED, ["a", "b", "c"], "validation")
    assert cm == {"split": "validation", "labels": ["a", "b", "c"], "matrix": [[1, 1, 0], [0, 2, 0], [1, 0, 1]]}
    pc = ev.per_class(Y_TRUE, Y_PRED, ["a", "b", "c"])
    assert pc[1] == {"label": "b", "precision": pytest.approx(2 / 3, abs=1e-5), "recall": 1.0,
                     "f1": 0.8, "support": 2}


def test_misclassified_is_capped_sorted_by_confidence_and_json_safe():
    import numpy as np

    n = 30
    sp = ev.SplitPredictions(y_true=["a"] * n, y_pred=["b"] * n, proba=[[1 - i / 100, i / 100] for i in range(n)],
                             inputs=[{"text": "x" * 500, "v": np.float32(1.5), "w": float("nan")} for _ in range(n)])
    rows = ev.misclassified(sp)
    assert len(rows) == ev.MAX_MISCLASSIFIED
    conf = [r["confidence"] for r in rows]
    assert conf == sorted(conf, reverse=True) and rows[0]["true"] == "a" and rows[0]["predicted"] == "b"
    assert rows[0]["inputs"]["text"].endswith("...") and rows[0]["inputs"]["v"] == 1.5 and rows[0]["inputs"]["w"] is None
    json.dumps(rows)
    right = ev.SplitPredictions(y_true=["a"], y_pred=["a"])
    assert ev.misclassified(right) == []


def test_misclassified_without_probabilities_has_null_confidence():
    rows = ev.misclassified(ev.SplitPredictions(y_true=["a"], y_pred=["b"]))
    assert rows == [{"sample_id": "validation:0", "inputs": {}, "true": "a", "predicted": "b", "confidence": None}]


def test_misclassified_rows_have_unique_ids_and_class_relative_paths(tmp_path):
    # BE-14: `022.png` exists under two class folders; the file name alone does not say which.
    root = tmp_path / "cifar-mini"
    files = [root / "airplane" / "022.png", root / "bird" / "022.png", root / "bird" / "002.png"]
    sp = ev.SplitPredictions(y_true=["airplane", "bird", "bird"], y_pred=["bird", "airplane", "cat"],
                             inputs=[{"image": str(f)} for f in files])
    rows = ev.misclassified(sp, media_root=root)
    assert sorted(r["path"] for r in rows) == ["airplane/022.png", "bird/002.png", "bird/022.png"]
    assert len({r["sample_id"] for r in rows}) == 3
    assert {r["sample_id"] for r in rows} == {"validation:0", "validation:1", "validation:2"}
    # no root (or a file outside it): the parent folder still names the class
    assert sorted(r["path"] for r in ev.misclassified(sp)) == ["airplane/022.png", "bird/002.png", "bird/022.png"]
    tabular = ev.misclassified(ev.SplitPredictions(y_true=["a"], y_pred=["b"], inputs=[{"text": "hello.png?"}]))
    assert "path" not in tabular[0]



def test_misclassified_sample_id_uses_the_split_row_ids():
    sp = ev.SplitPredictions(y_true=["a", "b"], y_pred=["b", "a"], row_ids=[41, 7])
    assert sorted(r["sample_id"] for r in ev.misclassified(sp)) == ["validation:41", "validation:7"]

def test_media_path():
    assert ev.media_path("/data/root/train/cat/1.jpg", "/data/root") == "train/cat/1.jpg"
    assert ev.media_path("/elsewhere/dog/7.wav", "/data/root") == "dog/7.wav"
    assert ev.media_path("not a file") is None and ev.media_path(3) is None


def test_misclassified_image_rows_carry_a_small_thumbnail(tmp_path):
    Image = pytest.importorskip("PIL.Image")
    import base64
    import io

    paths = []
    for i, colour in enumerate(("red", "blue")):
        p = tmp_path / f"{i}.png"
        Image.new("RGB", (320, 200), colour).save(p)
        paths.append(str(p))
    sp = ev.SplitPredictions(y_true=["cat", "dog", "cat"], y_pred=["dog", "cat", "cat"],
                             proba=[[0.3, 0.7], [0.9, 0.1], [0.8, 0.2]],
                             inputs=[{"image": paths[0]}, {"image": paths[1]}, {"image": paths[0]}])
    rows = ev.misclassified(sp)
    assert [r["true"] for r in rows] == ["dog", "cat"]  # most confident mistake first
    assert rows[0]["inputs"] == {"image": paths[1]}
    for row in rows:
        assert row["thumbnail"].startswith("data:image/jpeg;base64,")
        raw = base64.b64decode(row["thumbnail"].split(",", 1)[1])
        assert max(Image.open(io.BytesIO(raw)).size) == ev.THUMBNAIL_PX
        assert len(raw) < 5_000
    json.dumps(rows)


def test_thumbnail_skips_what_is_not_a_readable_image(tmp_path):
    pytest.importorskip("PIL.Image")
    broken = tmp_path / "broken.png"
    broken.write_bytes(b"not a png")
    assert ev.thumbnail("it was fine, I guess") is None
    assert ev.thumbnail(str(tmp_path / "missing.png")) is None
    assert ev.thumbnail(str(broken)) is None
    assert ev.thumbnail(1.5) is None
    rows = ev.misclassified(ev.SplitPredictions(y_true=["a"], y_pred=["b"], inputs=[{"image": str(broken)}]))
    assert "thumbnail" not in rows[0]


# ------------------------------------------------------------------ warnings

def _w(**kw):
    base = dict(task="tabular_classification", splits={"validation": {"n": 100}})
    base.update(kw)
    return ev.evaluation_warnings(**base)


def test_no_warnings_for_a_healthy_run():
    assert _w(label_dist={"train": {"a": 50, "b": 50}, "validation": {"a": 50, "b": 50}}, epochs=10, early_stop=5) == []


def test_tiny_validation_split_and_thin_classes():
    w = _w(splits={"validation": {"n": 12}}, label_dist={"validation": {"a": 9, "b": 3}})
    assert any("only 12 rows" in x for x in w)
    assert any("class a (9), b (3)" in x for x in w)
    assert not any("only" in x for x in _w(label_dist={"validation": {"a": 10, "b": 10}}))


def test_class_imbalance_over_seventy_percent_prefers_train():
    w = _w(label_dist={"train": {"a": 71, "b": 29}, "validation": {"a": 50, "b": 50}})
    assert any("'a' is 71% of the train rows" in x for x in w)
    assert not any("imbalance" in x for x in _w(label_dist={"train": {"a": 70, "b": 30}}))
    assert any("validation rows" in x for x in _w(label_dist={"validation": {"a": 80, "b": 20}}))


def test_auto_split_early_stop_and_text_only_tabular():
    assert any("validation_fraction" in x for x in _w(auto_split=True))
    assert any("early_stop=5 can never trigger with epochs=3" in x for x in _w(epochs=3, early_stop=5))
    assert any("early_stop=3" in x for x in _w(epochs=3, early_stop=3))
    assert not any("early_stop" in x for x in _w(epochs=10, early_stop=5))
    assert not any("early_stop" in x for x in _w(epochs=3, early_stop=-1))
    assert any("consider text_classification" in x for x in _w(input_types=["text"]))
    assert not any("text_classification" in x for x in _w(input_types=["text", "number"]))
    assert not any("text_classification" in x for x in _w(task="text_classification", input_types=["text"]))


def _resolved(hp=None, backend_config=None, task="tabular_classification", auto=True, input_types=("number",)):
    split = types.SimpleNamespace(validation_fraction=0.2 if auto else None)
    spec = types.SimpleNamespace(
        task=task, backend_config=backend_config or {}, data=types.SimpleNamespace(split=split),
        features=types.SimpleNamespace(inputs=[types.SimpleNamespace(type=t) for t in input_types]))
    return types.SimpleNamespace(spec=spec, hyperparameters=hp or {})


def test_training_params_reads_preset_then_trainer_override_then_ludwig_default():
    assert ev.training_params(_resolved({"epochs": 3, "early_stop": 5})) == (3, 5)
    assert ev.training_params(_resolved({"epochs": 3})) == (3, ev.DEFAULT_EARLY_STOP)
    assert ev.training_params(_resolved({"epochs": 3}, {"trainer": {"epochs": 50, "early_stop": 2}})) == (50, 2)
    assert ev.training_params(_resolved({"epochs": "many"})) == (None, None)


# ------------------------------------------------------------ shaping + build

def _classification_preds():
    return ev.ModelPredictions(kind="classification", classes=["a", "b", "c"], splits={
        "validation": ev.SplitPredictions(Y_TRUE, Y_PRED, PROBA, [{"text": f"row {i}"} for i in range(6)]),
        "train": ev.SplitPredictions(Y_TRUE * 2, Y_PRED * 2, PROBA * 2)})


def test_build_classification_shapes_the_contract_keys_in_split_order():
    block = ev.build_classification(_classification_preds())
    assert list(block["splits"]) == ["train", "validation"]
    assert block["classes"] == ["a", "b", "c"]
    assert block["label_distribution"]["train"] == {"a": 4, "b": 4, "c": 4}
    assert block["majority_baseline"] == {"train": pytest.approx(1 / 3, abs=1e-5), "validation": pytest.approx(1 / 3, abs=1e-5)}
    assert block["confusion_matrix"]["split"] == "validation"
    assert [r["label"] for r in block["per_class"]] == ["a", "b", "c"]
    assert block["misclassified"][0]["inputs"]["text"].startswith("row")


def test_build_regression_has_no_classification_keys():
    pred = ev.ModelPredictions(kind="regression", splits={
        "validation": ev.SplitPredictions([1.0, 2.0], [1.5, 2.5])})
    assert ev.build_regression(pred) == {"splits": {"validation": ev.regression_metrics([1.0, 2.0], [1.5, 2.5])}}


class _Backend:
    def __init__(self, pred=None, boom=False):
        self.pred, self.boom = pred, boom

    def predictions(self, result, resolved, data):
        if self.boom:
            raise RuntimeError("predict failed")
        return self.pred


def _data(n_train=8, n_val=4, test=None):
    return types.SimpleNamespace(train=list(range(n_train)), validation=list(range(n_val)),
                                 extra={"test": list(range(test))} if test else {})


def test_build_evaluation_end_to_end_is_json_and_carries_warnings():
    block = ev.build_evaluation(_Backend(_classification_preds()), types.SimpleNamespace(metrics={}),
                                _resolved({"epochs": 3, "early_stop": 5}), _data())
    json.dumps(block)
    assert block["splits"]["validation"]["n"] == 6
    w = block["warnings"]
    assert any("only 6 rows" in x for x in w) and any("validation_fraction" in x for x in w)
    assert any("early_stop=5" in x for x in w)


def test_build_evaluation_notes_from_the_backend_become_warnings():
    pred = _classification_preds()
    pred.notes.append("train metrics are computed on a 5000-row sample of 9000 rows")
    block = ev.build_evaluation(_Backend(pred), types.SimpleNamespace(metrics={}), _resolved(), _data())
    assert "train metrics are computed on a 5000-row sample of 9000 rows" in block["warnings"]


def test_generic_fallback_shapes_the_backends_own_metrics_per_split():
    be = types.SimpleNamespace()  # no `predictions` hook
    result = types.SimpleNamespace(metrics={"accuracy": 0.8, "loss": 0.4, "nan": float("nan")})
    block = ev.build_evaluation(be, result, _resolved(task="object_detection", auto=False), _data(test=3))
    assert block["splits"] == {"train": {"n": 8}, "validation": {"n": 4, "accuracy": 0.8, "loss": 0.4},
                               "test": {"n": 3}}
    assert any("only 4 rows" in x for x in block["warnings"])
    assert not any("validation_fraction" in x for x in block["warnings"])


def test_generic_fallback_without_metrics_or_sizes():
    assert ev.generic_evaluation({}, types.SimpleNamespace(train=object(), validation=None, extra={})) == {"splits": {}}


def test_a_hook_returning_none_falls_back_to_generic():
    block = ev.build_evaluation(_Backend(None), types.SimpleNamespace(metrics={"rmse": 1.0}), _resolved(), _data())
    assert block["splits"]["validation"]["rmse"] == 1.0


def test_a_failing_hook_becomes_a_warning_never_an_exception():
    block = ev.build_evaluation(_Backend(boom=True), types.SimpleNamespace(metrics={}), _resolved(), _data())
    assert block["splits"] == {} and any("RuntimeError: predict failed" in x for x in block["warnings"])


def test_missing_scikit_learn_skips_the_block_with_a_warning(monkeypatch):
    import builtins

    real = builtins.__import__

    def fake(name, *a, **k):
        if name == "sklearn" or name.startswith("sklearn."):
            raise ImportError("no sklearn")
        return real(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", fake)
    block = ev.build_evaluation(_Backend(_classification_preds()), types.SimpleNamespace(metrics={"accuracy": 0.5}),
                                _resolved(), _data())
    assert any("scikit-learn is not installed" in x for x in block["warnings"])
    assert block["splits"]["validation"]["accuracy"] == 0.5


def test_non_finite_values_are_dropped_from_the_json():
    out: dict = {}
    ev._put(out, "a", float("nan"))
    ev._put(out, "b", float("inf"))
    ev._put(out, "c", "x")
    ev._put(out, "d", 0.123456789)
    assert out == {"d": 0.123457}


def test_auto_split_warning_says_how_the_holdout_was_drawn():
    assert any("drawn per class" in x for x in _w(auto_split=True, split_strategy="stratified"))
    assert any("random rows" in x for x in _w(auto_split=True))
    data = _data()
    data.extra = {"data_split": {"strategy": "stratified"}}
    block = ev.build_evaluation(_Backend(_classification_preds()), types.SimpleNamespace(metrics={}),
                                _resolved(), data)
    assert any("drawn per class" in x for x in block["warnings"])
