"""ONNX int8 as a measured export (SAK-0019): ``export: [onnx, onnx_int8]`` delivers the float graph and the
same graph with its linear weights quantised to int8, each listed with its size and re-scored on the
validation split. Covered: the three Hugging Face classifiers (one code path; the text one runs end to end)."""

from __future__ import annotations

import json
import os
import subprocess
import sys

import pytest

from sakura.atelier.registry import resolve
from sakura.atelier.spec import load_spec
from tests.atelier.test_init_random import CLASSIFIER
from tests.atelier.test_smoke_runs import REPO, CLASSIC, _CLASSIFY, _run
from tests.atelier.tiny_models import synthetic_words, tiny_bert_dir


def _classifier(export, base="some/base"):
    text = CLASSIFIER.format(uri="synthetic://text?rows=240&seed=3&classes=3", overrides=json.dumps({"base_model": base}))
    return text.replace("export: [onnx]", f"export: {export}")


def test_onnx_int8_is_an_export_format_delivered_next_to_the_float_graph():
    assert resolve(load_spec(_classifier("[onnx, onnx_int8]"))).spec.export == ["onnx", "onnx_int8"]
    with pytest.raises(ValueError, match="list onnx as well"):
        load_spec(_classifier("[onnx_int8]"))
    with pytest.raises(ValueError, match="list onnx as well"):
        load_spec(_classifier("[safetensors, onnx_int8]"))


def test_a_model_whose_onnx_has_no_weights_to_quantise_refuses_the_format():
    # classic_ml's ONNX graphs are tree ensembles: thresholds are node attributes, there is no weight matrix
    spec = CLASSIC.format(model="random_forest", limits="", **_CLASSIFY).replace("export: [joblib]", "export: [onnx, onnx_int8]")
    with pytest.raises(ValueError, match=r"model 'random_forest' cannot export \['onnx_int8'\]"):
        resolve(load_spec(spec))


def test_quantising_a_graph_without_a_linear_weight_is_an_error_not_an_empty_int8_file(tmp_path):
    onnx = pytest.importorskip("onnx")
    pytest.importorskip("onnxruntime")
    from onnx import TensorProto, helper

    from sakura.atelier.ort import quantize_int8

    graph = helper.make_graph([helper.make_node("Relu", ["x"], ["y"])], "g",
                              [helper.make_tensor_value_info("x", TensorProto.FLOAT, [None, 4])],
                              [helper.make_tensor_value_info("y", TensorProto.FLOAT, [None, 4])])
    src = tmp_path / "relu.onnx"
    onnx.save(helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)]), str(src))
    with pytest.raises(ValueError, match="no MatMul weight to quantise"):
        quantize_int8(src, tmp_path / "relu.int8.onnx")
    assert not (tmp_path / "relu.int8.onnx").exists()


def test_the_int8_graph_is_delivered_smaller_and_rescored_beside_the_float_one(tmp_path):
    pytest.importorskip("transformers")
    pytest.importorskip("onnxruntime")
    uri = "synthetic://text?rows=240&seed=3&classes=3"
    base = tiny_bert_dir(tmp_path / "bert", synthetic_words(uri, tmp_path / "words"))
    overrides = {"base_model": str(base), "init": "random", "epochs": 8, "batch_size": 16, "learning_rate": 0.003,
                 "weight_decay": 0.0, "init_config": {"num_hidden_layers": 2, "hidden_size": 64, "intermediate_size": 256,
                                                      "max_position_embeddings": 64}}
    spec = CLASSIFIER.format(uri=uri, overrides=json.dumps(overrides)).replace("export: [onnx]", "export: [onnx, onnx_int8]")
    report, out = _run(tmp_path, spec, "--no-notebook")
    assert report["status"] == "done"
    float_art, int8_art = report["artifacts"][0], report["artifacts"][1]
    assert (float_art["format"], float_art["path"]) == ("onnx", "artifacts/model.onnx")
    assert (int8_art["format"], int8_art["path"]) == ("onnx_int8", "artifacts/model.int8.onnx")
    # its own artifact, with its size: smaller than the float graph it was quantised from
    assert (out / int8_art["path"]).stat().st_size == int8_art["bytes"] == int8_art["size_bytes"]
    assert int8_art["bytes"] < 0.7 * float_art["bytes"]
    assert int8_art["float_bytes"] == float_art["bytes"]
    assert int8_art["size_ratio"] == round(int8_art["bytes"] / float_art["bytes"], 4)
    assert int8_art["quantization"] == "dynamic_int8" and int8_art["quantized_tensors"] > 0
    # re-scored like any other artifact, on the same rows as the float graph
    float_row, int8_row = report["runtime_eval"]
    assert (float_row["format"], int8_row["format"]) == ("onnx", "onnx_int8")
    assert float_row["n"] == int8_row["n"] == 60 and int8_row["runtime"] == "onnxruntime"
    assert float_row["matches_training"] is True and float_row["training_delta"] == 0.0
    trained = report["metrics"]["accuracy"]
    assert int8_row["training_delta"] == pytest.approx(int8_row["value"] - trained, abs=1e-6)
    # the quantised tolerance of CONTRACTS §2.3: within 0.05 of the training metric, or the report says how far
    assert int8_row["matches_training"] is (abs(int8_row["training_delta"]) <= 0.05)
    assert int8_row["value"] > 0.8  # chance is 1/3
    assert int8_art["validated"] is True and int8_art["eval_acc"] == int8_row["value"]
    assert int8_art["matches_training"] is int8_row["matches_training"]
    assert report["onnx_provider"] == float_row["provider"]
    # the int8 file answers on its own
    ask = tmp_path / "ask.json"
    ask.write_text(json.dumps(["some text"]))
    proc = subprocess.run([sys.executable, "-m", "sakura.atelier", "predict", str(out), "--input", str(ask),
                           "--format", "onnx_int8"], capture_output=True, text=True,
                          env={**os.environ, "PYTHONPATH": str(REPO), "CUDA_VISIBLE_DEVICES": ""})
    assert proc.returncode == 0, proc.stdout
    assert set(json.loads(proc.stdout)[0]["scores"]) == {"c0", "c1", "c2"}
