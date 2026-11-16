"""``init: random`` (SAK-0019): train ``hf-causal-lm`` and the Hugging Face classifiers from fresh weights.

The validation rules are unit tests on ``registry.resolve``. The two end-to-end tests run
``python -m sakura.atelier run`` on a CPU from a directory that holds a config and a tokenizer and no weights."""

from __future__ import annotations

import json
import os
import subprocess
import sys

import pytest

from sakura.atelier.registry import resolve
from sakura.atelier.spec import load_spec
from tests.atelier.test_smoke_runs import REPO, _resolved, _run, _tiny_causal_lm, _train_points
from tests.atelier.tiny_models import fuse_decoder, synthetic_words, tiny_bert_dir

LM = """
atelier: 1
name: scratch-lm
task: text_generation
data: {{uri: "synthetic://instruct?rows=64&seed=3&keys=4", format: jsonl, split: {{validation_fraction: 0.25}}}}
features:
  inputs: [{{name: prompt, type: text}}]
  output: {{name: completion, type: text}}
model: hf-causal-lm
hp: text_generation/lora-fast@1
overrides: {overrides}
export: [safetensors]
seed: 5
"""

CLASSIFIER = """
atelier: 1
name: scratch-classifier
task: text_classification
data: {{uri: "{uri}", format: csv, split: {{validation_fraction: 0.25}}}}
features:
  inputs: [{{name: text, type: text}}]
  output: {{name: label, type: category}}
model: hf-text-classifier
hp: text_classification/hf-finetune@1
overrides: {overrides}
export: [onnx]
seed: 5
"""


def _lm(**overrides):
    return load_spec(LM.format(overrides=json.dumps({"base_model": "some/base", **overrides})))


# ------------------------------------------------------------------------------------------- validation

def test_a_spec_without_init_resolves_as_before_and_init_random_is_kept_in_the_hyperparameters():
    assert "init" not in resolve(_lm()).hyperparameters
    hp = resolve(_lm(init="random", init_config={"num_hidden_layers": 2})).hyperparameters
    assert hp["init"] == "random" and hp["init_config"] == {"num_hidden_layers": 2}
    assert resolve(_lm(init="pretrained")).hyperparameters["init"] == "pretrained"


@pytest.mark.parametrize("overrides, message", [
    ({"init": "zeros"}, "init must be one of ['pretrained', 'random']"),
    ({"init_config": {"hidden_size": 64}}, "init_config only applies with init: random"),
    ({"init": "random", "init_config": ["hidden_size"]}, "init_config must be a mapping"),
    ({"init": "random", "init_config": {"vocab_size": 100}}, "the vocabulary size comes from the tokenizer"),
    ({"init": "random", "init_config": {"hidden_size": None}}, "init_config.hidden_size must be"),
    ({"init": "random", "precision": "4bit"}, "init: random cannot be combined with precision: 4bit"),
])
def test_combinations_that_make_no_sense_are_refused_by_name(overrides, message):
    with pytest.raises(ValueError) as err:
        resolve(_lm(**overrides))
    assert message in str(err.value)


def test_init_random_is_refused_for_a_backend_that_cannot_build_from_a_config():
    spec = load_spec("""
atelier: 1
name: t
task: tabular_classification
data: {uri: "synthetic://tabular?rows=50", format: csv, split: {validation_fraction: 0.2}}
features:
  inputs: [{name: f0, type: number}]
  output: {name: label, type: category}
model: lightgbm
hp: tabular/lightgbm@1
overrides: {init: random}
export: [joblib]
""")
    with pytest.raises(ValueError, match="init: random is not supported for task 'tabular_classification'"):
        resolve(spec)


def test_validate_refuses_a_bad_init_without_running(tmp_path):
    spec = tmp_path / "s.yaml"
    spec.write_text(LM.format(overrides=json.dumps({"base_model": "some/base", "init": "random", "precision": "4bit"})))
    proc = subprocess.run([sys.executable, "-m", "sakura.atelier", "validate", str(spec)], capture_output=True,
                          text=True, env={**os.environ, "PYTHONPATH": str(REPO)})
    assert proc.returncode == 1 and "precision: 4bit" in proc.stderr


# ------------------------------------------------------------------------------------------- end to end

def test_a_causal_lm_trains_from_random_weights_in_full_and_the_artifact_reloads(tmp_path):
    pytest.importorskip("transformers")
    pytest.importorskip("peft")
    base = fuse_decoder(_tiny_causal_lm(tmp_path / "base"))
    (base / "model.safetensors").unlink()  # a config and a tokenizer only: there are no weights to load
    overrides = {"base_model": str(base), "init": "random", "epochs": 15, "batch_size": 8, "learning_rate": 0.005,
                 "init_config": {"num_hidden_layers": 1, "hidden_size": 48, "intermediate_size": 96},
                 "max_seq_length": 64}
    report, out = _run(tmp_path, LM.format(overrides=json.dumps(overrides)), "--no-notebook")
    assert report["status"] == "done" and report["backend"] == "peft"
    prov = report["provenance"]
    assert prov["init"] == "random" and prov["training"] == "full" and prov["base_commit"] is None
    assert prov["init_config"] == overrides["init_config"]
    config = json.loads((out / "artifacts" / "config.json").read_text())
    assert (config["num_hidden_layers"], config["hidden_size"], config["vocab_size"]) == (1, 48, 32)
    assert not (out / "checkpoint" / "adapter").exists()  # a full training: no adapter anywhere
    assert not list(out.rglob("adapter_config.json"))
    losses = [p["value"] for p in _train_points(out) if p["name"] == "loss"]
    assert sum(losses[-6:]) / 6 < 0.5 * sum(losses[:6]) / 6, (losses[:6], losses[-6:])
    assert report["metrics"]["exact_match"] >= 0.75  # 4 keys, each seen in training
    row = report["runtime_eval"][0]  # the delivered safetensors, reloaded by transformers
    assert row["format"] == "safetensors" and row["n"] == 16 and row["matches_training"] is True
    assert _resolved(out)["hyperparameters"]["init"] == "random"
    # the artifact answers in a fresh process
    ask = tmp_path / "ask.json"
    ask.write_text(json.dumps({"prompt": "x", "max_tokens": 4}))
    proc = subprocess.run([sys.executable, "-m", "sakura.atelier", "predict", str(out), "--input", str(ask)],
                          capture_output=True, text=True, env={**os.environ, "PYTHONPATH": str(REPO),
                                                               "CUDA_VISIBLE_DEVICES": ""})
    assert proc.returncode == 0 and "text" in json.loads(proc.stdout)


def test_a_text_classifier_trains_from_random_weights_and_its_onnx_rescored(tmp_path):
    pytest.importorskip("transformers")
    pytest.importorskip("onnxruntime")
    uri = "synthetic://text?rows=240&seed=3&classes=3"
    base = tiny_bert_dir(tmp_path / "bert", synthetic_words(uri, tmp_path / "words"))
    overrides = {"base_model": str(base), "init": "random", "epochs": 8, "batch_size": 16, "learning_rate": 0.003,
                 "weight_decay": 0.0, "init_config": {"num_hidden_layers": 1, "hidden_size": 48,
                                                      "intermediate_size": 96}}
    report, out = _run(tmp_path, CLASSIFIER.format(uri=uri, overrides=json.dumps(overrides)), "--no-notebook")
    assert report["status"] == "done" and report["backend"] == "hf_text_classification"
    prov = report["provenance"]
    assert prov["init"] == "random" and prov["init_config"] == overrides["init_config"] and prov["parameters"] > 0
    config = json.loads((out / "checkpoint" / "config.json").read_text())
    assert (config["num_hidden_layers"], config["hidden_size"]) == (1, 48)
    assert config["vocab_size"] == len(json.loads((base / "tokenizer.json").read_text())["model"]["vocab"])
    epochs = report["epoch_metrics"]
    assert epochs[-1]["train_loss"] < 0.5 * epochs[0]["train_loss"], epochs
    assert report["metrics"]["accuracy"] > 0.8  # three balanced classes: chance is 1/3
    row = report["runtime_eval"][0]
    assert row["format"] == "onnx" and row["n"] == 60 and row["matches_training"] is True
