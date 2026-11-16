"""Smoke runs (SAK-0019): a real ``python -m sakura.atelier run`` on ``synthetic://`` data with ``limits``,
one per backend whose stack is installed here. Each is a tiny training on a CPU, a few seconds.

classic_ml (scikit-learn) and risk_sim (numpy/scipy) need nothing heavy. peft, the Hugging Face classifiers and
Ludwig run where their own venv is (``docs/atelier/envs/``) and are skipped elsewhere. The backends with no run
here (whisper, deepspeech, open_clip, hf_detection, tabpfn) are covered at their step boundary by
``test_limits.py`` and ``test_wrap_up.py`` only."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from importlib import metadata
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parents[2]


def _run(tmp_path, spec_text, *extra, name="out", expect=0):
    """``python -m sakura.atelier run`` in a fresh process, as a runner starts it. Returns (report, out dir)."""
    spec = tmp_path / f"{name}.yaml"
    spec.write_text(spec_text)
    out = tmp_path / name
    env = {**os.environ, "PYTHONPATH": str(REPO), "CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "2"}
    env.pop("ATELIER_WRAP_UP_FILE", None)
    proc = subprocess.run([sys.executable, "-m", "sakura.atelier", "run", str(spec), "--out", str(out), *extra],
                          cwd=tmp_path, env=env, capture_output=True, text=True, timeout=900)
    report = json.loads((out / "report.json").read_text())
    assert proc.returncode == expect, f"exit {proc.returncode}: {report.get('error')}\n{proc.stderr[-3000:]}"
    return report, out


def _train_points(out):
    return [json.loads(line) for line in (out / "metrics.jsonl").read_text().splitlines()
            if json.loads(line)["split"] == "train"]


def _resolved(out):
    return yaml.safe_load((out / "resolved.yaml").read_text())


# --------------------------------------------------------------------------- classic_ml

CLASSIC = """
atelier: 1
name: smoke-{model}
task: {task}
data: {{uri: "synthetic://tabular?rows=400&seed=11&features=6{query}", format: csv, split: {{validation_fraction: 0.25}}}}
features:
  inputs: [{{name: f0, type: number}}, {{name: f1, type: number}}, {{name: f2, type: number}},
           {{name: f3, type: number}}, {{name: f4, type: number}}, {{name: f5, type: number}}]
  output: {{name: {target}, type: {otype}}}
model: {model}
hp: tabular/{model}@1
export: [joblib]
seed: 7
{limits}
"""
_CLASSIFY = dict(task="tabular_classification", query="&classes=3", target="label", otype="category")
_REGRESS = dict(task="tabular_regression", query="&task=regression", target="target", otype="number")
_ROUNDS = {"lightgbm": "n_estimators", "xgboost": "n_estimators", "catboost": "iterations",
           "random_forest": "n_estimators", "hist_gradient_boosting": "max_iter"}


def _estimator(out):
    import joblib

    return joblib.load(out / "artifacts" / "model.joblib").named_steps["est"]


@pytest.mark.parametrize("model", sorted(_ROUNDS))
def test_classic_ml_runs_capped_on_synthetic_data(model, tmp_path):
    pytest.importorskip("sklearn")
    pytest.importorskip({"lightgbm": "lightgbm", "xgboost": "xgboost", "catboost": "catboost"}.get(model, "sklearn"))
    limits = "limits: {max_steps: 12, max_train_rows: 150, max_eval_rows: 60}"
    spec = CLASSIC.format(model=model, limits=limits, **_CLASSIFY)
    classes = ["c0", "c1", "c2"]
    report, out = _run(tmp_path, spec, "--no-notebook")
    assert report["status"] == "done" and report["backend"] == "classic_ml"
    assert report["limits_applied"] == {
        "max_steps": 12, "max_train_rows": 150, "max_eval_rows": 60,
        "rows_before": {"train": 300, "validation": 100}, "rows_after": {"train": 150, "validation": 60}}
    assert report["data_split"]["train_rows"] == 150 and report["data_split"]["validation_rows"] == 60
    assert sorted(report["data_split"]["label_distribution"]["validation"]) == classes
    assert report["metrics"]["accuracy"] > 0.7  # balanced classes: chance is 1/2 at most
    assert _estimator(out).get_params()[_ROUNDS[model]] == 12  # the preset says 300 (CatBoost's: 500)
    resolved = _resolved(out)
    assert resolved["limits"] == {"max_steps": 12, "max_train_rows": 150, "max_eval_rows": 60}
    assert resolved["data"]["uri"].startswith("synthetic://tabular?") and resolved["data"]["sha256"] == report["data_sha256"]
    assert resolved["hyperparameters"]["n_estimators"] >= 300  # the preset is recorded as it is; limits say the cap


def test_classic_ml_regression_runs_capped_and_replays_identically(tmp_path):
    pytest.importorskip("sklearn")
    limits = "limits: {max_steps: 20, max_train_rows: 200}"
    first, out = _run(tmp_path, CLASSIC.format(model="hist_gradient_boosting", limits=limits, **_REGRESS), "--no-notebook")
    assert first["limits_applied"] == {"max_steps": 20, "max_train_rows": 200,
                                       "rows_before": {"train": 300, "validation": 100},
                                       "rows_after": {"train": 200, "validation": 100}}
    assert first["metrics"]["r2"] > 0.5
    # the replay of resolved.yaml: same spec hash, same data hash, same score
    again, _ = _run(tmp_path, (out / "resolved.yaml").read_text(), "--no-notebook", name="again")
    assert (again["spec_sha256"], again["data_sha256"]) == (first["spec_sha256"], first["data_sha256"])
    assert again["metrics"] == first["metrics"] and again["limits_applied"] == first["limits_applied"]


def test_classic_ml_without_limits_reports_none_and_trains_the_preset(tmp_path):
    pytest.importorskip("sklearn")
    report, out = _run(tmp_path, CLASSIC.format(model="random_forest", limits="", **_CLASSIFY), "--no-notebook")
    assert "limits_applied" not in report and "limits" not in _resolved(out)
    assert _estimator(out).get_params()["n_estimators"] == 300
    assert report["data_split"]["train_rows"] == 300


def test_classic_ml_onnx_export_of_a_capped_run_rescored(tmp_path):
    pytest.importorskip("skl2onnx")
    pytest.importorskip("onnxruntime")
    spec = CLASSIC.format(model="random_forest", limits="limits: {max_steps: 8, max_eval_rows: 40}", **_CLASSIFY)
    report, _ = _run(tmp_path, spec.replace("export: [joblib]", "export: [onnx]"), "--no-notebook")
    row = report["runtime_eval"][0]
    assert row["format"] == "onnx" and row["n"] == 40 and row["matches_training"] is True


def test_a_bad_synthetic_uri_fails_the_run_with_a_report(tmp_path):
    spec = CLASSIC.format(model="random_forest", limits="", **_CLASSIFY).replace("rows=400", "rows=400&colour=red")
    report, _ = _run(tmp_path, spec, "--no-notebook", expect=1)
    assert report["status"] == "failed" and "no parameter 'colour'" in report["error"]


def test_the_notebook_of_a_capped_synthetic_run_replays_it(tmp_path):
    pytest.importorskip("sklearn")
    pytest.importorskip("nbclient")
    limits = "limits: {max_steps: 10, max_train_rows: 120}"
    report, _ = _run(tmp_path, CLASSIC.format(model="hist_gradient_boosting", limits=limits, **_CLASSIFY))
    assert report["notebook_check"]["ok"] is True, report["notebook_check"]


# --------------------------------------------------------------------------- risk_sim

RISK = """
atelier: 1
name: smoke-risk
task: risk_simulation
model: scvar-copula
hp: risk_simulation/scvar@1
data: {{uri: "synthetic://risk?rows=240&seed=7", format: csv}}
export: [json]
seed: 39321
{limits}
"""


def test_risk_sim_runs_capped_on_synthetic_data(tmp_path):
    pytest.importorskip("scipy")
    report, out = _run(tmp_path, RISK.format(limits="limits: {max_steps: 3000}"), "--no-notebook")
    assert report["status"] == "done" and report["backend"] == "risk_sim"
    assert report["limits_applied"] == {"max_steps": 3000}
    results = json.loads((out / "artifacts" / "results.json").read_text())
    # the preset's ladder (10 000 then 100 000 draws) became one run of 3000
    assert [r["iterations"] for r in results["results"]["monte_carlo"]["runs"]] == [3000]
    assert results["manifest"]["hyperparameters"]["iterations"] == [3000]
    assert report["runtime_eval"][0]["matches_training"] is True  # the delivered file re-scores bit-exactly
    assert _resolved(out)["limits"] == {"max_steps": 3000}


def test_risk_sim_never_caps_below_the_smallest_sample_it_accepts(tmp_path):
    pytest.importorskip("scipy")
    report, out = _run(tmp_path, RISK.format(limits="limits: {max_steps: 5}"), "--no-notebook")
    results = json.loads((out / "artifacts" / "results.json").read_text())
    assert [r["iterations"] for r in results["results"]["monte_carlo"]["runs"]] == [1000]
    assert report["limits_applied"] == {"max_steps": 5}


def test_risk_sim_with_a_cap_above_its_ladder_is_unchanged(tmp_path):
    pytest.importorskip("scipy")
    spec = RISK.format(limits="limits: {max_steps: 50000}") + "overrides: {iterations: [2000, 4000]}\n"
    report, out = _run(tmp_path, spec, "--no-notebook")
    results = json.loads((out / "artifacts" / "results.json").read_text())
    assert [r["iterations"] for r in results["results"]["monte_carlo"]["runs"]] == [2000, 4000]
    assert "limits_applied" not in report


# --------------------------------------------------------------------------- peft (LoRA on a tiny offline Llama)


def _tiny_causal_lm(d):
    """A random 2-layer Llama with a character tokenizer over the synthetic text's alphabet: a few hundred
    KB, built offline (the same recipe as test_backend_peft's)."""
    import numpy as np
    from safetensors.numpy import save_file
    from tokenizers import Tokenizer, models, pre_tokenizers

    vocab = {t: i for i, t in enumerate(["<unk>", "<s>", "</s>", *"abcdefghijklmnopqrstuvwxyz :?"])}
    tok = Tokenizer(models.WordLevel(vocab=vocab, unk_token="<unk>"))
    tok.pre_tokenizer = pre_tokenizers.Split("", behavior="isolated")
    d.mkdir()
    tok.save(str(d / "tokenizer.json"))
    (d / "tokenizer_config.json").write_text(json.dumps({
        "tokenizer_class": "PreTrainedTokenizerFast", "bos_token": "<s>", "eos_token": "</s>", "unk_token": "<unk>"}))
    h, inter, layers, v = 64, 128, 2, len(vocab)
    (d / "config.json").write_text(json.dumps({
        "architectures": ["LlamaForCausalLM"], "model_type": "llama", "hidden_size": h, "intermediate_size": inter,
        "num_hidden_layers": layers, "num_attention_heads": 2, "num_key_value_heads": 2, "vocab_size": v,
        "rms_norm_eps": 1e-6, "rope_theta": 10000.0, "max_position_embeddings": 128, "tie_word_embeddings": False,
        "bos_token_id": 1, "eos_token_id": 2, "torch_dtype": "float32"}))
    rng = np.random.default_rng(0)

    def w(*shape):
        return (rng.standard_normal(shape) * 0.02).astype(np.float32)

    t = {"model.embed_tokens.weight": w(v, h), "model.norm.weight": np.ones(h, np.float32), "lm_head.weight": w(v, h)}
    for i in range(layers):
        p = f"model.layers.{i}."
        for n in ("q", "k", "v", "o"):
            t[f"{p}self_attn.{n}_proj.weight"] = w(h, h)
        t[f"{p}mlp.gate_proj.weight"], t[f"{p}mlp.up_proj.weight"] = w(inter, h), w(inter, h)
        t[f"{p}mlp.down_proj.weight"] = w(h, inter)
        t[f"{p}input_layernorm.weight"] = t[f"{p}post_attention_layernorm.weight"] = np.ones(h, np.float32)
    save_file(t, str(d / "model.safetensors"))
    return d


def test_peft_runs_capped_on_synthetic_instructions(tmp_path):
    pytest.importorskip("transformers")
    pytest.importorskip("peft")
    base = _tiny_causal_lm(tmp_path / "base")
    report, out = _run(tmp_path, f"""
atelier: 1
name: smoke-peft
task: text_generation
data: {{uri: "synthetic://instruct?rows=64&seed=3&keys=4", format: jsonl, split: {{validation_fraction: 0.25}}}}
features:
  inputs: [{{name: prompt, type: text}}]
  output: {{name: completion, type: text}}
model: hf-causal-lm
hp: text_generation/lora-fast@1
overrides: {{base_model: "{base}", epochs: 3, batch_size: 2, learning_rate: 0.001, lora_r: 4, lora_alpha: 8,
            lora_dropout: 0.0, lora_target_modules: [q_proj, v_proj], max_seq_length: 64}}
export: [safetensors]
seed: 5
limits: {{max_steps: 5, max_train_rows: 24, max_eval_rows: 4}}
""", "--no-notebook")
    assert report["status"] == "done" and report["backend"] == "peft"
    assert report["limits_applied"] == {
        "max_steps": 5, "max_train_rows": 24, "max_eval_rows": 4,
        "rows_before": {"train": 48, "validation": 16}, "rows_after": {"train": 24, "validation": 4}}
    assert report["stopped_early"]["reason"] == "max_steps" and report["stopped_early"]["step"] == 5
    assert [p["step"] for p in _train_points(out)] == [0, 1, 2, 3, 4]  # 3 epochs of 12 steps were planned
    assert report["runtime_eval"][0]["n"] == 4 and report["artifacts"][0]["format"] == "safetensors"


def test_a_lora_finetune_learns_the_instructions_and_the_merged_model_answers(tmp_path):
    """The whole text-generation path through the CLI, uncapped: a LoRA on the offline random Llama learns
    ``synthetic://instruct``, the merged safetensors model is delivered without an adapter, re-scored, and
    answers through ``predict`` in a fresh process."""
    pytest.importorskip("transformers")
    pytest.importorskip("peft")
    from tests.atelier.tiny_models import fuse_decoder

    base = fuse_decoder(_tiny_causal_lm(tmp_path / "base"))
    report, out = _run(tmp_path, f"""
atelier: 1
name: lora-e2e
task: text_generation
data: {{uri: "synthetic://instruct?rows=64&seed=3&keys=4", format: jsonl, split: {{validation_fraction: 0.25}}}}
features:
  inputs: [{{name: prompt, type: text}}]
  output: {{name: completion, type: text}}
model: hf-causal-lm
hp: text_generation/lora-fast@1
overrides: {{base_model: "{base}", epochs: 12, batch_size: 8, learning_rate: 0.003, lora_r: 8, lora_alpha: 16,
            lora_dropout: 0.0, max_seq_length: 64}}
export: [safetensors]
seed: 5
""", "--no-notebook")
    assert report["status"] == "done" and report["backend"] == "peft" and "limits_applied" not in report
    assert "init" not in report["provenance"] and report["provenance"]["resumed"] is False
    losses = [p["value"] for p in _train_points(out) if p["name"] == "loss"]
    assert len(losses) == 72  # 12 epochs of 6 batches
    # a LoRA on random weights learns slowly; the loss still falls from 3.2 to 2.3 (measured 2026-11-14)
    assert sum(losses[-6:]) / 6 < 0.8 * sum(losses[:6]) / 6, (losses[:6], losses[-6:])
    assert report["metrics"]["loss"] < 0.8 * sum(losses[:6]) / 6
    row = report["runtime_eval"][0]
    assert (row["format"], row["runtime"], row["n"]) == ("safetensors", "transformers", 16)
    assert row["value"] == report["metrics"]["exact_match"] and row["matches_training"] is True
    assert not list((out / "artifacts").rglob("adapter_*"))  # the adapter is merged: a plain dense model
    ask = tmp_path / "ask.json"
    ask.write_text(json.dumps({"prompt": "code for x:", "max_tokens": 4}))
    proc = subprocess.run([sys.executable, "-m", "sakura.atelier", "predict", str(out), "--input", str(ask)],
                          capture_output=True, text=True,
                          env={**os.environ, "PYTHONPATH": str(REPO), "CUDA_VISIBLE_DEVICES": ""})
    assert proc.returncode == 0 and isinstance(json.loads(proc.stdout)["text"], str)


# --------------------------------------------------------------------------- Hugging Face classifiers

VIT = "hf-internal-testing/tiny-random-ViTForImageClassification@5b3b44a3ed492070c273e481e30ecf4deddc5ec3"
W2V = "hf-internal-testing/tiny-random-Wav2Vec2ForSequenceClassification@2000b2022abcc37100241485f5872126b70164c9"
BERT = "hf-internal-testing/tiny-random-BertForSequenceClassification@4d5a3af95422fc0ebf732796c685c586b6d7420b"


def _hf_reachable():
    import urllib.request
    try:
        urllib.request.urlopen("https://huggingface.co/api/models/" + VIT.split("@")[0], timeout=5)
        return True
    except Exception:  # noqa: BLE001
        return False


@pytest.mark.parametrize("task, kind, fmt, itype, column, ref", [
    ("image_classification", "image?rows=48&seed=3&classes=3&size=32", "image_folder", "image", "image", VIT),
    ("audio_classification", "audio?rows=48&seed=3&classes=3&ms=300", "audio_folder", "audio", "audio", W2V),
    ("text_classification", "text?rows=48&seed=3&classes=3", "csv", "text", "text", BERT),
])
def test_hf_classifiers_run_capped_on_synthetic_data(task, kind, fmt, itype, column, ref, tmp_path):
    pytest.importorskip("transformers")
    pytest.importorskip("onnxruntime")
    if task == "audio_classification":
        pytest.importorskip("soundfile")
    if not _hf_reachable():
        pytest.skip("needs huggingface.co for the tiny random checkpoint")
    report, out = _run(tmp_path, f"""
atelier: 1
name: smoke-hf
task: {task}
data: {{uri: "synthetic://{kind}", format: {fmt}, split: {{validation_fraction: 0.25}}}}
features:
  inputs: [{{name: {column}, type: {itype}}}]
  output: {{name: label, type: category}}
model: "hf:{ref}"
hp: {task}/hf-finetune@1
overrides: {{epochs: 4, batch_size: 4}}
export: [onnx]
seed: 5
limits: {{max_steps: 3, max_train_rows: 18, max_eval_rows: 6}}
""", "--no-notebook")
    assert report["status"] == "done" and report["backend"] == f"hf_{task}"
    assert report["limits_applied"] == {
        "max_steps": 3, "max_train_rows": 18, "max_eval_rows": 6,
        "rows_before": {"train": 36, "validation": 12}, "rows_after": {"train": 18, "validation": 6}}
    assert report["stopped_early"] == {"reason": "max_steps", "epoch": 0.6, "step": 3}  # 5 steps per epoch
    assert len(report["epoch_metrics"]) == 1  # 4 epochs were planned
    assert set(report["data_split"]["label_distribution"]["train"]) == {"c0", "c1", "c2"}
    assert report["runtime_eval"][0]["n"] == 6


# --------------------------------------------------------------------------- hf_detection

YOLOS = "hf-internal-testing/tiny-random-YolosForObjectDetection@0a4aae25bfbe8b5edd4815cb00d697a6ba7d2126"


def test_hf_detection_runs_capped_on_a_tiny_yolo_dataset(tmp_path):
    """There is no synthetic detection kind: the dataset is a hand-made YOLO tree (a bright square on a dark
    image), capped by the list files `limits` writes."""
    pytest.importorskip("transformers")
    pytest.importorskip("onnxruntime")
    pytest.importorskip("torchmetrics")
    pytest.importorskip("faster_coco_eval")
    np = pytest.importorskip("numpy")
    from sakura.atelier.synthetic import _png

    if not _hf_reachable():
        pytest.skip("needs huggingface.co for the tiny random checkpoint")
    root = tmp_path / "ds"
    for split in ("train", "val"):
        (root / "images" / split).mkdir(parents=True)
        (root / "labels" / split).mkdir(parents=True)
        for i in range(8):
            arr = np.full((64, 64, 3), 30, dtype=np.uint8)
            arr[16:48, 16:48] = 220
            (root / "images" / split / f"{i}.png").write_bytes(_png(arr))
            (root / "labels" / split / f"{i}.txt").write_text("0 0.5 0.5 0.5 0.5\n")
    (root / "data.yaml").write_text("train: images/train\nval: images/val\nnames: {0: square}\n")
    report, out = _run(tmp_path, f"""
atelier: 1
name: smoke-detection
task: object_detection
data: {{uri: "file://{root}", format: yolo, split: {{train: train, validation: val}}}}
features:
  inputs: [{{name: image, type: image}}]
  output: {{name: boxes, type: boxes}}
model: "hf:{YOLOS}"
hp: object_detection/fast@1
overrides: {{epochs: 3, batch: 2, imgsz: 64}}
export: [onnx]
seed: 5
limits: {{max_steps: 2, max_train_rows: 6, max_eval_rows: 3}}
""", "--no-notebook")
    assert report["status"] == "done" and report["backend"] == "hf_detection"
    assert report["limits_applied"] == {
        "max_steps": 2, "max_train_rows": 6, "max_eval_rows": 3,
        "rows_before": {"train": 8, "validation": 8}, "rows_after": {"train": 6, "validation": 3}}
    assert report["stopped_early"]["reason"] == "max_steps" and report["stopped_early"]["step"] == 2
    assert [p["step"] for p in _train_points(out)] == [0, 1]  # 3 epochs of 3 steps were planned
    assert report["runtime_eval"][0]["n"] == 3  # the capped validation list is what is re-scored


# --------------------------------------------------------------------------- Ludwig


def _ludwig_major_minor():
    try:
        return tuple(int(x) for x in metadata.version("ludwig").split(".")[:2])
    except metadata.PackageNotFoundError:
        return None


LUDWIG = {
    "tabular": ("tabular_classification", "tabular?rows=600&seed=3&classes=3", "csv", """
  inputs: [{name: f0, type: number}, {name: f1, type: number}, {name: f2, type: number}, {name: f3, type: number}]
  output: {name: label, type: category}
model: ludwig_ecd
hp: tabular/ecd@1
overrides: {epochs: 10, batch_size: 32}"""),
    "text": ("text_classification", "text?rows=400&seed=3&classes=2", "csv", """
  inputs: [{name: text, type: text}]
  output: {name: label, type: category}
model: ludwig_text
hp: text_classification/ecd@1
overrides: {epochs: 10, batch_size: 32}"""),
    # random weights: the preset's pretrained ResNet-18 would be a download
    "image": ("image_classification", "image?rows=400&seed=3&classes=3&size=32", "image_folder", """
  inputs: [{name: image, type: image}]
  output: {name: label, type: category}
model: resnet18
hp: image_classification/fast@1
overrides: {pretrained: false, image_size: 32, epochs: 10, batch_size: 32}"""),
    "timeseries": ("time_series_forecasting", "timeseries?rows=600&seed=3", "csv", """
  inputs: [{name: value, type: timeseries}]
  output: {name: value, type: timeseries}
model: ludwig_forecaster
hp: time_series_forecasting/ecd@1
overrides: {window_size: 24, horizon: 6, epochs: 10, batch_size: 32}"""),
}


@pytest.mark.parametrize("case", sorted(LUDWIG))
def test_ludwig_runs_capped_on_synthetic_data(case, tmp_path):
    pytest.importorskip("ludwig")
    pytest.importorskip("onnxruntime")
    if case == "timeseries" and _ludwig_major_minor() < (0, 17):
        pytest.skip("time-series forecasting needs Ludwig >= 0.17")
    task, kind, fmt, features = LUDWIG[case]
    report, out = _run(tmp_path, f"""
atelier: 1
name: smoke-ludwig-{case}
task: {task}
data: {{uri: "synthetic://{kind}", format: {fmt}, split: {{validation_fraction: 0.25}}}}
features:{features}
export: [onnx]
seed: 5
limits: {{max_steps: 6, max_train_rows: 200, max_eval_rows: 60}}
""", "--no-notebook")
    assert report["status"] == "done" and report["backend"] == "ludwig"
    applied = report["limits_applied"]
    assert (applied["max_steps"], applied["max_train_rows"], applied["max_eval_rows"]) == (6, 200, 60)
    assert applied["rows_after"] == {"train": 200, "validation": 60}
    assert report["stopped_early"]["reason"] == "max_steps" and report["stopped_early"]["step"] == 6
    if case != "timeseries":
        # Ludwig's own `train_steps` ended these runs, with an evaluation round at the last step. Forecasting
        # counts windows, not rows, so there the watch stops the trainer and no round may have run.
        assert report["early_stop"]["reason"] == "max_steps" and report["early_stop"]["stopped_epoch"] == 1
        assert report["metrics"] and len(report["epoch_metrics"]) == 1  # 10 epochs of 7 steps were planned
        points = [json.loads(line) for line in (out / "metrics.jsonl").read_text().splitlines()]
        assert points and {p["step"] for p in points} == {6}  # one evaluation round, at the last step
        assert json.loads(next((out / "checkpoint").rglob("model_hyperparameters.json")).read_text())["trainer"]["train_steps"] == 6
    assert report["runtime_eval"][0]["format"] == "onnx" and report["runtime_eval"][0]["value"] is not None
