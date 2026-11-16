"""The engine's presets are exactly the canonical list the hub's planner and
compiler are tested against too (CONTRACTS.md §7). A preset renamed on one
side only fails a job at the runner, after the user confirmed and paid."""

from pathlib import Path

CANONICAL = {
    "image_classification/fast@1",
    "audio_classification/fast@1",
    "object_detection/fast@1",
    "tabular/ecd@1",
    "tabular/tabpfn@1",
    "text_generation/lora-fast@1",
    "text_generation/qlora-forge@1",  # the legacy La Forge LLM runner's recipe (2026-10-10)
    "text_generation/dpo@1",  # DPO on {prompt, chosen, rejected} rows, peft backend (2026-11-14)
    "text_generation/lora-tuned@1",  # schedule, periodic evaluation, best checkpoint, task metrics (2026-11-15)
    "speech_recognition/ctc-fast@1",
    "speech_recognition/whisper-lora@1",
    "time_series_forecasting/ecd@1",
    "text_classification/ecd@1",
    "image_regression/siglip2@1",
    "image_classification/hf-finetune@1",  # any HF classifier, model: hf:<repo>@<sha> (2026-11-09)
    "audio_classification/hf-finetune@1",
    "text_classification/hf-finetune@1",
    "tabular/lightgbm@1",  # classic_ml (2026-11-09)
    "tabular/xgboost@1",
    "tabular/catboost@1",
    "tabular/random_forest@1",
    "tabular/hist_gradient_boosting@1",
    "decision/lora-forge@1",  # the System One task (zak-forge FORGE-0021): LoRA on a 2B/4B base, calibrated
    "risk_simulation/scvar@1",  # supply-chain VaR: four estimators, Monte Carlo ladder 10k/100k/1M (2026-11-11)
    "reinforcement_learning/reinforce@1",  # rl: REINFORCE path walking on a knowledge graph (2026-11-14)
}

PRESETS = Path(__file__).resolve().parents[2] / "sakura" / "atelier" / "registry" / "presets"


def test_registry_presets_are_exactly_the_canonical_list():
    found = {f"{p.parent.name}/{p.stem}" for p in PRESETS.glob("*/*.yaml")}
    assert found == CANONICAL


def test_there_is_no_gbm_model():
    # Ludwig 0.11 has no GBM model type; an id that names one trains an ECD.
    models = (PRESETS.parent / "models.yaml").read_text()
    assert "ludwig_gbm" not in models


# The hub's MODEL_EXPORTS (api/forge/compiler/atelier.py) must agree with these:
# an export the engine cannot produce must be refused at quote time, not after training.
CANONICAL_EXPORTS = {
    "resnet18": ["onnx", "torchscript"],
    "stacked_cnn_audio": ["onnx", "torchscript"],
    "ludwig_ecd": ["onnx"],
    "tabpfn_v2": ["onnx"],  # a distilled student (issue #67), not TabPFN itself
    "rtdetr_r18": ["onnx"],
    "lfm2-350m": ["gguf", "safetensors", "mlx"],
    "hf-causal-lm": ["gguf", "safetensors", "mlx"],  # any HF causal LM, base_model required (2026-10-10)
    "deepspeech2": ["safetensors"],
    # templates of `hf:<repo>@<sha>` classifiers (2026-11-09)
    "hf-image-classifier": ["onnx", "safetensors", "onnx_int8"],
    "hf-audio-classifier": ["onnx", "safetensors", "onnx_int8"],
    "hf-text-classifier": ["onnx", "safetensors", "onnx_int8"],
    # classic_ml (2026-11-09): joblib = the pickled sklearn pipeline
    "lightgbm": ["onnx", "joblib"],
    "xgboost": ["onnx", "joblib"],
    "catboost": ["onnx", "joblib"],
    "random_forest": ["onnx", "joblib"],
    "hist_gradient_boosting": ["onnx", "joblib"],
    # ggml (whisper.cpp) added 2026-10-10: the hub's MODEL_EXPORTS must gain it in its own PR
    # before a hub job can request it (PR 4 of "llm/asr jobs on the Atelier engine").
    "whisper-small": ["safetensors", "ggml"],
    "whisper-large-v3-turbo": ["safetensors", "ggml"],
    "ludwig_forecaster": ["onnx"],
    "ludwig_text": ["onnx"],
    "siglip2_b16_256": ["onnx"],
    # safetensors only in v1: a gguf runtime cannot reproduce the restricted-letter readout or the
    # calibration temperature. The hub's MODEL_EXPORTS and the runner mirror this by hand.
    "hf-decision-lm": ["safetensors"],
    # risk_simulation (2026-11-11): results.json = every estimate + the run manifest; replayed, not re-scored
    "scvar-copula": ["json"],
    # reinforcement_learning (2026-11-14): the one-hop policy; graph.json is delivered beside it
    "kg-path-policy": ["torchscript", "onnx"],
}


def test_registry_exports_are_the_canonical_table():
    from sakura.atelier.registry import load_models

    assert {k: v.get("exports") for k, v in load_models().items()} == CANONICAL_EXPORTS


def _spec(model, hp, task, export):
    from sakura.atelier.spec import load_spec

    return load_spec(f"""
atelier: 1
name: t
task: {task}
data: {{uri: "file:///x.csv", format: csv, split: {{validation_fraction: 0.2}}}}
features:
  inputs: [{{name: a, type: number}}]
  output: {{name: y, type: category}}
model: {model}
hp: {hp}
export: {export}
""")


def test_resolve_refuses_an_export_the_model_cannot_produce():
    import pytest

    from sakura.atelier.registry import resolve

    with pytest.raises(ValueError, match="cannot export"):
        resolve(_spec("tabpfn_v2", "tabular/tabpfn@1", "tabular_classification", "[gguf]"))
    with pytest.raises(ValueError, match="export is empty"):
        resolve(_spec("ludwig_ecd", "tabular/ecd@1", "tabular_classification", "[]"))
    # tabpfn_v2 is `export_optional`: [] (accuracy reference) and [onnx]
    # (distilled student, issue #67) both resolve.
    assert resolve(_spec("tabpfn_v2", "tabular/tabpfn@1", "tabular_classification", "[]")).backend == "tabpfn"
    assert resolve(_spec("tabpfn_v2", "tabular/tabpfn@1", "tabular_classification", "[onnx]")).backend == "tabpfn"
    assert resolve(_spec("ludwig_ecd", "tabular/ecd@1", "tabular_classification", "[onnx]")).backend == "ludwig"


def test_hf_causal_lm_requires_a_base_model_and_records_it():
    import pytest

    from sakura.atelier.registry import resolve
    from sakura.atelier.spec import load_spec

    def spec(overrides):
        return load_spec(f"""
atelier: 1
name: t
task: text_generation
data: {{uri: "file:///x.csv", format: csv, split: {{validation_fraction: 0.2}}}}
features:
  inputs: [{{name: input, type: text}}]
  output: {{name: output, type: text}}
model: hf-causal-lm
hp: text_generation/qlora-forge@1
overrides: {overrides}
export: [gguf]
""")

    with pytest.raises(ValueError, match="requires hyperparameters \\['base_model'\\]"):
        resolve(spec("{}"))
    with pytest.raises(ValueError, match="base_model"):
        resolve(spec("{base_model: ''}"))
    r = resolve(spec("{base_model: Qwen/Qwen3-0.6B, base_revision: abc123, gguf_quant: MXQ4}"))
    assert r.backend == "peft"
    assert (r.hyperparameters["base_model"], r.hyperparameters["base_revision"]) == ("Qwen/Qwen3-0.6B", "abc123")
    # overrides win over the preset, which wins over the model's hp_defaults
    assert r.hyperparameters["gguf_quant"] == "MXQ4" and r.hyperparameters["precision"] == "4bit"


def test_lfm2_keeps_its_base_through_hp_defaults():
    from sakura.atelier.registry import resolve
    from sakura.atelier.spec import load_spec

    r = resolve(load_spec("""
atelier: 1
name: t
task: text_generation
data: {uri: "file:///x.jsonl", format: jsonl, split: {validation_fraction: 0.2}}
features:
  inputs: [{name: prompt, type: text}]
  output: {name: completion, type: text}
model: lfm2-350m
hp: text_generation/lora-fast@1
export: [gguf]
"""))
    assert r.hyperparameters["base_model"] == "LiquidAI/LFM2-350M" and r.hyperparameters["base_revision"] is None
    assert r.hyperparameters["max_seq_length"] == 512  # the preset value the backend now reads
    assert "precision" not in r.hyperparameters  # lora-fast@1 is untouched (immutable)


_TEXT_GENERATION = """
atelier: 1
name: t
task: text_generation
data: {{uri: "file:///x.jsonl", format: jsonl, split: {{validation_fraction: 0.2}}}}
features:
  inputs: [{{name: prompt, type: text}}]
  output: {{name: completion, type: text}}
model: lfm2-350m
hp: {hp}
overrides: {overrides}
export: [gguf]
"""


def _resolve_text_generation(hp="text_generation/lora-fast@1", overrides="{}"):
    from sakura.atelier.registry import resolve
    from sakura.atelier.spec import load_spec

    return resolve(load_spec(_TEXT_GENERATION.format(hp=hp, overrides=overrides)))


def test_lora_tuned_resolves_and_the_backend_reads_every_key_of_it():
    # A preset is immutable once released: a key the backend does not read would be dead weight for ever.
    from sakura.atelier.backends import peft
    from sakura.atelier.registry import load_preset

    read_before = {"lora_r", "lora_alpha", "lora_dropout", "epochs", "batch_size", "grad_accum", "learning_rate",
                   "max_seq_length", "precision"}
    preset = load_preset("text_generation/lora-tuned@1")
    assert set(preset) <= read_before | set(peft.TRAIN_EVAL_KEYS)
    hp = _resolve_text_generation("text_generation/lora-tuned@1").hyperparameters
    assert hp["base_model"] == "LiquidAI/LFM2-350M" and hp["lr_schedule"] == "cosine"
    o = peft._training_options(hp)
    assert (o.lora_r, o.lora_alpha, o.lora_dropout, o.epochs, o.batch_size, o.grad_accum) == (16, 32, 0.05, 3, 4, 4)
    assert (o.learning_rate, o.lr_schedule, o.warmup_ratio, o.min_lr_ratio, o.max_grad_norm) == (2e-4, "cosine", 0.03, 0.1, 1.0)
    assert (o.group_by_length, o.eval_every, o.eval_max_rows, o.gen_eval_rows, o.final_gen_rows) == (True, 0.1, 256, 32, 500)
    assert (o.eval_batch_size, o.early_stop_patience, o.keep_best, o.selection_metric) == (8, 5, True, "loss")
    assert (o.eval_max_new_tokens, o.metrics, o.baseline, o.max_seq_length, o.precision) == ("auto", "auto", True, 2048, "bf16")
    assert o.runtime_eval_rows == 50  # each export is re-scored on the first 50 final rows
    assert o.weight_decay is None and o.extended is True


def test_lora_fast_is_untouched_by_the_schedule_and_evaluation_keys():
    from sakura.atelier.backends import peft
    from sakura.atelier.registry import load_preset

    assert load_preset("text_generation/lora-fast@1") == {
        "lora_r": 8, "lora_alpha": 16, "lora_dropout": 0.05, "epochs": 1, "batch_size": 2, "learning_rate": 0.0002,
        "max_seq_length": 512}
    hp = _resolve_text_generation().hyperparameters
    assert not set(hp) & set(peft.TRAIN_EVAL_KEYS) and peft._training_options(hp).extended is False


def test_resolve_refuses_a_wrong_schedule_or_evaluation_value_before_a_run():
    import pytest

    with pytest.raises(ValueError, match="warmup_ratio must be a number from 0 to 0.5"):
        _resolve_text_generation(overrides="{warmup_ratio: 0.9}")
    with pytest.raises(ValueError, match="lr_schedule must be one of"):
        _resolve_text_generation(overrides="{lr_schedule: exponential}")
    with pytest.raises(ValueError, match="max_grad_norm does not apply with objective: dpo"):
        _resolve_text_generation("text_generation/dpo@1", "{max_grad_norm: 1.0}")
    # the same keys on another backend are that backend's own (open_clip and the classifiers read weight_decay)
    assert _resolve_text_generation(overrides="{lr_schedule: cosine, warmup_ratio: 0.1}").backend == "peft"
