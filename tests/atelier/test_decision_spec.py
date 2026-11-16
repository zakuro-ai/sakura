"""The `decision` (System One) task at the spec and registry layer: routing, a spec
without `features` or `split`, the model/preset/export tables and the base-model
allowlist. The data, loss and calibration tests are in test_decision_*.py."""

from __future__ import annotations

import pytest

from sakura.atelier.backends import MODEL_BACKENDS, TASK_BACKENDS, backend_name
from sakura.atelier.registry import list_presets, load_models, load_preset, resolve
from sakura.atelier.spec import dump_spec, load_spec

DECISION = """
atelier: 1
name: ticket-triage
task: decision
data: {uri: "file:///x.jsonl", sha256: "", format: jsonl}
model: hf-decision-lm
hp: decision/lora-forge@1
overrides: {base_model: Qwen/Qwen3.5-2B-Base, epochs: 3}
export: [safetensors]
seed: 42
"""


# ----------------------------------------------------------------- routing

def test_decision_and_its_model_route_to_peft():
    assert TASK_BACKENDS["decision"] == "peft"
    assert MODEL_BACKENDS["hf-decision-lm"] == "peft"
    assert backend_name("decision", "hf-decision-lm") == "peft"


def test_the_model_id_wins_over_the_task_as_for_hf_causal_lm():
    assert backend_name("text_generation", "hf-decision-lm") == "peft"
    assert backend_name("decision", "hf-decision-lm") == "peft"


def test_existing_routes_do_not_move():
    assert backend_name("text_classification", "ludwig_text") == "ludwig"
    assert backend_name("text_generation", "hf-causal-lm") == "peft"
    assert backend_name("text_generation", "lfm2-350m") == "peft"


#: The tables as the zak-forge runner (`ENGINE_TASK_BACKENDS`) and the hub copy them by hand.
#: Changing a line here means changing it there.
MIRRORED_TASK_BACKENDS = {
    "image_classification": "ludwig",
    "image_regression": "open_clip",
    "audio_classification": "ludwig",
    "tabular_classification": "ludwig",
    "tabular_regression": "ludwig",
    "object_detection": "hf_detection",
    "text_generation": "peft",
    "speech_recognition": "deepspeech",
    "time_series_forecasting": "ludwig",
    "text_classification": "ludwig",
    "decision": "peft",
    "risk_simulation": "risk_sim",
    "reinforcement_learning": "rl",
}
MIRRORED_MODEL_BACKENDS = {
    "tabpfn_v2": "tabpfn",
    "whisper-small": "whisper",
    "whisper-large-v3-turbo": "whisper",
    "hf-decision-lm": "peft",
    # templates of `hf:<repo>@<sha>` classifiers (2026-11-09): base_model required
    "hf-image-classifier": "hf_image_classification",
    "hf-audio-classifier": "hf_audio_classification",
    "hf-text-classifier": "hf_text_classification",
    # classic ML on tables (2026-11-09)
    "lightgbm": "classic_ml",
    "xgboost": "classic_ml",
    "catboost": "classic_ml",
    "random_forest": "classic_ml",
    "hist_gradient_boosting": "classic_ml",
}


def test_the_routing_tables_are_exactly_the_mirrored_ones():
    assert TASK_BACKENDS == MIRRORED_TASK_BACKENDS
    assert MODEL_BACKENDS == MIRRORED_MODEL_BACKENDS


# -------------------------------------------------------------------- spec

def test_a_decision_spec_has_no_features_and_no_split():
    spec = load_spec(DECISION)
    assert spec.task == "decision" and spec.features is None and spec.data.split is None


def test_the_dump_of_a_featureless_decision_spec_names_neither_key():
    dumped = dump_spec(load_spec(DECISION))
    assert "features" not in dumped and "split" not in dumped
    assert dump_spec(load_spec(dumped)) == dumped  # and it replays


def test_the_hub_spec_for_decision_loads_unchanged():
    # What the hub's planner writes (zak-marketplace decision_hf_decision_lm.yaml): no features, no split.
    hub = """
model_type: atelier
atelier: 1
name: ticket-triage
task: decision
data:
  uri: "zc://dataset/ticket-triage@sha256:%s"
  format: jsonl
guessed: []
confirmed: true
model: hf-decision-lm
hp: decision/lora-forge@1
overrides: {base_model: Qwen/Qwen3.5-2B-Base, epochs: 3}
seed: 42
export: [safetensors]
placement: {strategy: best_price, gpus: 1, min_vram_gb: 16}
""" % ("a" * 64)
    assert load_spec(hub).data.split is None


NON_DECISION = """
atelier: 1
name: t
task: tabular_classification
data: {uri: "file:///x.csv", format: csv, split: {validation_fraction: 0.2}}
features:
  inputs: [{name: a, type: number}]
  output: {name: y, type: category}
model: ludwig_ecd
hp: tabular/ecd@1
export: [onnx]
"""


def test_every_other_task_still_requires_features():
    without = NON_DECISION.replace("features:\n  inputs: [{name: a, type: number}]\n  output: {name: y, type: category}\n", "")
    assert "features" not in without
    with pytest.raises(ValueError, match="features"):
        load_spec(without)


def test_every_other_task_still_requires_a_split():
    with pytest.raises(ValueError, match="split"):
        load_spec(NON_DECISION.replace(", split: {validation_fraction: 0.2}", ""))


def test_decision_refuses_features_and_split_instead_of_ignoring_them():
    with_features = DECISION + "features:\n  inputs: [{name: a, type: text}]\n  output: {name: y, type: category}\n"
    with pytest.raises(ValueError, match="features does not apply to task decision"):
        load_spec(with_features)
    with pytest.raises(ValueError, match="splits by state"):
        load_spec(DECISION.replace('sha256: ""', 'sha256: "", split: {validation_fraction: 0.2}'))


def test_a_non_decision_dump_is_unchanged_by_the_optional_keys():
    # features/split stay in the dump of every other task, so their resolved.yaml and spec hash do not move
    dumped = dump_spec(load_spec(NON_DECISION))
    assert "features:" in dumped and "split:" in dumped


# ---------------------------------------------------------------- registry

def test_the_model_entry_is_peft_safetensors_only():
    entry = load_models()["hf-decision-lm"]
    assert (entry["backend"], entry["tasks"], entry["exports"]) == ("peft", ["decision"], ["safetensors"])
    assert entry["hp_required"] == ["base_model"]


def test_the_preset_is_listed_and_loads():
    assert "decision/lora-forge@1" in list_presets("decision")
    hp = load_preset("decision/lora-forge@1")
    assert hp["epochs"] == 3 and hp["lora_r"] > 0 and hp["learning_rate"] > 0
    assert hp["prompt_template"] == "decision/v1"
    allow = hp["base_model_allowlist"]
    assert set(allow) == {"Qwen/Qwen3.5-2B-Base", "Qwen/Qwen3.5-4B-Base"}


def test_resolve_a_decision_spec():
    r = resolve(load_spec(DECISION))
    assert r.backend == "peft"
    assert r.hyperparameters["base_model"] == "Qwen/Qwen3.5-2B-Base" and r.hyperparameters["epochs"] == 3


def test_the_base_model_is_required():
    with pytest.raises(ValueError, match=r"requires hyperparameters \['base_model'\]"):
        resolve(load_spec(DECISION.replace("base_model: Qwen/Qwen3.5-2B-Base, ", "")))


def test_a_base_off_the_allowlist_is_refused_with_a_named_error():
    with pytest.raises(ValueError, match="base model not on the allowlist"):
        resolve(load_spec(DECISION.replace("Qwen/Qwen3.5-2B-Base", "evil/model")))


def test_the_allowlist_names_the_base_and_what_is_allowed():
    with pytest.raises(ValueError) as e:
        resolve(load_spec(DECISION.replace("Qwen/Qwen3.5-2B-Base", "evil/model")))
    assert "evil/model" in str(e.value) and "Qwen/Qwen3.5-4B-Base" in str(e.value)


def test_an_override_cannot_widen_the_allowlist():
    widened = DECISION.replace("epochs: 3}", "epochs: 3, base_model_allowlist: {evil/model: null}}") \
                      .replace("Qwen/Qwen3.5-2B-Base", "evil/model")
    with pytest.raises(ValueError, match="base_model_allowlist"):
        resolve(load_spec(widened))


def test_the_allowlist_holds_in_a_replay_of_resolved_yaml():
    from sakura.atelier.registry import resolved_yaml_text

    resolved = resolve(load_spec(DECISION))
    replay = load_spec(resolved_yaml_text(resolved, "d" * 64))
    assert replay.hyperparameters is not None and resolve(replay).backend == "peft"
    tampered = replay.model_copy(update={"hyperparameters": {**replay.hyperparameters, "base_model": "evil/model"}})
    with pytest.raises(ValueError, match="base model not on the allowlist"):
        resolve(tampered)


def test_gguf_is_refused_before_anything_runs():
    with pytest.raises(ValueError, match=r"cannot export \['gguf'\]"):
        resolve(load_spec(DECISION.replace("[safetensors]", "[gguf]")))


def test_an_empty_export_is_refused():
    with pytest.raises(ValueError, match="export is empty"):
        resolve(load_spec(DECISION.replace("[safetensors]", "[]")))


def test_a_task_model_mismatch_is_refused_both_ways():
    with pytest.raises(ValueError, match="does not support task"):
        resolve(load_spec(DECISION.replace("hf-decision-lm", "hf-causal-lm")))
    other = DECISION.replace("task: decision", "task: text_generation").replace("decision/lora-forge@1", "text_generation/lora-fast@1")
    other = other.replace('data: {uri: "file:///x.jsonl", sha256: "", format: jsonl}',
                          'data: {uri: "file:///x.jsonl", format: jsonl, split: {validation_fraction: 0.2}}')
    other += "features:\n  inputs: [{name: prompt, type: text}]\n  output: {name: completion, type: text}\n"
    with pytest.raises(ValueError, match="does not support task"):
        resolve(load_spec(other))


PINNED_2B = "b1485b2fa6dfa1287294f269f5fb618e03d52d7c"


def test_an_unset_base_revision_resolves_to_the_pinned_commit():
    assert resolve(load_spec(DECISION)).hyperparameters["base_revision"] == PINNED_2B
    same = DECISION.replace("epochs: 3}", f"epochs: 3, base_revision: {PINNED_2B}}}")
    assert resolve(load_spec(same)).hyperparameters["base_revision"] == PINNED_2B


@pytest.mark.parametrize("revision", ["main", "abc123", "refs/pr/1"])
def test_any_other_revision_of_an_allowed_base_is_refused(revision):
    # the allowlist is a licence review of one commit; a later or community revision has not been reviewed
    other = DECISION.replace("epochs: 3}", f"epochs: 3, base_revision: {revision}}}")
    with pytest.raises(ValueError, match="not the reviewed commit"):
        resolve(load_spec(other))


def test_a_replay_cannot_carry_a_widened_allowlist_either():
    from sakura.atelier.registry import resolved_yaml_text

    replay = load_spec(resolved_yaml_text(resolve(load_spec(DECISION)), "d" * 64))
    assert replay.hyperparameters is not None
    widened = {**replay.hyperparameters, "base_model": "evil/model",
               "base_model_allowlist": {**replay.hyperparameters["base_model_allowlist"], "evil/model": None}}
    with pytest.raises(ValueError, match="base_model_allowlist"):
        resolve(replay.model_copy(update={"hyperparameters": widened}))
