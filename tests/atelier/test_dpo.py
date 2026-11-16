"""DPO post-training in the peft backend (SAK-0019): ``hp: text_generation/dpo@1`` (``objective: dpo``) on
``{prompt, chosen, rejected}`` rows.

The validation rules, the loss and the log-probability are unit tests. The end-to-end test runs
``python -m sakura.atelier run`` on ``synthetic://preference`` with a random two-layer Llama built offline and
checks that the policy ends up preferring ``chosen`` more often than the reference (its own starting weights)."""

from __future__ import annotations

import json
import math

import pytest

from sakura.atelier.data import materialise
from sakura.atelier.registry import list_presets, resolve
from sakura.atelier.spec import load_spec
from tests.atelier.test_smoke_runs import _resolved, _run, _tiny_causal_lm, _train_points

SPEC = """
atelier: 1
name: dpo
task: text_generation
data: {{uri: "{uri}", format: jsonl, split: {{validation_fraction: 0.25}}}}
features:
  inputs: [{{name: prompt, type: text}}]
  output: {{name: chosen, type: text}}
model: hf-causal-lm
hp: {hp}
overrides: {overrides}
export: [safetensors]
seed: 5
{extra}
"""
PREFERENCE = "synthetic://preference?rows=64&seed=3&keys=4"


def _spec(hp="text_generation/dpo@1", uri=PREFERENCE, extra="", **overrides):
    return load_spec(SPEC.format(hp=hp, uri=uri, extra=extra, overrides=json.dumps({"base_model": "some/base", **overrides})))


# ------------------------------------------------------------------------------------------- validation

def test_the_dpo_preset_resolves_to_the_objective_and_is_listed():
    assert "text_generation/dpo@1" in list_presets("text_generation")
    hp = resolve(_spec()).hyperparameters
    assert hp["objective"] == "dpo" and hp["dpo_beta"] == 0.1 and hp["lora_r"] == 8
    assert resolve(_spec(dpo_beta=0.5, rejected_column="worse")).hyperparameters["dpo_beta"] == 0.5
    # the objective is a hyperparameter: any text_generation preset can be turned into DPO by an override
    assert resolve(_spec(hp="text_generation/lora-fast@1", objective="dpo")).hyperparameters["objective"] == "dpo"
    assert "objective" not in resolve(_spec(hp="text_generation/lora-fast@1")).hyperparameters


@pytest.mark.parametrize("hp, overrides, message", [
    ("text_generation/lora-fast@1", {"objective": "ppo"}, "objective must be one of ['sft', 'dpo']"),
    ("text_generation/lora-fast@1", {"dpo_beta": 0.1}, "dpo_beta only applies with objective: dpo"),
    ("text_generation/lora-fast@1", {"rejected_column": "worse"}, "rejected_column only applies with objective: dpo"),
    ("text_generation/dpo@1", {"dpo_beta": 0}, "dpo_beta must be a number above 0"),
    ("text_generation/dpo@1", {"dpo_beta": "strong"}, "dpo_beta must be a number above 0"),
    ("text_generation/dpo@1", {"rejected_column": ""}, "rejected_column must name the dataset column"),
    ("text_generation/dpo@1", {"init": "random"}, "objective: dpo cannot be combined with init: random"),
    ("text_generation/dpo@1", {"teacher": "/runs/t"}, "objective: dpo cannot be combined with a teacher"),
])
def test_a_bad_dpo_recipe_is_refused_by_name(hp, overrides, message):
    with pytest.raises(ValueError) as err:
        resolve(_spec(hp=hp, **overrides))
    assert message in str(err.value)


def test_dpo_is_refused_outside_text_generation_on_peft():
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
overrides: {objective: dpo}
export: [joblib]
""")
    with pytest.raises(ValueError, match="objective: dpo is not supported for task 'tabular_classification'"):
        resolve(spec)


# ------------------------------------------------------------------------------------------- data

def test_preference_rows_materialise_with_the_rejected_completion(tmp_path):
    data = materialise(_spec(), tmp_path / "data")
    assert len(data.train) == 48 and len(data.validation) == 16
    assert set(data.train[0]) == {"prompt", "completion", "rejected"}
    assert all(r["completion"] != r["rejected"] for r in [*data.train, *data.validation])
    # the same file under the next-token objective has no rejected completion
    plain = materialise(_spec(hp="text_generation/lora-fast@1"), tmp_path / "plain")
    assert set(plain.train[0]) == {"prompt", "completion"}


def test_a_dataset_without_the_rejected_column_fails_by_name(tmp_path):
    with pytest.raises(ValueError, match="objective: dpo needs a 'rejected' column"):
        materialise(_spec(uri="synthetic://instruct?rows=32&seed=3").model_copy(update={
            "features": load_spec(SPEC.format(hp="text_generation/dpo@1", uri=PREFERENCE, extra="",
                                              overrides="{base_model: b}").replace("name: chosen", "name: completion")).features}),
            tmp_path / "data")
    path = tmp_path / "pairs.jsonl"
    path.write_text("".join(json.dumps({"prompt": f"q{i}", "chosen": "yes", "worse": "no"}) + "\n" for i in range(8)))
    data = materialise(_spec(uri=f"file://{path}", rejected_column="worse"), tmp_path / "named")
    assert data.train[0]["rejected"] == "no"


# ------------------------------------------------------------------------------------------- the loss

def test_the_dpo_loss_is_log_two_at_the_reference_and_falls_as_the_margin_grows():
    torch = pytest.importorskip("torch")
    from sakura.atelier.backends import dpo

    ref_c, ref_r = torch.tensor([-3.0, -5.0]), torch.tensor([-4.0, -2.0])
    assert float(dpo.dpo_loss(ref_c, ref_r, ref_c, ref_r, 0.1)) == pytest.approx(math.log(2))
    assert dpo.LOSS_AT_REFERENCE == pytest.approx(math.log(2))
    better = float(dpo.dpo_loss(ref_c + 2.0, ref_r - 2.0, ref_c, ref_r, 0.5))
    assert better == pytest.approx(math.log(1 + math.exp(-0.5 * 4.0)))
    worse = float(dpo.dpo_loss(ref_c - 2.0, ref_r + 2.0, ref_c, ref_r, 0.5))
    assert better < math.log(2) < worse
    stats = dpo.preference_stats([-1.0, -5.0, -2.0], [-2.0, -3.0, -2.0], [-1.0, -5.0, -2.0], [-2.0, -3.0, -2.0], 0.1)
    assert stats["preference_accuracy"] == pytest.approx(1 / 3)  # a tie is not a preference
    assert stats["loss"] == pytest.approx(math.log(2)) and stats["mean_margin"] == pytest.approx(-1 / 3)
    moved = dpo.preference_stats([0.0, 0.0], [-10.0, -10.0], [0.0, 0.0], [0.0, 0.0], 0.5)
    assert moved["loss"] == pytest.approx(math.log(1 + math.exp(-5.0))) and moved["preference_accuracy"] == 1.0


def test_the_completion_log_probability_sums_the_tokens_after_the_prompt(tmp_path):
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    from sakura.atelier.backends import dpo

    base = _tiny_causal_lm(tmp_path / "base")
    tok = transformers.AutoTokenizer.from_pretrained(str(base))
    tok.pad_token = tok.eos_token
    model = transformers.AutoModelForCausalLM.from_pretrained(str(base)).eval()
    pairs = [dpo.encode_pair(tok, "code for abc:", "xy", 64), dpo.encode_pair(tok, "q?", "longer answer", 64)]
    ids, prompt_len = pairs[0]
    assert prompt_len == len(tok("code for abc:")["input_ids"]) and len(ids) == prompt_len + 3  # x, y, eos
    with torch.no_grad():
        batch = dpo.completion_logprob(model, tok, pairs, "cpu")
        logp = torch.log_softmax(model(input_ids=torch.tensor([ids])).logits[0], -1)
    by_hand = sum(float(logp[i - 1, ids[i]]) for i in range(prompt_len, len(ids)))
    assert float(batch[0]) == pytest.approx(by_hand, rel=1e-4)  # padding the batch changes nothing
    assert float(batch[1]) < 0 and batch.shape == (2,)


# ------------------------------------------------------------------------------------------- end to end

def test_dpo_makes_the_policy_prefer_the_chosen_completion_more_often_than_its_reference(tmp_path):
    pytest.importorskip("transformers")
    pytest.importorskip("peft")
    base = _tiny_causal_lm(tmp_path / "base")
    overrides = {"base_model": str(base), "epochs": 10, "batch_size": 8, "learning_rate": 0.003, "dpo_beta": 2.0,
                 "lora_r": 8, "lora_alpha": 16, "max_seq_length": 64}
    report, out = _run(tmp_path, SPEC.format(hp="text_generation/dpo@1", uri=PREFERENCE, extra="",
                                             overrides=json.dumps(overrides)), "--no-notebook")
    assert report["status"] == "done" and report["backend"] == "peft"
    block = report["preference"]
    assert block["objective"] == "dpo" and block["beta"] == 2.0 and block["n"] == 16
    reference, policy = block["reference"], block["policy"]
    assert reference["loss"] == pytest.approx(math.log(2))  # the reference against itself
    assert policy["preference_accuracy"] > reference["preference_accuracy"]
    assert policy["preference_accuracy"] >= 0.9 and policy["loss"] < reference["loss"]
    assert policy["mean_margin"] > reference["mean_margin"]
    assert report["metrics"] == {"loss": policy["loss"], "preference_accuracy": policy["preference_accuracy"]}
    assert report["provenance"]["objective"] == "dpo"
    train = [p["value"] for p in _train_points(out) if p["name"] == "loss"]
    assert train[0] == pytest.approx(math.log(2), abs=1e-3) and train[-1] < train[0]
    # the delivered merged model, reloaded, prefers the same completions
    row = report["runtime_eval"][0]
    assert row["format"] == "safetensors" and row["metric"] == "preference_accuracy" and row["n"] == 16
    assert row["value"] == pytest.approx(policy["preference_accuracy"]) and row["matches_training"] is True
    assert json.loads((out / "artifacts" / "preprocess.json").read_text())["objective"] == "dpo"
    resolved = _resolved(out)
    assert resolved["hyperparameters"]["objective"] == "dpo" and resolved["hp"] == "text_generation/dpo@1"


def test_a_capped_dpo_run_stops_on_the_step_limit(tmp_path):
    pytest.importorskip("transformers")
    pytest.importorskip("peft")
    base = _tiny_causal_lm(tmp_path / "base")
    overrides = {"base_model": str(base), "epochs": 3, "batch_size": 4, "max_seq_length": 64}
    report, out = _run(tmp_path, SPEC.format(hp="text_generation/dpo@1", uri=PREFERENCE, overrides=json.dumps(overrides),
                                             extra="limits: {max_steps: 3, max_train_rows: 16, max_eval_rows: 4}"),
                       "--no-notebook")
    assert report["stopped_early"]["reason"] == "max_steps" and report["stopped_early"]["step"] == 3
    assert [p["step"] for p in _train_points(out)] == [0, 1, 2]
    assert report["preference"]["n"] == 4 and report["limits_applied"]["max_train_rows"] == 16
