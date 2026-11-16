"""`materialise` for task decision, and the options the peft backend reads for it."""

from __future__ import annotations

import json

import pytest

from sakura.atelier.data import materialise
from sakura.atelier.decision import DecisionDataError, decision_options
from sakura.atelier.spec import load_spec
from tests.atelier.decision_fakes import TOPIC, write_dataset

SPEC = """
atelier: 1
name: ticket-triage
task: decision
data: {uri: "data.jsonl", format: jsonl}
model: hf-decision-lm
hp: decision/lora-forge@1
overrides: {base_model: Qwen/Qwen3.5-2B-Base}
export: [safetensors]
seed: 7
"""


@pytest.fixture
def here(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    return tmp_path


def test_a_decision_spec_materialises_into_train_eval_and_calibration(here):
    write_dataset(here / "data.jsonl")
    data = materialise(load_spec(SPEC), here / "data")
    assert (len(data.train), len(data.validation), len(data.extra["calibration"])) == (72, 9, 9)
    assert data.labels is None and len(data.manifest_sha256) == 64
    assert data.extra["decision"]["stats"]["usable_rows"] == 90
    assert any("only 90 usable rows" in w for w in data.extra["decision"]["warnings"])


def test_no_state_is_in_two_of_the_three_splits(here):
    write_dataset(here / "data.jsonl")
    data = materialise(load_spec(SPEC), here / "data")
    parts = [{r["state"] for r in p} for p in (data.train, data.validation, data.extra["calibration"])]
    assert not (parts[0] & parts[1] or parts[0] & parts[2] or parts[1] & parts[2])


def test_the_split_follows_the_spec_seed(here):
    write_dataset(here / "data.jsonl")
    a = materialise(load_spec(SPEC), here / "d1")
    b = materialise(load_spec(SPEC), here / "d2")
    c = materialise(load_spec(SPEC.replace("seed: 7", "seed: 8")), here / "d3")
    assert a.validation == b.validation and a.validation != c.validation


def test_the_dataset_bytes_are_pinned_by_the_manifest(here):
    write_dataset(here / "data.jsonl")
    sha = materialise(load_spec(SPEC), here / "d1").manifest_sha256
    (here / "data.jsonl").write_text((here / "data.jsonl").read_text() + json.dumps(
        {"state": "one more", "questions": {"topic": TOPIC}, "answers": {"topic": "bug"}}) + "\n")
    assert materialise(load_spec(SPEC), here / "d2").manifest_sha256 != sha
    pinned = SPEC.replace('format: jsonl}', f'format: jsonl, sha256: "{sha}"}}')
    with pytest.raises(ValueError, match="data.sha256 mismatch"):
        materialise(load_spec(pinned), here / "d3")


def test_a_short_dataset_fails_by_name(here):
    write_dataset(here / "data.jsonl", n=20)
    with pytest.raises(DecisionDataError, match=r"too few usable rows \(20; at least 50 needed\)"):
        materialise(load_spec(SPEC), here / "data")


def test_a_malformed_row_fails_with_its_line_number(here):
    rows = write_dataset(here / "data.jsonl")
    rows[4]["answers"]["topic"] = "refund"
    (here / "data.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    with pytest.raises(DecisionDataError, match="row 5: answer 'refund' to 'topic'"):
        materialise(load_spec(SPEC), here / "data")


@pytest.mark.parametrize("fmt", ["csv", "parquet", "hf"])
def test_only_jsonl_is_a_decision_format(here, fmt):
    with pytest.raises(ValueError, match="decision data must be jsonl"):
        materialise(load_spec(SPEC.replace("format: jsonl", f"format: {fmt}")), here / "data")


# ------------------------------------------------------------------ options

HP = {"base_model": "Qwen/Qwen3.5-2B-Base", "base_revision": "abc",
      "base_model_allowlist": {"Qwen/Qwen3.5-2B-Base": "abc"}}


def test_options_default_to_the_preset_values():
    o = decision_options(HP)
    assert (o.epochs, o.lora_r, o.lora_alpha, o.precision, o.max_seq_length) == (3, 16, 32, "bf16", 1024)
    assert o.base_model == "Qwen/Qwen3.5-2B-Base" and o.base_revision == "abc"
    assert o.lora_target_modules == "all-linear" and o.grad_accum == 1
    assert o.gradient_checkpointing is True


def test_gradient_checkpointing_can_be_turned_off():
    assert decision_options({**HP, "gradient_checkpointing": False}).gradient_checkpointing is False


def test_options_read_the_hyperparameters():
    o = decision_options({**HP, "epochs": 2, "learning_rate": 0.01, "batch_size": 4, "grad_accum": 3,
                          "lora_target_modules": ["q_proj", "v_proj"], "precision": "fp32", "max_seq_length": 64})
    assert (o.epochs, o.learning_rate, o.batch_size, o.grad_accum, o.precision, o.max_seq_length) == (2, 0.01, 4, 3, "fp32", 64)
    assert o.lora_target_modules == ["q_proj", "v_proj"]


def test_a_base_off_the_allowlist_is_refused_again_at_train_time():
    with pytest.raises(DecisionDataError, match="base model not on the allowlist"):
        decision_options({**HP, "base_model": "evil/model"})
    with pytest.raises(DecisionDataError, match="base model not on the allowlist"):
        decision_options({"base_model": "Qwen/Qwen3.5-2B-Base"})  # no allowlist at all is not "allow everything"


def test_a_revision_other_than_the_reviewed_one_is_refused_at_train_time_too():
    with pytest.raises(DecisionDataError, match="not the reviewed commit"):
        decision_options({**HP, "base_revision": "somewhere-else"})


@pytest.mark.parametrize("extra,match", [
    ({"prompt_template": "alpaca"}, "prompt_template"),
    ({"precision": "fp8"}, "precision"),
    ({"epochs": 0}, "epochs"),
    ({"batch_size": 0}, "batch_size"),
])
def test_bad_options_are_refused(extra, match):
    with pytest.raises(ValueError, match=match):
        decision_options({**HP, **extra})
