"""Rows without a target and the ``predictions.jsonl`` artifact of the peft text-generation run
(CONTRACTS §2.2, #238 and zakuro-ai/zak-marketplace#969).

A conversation that ends with a user turn has an evaluation-only row: the full context, an empty completion. The
run never trains or measures a loss on it, generates from it at the final evaluation and for the baseline, and
scores it only with the metrics that need no target. Generation is replaced by a table of fixed replies.

Needs ``transformers`` and ``peft`` (the backend's own venv); skipped elsewhere."""

from __future__ import annotations

import json

import pytest

from sakura.atelier.backends import peft as backend_mod
from sakura.atelier.types import RunDir
from tests.atelier.test_peft_training_eval import ALPHABET, _hp, _Metrics
from tests.atelier.tiny_models import tiny_llama_dir

pytest.importorskip("transformers")
pytest.importorskip("peft")

RIGHT = '{"a":1}'
NO_REF = "\n  eval: {final: last_turn, group: conversation, keep: [turn]}"
WITH_REF = "\n  eval: {reference: ref, final: last_turn, group: conversation, keep: [turn]}"
#: validation row i (i = 0..5): conversation c<i // 2>; the even row is an assistant reply (completion RIGHT), the odd
#: one the trailing evaluation-only row (empty completion, the last turn)
TUNED = {0: RIGHT, 1: RIGHT, 2: RIGHT, 3: RIGHT, 4: RIGHT, 5: "not json"}
BASE = {0: "nope", 1: RIGHT, 2: "nope", 3: "nope", 4: "nope", 5: "nope"}


def _table(tmp_path, *, eval_only=True, empty_train=0, val_n=6, only=None):
    train = [{"prompt": f"t{i}=", "completion": RIGHT, "conversation": f"x{i}", "turn": 0, "last_turn": True,
              "ref": RIGHT} for i in range(6)]
    train += [{"prompt": f"e{i}=", "completion": "", "conversation": f"y{i}", "turn": 0, "last_turn": True,
               "ref": RIGHT} for i in range(empty_train)]
    val = []
    for i in range(val_n):
        trailing = (i in only) if only is not None else eval_only and i % 2 == 1
        val.append({"prompt": f"q{i}=", "completion": "" if trailing else RIGHT, "conversation": f"c{i // 2}",
                    "turn": i % 2, "last_turn": trailing or (not eval_only and i % 2 == 1), "ref": RIGHT})
    path = tmp_path / "t.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in [*train, *val]))
    return path, len(train)


def _is_tuned(net):
    """Whether ``net`` answers as the fine-tuned model: the adapter is there and on. The baseline is scored on
    the same wrapped model with its adapter switched off (the starting weights), so the class alone does not say."""
    # `is True`: on an adapter layer `disable_adapters` is a flag; on the transformers model it is a method
    return type(net).__name__.startswith("Peft") and not any(
        getattr(module, "disable_adapters", False) is True for module in net.modules())


def _run(tmp_path, monkeypatch, hp_over, eval_block=NO_REF, *, name="run", eval_only=True, empty_train=0, tuned=TUNED,
         base=BASE, val_n=6, only=None):
    import torch

    from sakura.atelier.data import materialise
    from sakura.atelier.registry import resolve
    from sakura.atelier.spec import load_spec

    path, n_train = _table(_mkdir(tmp_path / name), eval_only=eval_only, empty_train=empty_train, val_n=val_n,
                             only=only)
    spec = load_spec(f"""
atelier: 1
name: conv
task: text_generation
data: {{uri: "file://{path}", format: jsonl, split: {{train: "0:{n_train}", validation: "{n_train}:"}}}}
features:
  inputs: [{{name: prompt, type: text}}]
  output: {{name: completion, type: text}}{eval_block}
model: hf-causal-lm
hp: text_generation/lora-fast@1
export: [safetensors]
seed: 7
""")
    data = materialise(spec, tmp_path / name / "data")

    def fake(net, tokenizer, prompts, max_new_tokens, device):
        table = tuned if _is_tuned(net) else base
        return [table[int(p[1:-1])] for p in prompts]

    monkeypatch.setattr(backend_mod, "_generate", fake)
    hp = _hp(tiny_llama_dir(tmp_path / name / "base", ALPHABET), epochs=1, **hp_over)
    metrics = _Metrics()
    torch.manual_seed(0)
    out = RunDir(tmp_path / name / "run").ensure()
    result = backend_mod.BACKEND.train(resolve(spec.model_copy(update={"hyperparameters": hp})), data, out, metrics, None)
    return result, data, out, metrics


def _mkdir(path):
    path.mkdir(parents=True)
    return path


def _predictions(out):
    return [json.loads(line) for line in (out.artifacts / "predictions.jsonl").read_text().splitlines()]


def test_an_evaluation_only_row_has_no_loss_is_generated_from_and_scored_without_a_target(tmp_path, monkeypatch):
    result, data, out, _ = _run(tmp_path, monkeypatch, {"metrics": ["json_match", "json_valid"], "baseline": True})
    assert [r["completion"] for r in data.validation] == [RIGHT, "", RIGHT, "", RIGHT, ""]

    # reported apart from the rows without a reply token: three evaluation-only rows, three rows the loss uses
    assert result.report["truncation"]["validation"] == {
        "rows": 6, "rows_truncated": 0, "rows_without_target": 0, "rows_eval_only": 3, "rows_used": 3}
    assert result.report["truncation"]["train"] == {
        "rows": 6, "rows_truncated": 0, "rows_without_target": 0, "rows_dropped_empty": 0, "rows_used": 6}
    ge = result.report["generation_eval"]
    assert (ge["rows"], ge["loss_rows"], ge["rows_without_target"]) == (6, 3, 3)
    # the rows that have a headline score: the three evaluation-only rows have none, for either model
    assert (ge["rows_scored"], result.report["baseline"]["n_scored"]) == (3, 3) and result.report["baseline"]["n"] == 6

    # json_match (the headline) is the mean over the three rows that have a target; json_valid over all six
    assert result.metrics["json_match"] == 1.0 and result.metrics["json_valid"] == pytest.approx(5 / 6)
    assert result.report["baseline"]["json_match"] == 0.0 and result.report["baseline"]["json_valid"] == pytest.approx(1 / 6)
    # the evaluation-only rows are in no group for the headline: three groups, each scored on its reply
    assert ge["groups"]["n"] == 3 and ge["groups"]["all_correct"] == 3
    assert ge["groups"]["baseline_all_correct"] == 0

    rows = _predictions(out)
    assert [r["index"] for r in rows] == [0, 1, 2, 3, 4, 5]
    assert rows[1] == {
        "index": 1, "group": "c0", "final": True, "keep": {"turn": 1}, "has_target": False,
        "prediction": RIGHT, "prediction_chars": 7, "scores": {"json_match": None, "json_valid": 1.0},
        "baseline_prediction": RIGHT, "baseline_prediction_chars": 7,
        "baseline_scores": {"json_match": None, "json_valid": 1.0}}
    assert rows[0]["has_target"] is True and rows[0]["final"] is False
    assert rows[0]["scores"] == {"json_match": 1.0, "json_valid": 1.0}
    assert rows[0]["baseline_prediction"] == "nope" and rows[0]["baseline_scores"] == {"json_match": 0.0, "json_valid": 0.0}
    assert rows[5]["prediction"] == "not json" and rows[5]["scores"] == {"json_match": None, "json_valid": 0.0}
    assert not any("prompt" in r or "completion" in r for r in rows)  # no prompt text, no completion text
    assert "q0=" not in (out.artifacts / "predictions.jsonl").read_text()


def test_the_predictions_file_is_a_listed_artifact(tmp_path, monkeypatch):
    import hashlib

    result, _, out, _ = _run(tmp_path, monkeypatch, {"metrics": ["json_valid"]})
    extras = backend_mod.BACKEND.extra_artifacts(result, out)
    path = out.artifacts / "predictions.jsonl"
    assert [(a.path, a.format, a.sha256, a.bytes) for a in extras] == [
        (path, "predictions", hashlib.sha256(path.read_bytes()).hexdigest(), path.stat().st_size)]
    assert "baseline_prediction" not in _predictions(out)[0]  # no baseline: no base-model fields


def test_a_reference_makes_an_evaluation_only_row_scorable_by_every_metric(tmp_path, monkeypatch):
    result, _, out, _ = _run(tmp_path, monkeypatch, {"metrics": ["json_match", "json_valid"]}, WITH_REF)
    rows = _predictions(out)
    assert rows[1]["has_target"] is True and rows[1]["scores"] == {"json_match": 1.0, "json_valid": 1.0}
    assert rows[5]["scores"] == {"json_match": 0.0, "json_valid": 0.0}
    assert result.metrics["json_match"] == pytest.approx(5 / 6)  # six rows, one wrong
    assert result.report["generation_eval"]["rows_without_target"] == 0
    # still no loss on them: an empty completion has nothing to train on
    assert result.report["truncation"]["validation"]["rows_eval_only"] == 3
    assert result.report["generation_eval"]["loss_rows"] == 3


def test_a_run_with_no_evaluation_only_row_writes_the_file_too(tmp_path, monkeypatch):
    result, _, out, _ = _run(tmp_path, monkeypatch, {"metrics": ["json_match"]}, eval_only=False)
    rows = _predictions(out)
    assert len(rows) == 6 and all(r["has_target"] for r in rows)
    assert result.report["truncation"]["validation"]["rows_eval_only"] == 0
    assert result.report["generation_eval"]["rows_without_target"] == 0


def test_metrics_auto_and_the_length_use_the_rows_that_have_a_target(tmp_path, monkeypatch):
    result, _, _, _ = _run(tmp_path, monkeypatch, {"metrics": "auto", "eval_max_new_tokens": "auto"})
    ge = result.report["generation_eval"]
    # the three empty completions would make every target text, not JSON, and pull the length to 0
    assert ge["metrics"] == ["json_field_f1", "json_match", "json_valid"] and ge["max_new_tokens"] == 7 + 8


def test_training_rows_with_an_empty_completion_are_dropped_and_counted(tmp_path, monkeypatch):
    result, _, _, metrics = _run(tmp_path, monkeypatch, {"batch_size": 2}, empty_train=2)
    assert result.report["truncation"]["train"] == {
        "rows": 8, "rows_truncated": 0, "rows_without_target": 0, "rows_dropped_empty": 2, "rows_used": 6}
    assert len(metrics.series("loss", "train")) == 3  # six rows in batches of two
    assert result.report["training"]["steps_planned"] == 3


def test_without_features_eval_an_empty_completion_is_a_row_as_before(tmp_path, monkeypatch):
    result, data, out, metrics = _run(tmp_path, monkeypatch, {"batch_size": 2}, "", empty_train=2)
    assert result.report["truncation"]["train"] == {"rows": 8, "rows_truncated": 0, "rows_without_target": 0, "rows_used": 8}
    assert result.report["truncation"]["validation"]["rows_used"] == 6 and len(metrics.series("loss", "train")) == 4
    assert not (out.artifacts / "predictions.jsonl").exists()


def test_a_spec_without_the_extended_keys_gets_no_predictions_file(tmp_path, monkeypatch):
    result, _, out, _ = _run(tmp_path, monkeypatch, {})
    assert not (out.artifacts / "predictions.jsonl").exists()
    assert backend_mod.BACKEND.extra_artifacts(result, out) == []
    assert set(result.report) == {"truncation", "training"}


def test_the_stored_prediction_text_is_capped(tmp_path, monkeypatch):
    monkeypatch.setattr(backend_mod, "PREDICTION_CHARS_CAP", 5)
    _, _, out, _ = _run(tmp_path, monkeypatch, {"metrics": ["json_valid"], "baseline": True})
    row = _predictions(out)[0]
    assert row["prediction"] == RIGHT[:5] and row["prediction_chars"] == 7
    assert row["baseline_prediction"] == "nope"[:5] and row["baseline_prediction_chars"] == 4


def test_the_cap_is_8192_characters():
    import importlib

    assert importlib.import_module("sakura.atelier.backends.peft").PREDICTION_CHARS_CAP == 8192


def test_a_validation_set_of_only_evaluation_only_rows_is_refused_in_words(tmp_path, monkeypatch):
    import json as _json

    path, n_train = _table(_mkdir(tmp_path / "r"))
    rows = [_json.loads(line) for line in path.read_text().splitlines()]
    rows = rows[:n_train] + [{**r, "completion": ""} for r in rows[n_train:]]
    path.write_text("".join(_json.dumps(r) + "\n" for r in rows))
    from sakura.atelier.data import materialise
    from sakura.atelier.registry import resolve
    from sakura.atelier.spec import load_spec

    spec = load_spec(f"""
atelier: 1
name: conv
task: text_generation
data: {{uri: "file://{path}", format: jsonl, split: {{train: "0:{n_train}", validation: "{n_train}:"}}}}
features:
  inputs: [{{name: prompt, type: text}}]
  output: {{name: completion, type: text}}{NO_REF}
model: hf-causal-lm
hp: text_generation/lora-fast@1
export: [safetensors]
""")
    data = materialise(spec, tmp_path / "d")
    hp = _hp(tiny_llama_dir(tmp_path / "base", ALPHABET), metrics=["json_valid"])
    with pytest.raises(ValueError, match="every validation row is evaluation-only"):
        backend_mod.BACKEND.train(resolve(spec.model_copy(update={"hyperparameters": hp})), data,
                                  RunDir(tmp_path / "run").ensure(), _Metrics(), None)


def test_runtime_eval_skips_evaluation_only_rows_and_reports_how_many(tmp_path, monkeypatch):
    result, data, out, _ = _run(tmp_path, monkeypatch, {"metrics": ["json_valid", "json_match"]})
    # the training-side headline (json_valid) also counted the evaluation-only rows: 5 of 6 are valid JSON ...
    assert result.metrics["json_valid"] == pytest.approx(5 / 6)
    artifact = backend_mod.BACKEND.export(result, "safetensors", out)
    monkeypatch.setattr(backend_mod, "_hf_generate", lambda model_dir, prompt, n: TUNED[int(prompt[1:-1])])
    ev = backend_mod.BACKEND.runtime_eval(artifact, data)
    # ... and the export is re-scored on the three rows that have a target (all valid), compared on those same rows
    assert (ev.metric, ev.n, ev.value, ev.rows_eval_only) == ("json_valid", 3, 1.0, 3)
    assert ev.train_n == 3 and ev.train_value == 1.0 and ev.matches_training is None


# ------------------------------------------------------------------ evaluation-only rows are never sampled away

ONLY = {3, 11, 17, 25, 33}
EVERY_RIGHT = {i: RIGHT for i in range(40)}


def _generated(monkeypatch):
    """Records the validation rows each generation pass reads: ``(is the tuned model, [row indices])``."""
    calls = []
    real = backend_mod._generate_bounded

    def wrapped(net, tokenizer, prompts, *a, **kw):
        calls.append((_is_tuned(net), sorted(int(p[1:-1]) for p in prompts)))
        return real(net, tokenizer, prompts, *a, **kw)

    monkeypatch.setattr(backend_mod, "_generate_bounded", wrapped)
    return calls


def _sampled(tmp_path, monkeypatch, name, **over):
    calls = _generated(monkeypatch)
    result, _, out, _ = _run(tmp_path, monkeypatch, {"metrics": ["json_valid"], "baseline": True,
                                                    "final_gen_rows": 10, **over},
                             name=name, val_n=40, only=ONLY, tuned=EVERY_RIGHT, base=EVERY_RIGHT)
    return result, out, calls


def test_the_final_evaluation_and_the_baseline_generate_from_the_sample_and_every_evaluation_only_row(tmp_path, monkeypatch):
    result, out, calls = _sampled(tmp_path, monkeypatch, "all")
    lines = _predictions(out)
    assert len(lines) == 15 and sum(not r["has_target"] for r in lines) == 5
    assert {r["index"] for r in lines if not r["has_target"]} == ONLY
    final = [rows for tuned, rows in calls if tuned][-1]
    baseline = [rows for tuned, rows in calls if not tuned]
    assert sorted(r["index"] for r in lines) == final and baseline == [final]
    assert ONLY <= set(final) and len(set(final) - ONLY) == 10
    # the periodic evaluation (the other tuned pass) reads none of them
    periodic = [rows for tuned, rows in calls if tuned][:-1]
    assert periodic and not any(ONLY & set(rows) for rows in periodic)
    ge = result.report["generation_eval"]
    assert (ge["rows"], ge["rows_eval_only"]) == (15, 5)


def test_eval_only_gen_rows_caps_the_evaluation_only_rows_added_in_the_seeded_order(tmp_path, monkeypatch):
    _, out, calls = _sampled(tmp_path, monkeypatch, "cap", eval_only_gen_rows=2)
    only = [r["index"] for r in _predictions(out) if not r["has_target"]]
    first_two = [j for j in backend_mod._eval_order(40, 7) if j in ONLY][:2]
    assert sorted(only) == sorted(first_two) and len(_predictions(out)) == 12
    _, out0, _ = _sampled(tmp_path, monkeypatch, "none", eval_only_gen_rows=0)
    assert len(_predictions(out0)) == 10


def test_eval_only_gen_rows_must_be_a_whole_number_from_zero_or_null():
    assert backend_mod._training_options({"eval_only_gen_rows": 0}).eval_only_gen_rows == 0
    assert backend_mod._training_options({"eval_only_gen_rows": None}).eval_only_gen_rows is None
    with pytest.raises(ValueError, match="eval_only_gen_rows must be a whole number"):
        backend_mod._training_options({"eval_only_gen_rows": -1})
