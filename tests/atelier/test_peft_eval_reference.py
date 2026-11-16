"""Per-row reference, final-turn scope and per-group aggregates of the peft text-generation evaluation
(CONTRACTS §2.2, #238), on a tiny random model on a CPU. Generation is replaced by a table of
fixed replies, so every score below is computed by hand.

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

RIGHT, WRONG = '{"a":1}', '{"a":2}'
#: reply of the model to the validation prompt "q<i>=" (i = 0..5): conversations c0 = rows 0-1, c1 = 2-3, c2 = 4-5;
#: the second row of each is the final turn. Hand-computed against the reference {"a":1}:
REPLIES = {0: RIGHT, 1: RIGHT, 2: RIGHT, 3: WRONG, 4: WRONG, 5: RIGHT}


def _table(tmp_path):
    rows = [{"prompt": f"t{i}=", "completion": "ok", "conversation": f"x{i}", "turn": 0, "last_turn": True,
             "ref": RIGHT} for i in range(6)]  # the training rows
    rows += [{"prompt": f"q{i}=", "completion": "ok", "conversation": f"c{i // 2}", "turn": i % 2,
              "last_turn": i % 2 == 1, "ref": RIGHT, "state": json.dumps({"i": i})} for i in range(6)]
    path = tmp_path / "t.jsonl"
    path.write_text("".join(json.dumps({"state": "", **r}) + "\n" for r in rows))
    return path


def _spec_text(path, eval_block):
    return f"""
atelier: 1
name: conv
task: text_generation
data: {{uri: "file://{path}", format: jsonl, split: {{train: "0:6", validation: "6:"}}}}
features:
  inputs: [{{name: prompt, type: text}}]
  output: {{name: completion, type: text}}{eval_block}
model: hf-causal-lm
hp: text_generation/lora-fast@1
export: [safetensors]
seed: 7
"""


EVAL = "\n  eval: {reference: ref, final: last_turn, group: conversation, keep: [turn, state]}"


def _run(tmp_path, hp_over, eval_block=EVAL, name="run", replies=None, monkeypatch=None):
    import torch

    from sakura.atelier.data import materialise
    from sakura.atelier.registry import resolve
    from sakura.atelier.spec import load_spec

    spec = load_spec(_spec_text(_table(tmp_path), eval_block))
    data = materialise(spec, tmp_path / name / "data")
    seen: list[str] = []
    table = REPLIES if replies is None else replies

    def fake(net, tokenizer, prompts, max_new_tokens, device):
        seen.extend(prompts)
        return [table[int(p[1:-1])] for p in prompts]

    monkeypatch.setattr(backend_mod, "_generate", fake)
    hp = _hp(tiny_llama_dir(tmp_path / name / "base", ALPHABET), epochs=1, eval_max_new_tokens=8, **hp_over)
    metrics = _Metrics()
    torch.manual_seed(0)
    result = backend_mod.BACKEND.train(resolve(spec.model_copy(update={"hyperparameters": hp})), data,
                                       RunDir(tmp_path / name / "run").ensure(), metrics, None)
    out = RunDir(tmp_path / name / "run")
    return result, data, out, seen


def test_the_headline_is_computed_against_the_reference_on_every_row_and_per_group(tmp_path, monkeypatch):
    result, data, out, seen = _run(tmp_path, {"metrics": ["json_match", "json_valid"]}, monkeypatch=monkeypatch)
    # completion is "ok" on every row: only the reference can make a prediction right
    assert result.metrics["json_match"] == pytest.approx(4 / 6) and result.metrics["json_valid"] == 1.0
    ge = result.report["generation_eval"]
    assert ge["scope"] == "all" and ge["rows"] == 6
    assert ge["columns"] == {"reference": "ref", "final": "last_turn", "group": "conversation",
                             "keep": ["turn", "state"]}
    # c0 (rows 0, 1) is right on both; c1 and c2 each have a wrong row
    assert ge["groups"] == {"column": "conversation", "metric": "json_match", "n": 3, "all_correct": 1,
                            "all_correct_share": pytest.approx(1 / 3)}
    assert all(not p.startswith("{") for p in seen)  # the reference is never fed to the model


def test_final_scope_scores_generation_on_final_rows_only_and_the_loss_keeps_every_row(tmp_path, monkeypatch):
    result, data, out, seen = _run(tmp_path, {"metrics": ["json_match"], "eval_scope": "final"}, monkeypatch=monkeypatch)
    assert result.metrics["json_match"] == pytest.approx(2 / 3)  # rows 1, 3, 5: right, wrong, right
    ge = result.report["generation_eval"]
    assert ge["scope"] == "final" and ge["rows"] == 3 and ge["loss_rows"] == 6
    assert ge["groups"]["n"] == 3 and ge["groups"]["all_correct"] == 2
    assert ge["groups"]["all_correct_share"] == pytest.approx(2 / 3)
    assert set(seen) == {"q1=", "q3=", "q5="}  # the non-final prompts were never generated from
    assert json.loads((out.artifacts / "preprocess.json").read_text())["eval"]["rows"] == [1, 3, 5]
    assert result.report["truncation"]["validation"]["rows_used"] == 6


def test_final_gen_rows_counts_the_rows_scored_in_final_scope(tmp_path, monkeypatch):
    result, _, _, seen = _run(tmp_path, {"metrics": ["json_match"], "eval_scope": "final", "final_gen_rows": 2},
                              monkeypatch=monkeypatch)
    assert result.report["generation_eval"]["rows"] == 2
    assert len(set(seen[-2:])) == 2  # the final evaluation (last): two scored rows, not two of the six
    assert all(int(p[1:-1]) % 2 == 1 for p in seen)


def test_the_reference_is_what_the_export_is_re_scored_against(tmp_path, monkeypatch):
    result, data, out, _ = _run(tmp_path, {"metrics": ["json_match"], "eval_scope": "final"}, monkeypatch=monkeypatch)
    artifact = backend_mod.BACKEND.export(result, "safetensors", out)
    monkeypatch.setattr(backend_mod, "_hf_generate", lambda model_dir, prompt, n: REPLIES[int(prompt[1:-1])])
    ev = backend_mod.BACKEND.runtime_eval(artifact, data)
    assert ev.metric == "json_match" and ev.n == 3 and ev.value == pytest.approx(2 / 3)


def test_final_scope_without_a_final_column_is_refused_before_the_model_loads(tmp_path, monkeypatch):
    with pytest.raises(ValueError, match=r"eval_scope: final needs features.eval.final"):
        _run(tmp_path, {"metrics": ["json_match"], "eval_scope": "final"}, eval_block="\n  eval: {reference: ref}",
             monkeypatch=monkeypatch)


def test_final_scope_with_no_final_row_is_refused_in_words(tmp_path, monkeypatch):
    path = _table(tmp_path)
    lines = [json.loads(line) for line in path.read_text().splitlines()]
    path.write_text("".join(json.dumps({**r, "last_turn": False}) + "\n" for r in lines))
    from sakura.atelier.data import materialise
    from sakura.atelier.registry import resolve
    from sakura.atelier.spec import load_spec

    spec = load_spec(_spec_text(path, EVAL))
    data = materialise(spec, tmp_path / "d")
    hp = _hp(tiny_llama_dir(tmp_path / "base", ALPHABET), epochs=1, metrics=["json_match"], eval_scope="final")
    with pytest.raises(ValueError, match="no validation row has final true"):
        backend_mod.BACKEND.train(resolve(spec.model_copy(update={"hyperparameters": hp})), data,
                                  RunDir(tmp_path / "run").ensure(), _Metrics(), None)


@pytest.mark.parametrize("value, ok", [("all", True), ("final", True), ("last", False), (True, False), (None, True)])
def test_eval_scope_is_all_or_final(value, ok):
    hp = {"base_model": "x", "eval_scope": value}
    if ok:
        assert backend_mod.train_eval_options(hp)["eval_scope"] == (value or "all")
    else:
        with pytest.raises(ValueError, match="eval_scope"):
            backend_mod.train_eval_options(hp)
    assert backend_mod.train_eval_options({"base_model": "x"})["eval_scope"] == "all"
    assert backend_mod.train_eval_options({"base_model": "x", "eval_scope": "all"})["extended"] is True


def test_a_spec_with_the_evaluation_keys_but_none_of_the_new_ones_reports_as_before(tmp_path, monkeypatch):
    result, _, out, _ = _run(tmp_path, {"metrics": ["json_match"]}, eval_block="", monkeypatch=monkeypatch)
    ge = result.report["generation_eval"]
    assert set(ge) == {"metric", "metrics", "max_new_tokens", "rows", "loss_rows", "rows_scored"}  # no scope, columns or groups
    assert result.metrics["json_match"] == 0.0  # no reference: the replies are compared with the completion "ok"
