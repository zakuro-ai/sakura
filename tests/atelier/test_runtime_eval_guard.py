"""A runtime check that crashes costs one row of ``runtime_eval``, never the training that finished
(CONTRACTS §2.1). No training stack is needed: the backend is a fake."""

from __future__ import annotations

import json

import sakura.atelier.__main__ as cli

SPEC = """
atelier: 1
name: guarded
task: text_generation
data: {{uri: "file://{data}", format: jsonl, split: {{validation_fraction: 0.25}}}}
features:
  inputs: [{{name: prompt, type: text}}]
  output: {{name: completion, type: text}}
model: lfm2-350m
hp: text_generation/lora-fast@1
export: [safetensors, gguf]
seed: 1
"""


def _run(tmp_path, monkeypatch, raises, capsys):
    monkeypatch.setenv("SAKURA_ATELIER_BACKEND_MODULE_PEFT", "tests.atelier.fake_runtime_eval_backend")
    monkeypatch.setenv("SAKURA_ATELIER_NOTEBOOK", "0")
    monkeypatch.setenv("FAKE_RUNTIME_EVAL_RAISES", raises)
    data = tmp_path / "data.jsonl"
    data.write_text("".join(json.dumps({"prompt": f"p{i}", "completion": f"c{i}"}) + "\n" for i in range(20)))
    spec = tmp_path / "spec.yaml"
    spec.write_text(SPEC.format(data=data))
    out = tmp_path / "out"
    rc = cli.main(["run", str(spec), "--out", str(out)])
    return rc, json.loads((out / "report.json").read_text()), capsys.readouterr().err


def test_a_crashing_check_for_one_format_leaves_the_finished_run_and_the_other_rows(tmp_path, monkeypatch, capsys):
    rc, report, err = _run(tmp_path, monkeypatch, "gguf", capsys)
    assert rc == 0 and report["status"] == "done" and report["error"] is None
    rows = {r["format"]: r for r in report["runtime_eval"]}
    assert list(rows) == ["safetensors", "gguf"]  # both formats have a row, in the order of the export list
    assert (rows["safetensors"]["value"], rows["safetensors"]["n"], rows["safetensors"]["matches_training"]) == (0.5, 3, True)
    bad = rows["gguf"]
    assert (bad["value"], bad["n"], bad["matches_training"]) == (None, 0, None)
    assert bad["reason"].startswith("RuntimeError: the runtime ran out of memory")
    assert "\n" not in bad["reason"] and len(bad["reason"]) <= 300
    assert "Traceback" in err and "the runtime ran out of memory" in err  # the traceback is logged
    # both files are still delivered; the one that was not re-scored says why
    artifacts = {a["format"]: a for a in report["artifacts"]}
    assert {"safetensors", "gguf"} <= set(artifacts)
    assert artifacts["gguf"]["validated"] is False and artifacts["gguf"]["validation_reason"] == bad["reason"]
    assert artifacts["safetensors"]["validated"] is True


def test_when_every_check_works_the_rows_are_untouched(tmp_path, monkeypatch, capsys):
    rc, report, _ = _run(tmp_path, monkeypatch, "", capsys)
    assert rc == 0 and [r["value"] for r in report["runtime_eval"]] == [0.5, 0.5]
    assert all(r["reason"] is None for r in report["runtime_eval"])


def test_a_check_on_fewer_rows_is_compared_with_the_training_score_on_the_same_rows(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("FAKE_RUNTIME_EVAL_TRAIN_VALUE", "0.88")
    rc, report, _ = _run(tmp_path, monkeypatch, "", capsys)
    assert rc == 0
    for row in report["runtime_eval"]:
        # 0.9 against the run's own 0.5 would not match; against 0.88 on the same 3 rows it does
        assert (row["value"], row["matches_training"], row["training_value"], row["training_n"]) == (0.9, True, 0.88, 3)
        assert row["training_delta"] == 0.02 and row["n"] == 3
