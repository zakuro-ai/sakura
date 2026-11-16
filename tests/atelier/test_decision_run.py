"""A whole decision run through the CLI: prepare, train, calibrate, report, artifacts, predict, serve.

The peft backend's own decision code runs; only its model loading is swapped for a tiny fake LM
(`fake_decision_backend`, which never reads the base it is named after), so none of this needs a GPU or
`transformers`. What these tests prove is the
mechanics: that the right rows go to the right step, that the report says what was measured, and that
`predict` and `serve` answer in the documented shape. That the probabilities of a REAL model are calibrated
is not something a fake can show; see the PR for the one real run that was made."""

from __future__ import annotations

import hashlib
import io
import json
import math
import os
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

import sakura.atelier.__main__ as cli  # noqa: E402
from sakura.atelier import decision as dec  # noqa: E402
from sakura.atelier.backends import load_backend  # noqa: E402
from sakura.atelier.data import materialise  # noqa: E402
from sakura.atelier.spec import load_spec  # noqa: E402
from sakura.atelier.types import Artifact, RunDir  # noqa: E402
from tests.atelier.decision_fakes import TOPIC, TONE, URGENT, write_dataset  # noqa: E402
from tests.atelier.fake_decision_backend import FakePEFT  # noqa: E402

SPEC = """
atelier: 1
name: ticket-triage
task: decision
data: {uri: "data.jsonl", format: jsonl}
model: hf-decision-lm
hp: decision/lora-forge@1
overrides: {base_model: Qwen/Qwen3.5-2B-Base, epochs: 2, batch_size: 8, learning_rate: 0.05, max_seq_length: 512, precision: fp32}
export: [safetensors]
seed: 7
"""
REQUEST = {"state": "refund for the invoice, charged twice",
           "questions": {"topic": TOPIC, "urgent": URGENT, "tone": TONE}}


@pytest.fixture(autouse=True)
def env(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("SAKURA_ATELIER_BACKEND_MODULE_PEFT", "tests.atelier.fake_decision_backend")
    monkeypatch.setenv("SAKURA_ATELIER_NOTEBOOK", "0")
    # `serve` empties CUDA_VISIBLE_DEVICES for a decision run, in this very process: record it so it is put back
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", os.environ.get("CUDA_VISIBLE_DEVICES", ""))
    monkeypatch.setattr(FakePEFT, "loads", [])
    monkeypatch.setattr(FakePEFT, "scale", 1.0)
    return tmp_path


def run(tmp_path: Path, spec: str = SPEC, out: str = "out", rows: int = 90) -> tuple[int, dict, Path]:
    if not (tmp_path / "data.jsonl").exists():
        write_dataset(tmp_path / "data.jsonl", n=rows)
    (tmp_path / "spec.yaml").write_text(spec)
    rc = cli.main(["run", str(tmp_path / "spec.yaml"), "--out", str(tmp_path / out)])
    return rc, json.loads((tmp_path / out / "report.json").read_text()), tmp_path / out


def answers_of(report_dir: Path, request: dict, capsys, fmt: str = "predict") -> dict:
    f = report_dir / "req.json"
    f.write_text(json.dumps(request))
    capsys.readouterr()
    assert cli.main(["predict", str(report_dir), "--input", str(f)]) == 0
    return json.loads(capsys.readouterr().out)


# ------------------------------------------------------------------ the report

def test_the_run_reports_the_documented_keys(env):
    rc, report, _ = run(env)
    assert rc == 0 and report["status"] == "done" and report["error"] is None
    for key in ("atelier", "task", "backend", "model", "hp", "spec_sha256", "data_sha256", "billable_seconds",
                "train_seconds", "metrics", "runtime_eval", "evaluation", "artifacts", "resumed_from"):
        assert key in report, key  # the existing keys stay
    assert (report["task"], report["backend"], report["model"]) == ("decision", "peft", "hf-decision-lm")
    assert set(report["metrics"]) == {"accuracy", "log_loss", "ece"}
    ev = report["evaluation"]
    assert {"n", "accuracy", "log_loss", "ece", "by_type", "by_question"} <= set(ev["splits"]["eval"])
    assert set(ev["splits"]["eval"]["by_type"]) == {"choice", "noul", "score"}
    assert set(ev["splits"]["eval"]["by_question"]) == {"topic", "urgent", "tone"}
    assert any("only 90 usable rows" in w for w in ev["warnings"])  # the small-data case
    cal = report["calibration"]
    assert cal["method"] == "temperature" and set(cal["temperatures"]) == {"choice", "noul", "score"}
    assert set(cal["n_calib"]) == {"choice", "noul", "score"} and {"ece_before", "ece_after"} <= set(cal)
    prov = report["provenance"]
    assert (prov["base_model"], prov["precision"], prov["prompt_template"], prov["resumed"]) == ("Qwen/Qwen3.5-2B-Base", "fp32", "decision/v1", False)
    assert prov["base_revision"] == "b1485b2fa6dfa1287294f269f5fb618e03d52d7c"  # the allowlist pin


def test_a_wrap_up_before_the_first_step_still_delivers_a_calibrated_model(env, monkeypatch):
    """Graceful wrap-up (zak-forge#167): the loop stops at its first step boundary, then merge, calibration,
    evaluation and the artifacts run as usual on what training left, and the report says it stopped early."""
    signal = env / "wrap-up-requested"
    signal.write_text('{"reason": "budget"}')
    monkeypatch.setenv("ATELIER_WRAP_UP_FILE", str(signal))
    rc, report, out = run(env)
    assert rc == 0 and report["status"] == "done" and report["error"] is None
    assert report["stopped_early"] == {"reason": "wrap_up", "epoch": 0.0, "step": 0}
    assert report["calibration"]["method"] == "temperature" and "eval" in report["evaluation"]["splits"]
    assert report["artifacts"] and (out / "artifacts" / "model.safetensors").exists()
    lines = [json.loads(line) for line in (out / "metrics.jsonl").read_text().splitlines()]
    assert not [p for p in lines if p["split"] == "train"]  # no step was trained


def test_a_run_nobody_wrapped_up_has_no_stopped_early_field(env, monkeypatch):
    monkeypatch.delenv("ATELIER_WRAP_UP_FILE", raising=False)
    _, report, _ = run(env)
    assert "stopped_early" not in report


def test_ece_after_is_the_ece_of_the_eval_split(env):
    _, report, _ = run(env)
    eval_split = report["evaluation"]["splits"]["eval"]
    assert report["metrics"] == {k: eval_split[k] for k in ("accuracy", "log_loss", "ece")}
    assert report["calibration"]["ece_after"] == report["metrics"]["ece"]
    answers = sum(len(r["answers"]) for r in materialise(load_spec(SPEC), env / "again").validation)
    assert eval_split["n"] == answers  # the eval split, not the calibration or train rows


def test_the_split_sizes_are_in_the_evaluation_block(env):
    _, report, _ = run(env)
    assert report["evaluation"]["data"]["split_states"] == {"train": 72, "calibration": 9, "eval": 9}


def test_the_resolved_spec_names_neither_features_nor_a_split(env):
    _, _, out = run(env)
    text = (out / "resolved.yaml").read_text()
    assert "features" not in text and "split:" not in text and "task: decision" in text
    assert "base_model_allowlist" in text  # the allowlist the run used is on record


def test_metrics_jsonl_has_the_existing_line_shape(env):
    _, _, out = run(env)
    lines = [json.loads(x) for x in (out / "metrics.jsonl").read_text().splitlines()]
    assert {tuple(sorted(x)) for x in lines} == {("epoch", "name", "split", "step", "t", "value")}
    assert {x["name"] for x in lines if x["split"] == "train"} == {"loss"}
    val = [x for x in lines if x["split"] == "validation"]
    assert {x["name"] for x in val} == {"log_loss", "accuracy"} and len(val) == 2 * 2  # 2 epochs


# ------------------------------------------------------------------ artifacts

def test_the_artifacts_are_the_model_the_config_and_nothing_else_when_the_notebook_is_off(env):
    _, report, out = run(env)
    assert [a["format"] for a in report["artifacts"]] == ["safetensors", "decision_config", "license"]
    cfg_entry = report["artifacts"][1]
    cfg_path = out / cfg_entry["path"]
    assert cfg_path.name == "decision_config.json"
    assert cfg_entry["sha256"] == hashlib.sha256(cfg_path.read_bytes()).hexdigest() and cfg_entry["bytes"] == cfg_path.stat().st_size
    cfg = dec.read_decision_config(cfg_path)
    assert cfg["temperatures"] == report["calibration"]["temperatures"]
    assert (cfg["template"], cfg["base_model"], cfg["max_seq_length"]) == ("decision/v1", "Qwen/Qwen3.5-2B-Base", 512)
    assert (out / report["artifacts"][0]["path"]).is_file()


def test_the_runtime_eval_row_rescores_the_exported_model(env):
    _, report, _ = run(env)
    (row,) = report["runtime_eval"]
    assert (row["format"], row["runtime"], row["metric"]) == ("safetensors", "transformers", "accuracy")
    assert row["value"] == pytest.approx(report["metrics"]["accuracy"], abs=1e-6)  # the report rounds to 6 places
    assert row["n"] == report["evaluation"]["splits"]["eval"]["n"]
    assert row["matches_training"] is True


# ----------------------------------------------------------- predict and serve

def test_predict_answers_in_the_documented_shape(env, capsys):
    _, _, out = run(env)
    got = answers_of(out, REQUEST, capsys)
    a = got["answers"]
    assert set(a) == {"topic", "urgent", "tone"}
    assert a["topic"]["choice"] in {"billing", "bug", "other"} and set(a["topic"]["probs"]) == {"billing", "bug", "other"}
    assert sum(a["topic"]["probs"].values()) == pytest.approx(1.0, abs=1e-4)
    assert set(a["urgent"]) == {"noul"} and 0.0 <= a["urgent"]["noul"] <= 1.0
    assert set(a["tone"]) == {"score", "probs"} and 0.0 <= a["tone"]["score"] <= 2.0


def test_serve_answers_the_same_shape_and_the_same_numbers(env, capsys, monkeypatch):
    _, _, out = run(env)
    cold = answers_of(out, REQUEST, capsys)
    req = out / "serve-req.json"
    req.write_text(json.dumps(REQUEST))
    monkeypatch.setattr("sys.stdin", io.StringIO(json.dumps({"input": str(req)}) + "\n" + json.dumps({"input": str(req)}) + "\n"))
    capsys.readouterr()
    assert cli.main(["serve", str(out)]) == 0
    warm = [json.loads(x) for x in capsys.readouterr().out.splitlines()]
    assert warm == [cold, cold]


def test_the_hub_inputs_wrapper_is_accepted(env, capsys):
    _, _, out = run(env)
    assert answers_of(out, {"inputs": REQUEST}, capsys) == answers_of(out, REQUEST, capsys)


def test_a_bad_request_gets_a_clear_error_not_a_crash(env, capsys):
    _, _, out = run(env)
    f = out / "bad.json"
    f.write_text(json.dumps({"state": "x", "questions": {"q": {"type": "rank"}}}))
    capsys.readouterr()
    assert cli.main(["predict", str(out), "--input", str(f)]) == 1
    assert "type 'rank'" in json.loads(capsys.readouterr().out)["error"]


def test_predict_agrees_with_the_report_on_the_eval_rows(env):
    # Score every eval row through the delivered artifact, as a user of `predict` would, and recompute the
    # headline numbers from the probabilities it returns: they must be the ones the report states.
    _, report, out = run(env)
    backend = load_backend("peft")
    entry = report["artifacts"][0]
    artifact = Artifact(path=out / entry["path"], format="safetensors", sha256=entry["sha256"], bytes=entry["bytes"])
    rows = materialise(load_spec(SPEC), env / "again").validation
    hits, losses, conf, ok = [], [], [], []
    for row in rows:
        got = backend.predict(artifact, {"state": row["state"], "questions": row["questions"]})["answers"]
        for qid, gold in row["answers"].items():
            if "noul" in got[qid]:
                p = [got[qid]["noul"], 1 - got[qid]["noul"]]
                gi = 0 if gold else 1
            else:
                labels = row["questions"][qid].get("options") or row["questions"][qid]["levels"]
                p = [got[qid]["probs"][lab] for lab in labels]
                gi = labels.index(gold)
            hit = p.index(max(p)) == gi
            hits.append(hit)
            losses.append(-math.log(max(p[gi], 1e-12)))
            conf.append(max(p))
            ok.append(hit)
    m = report["metrics"]
    assert len(hits) == report["evaluation"]["splits"]["eval"]["n"]
    assert sum(hits) / len(hits) == pytest.approx(m["accuracy"], abs=1e-6)
    assert sum(losses) / len(losses) == pytest.approx(m["log_loss"], abs=1e-3)
    assert dec.expected_calibration_error(conf, ok) == pytest.approx(m["ece"], abs=2e-3)


# ---------------------------------------------------------- the fit sees no eval rows

def flip(row):
    """The same state with its answers changed to other valid labels."""
    out = json.loads(json.dumps(row))
    out["answers"] = {"topic": {"billing": "bug", "bug": "other", "other": "billing"}[row["answers"]["topic"]],
                      "urgent": not row["answers"]["urgent"],
                      "tone": {"calm": "angry", "annoyed": "calm", "angry": "annoyed"}[row["answers"]["tone"]]}
    return out


def test_changing_only_the_eval_labels_changes_the_eval_numbers_but_not_the_temperatures(env):
    rows = write_dataset(env / "data.jsonl")
    _, first, _ = run(env, out="one")
    eval_states = {r["state"] for r in materialise(load_spec(SPEC), env / "peek").validation}
    (env / "data.jsonl").write_text("".join(json.dumps(flip(r) if r["state"] in eval_states else r) + "\n" for r in rows))
    _, second, _ = run(env, out="two")
    assert first["calibration"]["temperatures"] == second["calibration"]["temperatures"]
    assert first["calibration"]["n_calib"] == second["calibration"]["n_calib"]
    assert first["metrics"] != second["metrics"] and first["calibration"]["ece_before"] != second["calibration"]["ece_before"]


def test_the_fit_is_given_the_calibration_rows_and_the_report_the_eval_rows(env, monkeypatch):
    seen = {}
    real = dec.report_blocks

    def spy(calibration, eval_items, warnings):
        seen["calibration"], seen["eval"] = len(calibration), len(eval_items)
        return real(calibration, eval_items, warnings)

    monkeypatch.setattr(dec, "report_blocks", spy)
    _, report, _ = run(env)
    sizes = report["evaluation"]["data"]["split_answers"]
    assert (seen["calibration"], seen["eval"]) == (sizes["calibration"], sizes["eval"])


def test_an_overconfident_model_is_cooled_by_the_fit(env, monkeypatch):
    # mechanics only: a fake whose logits are scaled up is overconfident by construction, and the fitted
    # temperature must come out above 1 and move the eval ECE down. No claim about a real model.
    monkeypatch.setattr(FakePEFT, "scale", 40.0)
    _, report, _ = run(env, rows=500)  # 50 calibration states: over the 30 answers a type needs to be fitted
    cal = report["calibration"]
    assert max(cal["temperatures"].values()) > 1.5
    assert cal["ece_after"] <= cal["ece_before"]


# --------------------------------------------------------------------- failures

def failed(env, spec=SPEC, rows=90):
    rc, report, _ = run(env, spec, rows=rows)
    assert rc == 1 and report["status"] == "failed"
    return report["error"]


def test_a_gguf_request_fails_as_customer_data_before_anything_runs(env):
    error = failed(env, SPEC.replace("[safetensors]", "[gguf]"))
    assert "cannot export ['gguf']" in error and error.startswith("ValueError")
    assert FakePEFT.loads == []  # no model was loaded for it


def test_the_backend_refuses_gguf_by_name_too(env):
    _, _, out = run(env)
    fitted = type("R", (), {"model": type("F", (), {"decision": {"x": 1}, "model_dir": out})()})()
    with pytest.raises(dec.DecisionDataError, match="safetensors only"):
        load_backend("peft").export(fitted, "gguf", RunDir(out))


def test_a_base_off_the_allowlist_fails_by_name(env):
    error = failed(env, SPEC.replace("Qwen/Qwen3.5-2B-Base", "evil/model"))
    assert "base model not on the allowlist" in error and "evil/model" in error and FakePEFT.loads == []


def test_too_few_rows_fail_by_name_before_the_model_loads(env):
    error = failed(env, rows=30)
    assert "DecisionDataError" in error and "too few usable rows (30; at least 50 needed)" in error
    assert FakePEFT.loads == []


def test_a_malformed_dataset_fails_by_name_with_the_row(env):
    rows = write_dataset(env / "data.jsonl")
    rows[2]["answers"]["topic"] = "refund"
    (env / "data.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    assert "row 3: answer 'refund' to 'topic'" in failed(env)


# ------------------------------------------------------------------- resume, serve

def test_a_resume_trains_on_from_the_previous_merged_weights(env):
    _, first, out = run(env, out="one")
    assert first["provenance"]["resumed"] is False
    second_spec = SPEC + f"resume_from: {{uri: '{(out / 'checkpoint').as_posix()}'}}\n"
    _, second, _ = run(env, second_spec, out="two")
    assert second["provenance"]["resumed"] is True
    assert FakePEFT.loads[-1][0] == str(out / "checkpoint" / "merged")


def test_serve_is_cpu_only_for_a_decision_run(env, monkeypatch, capsys):
    _, _, out = run(env)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    monkeypatch.setenv("SAKURA_ATELIER_BACKEND_MODULE_PEFT", "tests.atelier.fake_echo_backend")
    req = out / "r.json"
    req.write_text("{}")
    monkeypatch.setattr("sys.stdin", io.StringIO(json.dumps({"input": str(req)}) + "\n"))
    capsys.readouterr()
    assert cli.main(["serve", str(out)]) == 0
    assert json.loads(capsys.readouterr().out)["cuda_visible_devices"] == ""


def test_serve_leaves_other_tasks_on_the_gpu(env, monkeypatch, capsys):
    out = RunDir(env / "text").ensure()
    (out.root / "report.json").write_text(json.dumps({
        "status": "done", "task": "text_generation", "backend": "peft",
        "artifacts": [{"path": "artifacts/model.safetensors", "format": "safetensors", "sha256": "x", "bytes": 1}]}))
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    monkeypatch.setenv("SAKURA_ATELIER_BACKEND_MODULE_PEFT", "tests.atelier.fake_echo_backend")
    req = env / "r.json"
    req.write_text("{}")
    monkeypatch.setattr("sys.stdin", io.StringIO(json.dumps({"input": str(req)}) + "\n"))
    capsys.readouterr()
    assert cli.main(["serve", str(out.root)]) == 0
    assert json.loads(capsys.readouterr().out)["cuda_visible_devices"] == "0"
