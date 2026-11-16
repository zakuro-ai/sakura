"""``teacher:`` logit distillation (SAK-0019) for the Hugging Face classifiers and the causal LM.

The validation rules and the loss are unit tests. The end-to-end tests train a teacher through
``python -m sakura.atelier run`` and then a smaller student on the teacher's logits ALONE
(``distill_alpha: 1.0``: the labels do not enter the loss), so a student that scores well learned it from
the teacher."""

from __future__ import annotations

import json
import math

import pytest

from sakura.atelier.recipe import teacher_source
from sakura.atelier.registry import resolve
from sakura.atelier.spec import load_spec
from tests.atelier.test_init_random import CLASSIFIER, LM
from tests.atelier.test_smoke_runs import _resolved, _run, _tiny_causal_lm, _train_points
from tests.atelier.tiny_models import fuse_decoder, synthetic_words, tiny_bert_dir

SHA = "4d5a3af95422fc0ebf732796c685c586b6d7420b"


def _lm(**overrides):
    return load_spec(LM.format(overrides=json.dumps({"base_model": "some/base", **overrides})))


# ------------------------------------------------------------------------------------------- validation

def test_a_teacher_and_its_two_weights_are_kept_in_the_hyperparameters():
    hp = resolve(_lm(teacher=f"hf:owner/repo@{SHA}", distill_alpha=0.7, distill_temperature=3)).hyperparameters
    assert (hp["teacher"], hp["distill_alpha"], hp["distill_temperature"]) == (f"hf:owner/repo@{SHA}", 0.7, 3)
    assert "teacher" not in resolve(_lm()).hyperparameters
    assert resolve(_lm(teacher="/runs/teacher")).hyperparameters["teacher"] == "/runs/teacher"
    assert resolve(_lm(teacher="forge://job-1")).hyperparameters["teacher"] == "forge://job-1"


@pytest.mark.parametrize("overrides, message", [
    ({"teacher": ""}, "teacher must name a model"),
    ({"teacher": 7}, "teacher must name a model"),
    ({"teacher": "hf:owner/repo"}, "expected hf:<owner>/<repo>@<40-hex commit sha>"),
    ({"teacher": "hf:owner/repo@main"}, "expected hf:<owner>/<repo>@<40-hex commit sha>"),
    ({"distill_alpha": 0.5}, "distill_alpha only applies with a teacher"),
    ({"distill_temperature": 2}, "distill_temperature only applies with a teacher"),
    ({"teacher": "/t", "distill_alpha": 1.5}, "distill_alpha must be a number from 0 to 1"),
    ({"teacher": "/t", "distill_alpha": "half"}, "distill_alpha must be a number from 0 to 1"),
    ({"teacher": "/t", "distill_temperature": 0}, "distill_temperature must be a number above 0"),
    ({"teacher": "/t", "distill_temperature": True}, "distill_temperature must be a number above 0"),
])
def test_a_bad_teacher_option_is_refused_by_name(overrides, message):
    with pytest.raises(ValueError) as err:
        resolve(_lm(**overrides))
    assert message in str(err.value)


def test_a_teacher_is_refused_for_a_backend_that_cannot_distil():
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
overrides: {teacher: /runs/t}
export: [joblib]
""")
    with pytest.raises(ValueError, match="teacher is not supported for task 'tabular_classification'"):
        resolve(spec)


def test_a_teacher_reference_resolves_to_a_pinned_repo_or_a_model_directory(tmp_path):
    assert teacher_source(f"hf:owner/repo@{SHA}") == ("owner/repo", SHA)
    run = tmp_path / "run"
    (run / "checkpoint" / "merged").mkdir(parents=True)
    (run / "checkpoint" / "merged" / "config.json").write_text("{}")
    assert teacher_source(str(run)) == (str(run / "checkpoint" / "merged"), None)       # a causal LM run
    assert teacher_source(f"file://{run}/checkpoint") == (str(run / "checkpoint" / "merged"), None)
    (run / "checkpoint" / "config.json").write_text("{}")
    assert teacher_source(str(run)) == (str(run / "checkpoint" / "merged"), None)
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text("{}")
    assert teacher_source(str(model)) == (str(model), None)                             # a plain model directory
    with pytest.raises(ValueError, match="does not exist"):
        teacher_source(str(tmp_path / "nowhere"))
    with pytest.raises(ValueError, match="holds no model"):
        teacher_source(str(tmp_path))
    with pytest.raises(ValueError, match="forge://job-1 .* the runner"):
        teacher_source("forge://job-1")


# ------------------------------------------------------------------------------------------- the loss

def test_the_soft_loss_is_the_temperature_scaled_kl_and_zero_for_equal_logits():
    torch = pytest.importorskip("torch")
    from sakura.atelier.backends.distill import blend, kd_loss

    student = torch.tensor([[2.0, 0.0, -1.0], [0.5, 0.5, 0.5]])
    teacher = torch.tensor([[1.0, 1.0, 0.0], [3.0, 0.0, 0.0]])
    assert float(kd_loss(teacher, teacher, 2.0)) == pytest.approx(0.0, abs=1e-6)
    for t in (1.0, 4.0):
        p = torch.softmax(teacher / t, -1)
        want = (p * (torch.log(p) - torch.log_softmax(student / t, -1))).sum(-1).mean() * t * t
        assert float(kd_loss(student, teacher, t)) == pytest.approx(float(want), rel=1e-5)
    assert float(blend(torch.tensor(2.0), torch.tensor(4.0), 0.25)) == pytest.approx(2.5)


def test_label_spaces_are_matched_by_name_and_a_mismatch_names_both_sides():
    from sakura.atelier.backends.distill import label_permutation

    assert label_permutation(["b", "c", "a"], ["a", "b", "c"], "t") == [2, 0, 1]
    with pytest.raises(ValueError) as err:
        label_permutation(["a", "b", "d"], ["a", "b", "c"], "/runs/t")
    assert "only in the teacher: ['d']" in str(err.value) and "only in the data: ['c']" in str(err.value)


# ------------------------------------------------------------------------------------------- end to end

def test_a_classifier_student_learns_from_its_teachers_logits_alone(tmp_path):
    pytest.importorskip("transformers")
    pytest.importorskip("onnxruntime")
    uri = "synthetic://text?rows=240&seed=3&classes=3"
    base = tiny_bert_dir(tmp_path / "bert", synthetic_words(uri, tmp_path / "words"))
    scratch = {"base_model": str(base), "init": "random", "batch_size": 16, "learning_rate": 0.003, "weight_decay": 0.0}
    teacher_hp = {**scratch, "epochs": 8, "init_config": {"num_hidden_layers": 2, "hidden_size": 48, "intermediate_size": 96}}
    teacher, teacher_out = _run(tmp_path, CLASSIFIER.format(uri=uri, overrides=json.dumps(teacher_hp)),
                                "--no-notebook", name="teacher")
    assert teacher["metrics"]["accuracy"] > 0.8

    student_hp = {**scratch, "epochs": 8, "init_config": {"num_hidden_layers": 1, "hidden_size": 24, "intermediate_size": 48},
                  "teacher": str(teacher_out), "distill_alpha": 1.0, "distill_temperature": 2.0}
    report, out = _run(tmp_path, CLASSIFIER.format(uri=uri, overrides=json.dumps(student_hp)), "--no-notebook",
                       name="student")
    block = report["distillation"]
    assert block["teacher"] == str(teacher_out) and block["alpha"] == 1.0 and block["temperature"] == 2.0
    assert block["metric"] == "accuracy" and block["n"] == 60
    assert block["teacher_score"] == teacher["metrics"]["accuracy"]  # the same model on the same 60 rows
    assert block["student_score"] == report["metrics"]["accuracy"]
    # chance is 1/3 on three balanced classes; the student never saw a label
    assert block["student_score"] > 0.8 and block["agreement"] > 0.8
    assert report["provenance"]["teacher"] == str(teacher_out)
    assert _resolved(out)["hyperparameters"]["teacher"] == str(teacher_out)
    assert {p["name"] for p in _train_points(out)} >= {"loss", "distill_loss"}
    assert report["runtime_eval"][0]["matches_training"] is True  # the student's own ONNX, re-scored

    # a teacher trained on other classes is refused, naming both label spaces
    other = "synthetic://text?rows=120&seed=3&classes=2"
    failed, _ = _run(tmp_path, CLASSIFIER.format(uri=other, overrides=json.dumps(student_hp)), "--no-notebook",
                     name="mismatch", expect=1)
    assert "does not have the student's label space" in failed["error"] and "only in the teacher: ['c2']" in failed["error"]


def test_a_causal_lm_student_learns_from_its_teachers_logits_alone(tmp_path):
    pytest.importorskip("transformers")
    pytest.importorskip("peft")
    base = fuse_decoder(_tiny_causal_lm(tmp_path / "base"))
    (base / "model.safetensors").unlink()
    scratch = {"base_model": str(base), "init": "random", "batch_size": 8, "learning_rate": 0.005, "max_seq_length": 64}
    teacher_hp = {**scratch, "epochs": 15, "init_config": {"num_hidden_layers": 1, "hidden_size": 48, "intermediate_size": 96}}
    teacher, teacher_out = _run(tmp_path, LM.format(overrides=json.dumps(teacher_hp)), "--no-notebook", name="teacher")
    assert teacher["metrics"]["exact_match"] >= 0.75

    student_hp = {**scratch, "epochs": 15, "init_config": {"num_hidden_layers": 1, "hidden_size": 32, "intermediate_size": 64},
                  "teacher": str(teacher_out), "distill_alpha": 1.0, "distill_temperature": 1.0}
    report, out = _run(tmp_path, LM.format(overrides=json.dumps(student_hp)), "--no-notebook", name="student")
    block = report["distillation"]
    assert block["teacher"] == str(teacher_out) and block["alpha"] == 1.0 and block["metric"] == "exact_match"
    assert block["n"] == 16 and block["teacher_score"] == teacher["metrics"]["exact_match"]
    assert block["student_score"] == report["metrics"]["exact_match"] >= 0.75
    # a model that guesses tokens at random has a loss of ln(32) = 3.47 on these rows
    assert block["student_loss"] < 0.5 < math.log(32) and block["agreement"] > 0.9
    assert report["provenance"]["teacher"] == str(teacher_out) and report["provenance"]["training"] == "full"

    # a teacher with another tokenizer is refused: the two distributions are over different tokens
    other = fuse_decoder(_tiny_causal_lm(tmp_path / "other"))
    tok = json.loads((other / "tokenizer.json").read_text())
    vocab = tok["model"]["vocab"]
    vocab["a"], vocab["b"] = vocab["b"], vocab["a"]
    (other / "tokenizer.json").write_text(json.dumps(tok))
    failed, _ = _run(tmp_path, LM.format(overrides=json.dumps({**student_hp, "teacher": str(other)})), "--no-notebook",
                     name="mismatch", expect=1)
    assert "does not share the student's vocabulary" in failed["error"]
