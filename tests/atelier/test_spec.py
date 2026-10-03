"""AtelierSpec v1: what a replayable spec must refuse, and what it must keep."""

import pytest

from sakura.atelier.spec import dump_spec, load_spec, spec_sha256

BASE = """
atelier: 1
name: cifar10-resnet18
task: image_classification
data:
  uri: hf://uoft-cs/cifar10
  format: hf
  split: {train: "train[:1000]", validation: "test[:200]"}
features:
  inputs: [{name: img, type: image}]
  output: {name: label, type: category}
model: resnet18
hp: image_classification/fast@1
export: [onnx]
"""


def test_a_minimal_spec_loads_with_its_defaults():
    s = load_spec(BASE)
    assert s.seed == 42 and s.confirmed is False and s.guessed == []
    assert s.placement.strategy == "best_price"


def test_an_unknown_key_is_an_error_not_a_silent_no_op():
    with pytest.raises(ValueError, match="epochz"):
        load_spec(BASE + "epochz: 3\n")


def test_hp_must_be_a_versioned_preset_id():
    with pytest.raises(ValueError, match="hp must look like"):
        load_spec(BASE.replace("image_classification/fast@1", "fast"))


def test_a_preset_for_another_task_is_refused():
    with pytest.raises(ValueError, match="is for 'tabular'"):
        load_spec(BASE.replace("image_classification/fast@1", "tabular/ecd@1"))


def test_tabular_presets_cover_both_tabular_tasks():
    s = load_spec(BASE.replace("task: image_classification", "task: tabular_regression")
                  .replace("image_classification/fast@1", "tabular/ecd@1"))
    assert s.task == "tabular_regression"


def test_splits_are_explicit_or_a_fraction_never_both():
    with pytest.raises(ValueError, match="not both"):
        load_spec(BASE.replace('split: {train: "train[:1000]", validation: "test[:200]"}',
                               'split: {train: "train", validation_fraction: 0.2}'))


def test_a_data_hash_must_be_hex():
    with pytest.raises(ValueError, match="64 lowercase hex"):
        load_spec(BASE.replace("format: hf", "format: hf\n  sha256: nothex"))


def test_guesses_are_unconfirmed_until_confirmed():
    s = load_spec(BASE + "guessed: [features.output.name]\n")
    assert s.unconfirmed_guesses == ["features.output.name"]
    s = load_spec(BASE + "guessed: [features.output.name]\nconfirmed: true\n")
    assert s.unconfirmed_guesses == []


def test_dumping_is_canonical_so_a_spec_always_hashes_the_same():
    a, b = load_spec(BASE), load_spec(BASE)
    assert dump_spec(a) == dump_spec(b)
    assert spec_sha256(dump_spec(a)) == spec_sha256(dump_spec(load_spec(dump_spec(a))))


HUB_SPEC = """
model_type: atelier
atelier: 1
name: churn
task: tabular_classification
data:
  uri: zc://dataset/d-1@sha256:%s
  sha256: ""
  format: csv
  validation_fraction: 0.2
features:
  inputs: [{name: tenure, type: number}]
  output: {name: churned, type: binary}
guessed: []
confirmed: true
model: ludwig_ecd
hp: tabular/ecd@1
seed: 1
export: [onnx]
placement: {strategy: best_price, gpus: 1, min_vram_gb: null}
resume_from: {uri: "zc://forge-job/j-1", sha256: null}
""" % ("a" * 64)


def test_a_spec_downloaded_from_the_hub_loads_unchanged():
    # The hub's dialect: model_type, data.validation_fraction beside split,
    # nulls for "unset". The YAML alone must replay, so the engine reads it.
    spec = load_spec(HUB_SPEC)
    assert spec.model_type == "atelier"
    assert spec.data.split.validation_fraction == 0.2
    assert spec.placement.min_vram_gb == 0 and spec.resume_from.sha256 == ""


def test_both_split_spellings_at_once_are_refused():
    import pytest

    bad = HUB_SPEC.replace("  validation_fraction: 0.2", "  validation_fraction: 0.2\n  split: {train: a, validation: b}")
    with pytest.raises(Exception, match="not both"):
        load_spec(bad)
