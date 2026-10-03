"""registry.resolve: expanding hp -> hyperparameters, and the model/task
checks that keep an unsupported pairing from silently training anyway."""

import pytest

from sakura.atelier.registry import list_presets, load_models, resolve, resolved_yaml_text
from sakura.atelier.spec import load_spec

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


def test_resolve_expands_the_preset_into_hyperparameters():
    resolved = resolve(load_spec(BASE))
    assert resolved.backend == "ludwig"
    assert resolved.hyperparameters["epochs"] == 3
    assert resolved.hyperparameters["encoder"] == "resnet18"


def test_overrides_are_applied_on_top_of_the_preset():
    resolved = resolve(load_spec(BASE + "overrides: {epochs: 7}\n"))
    assert resolved.hyperparameters["epochs"] == 7
    assert resolved.hyperparameters["batch_size"] == 64  # untouched preset value


def test_unknown_model_is_refused():
    with pytest.raises(ValueError, match="unknown model"):
        resolve(load_spec(BASE.replace("model: resnet18", "model: not-a-model")))


def test_model_task_mismatch_is_refused():
    # yolo11n only supports object_detection; this spec's task is
    # image_classification -- a registered model used for the wrong task.
    spec_text = BASE.replace("model: resnet18", "model: yolo11n")
    with pytest.raises(ValueError, match="does not support task"):
        resolve(load_spec(spec_text))


def test_list_presets_filters_by_task_family():
    assert "image_classification/fast@1" in list_presets()
    assert "image_classification/fast@1" in list_presets("image_classification")
    assert "tabular/ecd@1" in list_presets("tabular_classification")
    assert "tabular/ecd@1" in list_presets("tabular_regression")
    assert "image_classification/fast@1" not in list_presets("tabular_classification")


def test_every_registry_model_lists_at_least_one_task():
    models = load_models()
    for model_id, entry in models.items():
        assert entry["tasks"], f"{model_id} has no tasks"


def test_resolved_yaml_round_trips_and_is_used_as_is_on_replay(tmp_path):
    resolved = resolve(load_spec(BASE))
    text = resolved_yaml_text(resolved, data_sha256="a" * 64)
    reloaded_spec = load_spec(text)
    assert reloaded_spec.hyperparameters == resolved.hyperparameters
    assert reloaded_spec.data.sha256 == "a" * 64

    # A spec that already carries hyperparameters (i.e. IS a resolved.yaml) is
    # used as-is -- resolve() must not need the preset file to still exist.
    reresolved = resolve(reloaded_spec)
    assert reresolved.hyperparameters == resolved.hyperparameters


def test_resolved_yaml_text_refuses_to_silently_disagree_with_the_manifest():
    pinned_spec = load_spec(BASE).model_copy(
        update={"data": load_spec(BASE).data.model_copy(update={"sha256": "a" * 64})}
    )
    resolved = resolve(pinned_spec)
    with pytest.raises(ValueError, match="does not match"):
        resolved_yaml_text(resolved, data_sha256="b" * 64)
