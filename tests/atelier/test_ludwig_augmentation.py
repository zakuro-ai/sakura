"""Recipe image augmentation in the Ludwig backend (#199): training split only, recorded in the report."""

from __future__ import annotations

import pytest

from sakura.atelier.backends.ludwig import (
    augmentation_report, compile_config, crop_train_images,
)
from sakura.atelier.evaluation import build_evaluation
from sakura.atelier.registry import resolve
from sakura.atelier.spec import dump_spec, load_spec
from sakura.atelier.types import TrainResult

SPEC = """
atelier: 1
name: cifar-mini
task: image_classification
data: {uri: file:///x, format: image_folder, split: {validation_fraction: 0.2}}
features:
  inputs: [{name: img, type: image}]
  output: {name: label, type: category}
model: resnet18
hp: image_classification/fast@1
export: [onnx]
"""


def _resolved(extra: str = ""):
    return resolve(load_spec(SPEC + extra))


def test_without_the_block_the_config_and_the_dump_are_unchanged():
    resolved = _resolved()
    assert "augmentation" not in compile_config(resolved)["input_features"][0]
    assert "augmentation" not in dump_spec(resolved.spec)
    assert augmentation_report(resolved.spec) is None


def test_disabled_block_changes_nothing_but_is_reported():
    resolved = _resolved("augmentation: {enabled: false}\n")
    assert compile_config(resolved) == compile_config(_resolved())
    assert augmentation_report(resolved.spec) == {"enabled": False, "ops": []}


def test_enabled_ops_are_ludwig_train_augmentations_on_the_image_input_only():
    resolved = _resolved("augmentation: {enabled: true, ops: [horizontal_flip, random_crop, rotate]}\n")
    config = compile_config(resolved)
    base = compile_config(_resolved())
    assert config["input_features"][0]["augmentation"] == [
        {"type": "random_horizontal_flip"}, {"type": "random_rotate", "degree": 15}]  # random_crop is not a Ludwig op
    config["input_features"][0].pop("augmentation")
    assert config == base  # on vs off differs only in the train pipeline
    assert augmentation_report(resolved.spec) == {"enabled": True, "ops": ["horizontal_flip", "random_crop", "rotate"]}


def test_default_ops_are_a_flip_and_a_random_crop():
    resolved = _resolved("augmentation: {enabled: true}\n")
    assert augmentation_report(resolved.spec) == {"enabled": True, "ops": ["horizontal_flip", "random_crop"]}


def test_unknown_op_is_rejected():
    with pytest.raises(ValueError, match="unknown augmentation"):
        _resolved("augmentation: {enabled: true, ops: [cutmix]}\n")


def test_other_tasks_do_not_augment():
    raw = SPEC.replace("image_classification/fast@1", "tabular_classification/ecd@1").replace(
        "task: image_classification", "task: tabular_classification")
    spec = load_spec(raw + "augmentation: {enabled: true}\n")
    assert augmentation_report(spec) == {"enabled": False, "ops": []}


def test_evaluation_block_reports_augmentation():
    resolved = _resolved("augmentation: {enabled: true}\n")
    result = TrainResult(model=None, metrics={}, train_seconds=0.0, augmentation=augmentation_report(resolved.spec))
    block = build_evaluation(object(), result, resolved, None)
    assert block["augmentation"] == {"enabled": True, "ops": ["horizontal_flip", "random_crop"]}
    plain = build_evaluation(object(), TrainResult(model=None, metrics={}, train_seconds=0.0), resolved, None)
    assert "augmentation" not in plain


def test_random_crop_rewrites_only_the_frame_it_is_given(tmp_path):
    pd = pytest.importorskip("pandas")
    Image = pytest.importorskip("PIL.Image")
    paths = []
    for i in range(3):
        p = tmp_path / f"a{i}.png"
        Image.new("RGB", (16, 16), (200, 10 * i, 5)).save(p)
        paths.append(str(p))
    train = pd.DataFrame({"img": paths, "label": ["a", "b", "a"]})
    out = crop_train_images(train, "img", tmp_path / "aug", seed=1)
    assert list(train["img"]) == paths  # the input frame is untouched
    assert all(str(tmp_path / "aug") in p for p in out["img"])
    assert list(out["label"]) == ["a", "b", "a"]
    with Image.open(out["img"][0]) as im:
        assert im.size == (16, 16)
    again = crop_train_images(train, "img", tmp_path / "aug2", seed=1)
    with Image.open(out["img"][1]) as a, Image.open(again["img"][1]) as b:
        assert a.tobytes() == b.tobytes()  # seeded
