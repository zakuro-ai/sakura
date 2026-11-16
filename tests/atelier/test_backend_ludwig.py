"""sakura.atelier.backends.ludwig: config compilation only. ``compile_config``
is pure dict logic with no ludwig import, so this runs in CI without ludwig
installed (train/export/runtime_eval/predict do import it, lazily, and are
exercised only by the GPU smoke runs on real hardware)."""

import sys

from sakura.atelier.backends.ludwig import compile_config
from sakura.atelier.registry import resolve
from sakura.atelier.spec import load_spec

IMAGE_SPEC = """
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

TABULAR_SPEC = """
atelier: 1
name: breast-cancer-gbm
task: tabular_classification
data:
  uri: file:///tmp/does-not-need-to-exist.csv
  format: csv
  split: {validation_fraction: 0.2}
features:
  inputs: [{name: x1, type: number}, {name: x2, type: number}]
  output: {name: y, type: category}
model: ludwig_ecd
hp: tabular/ecd@1
export: [onnx]
"""


def test_importing_the_ludwig_backend_module_does_not_import_ludwig():
    import importlib
    sys.modules.pop("ludwig", None)
    importlib.import_module("sakura.atelier.backends.ludwig")
    assert "ludwig" not in sys.modules


def test_image_classification_compiles_a_resnet_encoder():
    resolved = resolve(load_spec(IMAGE_SPEC))
    config = compile_config(resolved)
    assert config["input_features"][0]["type"] == "image"
    assert config["input_features"][0]["encoder"]["type"] == "resnet"
    assert config["input_features"][0]["encoder"]["model_variant"] == 18
    assert config["output_features"][0] == {"name": "label", "type": "category"}
    assert config["trainer"]["epochs"] == 3


def test_tabular_ecd_preset_uses_a_deeper_combiner():
    spec_text = TABULAR_SPEC.replace("task: tabular_classification", "task: tabular_regression") \
                             .replace("output: {name: y, type: category}", "output: {name: y, type: number}")
    resolved = resolve(load_spec(spec_text))
    config = compile_config(resolved)
    assert config["combiner"]["num_fc_layers"] == 2
    assert config["output_features"][0]["type"] == "number"


def test_backend_config_feature_lists_merge_by_name():
    """A normalized tabular target arrives as ``[{name, preprocessing}]``: it must
    keep the compiled ``type`` and leave the other features alone."""
    from sakura.atelier.backends.ludwig import _deep_merge

    compiled = {"input_features": [{"name": "parish", "type": "category"},
                                   {"name": "bedrooms", "type": "number"}],
                "output_features": [{"name": "log_price", "type": "number"}],
                "trainer": {"epochs": 3}}
    merged = _deep_merge(compiled, {
        "output_features": [{"name": "log_price", "preprocessing": {"normalization": "zscore"}}],
        "input_features": [{"name": "bedrooms", "preprocessing": {"normalization": "minmax"}},
                           {"name": "bathrooms", "type": "number"}],
    })
    assert merged["output_features"] == [
        {"name": "log_price", "type": "number", "preprocessing": {"normalization": "zscore"}}]
    assert merged["input_features"] == [
        {"name": "parish", "type": "category"},
        {"name": "bedrooms", "type": "number", "preprocessing": {"normalization": "minmax"}},
        {"name": "bathrooms", "type": "number"},
    ]
    assert compiled["output_features"] == [{"name": "log_price", "type": "number"}]  # not mutated


def test_backend_config_is_deep_merged_last():
    spec = load_spec(IMAGE_SPEC + 'backend_config: {trainer: {learning_rate: 0.5}}\n')
    config = compile_config(resolve(spec))
    assert config["trainer"]["learning_rate"] == 0.5          # overridden
    assert config["trainer"]["epochs"] == 3                    # untouched sibling key survives
    assert config["input_features"][0]["type"] == "image"      # untouched branch survives


def test_a_rank_1_category_input_is_fed_as_the_graphs_batch_by_1():
    """A category input is exported as [batch, 1]; the batcher yields [batch], which onnxruntime
    refuses ("Invalid rank"). Inputs already at the graph's rank pass through untouched."""
    import numpy as np

    from sakura.atelier.backends.ludwig import _fit_input_ranks

    cat, num, img = np.array([1, 2, 3]), np.array([0.5, 1.5, 2.5]), np.zeros((3, 2, 2))
    fed = _fit_input_ranks({"cat": cat, "num": num, "img": img}, {"cat": 2, "num": 1, "img": 3})
    assert fed["cat"].shape == (3, 1) and fed["cat"][:, 0].tolist() == [1, 2, 3]
    assert fed["num"] is num and fed["img"] is img
    assert _fit_input_ranks({"x": cat}, {})["x"] is cat
