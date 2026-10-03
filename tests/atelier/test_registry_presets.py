"""The engine's presets are exactly the canonical list the hub's planner and
compiler are tested against too (CONTRACTS.md §7). A preset renamed on one
side only fails a job at the runner, after the user confirmed and paid."""

from pathlib import Path

CANONICAL = {
    "image_classification/fast@1",
    "audio_classification/fast@1",
    "object_detection/fast@1",
    "tabular/ecd@1",
    "tabular/tabpfn@1",
    "text_generation/lora-fast@1",
    "speech_recognition/ctc-fast@1",
}

PRESETS = Path(__file__).resolve().parents[2] / "sakura" / "atelier" / "registry" / "presets"


def test_registry_presets_are_exactly_the_canonical_list():
    found = {f"{p.parent.name}/{p.stem}" for p in PRESETS.glob("*/*.yaml")}
    assert found == CANONICAL


def test_there_is_no_gbm_model():
    # Ludwig 0.11 has no GBM model type; an id that names one trains an ECD.
    models = (PRESETS.parent / "models.yaml").read_text()
    assert "ludwig_gbm" not in models


# The hub's MODEL_EXPORTS (api/forge/compiler/atelier.py) must agree with these:
# an export the engine cannot produce must be refused at quote time, not after training.
CANONICAL_EXPORTS = {
    "resnet18": ["onnx", "torchscript"],
    "stacked_cnn_audio": ["onnx", "torchscript"],
    "ludwig_ecd": ["onnx"],
    "tabpfn_v2": [],
    "yolo11n": ["onnx"],
    "lfm2-350m": ["gguf", "safetensors", "mlx"],
    "deepspeech2": ["safetensors"],
}


def test_registry_exports_are_the_canonical_table():
    from sakura.atelier.registry import load_models

    assert {k: v.get("exports") for k, v in load_models().items()} == CANONICAL_EXPORTS


def _spec(model, hp, task, export):
    from sakura.atelier.spec import load_spec

    return load_spec(f"""
atelier: 1
name: t
task: {task}
data: {{uri: "file:///x.csv", format: csv, split: {{validation_fraction: 0.2}}}}
features:
  inputs: [{{name: a, type: number}}]
  output: {{name: y, type: category}}
model: {model}
hp: {hp}
export: {export}
""")


def test_resolve_refuses_an_export_the_model_cannot_produce():
    import pytest

    from sakura.atelier.registry import resolve

    with pytest.raises(ValueError, match="evaluation-only"):
        resolve(_spec("tabpfn_v2", "tabular/tabpfn@1", "tabular_classification", "[onnx]"))
    with pytest.raises(ValueError, match="export is empty"):
        resolve(_spec("ludwig_ecd", "tabular/ecd@1", "tabular_classification", "[]"))
    assert resolve(_spec("tabpfn_v2", "tabular/tabpfn@1", "tabular_classification", "[]")).backend == "tabpfn"
    assert resolve(_spec("ludwig_ecd", "tabular/ecd@1", "tabular_classification", "[onnx]")).backend == "ludwig"
