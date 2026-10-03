"""Dataset materialisation from tiny local fixtures: manifest + sha256
verification (CONTRACTS §1.2), no network, no hf://."""

import json

import pandas as pd
import pytest
from PIL import Image

from sakura.atelier.data import materialise
from sakura.atelier.spec import load_spec

CSV_SPEC = """
atelier: 1
name: tiny-tabular
task: tabular_classification
data:
  uri: {uri}
  format: csv
  split: {{validation_fraction: 0.25}}
features:
  inputs: [{{name: x, type: number}}]
  output: {{name: y, type: category}}
model: ludwig_ecd
hp: tabular/ecd@1
export: [onnx]
"""

IMAGE_SPEC = """
atelier: 1
name: tiny-images
task: image_classification
data:
  uri: {uri}
  format: image_folder
  split: {{train: "train", validation: "validation"}}
features:
  inputs: [{{name: img, type: image}}]
  output: {{name: label, type: category}}
model: resnet18
hp: image_classification/fast@1
export: [onnx]
"""


def _write_csv(path):
    pd.DataFrame({"x": range(20), "y": ["a", "b"] * 10}).to_csv(path, index=False)


def test_csv_materialises_and_writes_a_verifiable_manifest(tmp_path):
    csv_path = tmp_path / "table.csv"
    _write_csv(csv_path)
    spec = load_spec(CSV_SPEC.format(uri=f"file://{csv_path}"))

    data = materialise(spec, tmp_path / "data")
    assert len(data.train) + len(data.validation) == 20
    manifest = json.loads((tmp_path / "data" / "manifest.json").read_text())
    assert all({"path", "bytes", "sha256"} <= set(e) for e in manifest)
    assert len(data.manifest_sha256) == 64


def test_a_pinned_sha256_that_does_not_match_fails_the_run(tmp_path):
    csv_path = tmp_path / "table.csv"
    _write_csv(csv_path)
    spec = load_spec(CSV_SPEC.format(uri=f"file://{csv_path}"))
    spec = spec.model_copy(update={"data": spec.data.model_copy(update={"sha256": "0" * 64})})

    with pytest.raises(ValueError, match="data.sha256 mismatch"):
        materialise(spec, tmp_path / "data")


def test_image_folder_materialises_paths_and_labels(tmp_path):
    root = tmp_path / "images"
    for split in ("train", "validation"):
        for label in ("cat", "dog"):
            d = root / split / label
            d.mkdir(parents=True)
            Image.new("RGB", (4, 4), color=(1, 2, 3)).save(d / "0.png")

    spec = load_spec(IMAGE_SPEC.format(uri=f"file://{root}"))
    data = materialise(spec, tmp_path / "data")

    assert set(data.labels) == {"cat", "dog"}
    assert len(data.train) == 2 and len(data.validation) == 2
    assert all(str(root) in p for p in data.train["img"])


def test_jsonl_is_read_as_one_record_per_line(tmp_path):
    jsonl_path = tmp_path / "rows.jsonl"
    jsonl_path.write_text("\n".join(
        json.dumps({"x": i, "y": "a" if i % 2 else "b"}) for i in range(20)
    ) + "\n")
    spec = load_spec(CSV_SPEC.format(uri=f"file://{jsonl_path}").replace("format: csv", "format: jsonl"))

    data = materialise(spec, tmp_path / "data")
    assert len(data.train) + len(data.validation) == 20
