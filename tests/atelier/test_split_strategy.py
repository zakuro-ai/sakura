"""A classification holdout is stratified per class by default (BE-18).

`validation_fraction: 0.2` on cifar-mini (500 images, 10 classes of 50) used
to hold out 4 to 17 images per class at random; drawn per class it holds out
10 of each, and the spec records how it was drawn (`data.split.strategy`)."""

from collections import Counter

import pandas as pd
import pytest
import yaml

from sakura.atelier.data import materialise
from sakura.atelier.registry import resolve, resolved_yaml_text
from sakura.atelier.spec import load_spec, split_strategy

Image = pytest.importorskip("PIL.Image")

FOLDER_SPEC = """
atelier: 1
name: cifar-mini
task: image_classification
data:
  uri: file://{root}
  format: image_folder
  split: {{validation_fraction: 0.2{extra}}}
features:
  inputs: [{{name: img, type: image}}]
  output: {{name: label, type: category}}
model: resnet18
hp: image_classification/fast@1
export: [onnx]
"""

TABLE_SPEC = """
atelier: 1
name: t
task: {task}
data:
  uri: file://{uri}
  format: csv
  split: {{validation_fraction: 0.2{extra}}}
features:
  inputs: [{{name: x, type: number}}]
  output: {{name: y, type: {out}}}
model: ludwig_ecd
hp: tabular/ecd@1
export: [onnx]
"""

CLASSES = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]


def _cifar_mini(root, per_class=50):
    for label in CLASSES:
        d = root / label
        d.mkdir(parents=True)
        for i in range(per_class):
            Image.new("RGB", (2, 2), color=(i, 0, 0)).save(d / f"{i:03d}.png")
    return root


def _folder_spec(root, extra=""):
    return load_spec(FOLDER_SPEC.format(root=root, extra=extra))


def test_cifar_mini_holds_out_ten_images_per_class(tmp_path):
    spec = _folder_spec(_cifar_mini(tmp_path / "cifar-mini"))
    data = materialise(spec, tmp_path / "d")
    per_class = Counter(data.validation["label"])
    assert set(per_class) == set(CLASSES)
    assert all(9 <= n <= 11 for n in per_class.values()), per_class
    assert len(data.train) + len(data.validation) == 500
    split = data.extra["data_split"]
    assert split["strategy"] == "stratified" and split["stratify_by"] == ["label"]
    assert split["mode"] == "stratified_fraction" and split["validation_rows"] == 100
    assert split["label_distribution"]["validation"]["bird"] == 10


def test_stratified_folder_split_is_deterministic(tmp_path):
    root = _cifar_mini(tmp_path / "c", per_class=12)
    a = materialise(_folder_spec(root), tmp_path / "a")
    b = materialise(_folder_spec(root), tmp_path / "b")
    assert sorted(a.validation["img"]) == sorted(b.validation["img"])
    assert a.manifest_sha256 == b.manifest_sha256


def test_random_strategy_keeps_the_old_folder_holdout(tmp_path):
    root = _cifar_mini(tmp_path / "c", per_class=12)
    spec = _folder_spec(root, ", strategy: random")
    data = materialise(spec, tmp_path / "d")
    rows = []
    for label in CLASSES:
        rows += [{"img": str((root / label / f"{i:03d}.png").resolve()), "label": label} for i in range(12)]
    expected = pd.DataFrame(rows).sample(frac=0.2, random_state=spec.seed)
    assert list(data.validation["img"]) == list(expected["img"])
    assert data.extra["data_split"]["strategy"] == "random"


def test_folder_records_where_validation_files_live(tmp_path):
    root = _cifar_mini(tmp_path / "c", per_class=5)
    data = materialise(_folder_spec(root), tmp_path / "d")
    assert data.extra["media_root"] == str(root.resolve())


def test_table_classification_is_stratified_by_its_label(tmp_path):
    csv = tmp_path / "t.csv"
    pd.DataFrame({"x": range(100), "y": ["a"] * 80 + ["b"] * 20}).to_csv(csv, index=False)
    spec = load_spec(TABLE_SPEC.format(task="tabular_classification", uri=csv, extra="", out="category"))
    data = materialise(spec, tmp_path / "d")
    assert Counter(data.validation["y"]) == {"a": 16, "b": 4}
    assert data.extra["data_split"]["strategy"] == "stratified"


def test_regression_stays_random(tmp_path):
    csv = tmp_path / "t.csv"
    pd.DataFrame({"x": range(50), "y": [float(i) for i in range(50)]}).to_csv(csv, index=False)
    spec = load_spec(TABLE_SPEC.format(task="tabular_regression", uri=csv, extra="", out="number"))
    data = materialise(spec, tmp_path / "d")
    expected = pd.read_csv(csv).sample(frac=0.2, random_state=spec.seed)
    assert data.validation["x"].tolist() == expected["x"].tolist()
    assert data.extra["data_split"]["strategy"] == "random"


def test_split_strategy_defaults_and_validation(tmp_path):
    assert split_strategy(_folder_spec(tmp_path)) == "stratified"
    assert split_strategy(_folder_spec(tmp_path, ", strategy: random")) == "random"
    reg = TABLE_SPEC.format(task="tabular_regression", uri="/x.csv", extra="", out="number")
    assert split_strategy(load_spec(reg)) == "random"
    with pytest.raises(ValueError, match="category or binary output"):
        load_spec(TABLE_SPEC.format(task="tabular_regression", uri="/x.csv", extra=", strategy: stratified",
                                    out="number"))
    with pytest.raises(ValueError, match="contradicts"):
        load_spec(TABLE_SPEC.format(task="tabular_classification", uri="/x.csv",
                                    extra=", strategy: random, stratify_by: [x]", out="category"))
    explicit = FOLDER_SPEC.format(root=tmp_path, extra="").replace(
        "{validation_fraction: 0.2}", "{train: train, validation: validation, strategy: random}")
    with pytest.raises(ValueError, match="needs validation_fraction"):
        load_spec(explicit)
    assert split_strategy(load_spec(explicit.replace(", strategy: random", ""))) is None


def test_resolved_yaml_records_the_strategy_and_replays(tmp_path):
    spec = _folder_spec(tmp_path)
    assert "strategy" not in spec.model_dump(mode="json")["data"]["split"]
    text = resolved_yaml_text(resolve(spec), "0" * 64)
    assert yaml.safe_load(text)["data"]["split"]["strategy"] == "stratified"
    assert load_spec(text).data.split.strategy == "stratified"
