"""An hf:// dataset's image and audio columns are written to files the training frame points at.

Every column used to be written to the same ``images/<split>/<i>.png``, so with two image columns (a
robot's two camera views) both pointed at whichever column was written last. Each column now has its
own directory when there are several; a lone column keeps the flat layout, so its manifest -- and the
``data.sha256`` a resolved spec pins -- does not change. Columns the spec does not name are not written."""
from __future__ import annotations

import io
import sys
import types
from pathlib import Path

import pytest

from sakura.atelier.data import _media_dirs, materialise
from sakura.atelier.spec import load_spec

PIL = pytest.importorskip("PIL.Image")

SPEC = """
atelier: 1
name: views
task: tabular_classification
data: {{uri: "{uri}", format: hf, split: {{train: "train", validation: "test"}}}}
features:
  inputs: [{inputs}]
  output: {{name: y, type: category}}
model: ludwig_ecd
hp: tabular/ecd@1
export: [onnx]
"""


def _png(value: int) -> bytes:
    buf = io.BytesIO()
    PIL.new("L", (2, 2), color=value).save(buf, format="PNG")
    return buf.getvalue()


def _pixel(path: str) -> int:
    with PIL.open(path) as im:
        return im.getpixel((0, 0))


def _fake_datasets(monkeypatch, columns: list[str]) -> None:
    """A stand-in `datasets` module serving 3 rows whose image column `c` holds pixel value 10*k+i."""

    class Image:  # a feature marker, as datasets.Image is
        pass

    class Table:
        def __init__(self):
            self.features = {**{c: Image() for c in columns}, "y": object()}
            self.column_names = [*columns, "y"]
            self.rows = [{**{c: {"bytes": _png(10 * k + i), "path": None} for k, c in enumerate(columns)},
                          "y": "ab"[i % 2]} for i in range(3)]

        def __len__(self):
            return len(self.rows)

        def to_pandas(self):
            import pandas as pd
            return pd.DataFrame(self.rows)

    fake = types.ModuleType("datasets")
    fake.load_dataset = lambda repo, **kwargs: Table()
    fake.Image = Image
    fake.Audio = fake.ClassLabel = type("_Other", (), {})
    monkeypatch.setitem(sys.modules, "datasets", fake)


def _inputs(*names: str) -> str:
    return ", ".join(f"{{name: {n}, type: image}}" for n in names)


def test_two_image_columns_each_keep_their_own_pixels(tmp_path, monkeypatch):
    _fake_datasets(monkeypatch, ["image", "wrist_image"])
    data = materialise(load_spec(SPEC.format(uri="hf://o/n", inputs=_inputs("image", "wrist_image"))),
                       tmp_path / "d")
    assert [_pixel(p) for p in data.train["image"]] == [0, 1, 2]
    assert [_pixel(p) for p in data.train["wrist_image"]] == [10, 11, 12]
    assert Path(data.train["image"][0]).parent.name == "image"
    assert Path(data.train["wrist_image"][0]).parent.name == "wrist_image"


def test_a_lone_image_column_keeps_the_flat_layout(tmp_path, monkeypatch):
    _fake_datasets(monkeypatch, ["img"])
    data = materialise(load_spec(SPEC.format(uri="hf://o/n", inputs=_inputs("img"))), tmp_path / "d")
    assert Path(data.train["img"][0]) == (tmp_path / "d" / "images" / "train" / "0.png").resolve()


def test_an_image_column_the_spec_does_not_name_is_not_written(tmp_path, monkeypatch):
    # The unused column comes last: before, it was written over the used one's files.
    _fake_datasets(monkeypatch, ["image", "mask"])
    data = materialise(load_spec(SPEC.format(uri="hf://o/n", inputs=_inputs("image"))), tmp_path / "d")
    assert [_pixel(p) for p in data.train["image"]] == [0, 1, 2]
    assert Path(data.train["image"][0]).parent == (tmp_path / "d" / "images" / "train").resolve()
    assert sorted(p.name for p in (tmp_path / "d" / "images" / "train").iterdir()) == [
        "0.png", "1.png", "2.png"]


def test_column_names_become_distinct_safe_directory_names():
    assert _media_dirs(["observation.images.top", "a/b", "a_b", ".."]) == {
        "observation.images.top": "observation.images.top", "a/b": "a_b", "a_b": "a_b_",
        "..": "column"}
    assert _media_dirs(["only"]) == {"only": ""}
    assert _media_dirs([]) == {}


def test_a_real_local_dataset_with_two_image_columns(tmp_path):
    datasets = pytest.importorskip("datasets")
    repo = tmp_path / "ds"
    for split, n in (("train", 4), ("test", 2)):
        ds = datasets.Dataset.from_dict(
            {"image": [_png(i) for i in range(n)], "wrist_image": [_png(100 + i) for i in range(n)],
             "y": ["ab"[i % 2] for i in range(n)]},
            features=datasets.Features({"image": datasets.Image(), "wrist_image": datasets.Image(),
                                        "y": datasets.Value("string")}))
        (repo / "data").mkdir(parents=True, exist_ok=True)
        ds.to_parquet(str(repo / "data" / f"{split}.parquet"))
    data = materialise(load_spec(SPEC.format(uri=f"hf://{repo}", inputs=_inputs("image", "wrist_image"))),
                       tmp_path / "d")
    assert [_pixel(p) for p in data.train["image"]] == [0, 1, 2, 3]
    assert [_pixel(p) for p in data.train["wrist_image"]] == [100, 101, 102, 103]
    assert [_pixel(p) for p in data.validation["wrist_image"]] == [100, 101]
