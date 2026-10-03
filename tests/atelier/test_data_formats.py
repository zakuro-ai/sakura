"""The rest of the data layer on CPU, from tiny local fixtures: every format
a backend consumes, in the exact shape that backend takes (data.materialise's
per-task contract), plus the refusals that keep a bad spec from training."""

import json

import pandas as pd
import pytest

from sakura.atelier import data as data_mod
from sakura.atelier.data import materialise
from sakura.atelier.spec import load_spec


def _spec(*, task, uri, fmt, split, inputs, output, model, hp, export="[onnx]"):
    return load_spec(f"""
atelier: 1
name: t
task: {task}
data: {{uri: "{uri}", format: {fmt}, split: {split}}}
features:
  inputs: {inputs}
  output: {output}
model: {model}
hp: {hp}
export: {export}
""")


def _tabular(uri, split="{validation_fraction: 0.25}", model="ludwig_ecd", hp="tabular/ecd@1", export="[onnx]",
             inputs="[{name: x, type: number}]"):
    return _spec(task="tabular_classification", uri=uri, fmt="csv", split=split, inputs=inputs,
                 output="{name: y, type: category}", model=model, hp=hp, export=export)


def _csv(tmp_path, n=20):
    p = tmp_path / "t.csv"
    pd.DataFrame({"x": range(n), "z": range(n), "y": ["a", "b"] * (n // 2)}).to_csv(p, index=False)
    return p


def test_audio_folder_split_by_fraction_keeps_every_clip_once(tmp_path):
    root = tmp_path / "clips"
    for word in ("yes", "no", "up"):
        (root / word).mkdir(parents=True)
        for i in range(4):
            (root / word / f"{i}.wav").write_bytes(b"RIFF")
        (root / word / "notes.txt").write_text("not audio")  # ignored by extension
    spec = _spec(task="audio_classification", uri=f"file://{root}", fmt="audio_folder",
                 split="{validation_fraction: 0.25}", inputs="[{name: audio, type: audio}]",
                 output="{name: label, type: category}", model="stacked_cnn_audio",
                 hp="audio_classification/fast@1")

    d = materialise(spec, tmp_path / "data")

    assert d.labels == ["no", "up", "yes"]
    assert len(d.train) == 9 and len(d.validation) == 3
    assert set(d.train["audio"]).isdisjoint(d.validation["audio"])
    assert all(p.endswith(".wav") for p in d.train["audio"])


def test_yolo_layout_is_pinned_verbatim_and_names_become_labels(tmp_path):
    src = tmp_path / "det"
    for split in ("train", "val"):
        (src / "images" / split).mkdir(parents=True)
        (src / "labels" / split).mkdir(parents=True)
        (src / "images" / split / "0.jpg").write_bytes(b"\xff\xd8")
        (src / "labels" / split / "0.txt").write_text("0 0.5 0.5 0.1 0.1\n")
    (src / "data.yaml").write_text("path: /somewhere/on/the/uploaders/disk\ntrain: images/train\n"
                                   "val: images/val\nnames:\n  0: cat\n  1: dog\n")
    spec = _spec(task="object_detection", uri=f"file://{src}", fmt="yolo",
                 split="{train: train, validation: val}", inputs="[{name: image, type: image}]",
                 output="{name: boxes, type: boxes}", model="yolo11n", hp="object_detection/fast@1")

    d = materialise(spec, tmp_path / "data")

    assert d.labels == ["cat", "dog"]
    pinned = tmp_path / "data" / "yolo"
    assert d.extra["yolo_data_yaml"] == str(pinned / "data.yaml")
    assert (pinned / "labels" / "val" / "0.txt").exists()
    import yaml
    assert yaml.safe_load((pinned / "data.yaml").read_text())["path"] == str(pinned.resolve())
    manifest = json.loads((tmp_path / "data" / "manifest.json").read_text())
    assert any(e["path"].endswith("labels/val/0.txt") for e in manifest)


def test_yolo_without_a_data_yaml_is_refused(tmp_path):
    (tmp_path / "det").mkdir()
    spec = _spec(task="object_detection", uri=f"file://{tmp_path / 'det'}", fmt="yolo",
                 split="{train: train, validation: val}", inputs="[{name: image, type: image}]",
                 output="{name: boxes, type: boxes}", model="yolo11n", hp="object_detection/fast@1")
    with pytest.raises(ValueError, match="needs a data.yaml"):
        materialise(spec, tmp_path / "data")


def _addition(tmp_path, n=10):
    p = tmp_path / "rows.jsonl"
    p.write_text("".join(json.dumps({"prompt": f"{i}+1=", "completion": str(i + 1)}) + "\n" for i in range(n)))
    return p


def _textgen(uri, split):
    return _spec(task="text_generation", uri=uri, fmt="jsonl", split=split,
                 inputs="[{name: prompt, type: text}]", output="{name: completion, type: text}",
                 model="lfm2-350m", hp="text_generation/lora-fast@1", export="[gguf]")


def test_text_generation_row_slices_give_prompt_completion_records(tmp_path):
    d = materialise(_textgen(f"file://{_addition(tmp_path)}", '{train: "0:8", validation: "8:"}'), tmp_path / "data")

    assert d.labels is None
    assert len(d.train) == 8 and d.validation == [{"prompt": "8+1=", "completion": "9"},
                                                   {"prompt": "9+1=", "completion": "10"}]


def test_text_generation_fraction_split_is_seeded(tmp_path):
    spec = _textgen(f"file://{_addition(tmp_path)}", "{validation_fraction: 0.3}")
    a = materialise(spec, tmp_path / "a").validation
    b = materialise(spec, tmp_path / "b").validation
    assert a == b and len(a) == 3


def test_csv_row_slices_and_labels(tmp_path):
    d = materialise(_tabular(f"file://{_csv(tmp_path)}", split='{train: "0:15", validation: "15:20"}'),
                    tmp_path / "data")
    assert len(d.train) == 15 and len(d.validation) == 5
    assert d.labels == ["a", "b"]
    assert list(d.train.columns) == ["x", "y"]  # only the spec's columns, in its order


def test_tabpfn_gets_x_y_pairs_not_a_frame(tmp_path):
    spec = _tabular(f"file://{_csv(tmp_path)}", model="tabpfn_v2", hp="tabular/tabpfn@1", export="[]",
                    inputs="[{name: x, type: number}, {name: z, type: number}]")
    d = materialise(spec, tmp_path / "data")
    (x_tr, y_tr), (x_va, y_va) = d.train, d.validation
    assert list(x_tr.columns) == ["x", "z"] and y_tr.name == "y"
    assert len(x_tr) + len(x_va) == 20 and len(y_va) == len(x_va)


def test_a_column_the_spec_names_but_the_data_lacks_is_refused(tmp_path):
    spec = _tabular(f"file://{_csv(tmp_path)}", inputs="[{name: missing_col, type: number}]")
    with pytest.raises(ValueError, match="missing"):
        materialise(spec, tmp_path / "data")


def test_parquet_is_read(tmp_path):
    pytest.importorskip("pyarrow")
    p = tmp_path / "t.parquet"
    pd.read_csv(_csv(tmp_path)).to_parquet(p)
    spec = _tabular(f"file://{p}")
    spec = spec.model_copy(update={"data": spec.data.model_copy(update={"format": "parquet"})})
    d = materialise(spec, tmp_path / "data")
    assert len(d.train) + len(d.validation) == 20


def test_an_unsupported_uri_scheme_is_refused(tmp_path):
    with pytest.raises(ValueError, match="scheme 's3' is not supported"):
        materialise(_tabular("s3://bucket/t.csv"), tmp_path / "data")


def test_https_sources_are_downloaded_once(tmp_path, monkeypatch):
    src = _csv(tmp_path)
    calls = []

    def fake_urlretrieve(url, dest):
        calls.append(url)
        dest.write_bytes(src.read_bytes())

    monkeypatch.setattr(data_mod.urllib.request, "urlretrieve", fake_urlretrieve)
    spec = _tabular("https://example.test/data/t.csv")
    materialise(spec, tmp_path / "data")
    materialise(spec, tmp_path / "data")
    assert calls == ["https://example.test/data/t.csv"]


def test_the_manifest_hash_is_stable_across_runs(tmp_path):
    spec = _tabular(f"file://{_csv(tmp_path)}")
    assert materialise(spec, tmp_path / "a").manifest_sha256 == materialise(spec, tmp_path / "b").manifest_sha256



def test_a_pre_split_tree_split_by_fraction_is_refused_with_the_fix(tmp_path):
    root = tmp_path / "clips"
    for split in ("train", "validation"):
        for word in ("yes", "no"):
            (root / split / word).mkdir(parents=True)
            (root / split / word / "0.wav").write_bytes(b"RIFF")
    spec = _spec(task="audio_classification", uri=f"file://{root}", fmt="audio_folder",
                 split="{validation_fraction: 0.2}", inputs="[{name: audio, type: audio}]",
                 output="{name: label, type: category}", model="stacked_cnn_audio",
                 hp="audio_classification/fast@1")
    with pytest.raises(ValueError, match="already split"):
        materialise(spec, tmp_path / "data")
