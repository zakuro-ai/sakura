"""The rest of the data layer on CPU, from tiny local fixtures: every format
a backend consumes, in the exact shape that backend takes (data.materialise's
per-task contract), plus the refusals that keep a bad spec from training."""

import json

import pandas as pd
import pytest

from sakura.atelier import data as data_mod
from sakura.atelier.data import materialise
from sakura.atelier.spec import Curation, load_spec


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
                 output="{name: boxes, type: boxes}", model="rtdetr_r18", hp="object_detection/fast@1")

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
                 output="{name: boxes, type: boxes}", model="rtdetr_r18", hp="object_detection/fast@1")
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


def test_text_generation_ignores_the_columns_a_prepare_step_writes_beside_prompt_and_completion(tmp_path):
    # The hub's prepare step keeps `conversation`, `turn`, `last_turn` and source fields next to the two columns
    # the spec names. They are not features: the records are still exactly {prompt, completion}.
    p = tmp_path / "prepared.jsonl"
    p.write_text("".join(json.dumps({
        "prompt": f"{i}+1=", "completion": str(i + 1), "conversation": f"c{i // 2}", "turn": i % 2,
        "last_turn": bool(i % 2), "exp_cal_state": {"events": [{"title": "sync", "start": f"{9 + i}:00"}]},
    }) + "\n" for i in range(10)))
    d = materialise(_textgen(f"file://{p}", '{train: "0:8", validation: "8:"}'), tmp_path / "data")
    assert len(d.train) == 8 and all(set(row) == {"prompt", "completion"} for row in d.train)
    assert d.validation == [{"prompt": "8+1=", "completion": "9"}, {"prompt": "9+1=", "completion": "10"}]


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


def _image_table(tmp_path, n=8):
    """An extracted hub archive: <dir>/ds/table.csv + images/, one or two photos per record."""
    from PIL import Image

    root = tmp_path / "archive" / "ds"
    (root / "images").mkdir(parents=True)
    rows = []
    for i in range(n):
        names = [f"images/{i}_{k}.png" for k in range(1 + i % 2)]
        for name in names:
            Image.new("RGB", (8, 8), (i * 20, 0, 0)).save(root / name)
        rows.append({"images": json.dumps(names), "bf": 10.0 + i, "split": "val" if i < 2 else "train"})
    pd.DataFrame(rows).to_csv(root / "table.csv", index=False)
    return root


def _image_regression(uri, split="{validation_fraction: 0.25}"):
    return _spec(task="image_regression", uri=uri, fmt="csv", split=split,
                 inputs="[{name: images, type: image}]", output="{name: bf, type: number}",
                 model="siglip2_b16_256", hp="image_regression/siglip2@1")


def test_image_table_splits_records_before_exploding_their_photos(tmp_path, monkeypatch):
    root = _image_table(tmp_path)
    # csv has no list type: the fixture stores the list as JSON text, as a parquet list column reads back
    real = pd.read_csv
    monkeypatch.setattr(pd, "read_csv", lambda p: real(p).assign(images=lambda d: d["images"].map(json.loads)))
    d = materialise(_image_regression(f"file://{tmp_path / 'archive'}", "{train: train, validation: val}"),
                    tmp_path / "work")
    # records 0,1 are val (1 + 2 photos), 2..7 train (12 - 3 = 9 photos): no record in both
    assert len(d.validation) == 3 and len(d.train) == 9
    assert set(d.validation["bf"]) == {10.0, 11.0}
    assert not set(d.train["images"]) & set(d.validation["images"])
    assert all(p.startswith(str(root.resolve())) for p in d.train["images"])
    assert d.labels is None
    pinned = json.loads((tmp_path / "work" / "manifest.json").read_text())
    assert {e["path"] for e in pinned} == {"table.csv"} | {f"images/{i}_{k}.png" for i in range(8) for k in range(1 + i % 2)}


def test_image_table_refuses_a_missing_image(tmp_path):
    root = _image_table(tmp_path, n=4)
    pd.DataFrame({"images": ["images/nope.png"], "bf": [1.0]}).to_csv(root / "table.csv", index=False)
    with pytest.raises(ValueError, match="does not exist"):
        materialise(_image_regression(f"file://{root}", "{train: '0:1', validation: '0:1'}"), tmp_path / "work")


def test_image_table_directory_needs_exactly_one_table_of_its_format(tmp_path):
    root = _image_table(tmp_path, n=4)
    (root / "other.csv").write_text("a\n1\n")
    with pytest.raises(ValueError, match="exactly one .csv"):
        materialise(_image_regression(f"file://{root}"), tmp_path / "work")


def test_image_table_ignores_a_copy_in_another_format(tmp_path, monkeypatch):
    # bulkorcut-style archives ship the table as both parquet and csv
    root = _image_table(tmp_path, n=4)
    (root / "table.parquet").write_bytes(b"not read: data.format is csv")
    real = pd.read_csv
    monkeypatch.setattr(pd, "read_csv", lambda p: real(p).assign(images=lambda d: d["images"].map(json.loads)))
    d = materialise(_image_regression(f"file://{root}"), tmp_path / "work")
    assert len(d.train) + len(d.validation) == 6


def _with_lone_surrogate(tmp_path, n=10):
    # A JSON `\udce1` escape decodes to a lone surrogate: text no UTF-8 file can hold.
    p = tmp_path / "rows.jsonl"
    rows = [{"prompt": f"{i}+1=", "completion": str(i + 1)} for i in range(n)]
    rows[3]["completion"] = "four \udce1 and \udce2"
    p.write_text("".join(json.dumps(r) + "\n" for r in rows))
    assert "\\udce1" in p.read_text()
    return p


def test_text_generation_replaces_a_lone_surrogate_and_counts_the_rows(tmp_path):
    spec = _textgen(f"file://{_with_lone_surrogate(tmp_path)}", '{train: "0:8", validation: "8:"}')
    d = materialise(spec, tmp_path / "data")

    assert d.train[3]["completion"] == "four \ufffd and \ufffd"
    assert d.extra["data_encoding"] == {"rows_with_invalid_unicode": 1, "characters_replaced": 2,
                                        "columns": ["completion"]}
    for r in d.train + d.validation:  # every record can be written as UTF-8, as the notebook does
        json.dumps(r, ensure_ascii=False).encode("utf-8")


def test_text_generation_with_curation_survives_a_lone_surrogate(tmp_path):
    spec = _textgen(f"file://{_with_lone_surrogate(tmp_path)}", '{train: "0:8", validation: "8:"}')
    spec = spec.model_copy(update={"curation": Curation(exact_duplicates=True)})
    d = materialise(spec, tmp_path / "data")
    assert d.extra["data_encoding"]["rows_with_invalid_unicode"] == 1


def test_a_clean_text_generation_table_reports_no_data_encoding(tmp_path):
    d = materialise(_textgen(f"file://{_addition(tmp_path)}", '{train: "0:8", validation: "8:"}'), tmp_path / "data")
    assert "data_encoding" not in d.extra


# --- text_generation from csv / parquet (the hub's La Forge LLM datasets) ------

def _textgen_table(tmp_path, fmt, n=10, inputs="[{name: input, type: text}]"):
    df = pd.DataFrame({"input": [f"{i}+1=" for i in range(n)], "output": [str(i + 1) for i in range(n)],
                       "note": ["ignored"] * n})
    p = tmp_path / f"t.{fmt}"
    (df.to_csv(p, index=False) if fmt == "csv" else df.to_parquet(p, index=False))
    return _spec(task="text_generation", uri=f"file://{p}", fmt=fmt, split="{validation_fraction: 0.3}",
                 inputs=inputs, output="{name: output, type: category}",
                 model="lfm2-350m", hp="text_generation/lora-fast@1", export="[gguf]")


@pytest.mark.parametrize("fmt", ["csv", "parquet"])
def test_text_generation_reads_a_table_into_prompt_completion_records(tmp_path, fmt):
    pytest.importorskip("pyarrow") if fmt == "parquet" else None
    d = materialise(_textgen_table(tmp_path, fmt), tmp_path / "data")

    assert d.labels is None and len(d.train) == 7 and len(d.validation) == 3
    assert all(set(r) == {"prompt", "completion"} for r in d.train + d.validation)
    assert all(r["completion"] == str(int(r["prompt"][:-3]) + 1) for r in d.train + d.validation)
    assert {r["prompt"] for r in d.train}.isdisjoint({r["prompt"] for r in d.validation})
    assert (tmp_path / "data" / f"t.{fmt}").is_file()  # pinned copy, hashed by the manifest


def test_text_generation_table_fraction_split_is_seeded(tmp_path):
    spec = _textgen_table(tmp_path, "csv")
    assert materialise(spec, tmp_path / "a").validation == materialise(spec, tmp_path / "b").validation


def test_text_generation_table_with_several_inputs_renders_named_lines(tmp_path):
    df = pd.DataFrame({"title": ["a", "b"], "body": ["x", "y"], "output": ["1", "2"]})
    p = tmp_path / "t.csv"
    df.to_csv(p, index=False)
    spec = _spec(task="text_generation", uri=f"file://{p}", fmt="csv", split='{train: "0:1", validation: "1:"}',
                 inputs="[{name: title, type: text}, {name: body, type: text}]",
                 output="{name: output, type: text}", model="lfm2-350m",
                 hp="text_generation/lora-fast@1", export="[gguf]")
    d = materialise(spec, tmp_path / "data")
    assert d.train == [{"prompt": "title: a\nbody: x", "completion": "1"}]
    assert d.validation == [{"prompt": "title: b\nbody: y", "completion": "2"}]


def test_text_generation_table_missing_a_feature_column_is_refused(tmp_path):
    with pytest.raises(ValueError, match="missing \\['text'\\]"):
        materialise(_textgen_table(tmp_path, "csv", inputs="[{name: text, type: text}]"), tmp_path / "data")
