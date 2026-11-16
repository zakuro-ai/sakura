"""The spec's ``curation`` block (SAK-0019, issue 202): validation rules, each rule on a table and on
a folder, the report block and the curated manifest, and real runs through ``python -m sakura.atelier run``."""

from __future__ import annotations

import json

import pandas as pd
import pytest
import yaml

from sakura.atelier import curation as cur
from sakura.atelier.data import materialise
from sakura.atelier.spec import dump_spec, load_spec
from tests.atelier.test_smoke_runs import _resolved, _run

TABLE = """
atelier: 1
name: curated
task: text_classification
data: {{uri: "file://{uri}", format: csv, split: {{validation_fraction: 0.25}}}}
features:
  inputs: [{{name: text, type: text}}]
  output: {{name: label, type: category}}
model: ludwig_text
hp: text_classification/ecd@1
export: [onnx]
{extra}
"""


def _spec(extra="", uri="/tmp/none.csv", **replace):
    text = TABLE.format(uri=uri, extra=extra)
    for old, new in replace.items():
        text = text.replace(old, new)
    return load_spec(text)


# ------------------------------------------------------------------------------------------- validation

def test_a_spec_without_curation_dumps_as_before_and_a_set_block_keeps_only_what_it_sets():
    assert "curation" not in dump_spec(_spec())
    spec = _spec("curation: {exact_duplicates: true, near_duplicates: {threshold: 0.8}, exclude_samples: ['validation:3', 7]}")
    assert yaml.safe_load(dump_spec(spec))["curation"] == {
        "exact_duplicates": True, "near_duplicates": {"threshold": 0.8}, "exclude_samples": ["validation:3", 7]}
    assert "curation" not in dump_spec(_spec("curation: {}"))
    # `near_duplicates: true` is the default threshold
    assert _spec("curation: {near_duplicates: true}").curation.near_duplicates.threshold == 0.9


@pytest.mark.parametrize("block, message", [
    ("{dedupe: true}", "Extra inputs are not permitted"),
    ("{near_duplicates: {threshold: 0}}", "greater than 0"),
    ("{near_duplicates: {threshold: 1.5}}", "less than or equal to 1"),
    ("{near_duplicates: {column: body}}", "curation.near_duplicates.column 'body' is not a column of spec.features"),
    ("{text_length: {}}", "curation.text_length needs min or max"),
    ("{text_length: {min: 10, max: 5}}", "curation.text_length.min is above max"),
    ("{text_length: {min: -1}}", "greater than or equal to 0"),
    ("{classes: {}}", "curation.classes needs allow or deny"),
    ("{classes: {allow: [a], deny: [b]}}", "curation.classes takes allow or deny, not both"),
    ("{classes: {allow: []}}", "curation.classes.allow lists no class"),
    ("{exclude_samples: [true]}", "exclude_samples"),
])
def test_a_bad_curation_block_is_refused_by_name(block, message):
    with pytest.raises(ValueError) as err:
        _spec(f"curation: {block}")
    assert message in str(err.value)


def test_curation_is_refused_where_it_does_not_apply():
    with pytest.raises(ValueError, match="curation is not supported for task 'text_classification' with data.format 'hf'"):
        _spec("curation: {exact_duplicates: true}", **{"format: csv": "format: hf"})
    folder = {"format: csv": "format: image_folder", "task: text_classification": "task: image_classification",
              "type: text": "type: image", "ludwig_text": "resnet18", "text_classification/ecd@1": "image_classification/fast@1"}
    assert _spec("curation: {exact_duplicates: true, exclude_samples: [a/1.png], classes: {deny: [b]}}", **folder).curation
    with pytest.raises(ValueError, match="curation.near_duplicates does not apply to a folder dataset"):
        _spec("curation: {near_duplicates: {threshold: 0.9}}", **folder)
    with pytest.raises(ValueError, match="curation.drop_missing_target does not apply to a folder dataset"):
        _spec("curation: {drop_missing_target: true}", **folder)
    with pytest.raises(ValueError, match="curation.classes needs a category or binary output"):
        _spec("curation: {classes: {deny: [a]}}", **{"type: category": "type: number", "ludwig_text": "ludwig_ecd",
                                                     "task: text_classification": "task: tabular_regression",
                                                     "text_classification/ecd@1": "tabular/ecd@1"})


# ------------------------------------------------------------------------------------------- text helpers

def test_near_duplicate_text_is_found_by_normalised_5_gram_similarity():
    texts = ["The quick brown fox jumps over the lazy dog.", "the quick  brown fox jumps over the lazy dog",
             "the quick brown fox jumps over a lazy dog", "an entirely different sentence about cats",
             "THE QUICK BROWN FOX JUMPS OVER THE LAZY DOG!!"]
    assert cur.normalise(texts[0]) == cur.normalise(texts[1]) == cur.normalise(texts[4])
    assert cur.near_duplicates(texts, 0.95) == {1: 0, 4: 0}
    assert cur.near_duplicates(texts, 0.7) == {1: 0, 2: 0, 4: 0}  # one word changed: similarity 0.73
    assert cur.near_duplicates(texts, 0.7) == cur.near_duplicates(texts, 0.7)
    assert cur.jaccard(cur.shingles("abc"), cur.shingles("ABC ")) == 1.0  # shorter than a 5-gram: the whole text


def test_a_table_sample_id_names_a_source_row_in_every_spelling():
    assert [cur.table_row_of(v) for v in (17, "17", "row:17", "validation:17", "train:17")] == [17] * 5
    assert [cur.table_row_of(v) for v in ("cat/1.png", "validation:x", True, "")] == [None] * 4


# ------------------------------------------------------------------------------------------- tables

def _table(tmp_path):
    rows = [
        ("the service was excellent and fast", "pos"),         # 0
        ("the service was excellent and fast", "pos"),         # 1  exact duplicate of 0
        ("The service was excellent, and fast!", "pos"),       # 2  near duplicate of 0
        ("terrible food and a rude waiter", "neg"),            # 3
        ("ok", "neg"),                                         # 4  too short
        ("no label on this one at all", None),                 # 5  missing target
        ("a spam row that should not be here", "spam"),        # 6  denied class
        ("we waited an hour for cold soup", "neg"),            # 7  excluded by id (a triage call)
        ("lovely terrace and friendly staff", "pos"),          # 8
        ("the bill was wrong twice", "neg"),                   # 9
        ("best pasta in town by far", "pos"),                  # 10
        ("never coming back to this place", "neg"),            # 11
    ]
    path = tmp_path / "reviews.csv"
    pd.DataFrame(rows, columns=["text", "label"]).to_csv(path, index=False)
    return path


CURATION = """curation:
  exclude_samples: ["validation:7", "row:99", "cat/1.png"]
  drop_missing_target: true
  classes: {deny: [spam]}
  text_length: {min: 5}
  exact_duplicates: true
  near_duplicates: {threshold: 0.8}
"""


def test_every_rule_runs_once_before_the_split_and_the_manifest_lists_each_row(tmp_path):
    spec = _spec(CURATION, uri=str(_table(tmp_path)))
    data = materialise(spec, tmp_path / "data")
    block = data.extra["curation"]
    report, manifest = block["report"], block["manifest"]
    assert (report["rows_before"], report["rows_after"]) == (12, 6)
    assert report["rules"] == {"exclude_samples": 1, "missing_target": 1, "classes": 1, "text_length": 1,
                               "exact_duplicates": 1, "near_duplicates": 1}
    assert report["exclude_samples_not_found"] == ["row:99", "cat/1.png"]
    assert report["near_duplicates"]["column"] == "text" and report["near_duplicates"]["threshold"] == 0.8
    assert manifest["kept"] == ["row:0", "row:3", "row:8", "row:9", "row:10", "row:11"]
    assert manifest["dropped"] == [
        {"id": "row:1", "rule": "exact_duplicates", "of": "row:0"},
        {"id": "row:2", "rule": "near_duplicates", "of": "row:0"},
        {"id": "row:4", "rule": "text_length"},
        {"id": "row:5", "rule": "missing_target"},
        {"id": "row:6", "rule": "classes"},
        {"id": "row:7", "rule": "exclude_samples"},
    ]
    assert manifest["schema"] == 1 and manifest["kept_sha256"] == report["kept_sha256"]
    # the split is drawn from the curated rows only, and the denied class is gone from the labels
    assert len(data.train) + len(data.validation) == 6 and data.labels == ["neg", "pos"]
    assert sorted([*data.train.index, *data.validation.index]) == [0, 3, 8, 9, 10, 11]
    assert data.extra["data_split"]["train_rows"] + data.extra["data_split"]["validation_rows"] == 6
    # replayable: the same spec gives the same manifest
    again = materialise(spec, tmp_path / "again").extra["curation"]
    assert again["manifest"] == manifest


def test_curation_runs_before_limits_and_a_rule_that_drops_nothing_reports_zero(tmp_path):
    spec = _spec("curation: {exact_duplicates: true, classes: {allow: [pos, neg, spam]}}\n"
                 "limits: {max_train_rows: 4}", uri=str(_table(tmp_path)))
    data = materialise(spec, tmp_path / "data")
    report = data.extra["curation"]["report"]
    assert (report["rows_before"], report["rows_after"]) == (12, 10)  # the duplicate and the unlabelled row
    assert report["rules"] == {"classes": 1, "exact_duplicates": 1}
    assert len(data.train) == 4 and data.extra["limits_applied"]["rows_before"]["train"] > 4
    untouched = materialise(_spec("curation: {text_length: {max: 1000}}", uri=str(_table(tmp_path))), tmp_path / "u")
    assert untouched.extra["curation"]["report"]["rules"] == {"text_length": 0}
    assert untouched.extra["curation"]["manifest"]["dropped"] == []


def test_a_curation_that_removes_every_row_fails_and_says_so(tmp_path):
    spec = _spec("curation: {classes: {allow: [nothing]}}", uri=str(_table(tmp_path)))
    with pytest.raises(ValueError, match="curation removed every row"):
        materialise(spec, tmp_path / "data")


def test_text_generation_rows_are_curated_on_the_prompt(tmp_path):
    path = tmp_path / "pairs.jsonl"
    rows = [{"prompt": f"what is the code for key {i % 6}?", "completion": f"code{i % 6}"} for i in range(24)]
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    spec = load_spec(f"""
atelier: 1
name: t
task: text_generation
data: {{uri: "file://{path}", format: jsonl, split: {{validation_fraction: 0.5}}}}
features:
  inputs: [{{name: prompt, type: text}}]
  output: {{name: completion, type: text}}
model: lfm2-350m
hp: text_generation/lora-fast@1
export: [safetensors]
curation: {{exact_duplicates: true}}
""")
    data = materialise(spec, tmp_path / "data")
    assert data.extra["curation"]["report"]["rows_after"] == 6
    assert len(data.train) + len(data.validation) == 6


# ------------------------------------------------------------------------------------------- folders

def _folder(tmp_path):
    from sakura.atelier.synthetic import _png
    import numpy as np

    root = tmp_path / "images"
    for cls, base in (("cat", 40), ("dog", 160)):
        (root / cls).mkdir(parents=True)
        for i in range(6):
            arr = np.full((8, 8, 3), base + i, dtype=np.uint8)
            (root / cls / f"{i}.png").write_bytes(_png(arr))
    (root / "dog" / "copy.png").write_bytes((root / "cat" / "0.png").read_bytes())  # a cat filed under dog
    (root / "junk").mkdir()
    (root / "junk" / "0.png").write_bytes(_png(np.zeros((8, 8, 3), dtype=np.uint8)))
    return root


def test_a_folder_dataset_drops_excluded_files_duplicate_bytes_and_denied_classes(tmp_path):
    root = _folder(tmp_path)
    spec = load_spec(f"""
atelier: 1
name: t
task: image_classification
data: {{uri: "file://{root}", format: image_folder, split: {{validation_fraction: 0.25}}}}
features:
  inputs: [{{name: image, type: image}}]
  output: {{name: label, type: category}}
model: resnet18
hp: image_classification/fast@1
export: [onnx]
curation: {{exclude_samples: ["cat/3.png", "bird/1.png"], exact_duplicates: true, classes: {{deny: [junk]}}}}
""")
    data = materialise(spec, tmp_path / "data")
    block = data.extra["curation"]
    assert (block["report"]["rows_before"], block["report"]["rows_after"]) == (14, 11)
    assert block["report"]["rules"] == {"exclude_samples": 1, "classes": 1, "exact_duplicates": 1}
    assert block["report"]["exclude_samples_not_found"] == ["bird/1.png"]
    assert block["manifest"]["dropped"] == [{"id": "cat/3.png", "rule": "exclude_samples"},
                                            {"id": "dog/copy.png", "rule": "exact_duplicates", "of": "cat/0.png"},
                                            {"id": "junk/0.png", "rule": "classes"}]
    used = {"/".join(p.split("/")[-2:]) for p in [*data.train["image"], *data.validation["image"]]}
    assert len(used) == 11 and not used & {"cat/3.png", "dog/copy.png", "junk/0.png"} and data.labels == ["cat", "dog"]
    pinned = {e["path"] for e in json.loads((tmp_path / "data" / "manifest.json").read_text())}
    assert pinned == used  # the dataset pin lists only the files the curated split uses


def test_a_folder_with_split_directories_is_curated_across_both(tmp_path):
    root = _folder(tmp_path)
    import shutil

    for split in ("train", "val"):
        for cls in ("cat", "dog"):
            (root / split / cls).mkdir(parents=True)
    for cls in ("cat", "dog"):
        for i in range(6):
            shutil.move(str(root / cls / f"{i}.png"), str(root / ("val" if i >= 4 else "train") / cls / f"{i}.png"))
    shutil.rmtree(root / "cat"), shutil.rmtree(root / "dog"), shutil.rmtree(root / "junk")
    shutil.copy(root / "train" / "cat" / "0.png", root / "val" / "cat" / "leak.png")  # a training file in validation
    shutil.copy(root / "train" / "dog" / "1.png", root / "val" / "dog" / "5.png.bak.png")
    spec = load_spec(f"""
atelier: 1
name: t
task: image_classification
data: {{uri: "file://{root}", format: image_folder, split: {{train: train, validation: val}}}}
features:
  inputs: [{{name: image, type: image}}]
  output: {{name: label, type: category}}
model: resnet18
hp: image_classification/fast@1
export: [onnx]
curation: {{exclude_samples: ["dog/4.png"], exact_duplicates: true}}
""")
    data = materialise(spec, tmp_path / "data")
    dropped = data.extra["curation"]["manifest"]["dropped"]
    assert {"id": "val/dog/4.png", "rule": "exclude_samples"} in dropped  # the validation file is the one meant
    assert {"id": "val/cat/leak.png", "rule": "exact_duplicates", "of": "train/cat/0.png"} in dropped
    assert len(dropped) == 3 and len(data.train) == 8 and len(data.validation) == 3


# ------------------------------------------------------------------------------------------- end to end

RUN = """
atelier: 1
name: curated-run
task: tabular_classification
data: {{uri: "file://{uri}", format: csv, split: {{validation_fraction: 0.25}}}}
features:
  inputs: [{{name: f0, type: number}}, {{name: f1, type: number}}]
  output: {{name: label, type: category}}
model: random_forest
hp: tabular/random_forest@1
overrides: {{n_estimators: 20}}
export: [joblib]
seed: 7
curation:
  exclude_samples: ["validation:3", "validation:5", "row:100000"]
  drop_missing_target: true
  classes: {{deny: [junk]}}
  exact_duplicates: true
"""


def test_a_curated_run_reports_the_counts_and_delivers_the_manifest(tmp_path):
    pytest.importorskip("sklearn")
    import numpy as np

    rng = np.random.default_rng(0)
    x = rng.normal(size=(200, 2))
    label = np.where(x[:, 0] + x[:, 1] > 0, "a", "b").astype(object)
    frame = pd.DataFrame({"f0": x[:, 0], "f1": x[:, 1], "label": label})
    frame.loc[10:14, "label"] = None          # 5 rows without a target
    frame.loc[20:22, "label"] = "junk"        # 3 rows of a class to deny
    frame = pd.concat([frame, frame.iloc[30:40]], ignore_index=True)  # rows 200..209 repeat rows 30..39
    path = tmp_path / "table.csv"
    frame.to_csv(path, index=False)

    report, out = _run(tmp_path, RUN.format(uri=path), "--no-notebook")
    assert report["status"] == "done"
    block = report["curation"]
    assert (block["rows_before"], block["rows_after"]) == (210, 190)
    assert block["rules"] == {"exclude_samples": 2, "missing_target": 5, "classes": 3, "exact_duplicates": 10}
    assert block["exclude_samples_not_found"] == ["row:100000"]
    assert block["manifest"] == "artifacts/curated_manifest.json"
    entry = next(a for a in report["artifacts"] if a["format"] == "curated_manifest")
    assert entry["path"] == block["manifest"] and entry["sha256"] == block["manifest_sha256"]
    manifest = json.loads((out / block["manifest"]).read_text())
    assert len(manifest["kept"]) == 190 and len(manifest["dropped"]) == 20
    assert {"id": "row:205", "rule": "exact_duplicates", "of": "row:35"} in manifest["dropped"]
    assert {"id": "row:3", "rule": "exclude_samples"} in manifest["dropped"]
    assert manifest["kept_sha256"] == block["kept_sha256"]
    split = report["data_split"]
    assert split["train_rows"] + split["validation_rows"] == 190 and "junk" not in split["label_distribution"]["train"]
    assert report["metrics"]["accuracy"] > 0.8
    # resolved.yaml keeps the block, and replaying it curates the same rows
    resolved = _resolved(out)
    assert resolved["curation"]["exclude_samples"] == ["validation:3", "validation:5", "row:100000"]
    again, out2 = _run(tmp_path, (out / "resolved.yaml").read_text(), "--no-notebook", name="again")
    assert again["curation"] == block and again["spec_sha256"] == report["spec_sha256"]
    assert again["metrics"] == report["metrics"]
