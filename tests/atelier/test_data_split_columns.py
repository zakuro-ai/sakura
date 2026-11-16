"""A user-chosen split column and stratified holdouts (CONTRACTS §1.3)."""

import pandas as pd
import pytest

from sakura.atelier.data import materialise
from sakura.atelier.spec import dump_spec, load_spec

SPEC = """
atelier: 1
name: split-columns
task: tabular_classification
data:
  uri: {uri}
  format: {fmt}
  {data}
features:
  inputs: [{{name: x, type: number}}]
  output: {{name: y, type: category}}
model: ludwig_ecd
hp: tabular/ecd@1
export: [onnx]
"""


def _spec(tmp_path, data, df=None, fmt="csv"):
    table = tmp_path / f"t.{fmt}"
    df = df if df is not None else pd.DataFrame({
        "x": range(40), "y": ["a"] * 30 + ["b"] * 10,
        "part": ["train"] * 20 + ["test"] * 10 + ["dev"] * 10})
    (df.to_csv if fmt == "csv" else df.to_parquet)(table, index=False)
    return load_spec(SPEC.format(uri=f"file://{table}", fmt=fmt, data=data))


def _plain(spec_data):
    return load_spec(SPEC.format(uri="file:///x.csv", fmt="csv", data=spec_data))


# ---------------------------------------------------------------- column

def test_split_column_honours_the_values(tmp_path):
    spec = _spec(tmp_path, "split: {column: part, train: train, validation: [test, dev]}")
    data = materialise(spec, tmp_path / "d")
    assert list(data.train["x"]) == list(range(20))
    assert list(data.validation["x"]) == list(range(20, 40))
    block = data.extra["data_split"]
    assert block["mode"] == "column" and block["column"] == "part"
    assert (block["train_rows"], block["validation_rows"], block["excluded_rows"]) == (20, 20, 0)
    assert block["label_distribution"]["train"] == {"a": 20}


def test_split_column_exclude_drops_rows(tmp_path):
    spec = _spec(tmp_path, "split: {column: part, train: train, validation: dev, exclude: [test]}")
    data = materialise(spec, tmp_path / "d")
    assert (len(data.train), len(data.validation)) == (20, 10)
    assert data.extra["data_split"]["excluded_rows"] == 10


def test_split_column_works_on_parquet_and_numeric_values(tmp_path):
    pytest.importorskip("pyarrow")  # the CI image has no parquet engine
    df = pd.DataFrame({"x": range(10), "y": list("ababababab"), "fold": [0] * 7 + [1] * 3})
    spec = _spec(tmp_path, "split: {column: fold, train: 0, validation: 1}", df, fmt="parquet")
    data = materialise(spec, tmp_path / "d")
    assert (len(data.train), len(data.validation)) == (7, 3)


def test_unmapped_value_is_an_error_naming_it(tmp_path):
    spec = _spec(tmp_path, "split: {column: part, train: train, validation: dev}")
    with pytest.raises(ValueError, match=r"'test' \(10 rows\).*exclude"):
        materialise(spec, tmp_path / "d")


def test_missing_values_are_unmapped_too(tmp_path):
    df = pd.DataFrame({"x": range(4), "y": list("abab"), "part": ["train", "train", "val", None]})
    spec = _spec(tmp_path, "split: {column: part, train: train, validation: val}", df)
    with pytest.raises(ValueError, match="<null>"):
        materialise(spec, tmp_path / "d")


def test_missing_column_is_an_error(tmp_path):
    spec = _spec(tmp_path, "split: {column: nope, train: a, validation: b}")
    with pytest.raises(ValueError, match="'nope'.*not found"):
        materialise(spec, tmp_path / "d")


def test_a_side_with_no_rows_is_an_error(tmp_path):
    spec = _spec(tmp_path, "split: {column: part, train: train, validation: missing, exclude: [test, dev]}")
    with pytest.raises(ValueError, match="validation.*selects no rows"):
        materialise(spec, tmp_path / "d")


# ------------------------------------------------------------ stratified

def test_stratified_holdout_keeps_class_shares(tmp_path):
    spec = _spec(tmp_path, "split: {validation_fraction: 0.2, stratify_by: [y]}")
    data = materialise(spec, tmp_path / "d")
    assert data.validation["y"].value_counts().to_dict() == {"a": 6, "b": 2}
    assert len(data.train) + len(data.validation) == 40
    assert not set(data.train["x"]) & set(data.validation["x"])
    block = data.extra["data_split"]
    assert block["mode"] == "stratified_fraction" and block["stratify_by"] == ["y"]
    assert block["label_distribution"]["validation"] == {"a": 6, "b": 2}


def test_stratified_is_deterministic_and_seed_dependent(tmp_path):
    spec = _spec(tmp_path, "split: {validation_fraction: 0.25, stratify_by: [y, part]}")
    a = materialise(spec, tmp_path / "a").validation["x"].tolist()
    b = materialise(spec, tmp_path / "b").validation["x"].tolist()
    assert a == b
    other = spec.model_copy(update={"seed": 7})
    assert materialise(other, tmp_path / "c").validation["x"].tolist() != a


def test_hub_spelling_of_stratify_by_is_accepted(tmp_path):
    spec = _spec(tmp_path, "validation_fraction: 0.2\n  stratify_by: [y]")
    assert spec.data.split.stratify_by == ["y"] and spec.data.split.validation_fraction == 0.2


def test_single_row_stratum_stays_in_train_with_a_warning(tmp_path):
    df = pd.DataFrame({"x": range(11), "y": ["a"] * 10 + ["rare"]})
    spec = _spec(tmp_path, "split: {validation_fraction: 0.2, stratify_by: [y]}", df)
    data = materialise(spec, tmp_path / "d")
    assert "rare" in set(data.train["y"]) and "rare" not in set(data.validation["y"])
    assert "rare" in data.extra["data_split"]["warnings"][0]


def test_stratifying_by_a_unique_column_fails_clearly(tmp_path):
    spec = _spec(tmp_path, "split: {validation_fraction: 0.2, stratify_by: [x]}")
    with pytest.raises(ValueError, match="none with two or more rows"):
        materialise(spec, tmp_path / "d")


def test_stratify_column_must_exist(tmp_path):
    spec = _spec(tmp_path, "split: {validation_fraction: 0.2, stratify_by: [nope]}")
    with pytest.raises(ValueError, match="'nope'.*not found"):
        materialise(spec, tmp_path / "d")


# ------------------------------------------------------- spec validation

@pytest.mark.parametrize("data, msg", [
    ("split: {column: part, train: a}", "needs both train and validation"),
    ("split: {column: part, train: a, validation: a}", "both claim"),
    ("split: {column: part, train: a, validation: b, exclude: [b]}", "exclude lists"),
    ("split: {column: part, train: a, validation: [] }", "lists no values"),
    ("split: {column: part, validation_fraction: 0.2}", "needs both train and validation"),
    ("split: {column: part, train: a, validation: b, validation_fraction: 0.2}", "not both"),
    ("split: {column: part, train: a, validation: b, stratify_by: [y]}", "stratify_by needs validation_fraction"),
    ("split: {train: '0:5', validation: '5:9', stratify_by: [y]}", "stratify_by needs validation_fraction"),
    ("split: {validation_fraction: 0.2, stratify_by: []}", "one or more"),
    ("split: {validation_fraction: 0.2, stratify_by: [y, y]}", "distinct"),
    ("split: {train: '0:5', validation: '5:9', exclude: [a]}", "only applies with split.column"),
    ("split: {train: 1, validation: 2}", "split expression string"),
    ("validation_fraction: 0.2\n  split: {column: part, train: a, validation: b}", "not both"),
])
def test_invalid_combinations_are_rejected(data, msg):
    with pytest.raises(ValueError, match=msg):
        _plain(data)


def test_column_and_stratify_need_a_table_format():
    for extra in ("split: {column: c, train: a, validation: b}",
                  "split: {validation_fraction: 0.2, stratify_by: [y]}"):
        with pytest.raises(ValueError, match="csv, parquet or jsonl"):
            load_spec(SPEC.format(uri="hf://a/b", fmt="hf", data=extra))


def test_stratify_is_refused_for_forecasting():
    text = SPEC.format(uri="file:///x.csv", fmt="csv",
                       data="split: {validation_fraction: 0.2, stratify_by: [y]}")
    text = text.replace("tabular_classification", "time_series_forecasting")
    text = text.replace("tabular/ecd@1", "time_series_forecasting/fast@1")
    with pytest.raises(ValueError, match="chronological"):
        load_spec(text)


# ------------------------------------------------- backwards compatibility

def test_specs_without_the_new_fields_dump_exactly_as_before():
    spec = _plain("split: {validation_fraction: 0.25}")
    dumped = dump_spec(spec)
    assert spec.data.split.model_dump() == {"train": None, "validation": None, "validation_fraction": 0.25}
    assert "column:" not in dumped and "stratify_by" not in dumped and "exclude" not in dumped
    assert "train: null" in dumped and "validation_fraction: 0.25" in dumped


def test_random_fraction_split_is_unchanged(tmp_path):
    # A classification holdout is stratified by default now (BE-18); `strategy: random` is the old code path.
    spec = _spec(tmp_path, "split: {validation_fraction: 0.25, strategy: random}")
    data = materialise(spec, tmp_path / "d")
    table = pd.read_csv(tmp_path / "t.csv")
    expected = table.sample(frac=0.25, random_state=spec.seed)
    assert data.validation["x"].tolist() == expected["x"].tolist()
    assert data.extra["data_split"]["mode"] == "fraction"
    assert data.extra["data_split"]["strategy"] == "random"
    assert "label_distribution" in data.extra["data_split"]
