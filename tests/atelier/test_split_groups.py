"""Group-aware holdouts and the evaluation columns of a text-generation table (CONTRACTS §1.3, #238).

``data.split.group_by`` holds whole groups (a conversation, a patient) out; ``features.eval`` names the columns
that ride along on validation records and are never fed to the model."""

import json

import pandas as pd
import pytest

from sakura.atelier.data import materialise
from sakura.atelier.spec import dump_spec, load_spec

TABLE_SPEC = """
atelier: 1
name: grouped
task: text_generation
data:
  uri: file://{uri}
  format: {fmt}
  split: {split}
features:
  inputs: [{{name: prompt, type: text}}]
  output: {{name: completion, type: text}}{eval}
model: hf-causal-lm
hp: text_generation/lora-fast@1
export: [safetensors]
seed: 7
"""


def _rows(groups=20, turns=3):
    return [{"prompt": f"g{g} t{t}", "completion": f"reply {g}.{t}", "conversation": f"c{g}", "turn": t,
             "last_turn": t == turns - 1, "kind": "a" if g % 2 == 0 else "b",
             "exp": json.dumps({"events": g})} for g in range(groups) for t in range(turns)]


def _spec(tmp_path, split, rows=None, eval_block="", fmt="jsonl"):
    path = tmp_path / f"t.{fmt}"
    frame = pd.DataFrame(rows if rows is not None else _rows())
    if fmt == "jsonl":
        frame.to_json(path, orient="records", lines=True)
    else:
        frame.to_csv(path, index=False)
    return load_spec(TABLE_SPEC.format(uri=path, fmt=fmt, split=split, eval=eval_block))


def _groups(records_or_frame):
    return {r["conversation"] for r in records_or_frame}


GROUPED = "{validation_fraction: 0.25, group_by: conversation}"
EVAL = """
  eval: {reference: exp, final: last_turn, group: conversation, keep: [turn, kind]}"""


def test_no_group_is_on_both_sides_and_the_split_is_stable_for_a_seed(tmp_path):
    spec = _spec(tmp_path, GROUPED, eval_block="\n  eval: {group: conversation, keep: [conversation]}")
    first = materialise(spec, tmp_path / "d1")
    again = materialise(spec, tmp_path / "d2")
    train_groups = {r["prompt"].split()[0] for r in first.train}
    val_groups = {r["prompt"].split()[0] for r in first.validation}
    assert train_groups and val_groups and not (train_groups & val_groups)
    assert len(val_groups) == 5 and len(first.validation) == 15  # a quarter of 20 groups, all their rows
    assert first.validation == again.validation and first.train == again.train
    other = materialise(load_spec(dump_spec(spec).replace("seed: 7", "seed: 8")), tmp_path / "d3")
    assert other.validation != first.validation  # the seed moves the draw
    block = first.extra["data_split"]
    assert block["group_by"] == "conversation" and block["groups"] == {"train": 15, "validation": 5}
    assert (block["train_rows"], block["validation_rows"]) == (45, 15)


def test_a_csv_table_groups_the_same_way(tmp_path):
    spec = _spec(tmp_path, GROUPED, fmt="csv")
    data = materialise(spec, tmp_path / "d")
    assert not ({r["prompt"].split()[0] for r in data.train} & {r["prompt"].split()[0] for r in data.validation})


def test_stratify_by_works_when_the_stratum_is_constant_inside_a_group(tmp_path):
    spec = _spec(tmp_path, "{validation_fraction: 0.2, group_by: conversation, stratify_by: [kind]}")
    data = materialise(spec, tmp_path / "d")
    val_groups = {r["prompt"].split()[0] for r in data.validation}
    assert len(val_groups) == 4  # 2 of the 10 groups of each kind: the strata keep their share
    assert sum(int(g[1:]) % 2 == 0 for g in val_groups) == 2
    assert not (val_groups & {r["prompt"].split()[0] for r in data.train})
    assert data.extra["data_split"]["stratify_by"] == ["kind"]


def test_stratify_by_a_column_that_changes_inside_a_group_is_refused_in_words(tmp_path):
    rows = _rows()
    for r in rows:
        r["kind"] = "a" if r["turn"] == 0 else "b"
    spec = _spec(tmp_path, "{validation_fraction: 0.2, group_by: conversation, stratify_by: [kind]}", rows)
    with pytest.raises(ValueError, match=r"stratify_by \['kind'\].*not constant inside a group.*'c0'"):
        materialise(spec, tmp_path / "d")


def test_one_group_cannot_be_held_out_of_itself(tmp_path):
    rows = [r for r in _rows() if r["conversation"] == "c0"]
    with pytest.raises(ValueError, match="at least two distinct values"):
        materialise(_spec(tmp_path, GROUPED, rows), tmp_path / "d")


def test_a_missing_group_column_is_named(tmp_path):
    with pytest.raises(ValueError, match=r"group_by.*'nope'.*not found"):
        materialise(_spec(tmp_path, "{validation_fraction: 0.25, group_by: nope}"), tmp_path / "d")


@pytest.mark.parametrize("split, message", [
    ('{train: "0:10", validation: "10:", group_by: conversation}', "group_by needs validation_fraction"),
    ("{validation_fraction: 0.2, group_by: ''}", "group_by must name a column"),
])
def test_group_by_is_checked_by_validate(tmp_path, split, message):
    with pytest.raises(ValueError, match=message):
        _spec(tmp_path, split)


def test_group_by_needs_a_table(tmp_path):
    with pytest.raises(ValueError, match="group_by needs a csv, parquet or jsonl table"):
        load_spec(f"""
atelier: 1
name: x
task: image_classification
data: {{uri: "file://{tmp_path}", format: image_folder, split: {{validation_fraction: 0.2, group_by: g}}}}
features:
  inputs: [{{name: img, type: image}}]
  output: {{name: label, type: category}}
model: resnet18
hp: image_classification/fast@1
export: [onnx]
""")


# ------------------------------------------------------------------ the evaluation columns

def test_validation_records_carry_the_evaluation_fields_and_train_records_do_not(tmp_path):
    spec = _spec(tmp_path, GROUPED, eval_block=EVAL)
    data = materialise(spec, tmp_path / "d")
    assert all(set(r) == {"prompt", "completion"} for r in data.train)
    row = data.validation[0]
    assert set(row) == {"prompt", "completion", "eval"}
    group = row["prompt"].split()[0][1:]
    assert row["eval"] == {"reference": json.dumps({"events": int(group)}), "final": False, "group": f"c{group}",
                           "keep": {"turn": 0, "kind": "a" if int(group) % 2 == 0 else "b"}}
    assert [r["eval"]["final"] for r in data.validation[:3]] == [False, False, True]


def test_a_json_reference_cell_is_carried_as_json_text(tmp_path):
    rows = [{**r, "exp": {"events": [r["conversation"]]}} for r in _rows()]  # a parsed object in a jsonl cell
    data = materialise(_spec(tmp_path, GROUPED, rows, EVAL), tmp_path / "d")
    ref = data.validation[0]["eval"]["reference"]
    assert isinstance(ref, str) and json.loads(ref) == {"events": [data.validation[0]["eval"]["group"]]}


def test_an_evaluation_column_the_table_lacks_is_named(tmp_path):
    spec = _spec(tmp_path, GROUPED, eval_block="\n  eval: {reference: gold}")
    with pytest.raises(ValueError, match=r"features.eval.*'gold'.*not found"):
        materialise(spec, tmp_path / "d")


def test_features_eval_is_checked_by_validate(tmp_path):
    with pytest.raises(ValueError, match="features.eval needs a csv, parquet or jsonl table"):
        load_spec("""
atelier: 1
name: x
task: text_generation
data: {uri: "hf://org/repo", format: hf, split: {train: "train", validation: "test"}}
features:
  inputs: [{name: prompt, type: text}]
  output: {name: completion, type: text}
  eval: {reference: gold}
model: hf-causal-lm
hp: text_generation/lora-fast@1
export: [safetensors]
""")
    with pytest.raises(ValueError, match="features.eval only applies to task text_generation"):
        load_spec("""
atelier: 1
name: x
task: tabular_classification
data: {uri: "file://x.csv", format: csv, split: {validation_fraction: 0.2}}
features:
  inputs: [{name: x, type: number}]
  output: {name: y, type: category}
  eval: {reference: gold}
model: ludwig_ecd
hp: tabular/ecd@1
export: [onnx]
""")
    with pytest.raises(ValueError, match="features.eval must set at least one"):
        _spec(tmp_path, GROUPED, eval_block="\n  eval: {}")


def test_a_spec_without_the_new_keys_dumps_and_loads_as_before(tmp_path):
    spec = _spec(tmp_path, "{validation_fraction: 0.25}")
    dumped = dump_spec(spec)
    assert "group_by" not in dumped and "eval" not in dumped
    assert "eval" not in spec.features.model_dump() or spec.features.model_dump().get("eval") is None
    data = materialise(spec, tmp_path / "d")
    assert all(set(r) == {"prompt", "completion"} for r in [*data.train, *data.validation])
    assert "group_by" not in data.extra["data_split"] and "groups" not in data.extra["data_split"]
    # and a spec with the keys round-trips them
    grouped = _spec(tmp_path, GROUPED, eval_block=EVAL)
    again = load_spec(dump_spec(grouped))
    assert again.data.split.group_by == "conversation"
    assert again.features.eval.keep == ["turn", "kind"] and again.features.eval.reference == "exp"


def test_a_missing_completion_is_empty_text_only_when_features_eval_is_set(tmp_path):
    rows = _rows()
    rows[-1]["completion"] = None  # a CSV reads an empty cell as missing: the trailing evaluation-only row
    split = '{train: "0:50", validation: "50:"}'
    with_eval = materialise(_spec(tmp_path, split, rows, EVAL, fmt="csv"), tmp_path / "d1")
    assert with_eval.validation[-1]["completion"] == "" and with_eval.validation[0]["completion"] == "reply 16.2"
    without = materialise(_spec(tmp_path, split, rows, "", fmt="csv"), tmp_path / "d2")
    assert without.validation[-1]["completion"] == "nan"  # as before: a row like any other
