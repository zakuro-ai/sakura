"""The spec's ``limits`` block (SAK-0019): validation, the row cap at materialisation, the step cap and the
time budget at the wrap-up step boundary, and ``limits_applied`` in ``report.json``.

No training stack is needed: the loops are fakes, the datasets are tiny local files. The real runs with
limits are in ``test_smoke_runs.py``."""

from __future__ import annotations

import json
import sys
import time
import types

import pandas as pd
import pytest
import yaml

from sakura import atelier
from sakura.atelier import wrap_up
from sakura.atelier.data import _cap_positions, materialise
from sakura.atelier.registry import resolve, resolved_yaml_text
from sakura.atelier.spec import dump_spec, load_spec

BASE = """
atelier: 1
name: capped
task: tabular_classification
data: {{uri: "{uri}", format: csv, split: {{validation_fraction: 0.25}}}}
features:
  inputs: [{{name: x, type: number}}]
  output: {{name: y, type: category}}
model: ludwig_ecd
hp: tabular/ecd@1
export: [onnx]
seed: 3
"""


def _spec(uri="file:///tmp/none", **limits):
    return load_spec(BASE.format(uri=uri) + (f"limits: {json.dumps(limits)}\n" if limits else ""))


# --------------------------------------------------------------------------- the spec


def test_the_engine_advertises_limits_and_synthetic_data():
    assert {"wrap_up", "limits", "synthetic_data"} <= atelier.FEATURES
    assert {"init_random", "distillation", "dpo", "onnx_int8", "curation"} <= atelier.FEATURES


def test_limits_are_optional_and_every_field_is_too():
    assert _spec().limits is None
    lim = _spec(max_steps=5, max_train_rows=100, max_eval_rows=20, time_budget_s=30).limits
    assert (lim.max_steps, lim.max_train_rows, lim.max_eval_rows, lim.time_budget_s) == (5, 100, 20, 30.0)
    only = _spec(max_steps=1).limits
    assert only.max_steps == 1 and only.max_train_rows is None and only.time_budget_s is None
    assert _spec(time_budget_s=0.5).limits.time_budget_s == 0.5


@pytest.mark.parametrize("bad, match", [
    ({"max_steps": 0}, "max_steps"), ({"max_steps": -3}, "max_steps"), ({"max_steps": 1.5}, "max_steps"),
    ({"max_steps": "10"}, "max_steps"), ({"max_steps": True}, "max_steps"),
    ({"max_train_rows": 0}, "max_train_rows"), ({"max_eval_rows": -1}, "max_eval_rows"),
    ({"time_budget_s": 0}, "time_budget_s"), ({"time_budget_s": -2.0}, "time_budget_s"),
    ({"time_budget_s": "soon"}, "time_budget_s"), ({"time_budget_s": True}, "time_budget_s"),
    ({"max_epochs": 2}, "max_epochs"),
])
def test_a_bad_limit_is_refused(bad, match):
    with pytest.raises(ValueError, match=match):
        _spec(**bad)


def test_limits_must_be_a_mapping():
    with pytest.raises(ValueError, match="limits"):
        load_spec(BASE.format(uri="file:///tmp/none") + "limits: 5\n")


def test_a_spec_without_limits_dumps_exactly_as_before():
    assert "limits" not in dump_spec(_spec())
    assert "limits" not in dump_spec(load_spec(BASE.format(uri="file:///tmp/none") + "limits: {}\n"))


def test_only_the_limits_that_are_set_are_dumped_and_they_round_trip():
    dumped = yaml.safe_load(dump_spec(_spec(max_steps=4, time_budget_s=2.5)))
    assert dumped["limits"] == {"max_steps": 4, "time_budget_s": 2.5}
    assert load_spec(dump_spec(_spec(max_steps=4, time_budget_s=2.5))).limits.max_steps == 4


def test_resolved_yaml_keeps_the_limits_so_a_replay_is_capped_the_same_way():
    resolved = resolve(_spec(max_steps=4, max_train_rows=10))
    text = resolved_yaml_text(resolved, "0" * 64)
    assert yaml.safe_load(text)["limits"] == {"max_steps": 4, "max_train_rows": 10}
    assert load_spec(text).limits == resolved.spec.limits


RISK = """
atelier: 1
name: r
task: risk_simulation
model: scvar-copula
hp: risk_simulation/scvar@1
data: {uri: "file:///tmp/none", format: csv}
export: [json]
"""


def test_a_risk_simulation_takes_a_step_cap_but_no_row_cap():
    assert load_spec(RISK + "limits: {max_steps: 2000, time_budget_s: 60}\n").limits.max_steps == 2000
    for key in ("max_train_rows", "max_eval_rows"):
        with pytest.raises(ValueError, match=f"limits.{key} does not apply to task risk_simulation"):
            load_spec(RISK + f"limits: {{{key}: 10}}\n")


# --------------------------------------------------------------------------- which rows a cap keeps


def test_a_plain_cap_keeps_the_first_rows_and_a_tail_cap_the_last():
    assert _cap_positions(10, 3) == [0, 1, 2]
    assert _cap_positions(10, 3, tail=True) == [7, 8, 9]
    assert _cap_positions(4, 9) == [0, 1, 2, 3]  # nothing to cap


def test_a_class_cap_keeps_every_class_and_about_its_share():
    labels = ["a"] * 90 + ["b"] * 9 + ["c"]
    kept = _cap_positions(100, 10, labels)
    counts = pd.Series([labels[i] for i in kept]).value_counts().to_dict()
    assert len(kept) == 10 and counts == {"a": 7, "b": 2, "c": 1}
    assert kept == sorted(kept)  # table order is kept
    assert kept[:7] == list(range(7))  # the FIRST rows of a class, not a random draw
    # exactly as many rows as classes: one of each
    assert sorted(labels[i] for i in _cap_positions(100, 3, labels)) == ["a", "b", "c"]


def test_a_cap_below_the_number_of_classes_cannot_keep_them_all():
    assert _cap_positions(6, 2, ["a", "a", "b", "b", "c", "c"]) == [0, 1]


def test_a_class_cap_is_exact_for_any_size():
    labels = [f"c{i % 7}" for i in range(500)] + ["rare"] * 3
    for n in (8, 9, 50, 251, 502):
        kept = _cap_positions(len(labels), n, labels)
        assert len(kept) == n == len(set(kept))
        assert {labels[i] for i in kept} == set(labels)


# --------------------------------------------------------------------------- the cap at materialisation


def _class_table(path, n=240):
    # sorted by class: the first rows are all one class, the case a plain head() would get wrong
    pd.DataFrame({"x": range(n), "y": ["a"] * (n // 2) + ["b"] * (n // 4) + ["c"] * (n // 4)}).to_csv(path, index=False)
    return path


def test_rows_are_capped_once_after_the_split_and_reported(tmp_path):
    csv = _class_table(tmp_path / "t.csv")
    full = materialise(_spec(f"file://{csv}"), tmp_path / "full")
    capped = materialise(_spec(f"file://{csv}", max_train_rows=20, max_eval_rows=6), tmp_path / "capped")
    assert (len(full.train), len(full.validation)) == (180, 60)
    assert (len(capped.train), len(capped.validation)) == (20, 6)
    assert set(capped.train["y"]) == {"a", "b", "c"} == set(capped.validation["y"])
    # the capped rows are rows of the uncapped split, on the same side
    assert set(capped.train["x"]) <= set(full.train["x"]) and set(capped.validation["x"]) <= set(full.validation["x"])
    assert capped.extra["limits_applied"] == {
        "max_train_rows": 20, "max_eval_rows": 6,
        "rows_before": {"train": 180, "validation": 60}, "rows_after": {"train": 20, "validation": 6}}
    # data_split describes what the run used
    assert (capped.extra["data_split"]["train_rows"], capped.extra["data_split"]["validation_rows"]) == (20, 6)
    assert "limits_applied" not in full.extra
    # the table is pinned whole either way: a cap does not change the dataset hash
    assert capped.manifest_sha256 == full.manifest_sha256


def test_the_cap_is_deterministic_for_the_seed(tmp_path):
    csv = _class_table(tmp_path / "t.csv")
    a = materialise(_spec(f"file://{csv}", max_train_rows=20), tmp_path / "a")
    b = materialise(_spec(f"file://{csv}", max_train_rows=20), tmp_path / "b")
    assert list(a.train["x"]) == list(b.train["x"]) and list(a.validation["x"]) == list(b.validation["x"])


def test_a_cap_larger_than_the_split_changes_and_reports_nothing(tmp_path):
    csv = _class_table(tmp_path / "t.csv")
    data = materialise(_spec(f"file://{csv}", max_train_rows=10_000, max_eval_rows=10_000), tmp_path / "d")
    assert (len(data.train), len(data.validation)) == (180, 60) and "limits_applied" not in data.extra


def test_only_the_side_that_was_cut_is_named(tmp_path):
    csv = _class_table(tmp_path / "t.csv")
    data = materialise(_spec(f"file://{csv}", max_train_rows=10, max_eval_rows=10_000), tmp_path / "d")
    applied = data.extra["limits_applied"]
    assert applied["max_train_rows"] == 10 and "max_eval_rows" not in applied
    assert applied["rows_after"] == {"train": 10, "validation": 60}


def test_a_forecasting_cap_keeps_one_contiguous_stretch_in_time_order(tmp_path):
    csv = tmp_path / "s.csv"
    pd.DataFrame({"v": range(100)}).to_csv(csv, index=False)
    spec = load_spec(f"""
atelier: 1
name: ts
task: time_series_forecasting
data: {{uri: "file://{csv}", format: csv, split: {{validation_fraction: 0.2}}}}
features:
  inputs: [{{name: v, type: timeseries}}]
  output: {{name: v, type: timeseries}}
model: ludwig_forecaster
hp: time_series_forecasting/ecd@1
export: [onnx]
limits: {{max_train_rows: 30, max_eval_rows: 10}}
""")
    data = materialise(spec, tmp_path / "d")
    # train is the 30 rows just before the holdout, validation the 10 just after: no gap, no future in train
    assert list(data.train["v"]) == list(range(50, 80))
    assert list(data.validation["v"]) == list(range(80, 90))


def test_an_image_folder_cap_keeps_every_class_and_pins_only_the_kept_files(tmp_path):
    from sakura.atelier.synthetic import generate

    root = generate("synthetic://image?rows=40&classes=4&size=8", tmp_path, "image_folder")
    text = f"""
atelier: 1
name: img
task: image_classification
data: {{uri: "file://{root}", format: image_folder, split: {{validation_fraction: 0.25}}}}
features:
  inputs: [{{name: img, type: image}}]
  output: {{name: label, type: category}}
model: resnet18
hp: image_classification/fast@1
export: [onnx]
"""
    full = materialise(load_spec(text), tmp_path / "full")
    capped = materialise(load_spec(text + "limits: {max_train_rows: 8, max_eval_rows: 4}\n"), tmp_path / "capped")
    assert (len(capped.train), len(capped.validation)) == (8, 4)
    assert set(capped.train["label"]) == set(capped.validation["label"]) == {"c0", "c1", "c2", "c3"}
    assert len(json.loads((tmp_path / "capped" / "manifest.json").read_text())) == 12  # 8 + 4 files, not 40
    assert capped.manifest_sha256 != full.manifest_sha256
    assert capped.extra["limits_applied"]["rows_before"] == {"train": len(full.train), "validation": len(full.validation)}


def test_text_generation_records_are_capped(tmp_path):
    jsonl = tmp_path / "d.jsonl"
    jsonl.write_text("".join(json.dumps({"prompt": f"p{i}", "completion": f"c{i}"}) + "\n" for i in range(40)))
    spec = load_spec(f"""
atelier: 1
name: gen
task: text_generation
data: {{uri: "file://{jsonl}", format: jsonl, split: {{train: "0:30", validation: "30:40"}}}}
features:
  inputs: [{{name: prompt, type: text}}]
  output: {{name: completion, type: text}}
model: lfm2-350m
hp: text_generation/lora-fast@1
export: [safetensors]
limits: {{max_train_rows: 5, max_eval_rows: 2}}
""")
    data = materialise(spec, tmp_path / "data")
    assert [r["prompt"] for r in data.train] == [f"p{i}" for i in range(5)]
    assert [r["prompt"] for r in data.validation] == ["p30", "p31"]
    assert data.extra["limits_applied"]["rows_before"] == {"train": 30, "validation": 10}


def test_an_asr_manifest_cap_pins_only_the_kept_clips(tmp_path):
    from sakura.atelier.synthetic import generate

    root = generate("synthetic://asr?rows=20", tmp_path, "asr_manifest")
    spec = load_spec(f"""
atelier: 1
name: asr
task: speech_recognition
data: {{uri: "file://{root}", format: asr_manifest, split: {{validation_fraction: 0.25}}}}
features:
  inputs: [{{name: audio, type: audio}}]
  output: {{name: transcript, type: text}}
model: deepspeech2
hp: speech_recognition/ctc-fast@1
export: [safetensors]
limits: {{max_train_rows: 4, max_eval_rows: 2}}
""")
    data = materialise(spec, tmp_path / "data")
    assert (len(data.train), len(data.validation)) == (4, 2)
    assert len(json.loads((tmp_path / "data" / "manifest.json").read_text())) == 1 + 6  # the manifest + 6 clips
    assert data.extra["limits_applied"]["rows_before"] == {"train": 15, "validation": 5}


def test_a_decision_cap_keeps_the_calibration_split_whole(tmp_path, monkeypatch):
    from tests.atelier.decision_fakes import write_dataset

    monkeypatch.chdir(tmp_path)
    write_dataset(tmp_path / "data.jsonl")
    text = """
atelier: 1
name: ticket-triage
task: decision
data: {uri: "data.jsonl", format: jsonl}
model: hf-decision-lm
hp: decision/lora-forge@1
overrides: {base_model: Qwen/Qwen3.5-2B-Base}
export: [safetensors]
seed: 7
"""
    full = materialise(load_spec(text), tmp_path / "full")
    capped = materialise(load_spec(text + "limits: {max_train_rows: 10, max_eval_rows: 4}\n"), tmp_path / "capped")
    assert (len(capped.train), len(capped.validation)) == (10, 4)
    assert capped.train == full.train[:10] and capped.validation == full.validation[:4]
    assert capped.extra["calibration"] == full.extra["calibration"]  # the temperatures are fitted on all of it
    assert capped.extra["limits_applied"] == {
        "max_train_rows": 10, "max_eval_rows": 4,
        "rows_before": {"train": 72, "validation": 9}, "rows_after": {"train": 10, "validation": 4}}


def test_a_yolo_cap_points_data_yaml_at_a_list_of_the_first_images(tmp_path):
    src = tmp_path / "src"
    for split, n in (("train", 6), ("val", 4)):
        (src / "images" / split).mkdir(parents=True)
        (src / "labels" / split).mkdir(parents=True)
        for i in range(n):
            (src / "images" / split / f"{i}.png").write_bytes(b"png")
            (src / "labels" / split / f"{i}.txt").write_text("0 0.5 0.5 0.2 0.2\n")
    (src / "data.yaml").write_text(yaml.safe_dump({"train": "images/train", "val": "images/val", "names": ["thing"]}))
    spec = load_spec(f"""
atelier: 1
name: det
task: object_detection
data: {{uri: "file://{src}", format: yolo, split: {{train: train, validation: val}}}}
features:
  inputs: [{{name: image, type: image}}]
  output: {{name: boxes, type: boxes}}
model: rtdetr_r18
hp: object_detection/fast@1
export: [onnx]
limits: {{max_train_rows: 2}}
""")
    data = materialise(spec, tmp_path / "data")
    doc = yaml.safe_load(open(data.extra["yolo_data_yaml"]).read())
    listed = open(doc["train"]).read().split()
    assert [p.rsplit("/", 1)[-1] for p in listed] == ["0.png", "1.png"]
    assert doc["val"] == "images/val"  # the side without a cap is untouched
    assert data.extra["limits_applied"] == {"max_train_rows": 2, "rows_before": {"train": 6}, "rows_after": {"train": 2}}


# --------------------------------------------------------------------------- hf://


HF = """
atelier: 1
name: hf
task: tabular_classification
data: {{uri: "{uri}", format: hf, split: {split}}}
features:
  inputs: [{{name: x, type: number}}]
  output: {{name: y, type: category}}
model: ludwig_ecd
hp: tabular/ecd@1
export: [onnx]
{limits}
"""


def _fake_datasets(monkeypatch, calls):
    """A stand-in `datasets` module: records every load_dataset call and serves a 100-row table."""

    class _Table:
        features: dict = {}
        column_names = ["x", "y"]

        def __init__(self, rows):
            self.rows = rows

        def __len__(self):
            return len(self.rows)

        def __iter__(self):
            return iter(self.rows)

        def __getitem__(self, key):
            return [r[key] for r in self.rows]

        def select(self, positions):
            return _Table([self.rows[i] for i in positions])

        def to_pandas(self):
            return pd.DataFrame(self.rows)

    def load_dataset(repo, **kwargs):
        calls.append((repo, kwargs))
        return _Table([{"x": i, "y": "ab"[i % 2]} for i in range(100)])

    fake = types.ModuleType("datasets")
    fake.load_dataset = load_dataset
    fake.Image = fake.Audio = fake.ClassLabel = type("_Feature", (), {})
    fake.Dataset = types.SimpleNamespace(from_list=lambda rows, features=None: _Table(rows))
    monkeypatch.setitem(sys.modules, "datasets", fake)


def test_an_hf_uri_without_a_revision_loads_exactly_as_before(tmp_path, monkeypatch):
    calls: list = []
    _fake_datasets(monkeypatch, calls)
    spec = load_spec(HF.format(uri="hf://owner/name", split='{train: "train", validation: "test"}', limits=""))
    data = materialise(spec, tmp_path / "d")
    assert calls == [("owner/name", {"split": "train"}), ("owner/name", {"split": "test"})]
    assert len(data.train) == 100 and "limits_applied" not in data.extra


def test_an_hf_uri_revision_is_passed_to_load_dataset(tmp_path, monkeypatch):
    calls: list = []
    _fake_datasets(monkeypatch, calls)
    spec = load_spec(HF.format(uri="hf://owner/name@abc123", split='{train: "train", validation: "test"}', limits=""))
    materialise(spec, tmp_path / "d")
    assert calls == [("owner/name", {"split": "train", "revision": "abc123"}),
                     ("owner/name", {"split": "test", "revision": "abc123"})]


def test_a_capped_hf_source_is_streamed_and_only_the_capped_side(tmp_path, monkeypatch):
    calls: list = []
    _fake_datasets(monkeypatch, calls)
    spec = load_spec(HF.format(uri="hf://owner/name@v2", split='{train: "train[:60]", validation: "test"}',
                               limits="limits: {max_train_rows: 7}"))
    data = materialise(spec, tmp_path / "d")
    assert calls == [("owner/name", {"split": "train", "streaming": True, "revision": "v2"}),
                     ("owner/name", {"split": "test", "revision": "v2"})]
    assert list(data.train["x"]) == list(range(7)) and len(data.validation) == 100
    # the stream was cut at the cap: how many rows the split has is not known, and that is what is reported
    assert data.extra["limits_applied"] == {
        "max_train_rows": 7, "rows_before": {"train": None, "validation": 100},
        "rows_after": {"train": 7, "validation": 100}}


def test_a_capped_hf_holdout_streams_just_enough_rows_for_both_sides(tmp_path, monkeypatch):
    calls: list = []
    _fake_datasets(monkeypatch, calls)

    def train_test_split(self, test_size, seed, stratify_by_column):
        n = int(round(len(self) * test_size))
        return {"train": self.select(range(len(self) - n)), "test": self.select(range(len(self) - n, len(self)))}

    table_cls = type(sys.modules["datasets"].load_dataset("x"))
    table_cls.train_test_split = train_test_split
    calls.clear()
    spec = load_spec(HF.format(uri="hf://owner/name", split="{validation_fraction: 0.2}",
                               limits="limits: {max_train_rows: 16, max_eval_rows: 4}"))
    data = materialise(spec, tmp_path / "d")
    assert calls == [("owner/name", {"split": "train", "streaming": True})]
    assert (len(data.train), len(data.validation)) == (16, 4)


def test_a_slice_a_stream_cannot_take_is_refused_by_name(tmp_path, monkeypatch):
    _fake_datasets(monkeypatch, [])
    spec = load_spec(HF.format(uri="hf://owner/name", split='{train: "train[:10%]", validation: "test"}',
                               limits="limits: {max_train_rows: 7}"))
    with pytest.raises(ValueError, match=r"train\[:10%\].*cannot be streamed"):
        materialise(spec, tmp_path / "d")


def test_a_real_local_dataset_is_streamed_and_capped(tmp_path):
    pytest.importorskip("datasets")
    ds = tmp_path / "ds"
    ds.mkdir()
    pd.DataFrame({"x": range(300), "y": ["a", "b", "c"] * 100}).to_csv(ds / "train.csv", index=False)
    pd.DataFrame({"x": range(300, 360), "y": ["a", "b", "c"] * 20}).to_csv(ds / "test.csv", index=False)
    spec = load_spec(HF.format(uri=f"hf://{ds}", split='{train: "train", validation: "test[10:40]"}',
                               limits="limits: {max_train_rows: 9, max_eval_rows: 6}"))
    data = materialise(spec, tmp_path / "d")
    assert list(data.train["x"]) == list(range(9))
    assert list(data.validation["x"]) == list(range(310, 316))
    assert set(data.train["y"]) == {"a", "b", "c"}
    assert data.extra["limits_applied"]["rows_before"] == {"train": None, "validation": None}


# --------------------------------------------------------------------------- the step cap and the time budget


def test_a_watch_with_max_steps_stops_at_that_step_and_says_why(tmp_path, monkeypatch):
    monkeypatch.delenv(wrap_up.ENV_VAR, raising=False)
    watch = wrap_up.WrapUpWatch(tmp_path, max_steps=3)
    assert [watch.stop(s / 10, s) for s in range(3)] == [False, False, False]
    assert watch.last_step == 2
    assert watch.stop(0.3, 3) and watch.stopped_early == {"reason": "max_steps", "epoch": 0.3, "step": 3}
    assert not wrap_up.WrapUpWatch(tmp_path).stop(9.0, 10_000)  # no cap, no stop
    assert not wrap_up.WrapUpWatch(tmp_path, max_steps=3).stop(None, None)  # an unknown position is not a cap


def test_a_wrap_up_request_wins_over_the_step_cap(tmp_path, monkeypatch):
    monkeypatch.delenv(wrap_up.ENV_VAR, raising=False)
    (tmp_path / "wrap-up-requested").write_text("")
    watch = wrap_up.WrapUpWatch(tmp_path, max_steps=3)
    assert watch.stop(0.5, 5) and watch.stopped_early["reason"] == "wrap_up"


def test_max_steps_of_reads_a_resolved_spec_and_tolerates_a_fake():
    assert wrap_up.max_steps_of(resolve(_spec(max_steps=9))) == 9
    assert wrap_up.max_steps_of(resolve(_spec())) is None
    assert wrap_up.max_steps_of(types.SimpleNamespace(spec=types.SimpleNamespace())) is None


def test_a_time_budget_raises_the_wrap_up_signal_and_cleans_up(tmp_path, monkeypatch):
    monkeypatch.delenv(wrap_up.ENV_VAR, raising=False)
    watch = wrap_up.WrapUpWatch(tmp_path)
    with wrap_up.TimeBudget(tmp_path, 0.05) as budget:
        assert not watch.stop(0.0, 0)
        deadline = time.time() + 5
        while not watch.stop(0.1, 1) and time.time() < deadline:
            time.sleep(0.01)
        assert budget.fired and watch.stopped_early == {"reason": "wrap_up", "epoch": 0.1, "step": 1}
        assert wrap_up.read_reason(tmp_path) == "time_budget"
    assert not (tmp_path / "wrap-up-requested").exists()  # a later run here does not start stopped


def test_a_time_budget_that_does_not_run_out_leaves_no_trace(tmp_path, monkeypatch):
    monkeypatch.delenv(wrap_up.ENV_VAR, raising=False)
    with wrap_up.TimeBudget(tmp_path, 60) as budget:
        pass
    assert not budget.fired and not (tmp_path / "wrap-up-requested").exists()
    with wrap_up.TimeBudget(tmp_path, None) as none:
        pass
    assert not none.fired


def test_a_time_budget_leaves_somebody_elses_request_alone(tmp_path, monkeypatch):
    monkeypatch.delenv(wrap_up.ENV_VAR, raising=False)
    (tmp_path / "wrap-up-requested").write_text('{"reason": "operator"}')
    with wrap_up.TimeBudget(tmp_path, 0.01) as budget:
        time.sleep(0.2)
    assert not budget.fired and wrap_up.read_reason(tmp_path) == "operator"


# --------------------------------------------------------------------------- report.json


GEN = """
atelier: 1
name: capped-run
task: text_generation
data: {{uri: "file://{data}", format: jsonl, split: {{validation_fraction: 0.25}}}}
features:
  inputs: [{{name: prompt, type: text}}]
  output: {{name: completion, type: text}}
model: lfm2-350m
hp: text_generation/lora-fast@1
export: [safetensors]
seed: 1
{limits}
"""


def _run(tmp_path, monkeypatch, limits="", name="out"):
    import sakura.atelier.__main__ as cli

    monkeypatch.delenv(wrap_up.ENV_VAR, raising=False)
    monkeypatch.setenv("SAKURA_ATELIER_BACKEND_MODULE_PEFT", "tests.atelier.fake_limits_backend")
    monkeypatch.setenv("SAKURA_ATELIER_NOTEBOOK", "0")
    data = tmp_path / "data.jsonl"
    data.write_text("".join(json.dumps({"prompt": f"p{i}", "completion": f"c{i}"}) + "\n" for i in range(40)))
    spec = tmp_path / f"{name}.yaml"
    spec.write_text(GEN.format(data=data, limits=limits))
    out = tmp_path / name
    rc = cli.main(["run", str(spec), "--out", str(out)])
    return rc, json.loads((out / "report.json").read_text()), out


def test_a_run_without_limits_has_no_limits_applied(tmp_path, monkeypatch):
    rc, report, out = _run(tmp_path, monkeypatch)
    assert rc == 0 and report["status"] == "done"
    assert "limits_applied" not in report and "stopped_early" not in report
    assert "limits" not in yaml.safe_load((out / "resolved.yaml").read_text())


def test_limits_that_change_nothing_are_not_reported(tmp_path, monkeypatch):
    rc, report, out = _run(tmp_path, monkeypatch,
                           "limits: {max_steps: 1000, max_train_rows: 1000, max_eval_rows: 1000, time_budget_s: 600}")
    assert rc == 0 and "limits_applied" not in report and "stopped_early" not in report
    # ... but the block is still part of the replayable spec
    assert yaml.safe_load((out / "resolved.yaml").read_text())["limits"]["max_steps"] == 1000


def test_a_capped_run_reports_what_the_limits_changed(tmp_path, monkeypatch):
    from tests.atelier.fake_limits_backend import StepBackend

    rc, report, out = _run(tmp_path, monkeypatch, "limits: {max_steps: 4, max_train_rows: 12, max_eval_rows: 3}")
    assert rc == 0 and report["status"] == "done" and report["error"] is None
    assert report["limits_applied"] == {
        "max_steps": 4, "max_train_rows": 12, "max_eval_rows": 3,
        "rows_before": {"train": 30, "validation": 10}, "rows_after": {"train": 12, "validation": 3}}
    assert report["stopped_early"] == {"reason": "max_steps", "epoch": 1.0, "step": 4}
    assert StepBackend.seen == {"train_rows": 12, "validation_rows": 3, "steps": 4}
    assert report["artifacts"][0]["format"] == "safetensors"  # export and evaluation still ran
    resolved = yaml.safe_load((out / "resolved.yaml").read_text())
    assert resolved["limits"] == {"max_steps": 4, "max_train_rows": 12, "max_eval_rows": 3}


def test_replaying_resolved_yaml_gives_the_same_capped_run(tmp_path, monkeypatch):
    import sakura.atelier.__main__ as cli

    _, first, out = _run(tmp_path, monkeypatch, "limits: {max_steps: 4, max_train_rows: 12}")
    again = tmp_path / "again"
    assert cli.main(["run", str(out / "resolved.yaml"), "--out", str(again)]) == 0
    second = json.loads((again / "report.json").read_text())
    assert second["limits_applied"] == first["limits_applied"]
    assert second["spec_sha256"] == first["spec_sha256"] and second["data_sha256"] == first["data_sha256"]
    assert second["stopped_early"] == first["stopped_early"]


def test_a_time_budget_stops_the_loop_through_the_wrap_up_signal(tmp_path, monkeypatch):
    monkeypatch.setenv("FAKE_LIMITS_STEP_SECONDS", "0.02")
    rc, report, out = _run(tmp_path, monkeypatch, "limits: {time_budget_s: 0.1}")
    assert rc == 0 and report["status"] == "done"
    assert report["limits_applied"] == {"time_budget_s": 0.1}
    assert report["stopped_early"]["reason"] == "wrap_up" and 0 < report["stopped_early"]["step"] < 200
    assert not (out / "wrap-up-requested").exists()


def test_a_backend_without_a_loop_reports_its_own_step_cap(tmp_path, monkeypatch):
    monkeypatch.setenv("FAKE_LIMITS_NO_LOOP", "1")
    rc, report, _ = _run(tmp_path, monkeypatch, "limits: {max_steps: 4}")
    assert rc == 0 and report["limits_applied"] == {"max_steps": 4} and "stopped_early" not in report


# --------------------------------------------------------------------------- backends without a real stack


def test_ludwig_gets_train_steps_only_when_that_shortens_the_run():
    from sakura.atelier.backends.ludwig import apply_max_steps, compile_config

    def config():
        return compile_config(resolve(_spec()))  # tabular/ecd@1: 20 epochs, batch 128

    c = config()
    assert apply_max_steps(c, 5, 1000) and c["trainer"]["train_steps"] == 5  # 20 * 8 = 160 steps planned
    c = config()
    assert not apply_max_steps(c, 500, 1000) and "train_steps" not in c["trainer"]  # would LENGTHEN the run
    c = config()
    assert not apply_max_steps(c, None, 1000) and "train_steps" not in c["trainer"]
    c = config()
    c["trainer"]["batch_size"] = "auto"  # steps per epoch unknown: left to the watch
    assert not apply_max_steps(c, 5, 1000) and "train_steps" not in c["trainer"]
    c = config()
    c["trainer"]["train_steps"] = 50  # a spec's own backend_config: only ever lowered
    assert apply_max_steps(c, 5, 1000) and c["trainer"]["train_steps"] == 5
    assert not apply_max_steps(c, 80, 1000) and c["trainer"]["train_steps"] == 5


def test_ludwig_forecasting_is_left_to_the_watch():
    from sakura.atelier.backends.ludwig import apply_max_steps

    c = {"output_features": [{"name": "v_out", "type": "timeseries"}], "trainer": {"epochs": 30, "batch_size": 64}}
    assert not apply_max_steps(c, 2, 10_000) and "train_steps" not in c["trainer"]


def test_ludwig_adapter_stops_the_trainer_at_the_step_cap(tmp_path, monkeypatch):
    """The path for a run `train_steps` could not be set for: the watch carries the cap, and the adapter
    stops Ludwig at that step exactly as it does for a wrap-up request."""
    from sakura.adapters.ludwig import LudwigAdapter
    from sakura.atelier.types import MetricsSink
    from tests.atelier.test_wrap_up import _FakeLudwigTrainer

    monkeypatch.delenv(wrap_up.ENV_VAR, raising=False)
    watch = wrap_up.WrapUpWatch(tmp_path, max_steps=6)
    adapter = LudwigAdapter(MetricsSink(tmp_path / "m.jsonl"), output_feature="label", wrap_up=watch)
    trainer = _FakeLudwigTrainer(adapter, tmp_path)  # 12 steps, an evaluation round every 4
    trainer.run(None)
    assert trainer.steps == 6 and trainer.total_steps == 6
    assert trainer.best_written_at == [4]  # the round before the cap is the checkpoint that is kept
    assert watch.stopped_early == {"reason": "max_steps", "epoch": 1.25, "step": 6}


def test_ludwig_reports_max_steps_as_the_reason_a_run_stopped():
    from collections import namedtuple

    from sakura.atelier.backends.ludwig import early_stop_summary

    metric = namedtuple("Metric", ["epoch", "step", "value"])
    stats = {"validation": {"label": {"loss": [metric(0, 7, 2.0), metric(1, 14, 1.5)]}}}
    summary = early_stop_summary(stats, field="label", metric="loss", patience=5, max_epochs=30, wrap_up=True)
    assert summary["reason"] == "wrap_up"  # what the backend then rewrites to the watch's own reason


def test_deepspeech_loader_ends_the_epoch_at_the_step_cap(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from sakura.atelier.backends.deepspeech import _WrapUpLoader

    monkeypatch.delenv(wrap_up.ENV_VAR, raising=False)
    trainer = SimpleNamespace(_stop=False)
    watch = wrap_up.WrapUpWatch(tmp_path, max_steps=8)
    loader = _WrapUpLoader(list(range(6)), watch, [trainer])
    assert list(loader) == [0, 1, 2, 3, 4, 5] and not trainer._stop  # epoch 0: steps 0..5
    assert list(loader) == [0, 1] and trainer._stop  # epoch 1: steps 6 and 7, then the cap
    assert watch.stopped_early == {"reason": "max_steps", "epoch": 1 + 2 / 6, "step": 8}
    assert list(loader) == []


# --------------------------------------------------------------------------- torch loops


def test_open_clip_stops_at_the_step_cap_and_still_scores(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    from types import SimpleNamespace

    import numpy as np

    from sakura.atelier.backends import open_clip as oc
    from sakura.atelier.types import MetricsSink, RunDir

    monkeypatch.delenv(wrap_up.ENV_VAR, raising=False)
    out = RunDir(tmp_path / "run").ensure()
    spec = resolve(load_spec("""
atelier: 1
name: age
task: image_regression
data: {uri: "file:///x", format: csv, split: {validation_fraction: 0.2}}
features:
  inputs: [{name: img, type: image}]
  output: {name: age, type: number}
model: siglip2_b16_256
hp: image_regression/siglip2@1
export: [onnx]
overrides: {epochs: 3, batch_size: 4}
limits: {max_steps: 7}
"""))

    class Net(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.visual = torch.nn.Linear(2, 2)
            self.head = torch.nn.Linear(2, 1)

        def forward(self, x):
            return self.head(self.visual(x))

    class DL:
        def __len__(self):
            return 5

        def __iter__(self):
            return iter([(torch.zeros(4, 2), torch.zeros(4, 1))] * 5)

    monkeypatch.setattr(oc, "_build", lambda model: (Net(), {"size": 2}))
    monkeypatch.setattr(oc, "_loader", lambda *a, **k: DL())
    monkeypatch.setattr(oc, "_predict_torch", lambda *a, **k: np.zeros(4))
    data = SimpleNamespace(
        train=pd.DataFrame({"img": ["a"] * 4, "age": [1.0, 2.0, 3.0, 4.0]}),
        validation=pd.DataFrame({"img": ["a"] * 4, "age": [1.0, 2.0, 3.0, 4.0]}))
    with MetricsSink(out.metrics_path) as sink:
        result = oc.BACKEND.train(spec, data, out, sink, None)

    assert result.stopped_early == {"reason": "max_steps", "epoch": pytest.approx(1 + 2 / 5), "step": 7}
    points = [json.loads(line) for line in out.metrics_path.read_text().splitlines()]
    assert sum(p["name"] == "loss" for p in points) == 7  # 15 steps were planned
    assert (out.checkpoint / "model.pt").is_file() and "mae" in result.metrics  # still delivers a scored model


def test_decision_loop_stops_at_the_step_cap(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    from sakura.atelier.backends import decision_lm as D
    from sakura.atelier.decision import identity_order
    from sakura.atelier.types import MetricsSink
    from tests.atelier.decision_fakes import CharTokenizer, FakeLM
    from tests.atelier.test_decision_lm import cfg, easy_instances

    monkeypatch.delenv(wrap_up.ENV_VAR, raising=False)
    tok = CharTokenizer()
    model = FakeLM(seed=7)
    opt = torch.optim.Adam(model.parameters(), lr=0.05)
    with MetricsSink(tmp_path / "m.jsonl") as sink:
        summary = D.train_loop(model, tok, D.make_readout(tok), easy_instances(24), easy_instances(8),
                               cfg=cfg(epochs=6, batch_size=4), device="cpu", optimizer=opt, metrics=sink,
                               order_fn=lambda rng, k: identity_order(k),
                               watch=wrap_up.WrapUpWatch(tmp_path, max_steps=8))
    # 6 steps an epoch: one whole epoch (scored), then two steps of the next
    assert summary["stopped_early"] == {"reason": "max_steps", "epoch": pytest.approx(1 + 2 * 4 / 24), "step": 8}
    assert summary["log_loss"] is not None
