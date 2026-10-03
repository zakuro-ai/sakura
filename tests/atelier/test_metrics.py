"""The metrics file is tailed live by the runner, so what matters is that a
reader never sees a half-written point and never loses a whole one."""

import json

import pytest

from sakura.atelier.types import MetricsSink, read_metrics


def test_points_are_readable_as_soon_as_they_are_logged(tmp_path):
    p = tmp_path / "metrics.jsonl"
    with MetricsSink(p, clock=lambda: 100.0) as m:
        m.log("loss", 0.9, step=1, epoch=0.1)
        assert read_metrics(p) == ([{"t": 100.0, "step": 1, "epoch": 0.1, "split": "train",
                                     "name": "loss", "value": 0.9}], 1)
        m.log("accuracy", 0.5, step=10, split="validation")
    points, nxt = read_metrics(p, since=1)
    assert [x["name"] for x in points] == ["accuracy"] and nxt == 2


def test_a_half_written_line_waits_for_the_next_read(tmp_path):
    p = tmp_path / "metrics.jsonl"
    p.write_text(json.dumps({"step": 1}) + "\n" + '{"step": 2, "va')
    points, nxt = read_metrics(p)
    assert points == [{"step": 1}] and nxt == 1


def test_asking_past_the_end_returns_nothing_and_keeps_the_cursor(tmp_path):
    p = tmp_path / "metrics.jsonl"
    p.write_text('{"step": 1}\n')
    assert read_metrics(p, since=5) == ([], 5)
    assert read_metrics(tmp_path / "missing.jsonl", since=3) == ([], 3)


def test_a_nan_is_refused_rather_than_drawn_as_a_hole(tmp_path):
    with MetricsSink(tmp_path / "m.jsonl") as m, pytest.raises(ValueError, match="NaN"):
        m.log("loss", float("nan"), step=3)


def test_split_is_train_or_validation(tmp_path):
    with MetricsSink(tmp_path / "m.jsonl") as m, pytest.raises(ValueError, match="split"):
        m.log("loss", 1.0, step=1, split="test")
