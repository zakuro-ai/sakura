"""Decision data: validate, strip, dedupe, drop the unanswered, split BY STATE.

All pure Python: no model, no torch."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from sakura.atelier.decision import (
    MAX_OPTIONS,
    DecisionDataError,
    instances,
    load_rows,
    prepare_rows,
)

TOPIC = {"type": "choice", "instructions": "What is this about?", "options": ["billing", "bug", "other"]}
URGENT = {"type": "noul", "statement": "The customer needs a reply today."}
TONE = {"type": "score", "instructions": "How upset is the customer?", "levels": ["calm", "annoyed", "angry"]}


def row(i, answers=None, questions=None, state=None):
    return {"state": state if state is not None else f"ticket number {i}",
            "questions": questions or {"topic": TOPIC, "urgent": URGENT, "tone": TONE},
            "answers": answers or {"topic": "billing", "urgent": True, "tone": "annoyed"}}


def numbered(rows):
    return list(enumerate(rows, start=1))


def many(n, **kw):
    return [row(i, **kw) for i in range(n)]


# ------------------------------------------------------------------- split

def test_no_state_appears_in_two_splits():
    # every state has three rows, one per question: a split by question would put one state everywhere
    rows = []
    for s in range(60):
        for qid, spec, ans in (("topic", TOPIC, "bug"), ("urgent", URGENT, False), ("tone", TONE, "calm")):
            rows.append(row(0, state=f"state {s}", questions={qid: spec}, answers={qid: ans}))
    p = prepare_rows(numbered(rows), seed=1)
    states = {name: {r["state"] for r in getattr(p, name)} for name in ("train", "calibration", "eval")}
    assert not (states["train"] & states["calibration"] or states["train"] & states["eval"]
                or states["calibration"] & states["eval"])
    assert sum(len(v) for v in states.values()) == 60
    for name in ("train", "calibration", "eval"):  # all three of a state's rows stay together
        rows_of = [r for r in getattr(p, name)]
        assert len(rows_of) == 3 * len(states[name])


def test_the_split_is_80_10_10_by_state():
    p = prepare_rows(numbered(many(200)), seed=7)
    assert (len(p.train), len(p.calibration), len(p.eval)) == (160, 20, 20)
    assert p.stats["split_states"] == {"train": 160, "calibration": 20, "eval": 20}


def test_the_split_is_seeded_and_does_not_depend_on_row_order():
    rows = many(120)
    a = prepare_rows(numbered(rows), seed=3)
    b = prepare_rows(numbered(rows), seed=3)
    c = prepare_rows(numbered(list(reversed(rows))), seed=3)
    d = prepare_rows(numbered(rows), seed=4)
    assert a.eval == b.eval == c.eval
    assert a.eval != d.eval


def test_a_state_that_is_a_json_object_is_one_state_whatever_its_key_order():
    rows = many(60)
    rows.append(row(0, state={"b": 1, "a": 2}, questions={"topic": TOPIC}, answers={"topic": "bug"}))
    rows.append(row(0, state={"a": 2, "b": 1}, questions={"topic": TOPIC}, answers={"topic": "other"}))
    p = prepare_rows(numbered(rows), seed=0)
    homes = [name for name in ("train", "calibration", "eval") for r in getattr(p, name)
             if isinstance(r["state"], dict)]
    assert len(homes) == 2 and len(set(homes)) == 1


def test_too_few_states_to_fill_three_splits_is_refused_by_name():
    rows = [row(0, state="same state", questions={"q%d" % i: TOPIC}, answers={"q%d" % i: "bug"}) for i in range(60)]
    with pytest.raises(DecisionDataError, match="too few distinct states"):
        prepare_rows(numbered(rows), seed=0)


# ----------------------------------------------------------- usable rows

def test_fewer_than_fifty_usable_rows_is_a_named_customer_data_failure():
    with pytest.raises(DecisionDataError, match=r"too few usable rows \(49; at least 50 needed\)"):
        prepare_rows(numbered(many(49)), seed=0)


def test_fifty_to_299_rows_train_with_a_warning_and_300_do_not():
    p = prepare_rows(numbered(many(50)), seed=0)
    assert any("only 50 usable rows" in w and "300" in w for w in p.warnings)
    assert not any("usable rows" in w for w in prepare_rows(numbered(many(300)), seed=0).warnings)
    assert any("299" in w for w in prepare_rows(numbered(many(299)), seed=0).warnings)


def test_a_small_eval_split_is_warned_about():
    p = prepare_rows(numbered(many(60)), seed=0)  # 6 eval states x 3 answers = 18 answers < 30
    assert any("eval split has only" in w for w in p.warnings)


def test_exact_duplicates_are_dropped_before_counting():
    rows = many(49)
    rows += [dict(r) for r in rows[:10]]  # 10 copies
    with pytest.raises(DecisionDataError, match=r"too few usable rows \(49;"):
        prepare_rows(numbered(rows), seed=0)
    p = prepare_rows(numbered(many(60) + many(60)), seed=0)
    assert p.stats["usable_rows"] == 60 and p.stats["duplicates"] == 60


def test_the_same_state_with_other_answers_is_not_a_duplicate():
    rows = many(60)
    rows.append(row(0, state="ticket number 3", answers={"topic": "bug", "urgent": False, "tone": "calm"}))
    p = prepare_rows(numbered(rows), seed=0)
    assert p.stats["usable_rows"] == 61 and p.stats["duplicates"] == 0


def test_strip_makes_whitespace_variants_one_row():
    rows = many(60)
    twin = row(0, state="  ticket number 5\n")
    twin["questions"] = {"topic": {**TOPIC, "instructions": " What is this about? ", "options": [" billing", "bug ", "other"]},
                         "urgent": URGENT, "tone": TONE}
    p = prepare_rows(numbered(rows + [twin]), seed=0)
    assert p.stats["usable_rows"] == 60 and p.stats["duplicates"] == 1
    kept = next(r for part in (p.train, p.calibration, p.eval) for r in part if r["state"] == "ticket number 5")
    assert kept["questions"]["topic"]["options"] == ["billing", "bug", "other"]


def test_rows_with_no_answered_question_are_dropped_not_failed():
    rows = many(60)
    rows.append(row(0, state="nothing answered", answers={"topic": None}))
    empty = row(0, state="empty answers")
    empty["answers"] = {}
    rows.append(empty)
    p = prepare_rows(numbered(rows), seed=0)
    assert p.stats["usable_rows"] == 60 and p.stats["dropped_unanswered"] == 2
    assert any("no answered question" in w for w in p.warnings)


def test_a_question_left_unanswered_in_a_row_is_not_an_instance():
    r = row(0, answers={"topic": "bug"})
    assert [(i.qid, i.kind) for i in instances(r)] == [("topic", "choice")]


# ------------------------------------------------------------- instances

def test_instances_carry_the_target_in_the_given_label_order():
    r = row(0, answers={"topic": "bug", "urgent": False, "tone": "angry"})
    got = {i.qid: i for i in instances(r)}
    assert got["topic"].target == [0.0, 1.0, 0.0] and got["topic"].labels == ["billing", "bug", "other"]
    assert got["urgent"].target == [0.0, 1.0] and got["urgent"].labels == ["true", "false"]  # [true, false]
    assert got["tone"].target == [0.0, 0.0, 1.0]


def test_soft_labels_are_distributions_over_the_labels():
    r = row(0, answers={"topic": {"billing": 0.8, "other": 0.2}, "urgent": 0.9})
    got = {i.qid: i for i in instances(r)}
    assert got["topic"].target == [0.8, 0.0, 0.2]
    assert got["urgent"].target == pytest.approx([0.9, 0.1])


def test_a_soft_label_that_sums_to_slightly_off_one_is_renormalised():
    got = instances(row(0, answers={"topic": {"billing": 0.8, "bug": 0.205}}))[0]
    assert sum(got.target) == pytest.approx(1.0)


def test_a_noul_true_puts_all_the_mass_on_true():
    assert instances(row(0, answers={"urgent": True}))[0].target == [1.0, 0.0]


# ------------------------------------------------------------ validation

def bad(**fields):
    r = row(0)
    r.update(fields)
    return numbered(many(60) + [r])


@pytest.mark.parametrize("fields,match", [
    ({"questions": {"topic": {**TOPIC, "options": ["only one"]}}, "answers": {"topic": "only one"}}, r"1 options; 2 to 26"),
    ({"questions": {"topic": {**TOPIC, "options": [f"o{i}" for i in range(27)]}}, "answers": {"topic": "o1"}},
     r"27 options; 2 to 26"),
    ({"questions": {"tone": {**TONE, "levels": ["x"]}}, "answers": {"tone": "x"}}, r"1 levels; 2 to 26"),
    ({"questions": {"topic": {**TOPIC, "options": ["a", "a"]}}, "answers": {"topic": "a"}}, "not distinct"),
    ({"questions": {"topic": TOPIC}, "answers": {"topic": "refund"}}, r"answer 'refund' to 'topic' is not one of its 3 labels"),
    ({"questions": {"topic": TOPIC}, "answers": {"topic": {"refund": 1.0}}}, "names 'refund'"),
    ({"questions": {"topic": TOPIC}, "answers": {"topic": {"bug": 0.4, "other": 0.4}}}, "sums to 0.8, not 1"),
    ({"questions": {"topic": TOPIC}, "answers": {"topic": {"bug": -0.5, "other": 1.5}}}, "non-negative"),
    ({"questions": {"x": {"type": "rank"}}, "answers": {"x": "a"}}, "type 'rank'"),
    ({"questions": {"urgent": URGENT}, "answers": {"urgent": "yes"}}, "must be true or false"),
    ({"questions": {"urgent": {"type": "noul"}}, "answers": {"urgent": True}}, "needs a statement"),
    ({"questions": {"topic": TOPIC}, "answers": {"nope": "bug"}}, "no such question"),
    ({"state": "  "}, "state is missing or empty"),
    ({"questions": {}}, "questions must be a non-empty object"),
    ({"answers": ["bug"]}, "answers must be an object"),
])
def test_bad_rows_are_refused_with_a_clear_message(fields, match):
    with pytest.raises(DecisionDataError, match=match):
        prepare_rows(bad(**fields), seed=0)


def test_the_message_names_the_row_and_the_format():
    with pytest.raises(DecisionDataError, match=r"not in the decision format: row 61: answer 'refund' to 'topic'"):
        prepare_rows(bad(answers={"topic": "refund"}, questions={"topic": TOPIC}), seed=0)


def test_26_options_are_accepted_and_27_are_not():
    assert MAX_OPTIONS == 26
    q = {"topic": {**TOPIC, "options": [f"o{i}" for i in range(26)]}}
    prepare_rows(numbered(many(60) + [row(0, questions=q, answers={"topic": "o25"})]), seed=0)


def test_only_a_few_problems_are_spelled_out():
    rows = numbered(many(60) + [row(0, answers={"topic": "refund"}, questions={"topic": TOPIC}) for _ in range(9)])
    with pytest.raises(DecisionDataError) as e:
        prepare_rows(rows, seed=0)
    assert str(e.value).count("row ") <= 6 and "and 4 more row problems" in str(e.value)


# ---------------------------------------------------------------- loading

def test_load_rows_reads_jsonl_and_reports_a_bad_line_by_number(tmp_path: Path):
    f = tmp_path / "d.jsonl"
    f.write_text(json.dumps(row(1)) + "\n\n" + "{not json\n" + json.dumps(row(2)) + "\n", encoding="utf-8")
    with pytest.raises(DecisionDataError, match="row 3: invalid JSON"):
        load_rows(f)
    f.write_text(json.dumps(row(1)) + "\n\n" + json.dumps(row(2)) + "\n", encoding="utf-8")
    assert [n for n, _ in load_rows(f)] == [1, 3]  # blank lines are skipped, numbers are file lines
