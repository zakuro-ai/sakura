"""The shape `predict` and `serve` answer in, and the `decision_config` they read. Pure Python."""

from __future__ import annotations

import json
import math

import pytest

from sakura.atelier.decision import (
    DecisionDataError,
    answer_request,
    decision_config,
    parse_request,
    read_decision_config,
)

REQUEST = {
    "state": "Hi, I was charged twice for March.",
    "questions": {
        "topic": {"type": "choice", "instructions": "What is this about?", "options": ["billing", "bug", "other"]},
        "urgent": {"type": "noul", "statement": "The customer needs a reply today."},
        "tone": {"type": "score", "instructions": "How upset is the customer?", "levels": ["calm", "annoyed", "angry"]},
    },
}
TEMPS = {"choice": 2.0, "noul": 1.0, "score": 1.0}


class FakeScorer:
    """Logits by question id; remembers what it was asked so a test can see the state shared once."""

    def __init__(self, logits):
        self.logits = logits
        self.calls = []

    def __call__(self, state, instances):
        self.calls.append((state, [i.qid for i in instances]))
        return [self.logits[i.qid] for i in instances]


LOGITS = {"topic": [math.log(8), math.log(1), math.log(1)], "urgent": [math.log(3), 0.0], "tone": [0.0, math.log(3), 0.0]}


def test_answers_have_the_documented_shape():
    out = answer_request(REQUEST, TEMPS, FakeScorer(LOGITS))
    assert list(out) == ["answers"]
    a = out["answers"]
    assert set(a) == {"topic", "urgent", "tone"}
    assert set(a["topic"]) == {"choice", "probs"} and a["topic"]["choice"] == "billing"
    assert set(a["topic"]["probs"]) == {"billing", "bug", "other"}
    assert set(a["urgent"]) == {"noul"} and 0.0 < a["urgent"]["noul"] < 1.0
    assert set(a["tone"]) == {"score", "probs"} and set(a["tone"]["probs"]) == {"calm", "annoyed", "angry"}
    json.dumps(out)  # plain JSON types


def test_the_temperature_of_the_type_is_applied_to_the_logits():
    a = answer_request(REQUEST, TEMPS, FakeScorer(LOGITS))["answers"]
    # choice logits [ln 8, 0, 0] at T = 2 -> softmax([ln 8 / 2, 0, 0]) = [sqrt 8, 1, 1] / (sqrt 8 + 2)
    top = math.sqrt(8) / (math.sqrt(8) + 2)
    assert a["topic"]["probs"]["billing"] == pytest.approx(top, abs=1e-5)
    assert sum(a["topic"]["probs"].values()) == pytest.approx(1.0, abs=1e-5)
    assert a["urgent"]["noul"] == pytest.approx(0.75, abs=1e-5)  # T = 1: 3 / (3 + 1)


def test_noul_reports_p_true_and_score_an_expected_level_indexed_from_zero():
    a = answer_request(REQUEST, TEMPS, FakeScorer(LOGITS))["answers"]
    assert a["urgent"]["noul"] == pytest.approx(0.75, abs=1e-5)
    # probs over [calm, annoyed, angry] = [.2, .6, .2]  ->  0*.2 + 1*.6 + 2*.2 = 1.0
    assert a["tone"]["score"] == pytest.approx(1.0, abs=1e-5)
    skew = {**LOGITS, "tone": [0.0, 0.0, math.log(8)]}
    assert answer_request(REQUEST, TEMPS, FakeScorer(skew))["answers"]["tone"]["score"] > 1.2


def test_the_questions_of_one_state_are_scored_together_so_they_can_share_its_prefix():
    scorer = FakeScorer(LOGITS)
    answer_request(REQUEST, TEMPS, scorer)
    assert scorer.calls == [("Hi, I was charged twice for March.", ["topic", "urgent", "tone"])]


def test_the_inputs_wrapper_the_hub_posts_is_accepted():
    assert answer_request({"inputs": REQUEST}, TEMPS, FakeScorer(LOGITS)) == answer_request(REQUEST, TEMPS, FakeScorer(LOGITS))


def test_a_json_state_is_passed_through_as_given():
    scorer = FakeScorer(LOGITS)
    answer_request({"state": {"ticket": 7}, "questions": {"topic": REQUEST["questions"]["topic"]}}, TEMPS, scorer)
    assert scorer.calls[0][0] == {"ticket": 7}


@pytest.mark.parametrize("request_,match", [
    ("not an object", "request must be an object"),
    ({"questions": REQUEST["questions"]}, "state is missing or empty"),
    ({"state": "x"}, "questions must be a non-empty object"),
    ({"state": "x", "questions": {}}, "questions must be a non-empty object"),
    ({"state": "x", "questions": {"q": {"type": "rank"}}}, "type 'rank'"),
    ({"state": "x", "questions": {"q": {"type": "choice", "options": ["only"]}}}, "1 options; 2 to 26"),
])
def test_a_bad_request_is_refused_with_a_clear_message(request_, match):
    with pytest.raises(DecisionDataError, match=match):
        parse_request(request_)


def test_parse_request_keeps_the_given_option_order():
    _, instances = parse_request(REQUEST)
    assert [i.qid for i in instances] == ["topic", "urgent", "tone"]
    assert instances[0].labels == ["billing", "bug", "other"] and instances[1].labels == ["true", "false"]


# ------------------------------------------------------------ decision_config

def test_decision_config_names_what_a_reader_needs():
    cfg = decision_config(temperatures=TEMPS, base_model="Qwen/Qwen3.5-2B-Base", base_revision="abc",
                          base_commit="def", max_seq_length=1024, precision="bf16")
    assert cfg["schema"] == 1 and cfg["template"] == "decision/v1"
    assert cfg["temperatures"] == TEMPS and cfg["base_model"] == "Qwen/Qwen3.5-2B-Base"
    assert cfg["base_revision"] == "abc" and cfg["letters"] == "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    assert cfg["letter_token"] == "space+letter"
    json.dumps(cfg)


def test_read_decision_config_round_trips_and_refuses_a_future_schema(tmp_path):
    cfg = decision_config(temperatures=TEMPS, base_model="m", base_revision=None, base_commit=None,
                          max_seq_length=64, precision="fp32")
    f = tmp_path / "decision_config.json"
    f.write_text(json.dumps(cfg))
    assert read_decision_config(f) == cfg
    f.write_text(json.dumps({**cfg, "schema": 2}))
    with pytest.raises(ValueError, match="schema 2"):
        read_decision_config(f)
    f.write_text(json.dumps({**cfg, "template": "decision/v2"}))
    with pytest.raises(ValueError, match="decision/v2"):
        read_decision_config(f)


# ----------------------------------------------------------------- answer_hit

def _inst(kind, labels, target):
    from sakura.atelier.decision import Instance

    return Instance(state="s", qid="q", kind=kind, text="t", labels=labels, target=target)


def test_answer_hit_follows_the_same_tie_rule_as_the_report_metrics():
    from sakura.atelier.decision import answer_hit, shape_answer, softmax, split_metrics, Scored

    choice = _inst("choice", ["a", "b", "c"], [0.0, 1.0, 0.0])
    assert answer_hit(choice, shape_answer(choice, [0.0, 3.0, 0.0], 1.0))
    assert not answer_hit(choice, shape_answer(choice, [3.0, 0.0, 0.0], 1.0))
    noul = _inst("noul", ["true", "false"], [1.0, 0.0])
    assert answer_hit(noul, shape_answer(noul, [1.0, 0.0], 1.0))
    assert answer_hit(noul, {"noul": 0.5})  # a tie goes to the first label, true -- as in split_metrics
    assert split_metrics([Scored("q", "noul", [0.0, 0.0], [1.0, 0.0])], {})["accuracy"] == 1.0
    soft = _inst("score", ["lo", "mid", "hi"], [0.2, 0.5, 0.3])  # the modal label of a soft target is the gold
    assert answer_hit(soft, shape_answer(soft, [0.0, 2.0, 0.0], 1.0))
    assert softmax([0.0, 2.0, 0.0])[1] > 0.5
