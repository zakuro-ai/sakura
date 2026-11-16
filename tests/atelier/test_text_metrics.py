"""sakura.atelier.text_metrics: the generation metrics as pure functions, with no model and no tokenizer."""

import json

import pytest

from sakura.atelier import text_metrics as tm


def test_exact_match_compares_the_whole_completion_or_its_first_line():
    assert tm.exact_match("  42 \n", "42") == 1.0 and tm.exact_match("43", "42") == 0.0
    assert tm.exact_match("42\nand more", "42") == 0.0  # the whole completion counts ...
    assert tm.exact_match("  42\nand more", "42", first_line=True) == 1.0  # ... unless the 8-token rule applies
    assert tm.exact_match('{"a": 1}', '{"a":1}') == 0.0  # text, not JSON: that is json_match


@pytest.mark.parametrize("text, valid", [
    ('{"a": 1}', True), (' [1, 2, {"b": null}]\n', True), ("42", True), ('"text"', True),
    ('{"a": 1} and then prose', False), ('{"a": 1', False), ("{'a': 1}", False), ("", False),
    ("NaN", False), ('{"a": Infinity}', False), ("```json\n{}\n```", False),
])
def test_json_valid_needs_one_json_value_and_nothing_else(text, valid):
    assert tm.json_valid(text) == float(valid)


def test_json_match_compares_values_not_text():
    assert tm.json_match('{"b": [1, 2], "a": 1.0}', '{"a":1,"b":[1,2]}') == 1.0  # key order, spacing, 1 and 1.0
    assert tm.json_match('{"a": true}', '{"a": 1}') == 0.0  # a boolean is not a number
    assert tm.json_match('{"a": [2, 1]}', '{"a": [1, 2]}') == 0.0  # a list of scalars keeps its order
    assert tm.json_match('{"a": null}', '{"a": "null"}') == 0.0
    assert tm.json_match("not json", '{"a": 1}') == 0.0 and tm.json_match('{"a": 1}', "not json") == 0.0


def test_json_field_f1_counts_the_fields_that_agree():
    want = '{"title": "Lunch", "start": "12:00", "room": {"floor": 2, "name": "A"}}'
    assert tm.json_field_f1(want, want) == 1.0
    assert tm.json_field_f1('{"room": {"name": "A", "floor": 2.0}, "start": "12:00", "title": "Lunch"}', want) == 1.0
    # 3 of the 4 pairs agree on both sides: precision 3/4, recall 3/4
    assert tm.json_field_f1('{"title": "Lunch", "start": "13:00", "room": {"floor": 2, "name": "A"}}', want) == pytest.approx(0.75)
    # one extra field: precision 4/5, recall 4/4
    assert tm.json_field_f1(want[:-1] + ', "notes": "x"}', want) == pytest.approx(2 * 0.8 / 1.8)
    # one field missing: precision 3/3, recall 3/4
    assert tm.json_field_f1('{"title": "Lunch", "start": "12:00", "room": {"floor": 2}}', want) == pytest.approx(2 * 0.75 / 1.75)
    assert tm.json_field_f1('{"other": 1}', want) == 0.0
    assert tm.json_field_f1("Sure! Here is the JSON", want) == 0.0  # a completion that does not parse scores 0
    assert tm.json_field_f1(want, "the target is not JSON") == 0.0


def test_json_field_f1_matches_a_list_of_objects_as_a_multiset_of_whole_objects():
    want = json.dumps({"day": "mon", "events": [{"t": "a", "at": 9}, {"t": "b", "at": 10}, {"t": "b", "at": 10}]})
    reordered = json.dumps({"events": [{"at": 10, "t": "b"}, {"t": "a", "at": 9}, {"t": "b", "at": 10}], "day": "mon"})
    assert tm.json_field_f1(reordered, want) == 1.0  # the order of the objects does not count
    # one of the two identical events is missing: 5 of the 7 pairs are found, all 5 predicted are right
    once = json.dumps({"day": "mon", "events": [{"t": "a", "at": 9}, {"t": "b", "at": 10}]})
    assert tm.json_field_f1(once, want) == pytest.approx(2 * 1.0 * (5 / 7) / (1.0 + 5 / 7))
    # an object with one wrong field does not match at all: its two pairs are lost on both sides
    wrong = json.dumps({"day": "mon", "events": [{"t": "a", "at": 9}, {"t": "b", "at": 10}, {"t": "b", "at": 11}]})
    assert tm.json_field_f1(wrong, want) == pytest.approx(5 / 7)
    # fields are not traded between objects: swapping two events' times matches neither
    swapped = json.dumps({"day": "mon", "events": [{"t": "a", "at": 10}, {"t": "b", "at": 9}, {"t": "b", "at": 10}]})
    assert tm.json_field_f1(swapped, want) == pytest.approx(3 / 7)
    assert tm.json_field_f1('[{"t": "a"}, {"t": "b"}]', '[{"t": "b"}, {"t": "a"}]') == 1.0  # a top-level list too
    assert tm.json_field_f1('{"tags": ["x", "y"]}', '{"tags": ["y", "x"]}') == 0.0  # scalars are matched by position
    assert tm.json_field_f1('{"events": []}', '{"events": []}') == 1.0 and tm.json_field_f1("{}", "{}") == 1.0


def test_tokens_are_words_numbers_punctuation_marks_and_single_cjk_characters():
    assert tm.tokens("Hello, World! It's 12:30.") == ["hello", ",", "world", "!", "it", "'", "s", "12", ":", "30", "."]
    assert tm.tokens("東京でlunch at 12時") == ["東", "京", "で", "lunch", "at", "12", "時"]
    assert tm.tokens("  \n ") == []


def test_token_f1_is_the_overlap_of_the_two_bags_of_tokens():
    assert tm.token_f1("The cat sat", "the cat sat") == 1.0 and tm.token_f1("sat cat the", "the cat sat") == 1.0
    assert tm.token_f1("the cat", "the cat sat down") == pytest.approx(2 * 1.0 * 0.5 / 1.5)
    assert tm.token_f1("the the the", "the cat") == pytest.approx(2 * (1 / 3) * 0.5 / (1 / 3 + 0.5))  # counted once
    assert tm.token_f1("dog", "cat") == 0.0 and tm.token_f1("", "cat") == 0.0 and tm.token_f1("", "") == 1.0


def test_rouge_l_is_the_longest_common_subsequence_so_the_order_counts():
    assert tm.rouge_l("the cat sat on the mat", "the cat sat on the mat") == 1.0
    assert tm.rouge_l("a b c d", "a c d e") == pytest.approx(0.75)  # a, c, d in that order: 3 of 4 on both sides
    assert tm.rouge_l("sat cat the", "the cat sat") == pytest.approx(1 / 3)  # the same words, reversed
    assert tm.rouge_l("a b", "a x b y z") == pytest.approx(2 * 1.0 * 0.4 / 1.4)
    assert tm.rouge_l("", "a") == 0.0 and tm.rouge_l("", "") == 1.0


def test_score_is_the_mean_of_each_metric_over_the_rows():
    predictions, targets = ['{"a": 1}', "oops", '{"a": 2}'], ['{"a": 1}', '{"a": 1}', '{"a": 1}']
    assert tm.score(["json_valid", "json_match", "exact_match"], predictions, targets) == {
        "json_valid": pytest.approx(2 / 3), "json_match": pytest.approx(1 / 3), "exact_match": pytest.approx(1 / 3)}
    assert tm.score(["exact_match"], ["yes\nmore"], ["yes"], first_line=True) == {"exact_match": 1.0}
    assert tm.score(["exact_match"], ["yes\nmore"], ["yes"]) == {"exact_match": 0.0}
    assert tm.score(["token_f1", "rouge_l"], [], []) == {"token_f1": 0.0, "rouge_l": 0.0}  # no row: 0, not an error
    with pytest.raises(ValueError, match="unknown metric 'bleu'"):
        tm.score(["bleu"], ["a"], ["a"])
    assert set(tm.METRIC_NAMES) == {"exact_match", "json_valid", "json_match", "json_field_f1", "token_f1", "rouge_l"}


def test_auto_picks_the_metrics_from_the_validation_targets():
    assert tm.auto_metrics(['{"a": 1}', ' [{"b": 2}]\n'], [5, 7]) == ["json_field_f1", "json_match", "json_valid"]
    assert tm.auto_metrics(["positive", "negative"], [1, 2]) == ["exact_match"]
    assert tm.auto_metrics(["2", "4"], [1, 1]) == ["exact_match"]  # a bare number is a label, not a JSON document
    assert tm.auto_metrics(["positive", "a much longer label"], [1, 9]) == ["token_f1", "rouge_l"]  # over 8 tokens
    assert tm.auto_metrics(["two\nlines", "one"], [3, 1]) == ["token_f1", "rouge_l"]
    assert tm.auto_metrics(['{"a": 1}', "You're welcome!"], [5, 4]) == ["exact_match"]  # one target is not JSON
    assert tm.auto_metrics(['{"a": 1}', "You're welcome, see you tomorrow at nine!"], [5, 12]) == ["token_f1", "rouge_l"]
    assert tm.auto_metrics([], []) == ["exact_match"]  # no validation row: as before


def test_auto_generation_length_is_the_99th_percentile_of_the_targets_plus_8_capped():
    assert tm.auto_max_new_tokens(list(range(1, 201))) == 198 + 8  # nearest rank: the 198th of 200
    assert tm.auto_max_new_tokens([10] * 99 + [900]) == 10 + 8  # one outlier in a hundred does not set the length
    assert tm.auto_max_new_tokens([10] * 98 + [900, 900]) == 908
    assert tm.auto_max_new_tokens([5]) == 13 and tm.auto_max_new_tokens([]) == 8
    assert tm.auto_max_new_tokens([5000] * 10) == 1024


def test_a_row_without_a_target_is_scored_only_by_the_metrics_that_need_none():
    # rows: a normal one, and one with no target at all (an evaluation-only row, #238)
    rows = tm.score_rows(["json_match", "json_valid", "exact_match"], ['{"a": 1}', '{"a": 1}'], ['{"a":1}', ""],
                         scored=[True, False])
    assert rows == {"json_match": [1.0, None], "json_valid": [1.0, 1.0], "exact_match": [0.0, None]}
    assert tm.NO_TARGET_METRICS == ("json_valid",)
    # a mean skips the rows that were not scored; with none scored it is 0, as for no rows
    assert tm.mean([1.0, None, 0.0]) == 0.5 and tm.mean([None, None]) == 0.0
    assert tm.score(["json_match", "json_valid"], ['{"a": 1}', "x"], ['{"a":1}', ""], scored=[True, False]) == {
        "json_match": 1.0, "json_valid": 0.5}
    # without the mask every row has a target, as before
    assert tm.score_rows(["json_valid"], ["{}", "x"], ["", ""]) == {"json_valid": [1.0, 0.0]}
