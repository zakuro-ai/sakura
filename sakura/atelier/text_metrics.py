"""Generation metrics of the ``text_generation`` task: what a completion scores against its target.

Pure functions on strings (CONTRACTS §2.2), so they are tested without a model and a runtime that re-scores
an export uses the same code as training. Every metric is a number from 0 to 1 per row; a run reports the
mean over its rows. Nothing here imports a training stack.
"""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from collections.abc import Callable, Sequence
from typing import Any

#: The longest generation ``eval_max_new_tokens: auto`` asks for.
AUTO_MAX_NEW_TOKENS_CAP = 1024
#: Tokens added to the 99th percentile of the target lengths, and the length a short answer is generated at.
AUTO_MAX_NEW_TOKENS_SLACK = 8


def exact_match(prediction: str, target: str, *, first_line: bool = False) -> float:
    """1 when the completion is the target, both stripped. ``first_line`` is the rule of an 8-token
    evaluation: only the first line of the completion counts (what follows is the model running on)."""
    text = prediction.strip()
    if first_line:
        text = text.split("\n")[0].strip()
    return float(text == target.strip())


def _refuse_constant(name: str) -> Any:
    raise ValueError(f"{name} is not JSON")


def parse_json(text: str) -> tuple[bool, Any]:
    """``(True, value)`` when the stripped text is one JSON value with nothing after it; ``(False, None)``
    otherwise. ``NaN`` and ``Infinity`` are not JSON."""
    try:
        return True, json.loads(text.strip(), parse_constant=_refuse_constant)
    except (ValueError, RecursionError):
        return False, None


def json_valid(prediction: str, target: str = "") -> float:
    """1 when the completion parses as JSON."""
    return float(parse_json(prediction)[0])


def _canonical(value: Any) -> Any:
    """A JSON value in a form that compares as JSON does: key order ignored, ``1`` equal to ``1.0``, and
    ``true`` different from ``1`` (which Python would call equal)."""
    if isinstance(value, bool):
        return ("bool", value)
    if isinstance(value, float) and value.is_integer():
        return ("number", int(value))
    if isinstance(value, (int, float)):
        return ("number", value)
    if isinstance(value, list):
        return ("list", tuple(_canonical(v) for v in value))
    if isinstance(value, dict):
        return ("object", tuple(sorted((str(k), _canonical(v)) for k, v in value.items())))
    return ("string" if isinstance(value, str) else "null", value)


def json_match(prediction: str, target: str) -> float:
    """1 when both sides parse and are the same JSON value."""
    (ok_p, got), (ok_t, want) = parse_json(prediction), parse_json(target)
    return float(ok_p and ok_t and _canonical(got) == _canonical(want))


def _pairs(value: Any, path: str = "") -> list[tuple[Any, int]]:
    """A JSON value as ``(item, weight)`` entries. A scalar (or an empty object or list) at ``path`` is one
    entry of weight 1. A list of objects gives one entry per object, the whole object as the item, without its
    position, weighing its own number of pairs; any other list is entered position by position."""
    if isinstance(value, dict) and value:
        return [entry for key, inner in value.items() for entry in _pairs(inner, f"{path}.{key}" if path else str(key))]
    if isinstance(value, list) and value:
        if all(isinstance(inner, dict) for inner in value):
            objects = [_pairs(inner) for inner in value]
            return [((f"{path}[]", tuple(sorted(entries, key=repr))), sum(weight for _, weight in entries))
                    for entries in objects]
        return [entry for i, inner in enumerate(value) for entry in _pairs(inner, f"{path}[{i}]")]
    return [((path, _canonical(value)), 1)]


def json_field_f1(prediction: str, target: str) -> float:
    """F1 between the two sides' flattened ``path = value`` pairs (see :func:`_pairs`: a list of objects
    matches as a multiset of whole objects, whatever their order). 0 when either side does not parse."""
    (ok_p, got), (ok_t, want) = parse_json(prediction), parse_json(target)
    if not (ok_p and ok_t):
        return 0.0
    weight: dict[Any, int] = {}
    counts: list[Counter[Any]] = [Counter(), Counter()]
    for side, value in enumerate((got, want)):
        for item, w in _pairs(value):
            counts[side][item] += 1
            weight[item] = w
    overlap = sum(min(n, counts[1][item]) * weight[item] for item, n in counts[0].items())
    if not overlap:
        return 0.0
    precision = overlap / sum(n * weight[item] for item, n in counts[0].items())
    recall = overlap / sum(n * weight[item] for item, n in counts[1].items())
    return 2 * precision * recall / (precision + recall)


_CJK = "぀-ヿ㐀-䶿一-鿿豈-﫿가-힯"
#: One CJK character, or a run of other letters, digits and underscores, or one punctuation mark.
_TOKEN = re.compile(rf"[{_CJK}]|[^\W{_CJK}]+|[^\w\s]")


def tokens(text: str) -> list[str]:
    """The tokens ``token_f1`` and ``rouge_l`` compare: lower-cased words and numbers, each punctuation mark,
    and each CJK character on its own (those scripts put no space between words)."""
    return _TOKEN.findall(text.lower())


def _f1(common: int, predicted: int, wanted: int) -> float:
    if predicted == 0 or wanted == 0:
        return float(predicted == wanted)  # two empty texts agree
    if common == 0:
        return 0.0
    precision, recall = common / predicted, common / wanted
    return 2 * precision * recall / (precision + recall)


def token_f1(prediction: str, target: str) -> float:
    """F1 over the two sides' bags of tokens (the order of the words does not count)."""
    got, want = tokens(prediction), tokens(target)
    return _f1(sum((Counter(got) & Counter(want)).values()), len(got), len(want))


def rouge_l(prediction: str, target: str) -> float:
    """ROUGE-L: the F-measure of the longest common subsequence of the two sides' tokens (the order counts)."""
    got, want = tokens(prediction), tokens(target)
    row = [0] * (len(want) + 1)
    for a in got:
        diagonal = 0
        for j, b in enumerate(want, start=1):
            diagonal, row[j] = row[j], diagonal + 1 if a == b else max(row[j], row[j - 1])
    return _f1(row[-1], len(got), len(want))


_METRICS: dict[str, Callable[[str, str], float]] = {
    "exact_match": exact_match, "json_valid": json_valid, "json_match": json_match,
    "json_field_f1": json_field_f1, "token_f1": token_f1, "rouge_l": rouge_l,
}
METRIC_NAMES = tuple(_METRICS)


#: The metrics that compare nothing with a target, so they also score a row that has none (an evaluation-only row).
NO_TARGET_METRICS = ("json_valid",)


def score_rows(names: Sequence[str], predictions: Sequence[str], targets: Sequence[str], *,
               first_line: bool = False, scored: Sequence[bool] | None = None) -> dict[str, list[float | None]]:
    """Each metric in ``names`` on every row, in the order of the rows. ``scored`` says which rows have a
    target (default: all of them): on a row without one, a metric that needs a target is ``None`` (not scored)
    and a metric of :data:`NO_TARGET_METRICS` is scored as usual."""
    values: dict[str, list[float | None]] = {}
    for name in names:
        if name not in _METRICS:
            raise ValueError(f"unknown metric {name!r}: choose from {list(METRIC_NAMES)}")
        needs_target = name not in NO_TARGET_METRICS
        row_has_target = list(scored) if scored is not None else [True] * len(predictions)
        column: list[float | None] = []
        for p, t, has in zip(predictions, targets, row_has_target):
            if needs_target and not has:
                column.append(None)
            elif name == "exact_match":
                column.append(float(exact_match(p, t, first_line=first_line)))
            else:
                column.append(_METRICS[name](p, t))
        values[name] = column
    return values


def mean(values: Sequence[float | None]) -> float:
    """The mean of ``values`` that are numbers (a ``None`` is a row that was not scored), 0 when there are none."""
    kept = [v for v in values if v is not None]
    return sum(kept) / len(kept) if kept else 0.0


def score(names: Sequence[str], predictions: Sequence[str], targets: Sequence[str], *,
          first_line: bool = False, scored: Sequence[bool] | None = None) -> dict[str, float]:
    """The mean of each metric in ``names`` over the rows it scores (0 when there are none). ``first_line`` is
    ``exact_match``'s rule for an 8-token evaluation; ``scored`` as in :func:`score_rows`."""
    rows = score_rows(names, predictions, targets, first_line=first_line, scored=scored)
    return {name: mean(values) for name, values in rows.items()}


def auto_metrics(targets: Sequence[str], token_counts: Sequence[int]) -> list[str]:
    """``metrics: auto``: the metrics that fit the validation targets, the headline first.

    Every target is a JSON object or array: the three JSON metrics. Every target is one line of at most 8
    tokens (``token_counts``, the model tokenizer's): ``exact_match``. Anything else is free text:
    ``token_f1`` and ``rouge_l``. No target at all: ``exact_match``, as before."""
    if not targets:
        return ["exact_match"]
    if all(isinstance(parse_json(t)[1], (dict, list)) for t in targets):
        return ["json_field_f1", "json_match", "json_valid"]
    if all("\n" not in t.strip() for t in targets) and max(token_counts, default=0) <= AUTO_MAX_NEW_TOKENS_SLACK:
        return ["exact_match"]
    return ["token_f1", "rouge_l"]


def auto_max_new_tokens(token_counts: Sequence[int]) -> int:
    """``eval_max_new_tokens: auto``: the 99th percentile (nearest rank) of the validation targets' length
    in tokens, plus 8, at most 1024. 8 when there is no target."""
    if not token_counts:
        return AUTO_MAX_NEW_TOKENS_SLACK
    ranked = sorted(token_counts)
    p99 = ranked[max(0, math.ceil(0.99 * len(ranked)) - 1)]
    return min(AUTO_MAX_NEW_TOKENS_CAP, p99 + AUTO_MAX_NEW_TOKENS_SLACK)


__all__ = ["METRIC_NAMES", "NO_TARGET_METRICS", "auto_max_new_tokens", "auto_metrics", "exact_match",
           "json_field_f1", "json_match", "json_valid", "mean", "parse_json", "rouge_l", "score", "score_rows", "token_f1", "tokens"]
