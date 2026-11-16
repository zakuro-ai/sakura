"""The ``decision`` (System One) task, the parts that need no training stack.

A decision row is a ``state`` (text or any JSON value), typed ``questions`` about it
(``choice``, ``score``, ``noul``) and the gold ``answers``. The model reads the answer
off the logits of the option letters, so what it returns is a probability per option and
nothing generated. Contract: zak-forge ``docs/specs/2026-10-21-decision-task-design.md``
Part B; the torch half (loss, readout, training, predict) is in ``backends/peft.py``.

Everything here is plain Python so that ``import sakura.atelier`` stays free of any
training stack and so the data, split, template and calibration rules are testable
without a GPU.
"""

from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

TEMPLATE_VERSION = "decision/v1"

#: Fewer usable rows than this is a named customer-data failure; fewer than WARN_USABLE_ROWS
#: is a warning in the report.
MIN_USABLE_ROWS = 50
WARN_USABLE_ROWS = 300

QUESTION_TYPES = ("choice", "score", "noul")
#: A choice has 2 to 26 options and a score 2 to 26 ordered levels: one letter each. Scoring the
#: option NAMES instead of letters is weaker (reported by the open projects) and out of scope.
MIN_OPTIONS, MAX_OPTIONS = 2, 26
LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
#: A ``noul`` is a two-option question; its labels in this order, and P(true) is the first.
NOUL_LABELS = ["true", "false"]
#: 80% train, 10% calibration, 10% eval, by state.
EVAL_FRACTION = CALIBRATION_FRACTION = 0.1
#: A state per split at the least.
MIN_STATES = 3
#: An eval split with fewer answers than this gives accuracy / ECE too noisy to compare.
MIN_EVAL_ANSWERS = 30
#: A soft label's probabilities must sum to 1 within this; they are renormalised after.
_DISTRIBUTION_TOLERANCE = 0.01
#: How many distinct row problems are spelled out; the rest are counted.
MAX_ERRORS_SHOWN = 5


class DecisionDataError(ValueError):
    """The customer's data (or base-model choice) cannot be trained on: a named, non-platform failure."""


# ---------------------------------------------------------------- base model

def check_base_model(hp: dict[str, Any], preset_allowlist: dict[str, Any]) -> dict[str, Any]:
    """``hp`` with the base model checked against the allowlist and its revision pinned.

    ``preset_allowlist`` is the preset's ``base_model_allowlist`` (``{repo id: pinned commit}``) read
    from the preset file. The allowlist in ``hp`` (a replay of ``resolved.yaml`` carries one) may not
    differ from it: otherwise an ``overrides`` or a hand-written ``hyperparameters`` block could widen
    the list of bases a licence review covered. A ``base_revision`` left unset becomes the pinned
    commit, so ``resolved.yaml`` names the exact weights. Any OTHER revision is refused: the allowlist is a
    licence review of that commit, and a later (or community) revision of the same repo has not been reviewed.
    A new review is a new preset version.

    Raises ``DecisionDataError`` (a ``ValueError``) with ``base model not on the allowlist`` for a base off
    the list."""
    out = dict(hp)
    if not isinstance(preset_allowlist, dict) or not preset_allowlist:
        raise DecisionDataError("the decision preset carries no base_model_allowlist")
    carried = out.get("base_model_allowlist", preset_allowlist)
    if carried != preset_allowlist:
        raise DecisionDataError("base_model_allowlist is set by the preset and cannot be overridden "
                                f"(got {carried!r})")
    base = str(out.get("base_model") or "")
    if base not in preset_allowlist:
        raise DecisionDataError(f"base model not on the allowlist ({base!r}; allowed: "
                                f"{', '.join(sorted(preset_allowlist))})")
    out["base_model_allowlist"] = dict(preset_allowlist)
    out["base_revision"] = _reviewed_revision(base, out.get("base_revision"), preset_allowlist[base])
    return out


def _reviewed_revision(base: str, requested: Any, pinned: str | None) -> str | None:
    """The revision to load: the pinned commit when none is asked for, and only the pinned commit otherwise
    (an allowlist entry pinned to ``None`` reviews no particular commit and takes any)."""
    if requested in (None, ""):
        return pinned
    if pinned and str(requested) != pinned:
        raise DecisionDataError(f"base_revision {str(requested)!r} is not the reviewed commit {pinned} of {base}; "
                                "the allowlist covers that commit only")
    return str(requested)


PRECISIONS = ("fp32", "bf16", "4bit")


@dataclass(frozen=True)
class DecisionOptions:
    """The peft backend's reading of ``spec.hyperparameters`` for a decision run."""

    base_model: str
    base_revision: str | None
    epochs: int
    learning_rate: float
    batch_size: int
    grad_accum: int
    lora_r: int
    lora_alpha: int
    lora_dropout: float
    lora_target_modules: str | list[str]
    max_seq_length: int
    precision: str
    #: Recompute each layer's activations in the backward pass instead of keeping them. Same gradients; what
    #: lets the 2B base train at 8 x 1024 tokens on an 11 GB card (the GPU host 2080 Ti: 7.5 GiB peak with it, out of
    #: memory at 1 x 1024 without). Not a preset key: on by default, off only through `overrides`.
    gradient_checkpointing: bool = True


def decision_options(hp: dict[str, Any]) -> DecisionOptions:
    """Parse and check the hyperparameters of a decision run. Defaults are the values of ``decision/lora-forge@1``.

    The base model is checked against the allowlist the hyperparameters carry (``resolve`` already did, against
    the preset file; a backend called by other means must not train on a base nobody reviewed), and a run with
    no allowlist at all is refused rather than taken to allow everything."""
    allow = hp.get("base_model_allowlist")
    base = str(hp.get("base_model") or "")
    if not isinstance(allow, dict) or base not in allow:
        raise DecisionDataError(f"base model not on the allowlist ({base!r}; allowed: "
                                f"{', '.join(sorted(allow)) if isinstance(allow, dict) else 'none given'})")
    template = str(hp.get("prompt_template", TEMPLATE_VERSION))
    if template != TEMPLATE_VERSION:
        raise ValueError(f"prompt_template must be {TEMPLATE_VERSION!r} for the decision task, got {template!r}")
    precision = str(hp.get("precision", "bf16"))
    if precision not in PRECISIONS:
        raise ValueError(f"precision must be one of {PRECISIONS}, got {precision!r}")
    epochs, batch_size = int(hp.get("epochs", 3)), int(hp.get("batch_size", 8))
    if epochs < 1:
        raise ValueError(f"epochs must be at least 1, got {epochs}")
    if batch_size < 1:
        raise ValueError(f"batch_size must be at least 1, got {batch_size}")
    targets = hp.get("lora_target_modules", "all-linear")
    return DecisionOptions(
        base_model=base,
        base_revision=_reviewed_revision(base, hp.get("base_revision"), allow[base]),
        epochs=epochs, learning_rate=float(hp.get("learning_rate", 1e-4)), batch_size=batch_size,
        grad_accum=max(1, int(hp.get("grad_accum", 1))), lora_r=int(hp.get("lora_r", 16)),
        lora_alpha=int(hp.get("lora_alpha", 32)), lora_dropout=float(hp.get("lora_dropout", 0.05)),
        lora_target_modules=targets if isinstance(targets, str) else [str(t) for t in targets],
        max_seq_length=int(hp.get("max_seq_length", 1024)), precision=precision,
        gradient_checkpointing=bool(hp.get("gradient_checkpointing", True)))


# ------------------------------------------------------------------ rows

@dataclass(frozen=True)
class Instance:
    """One (state, question) with its target: what the model is trained and scored on."""

    state: Any
    qid: str
    kind: str
    #: ``instructions`` for a choice / score, the ``statement`` for a noul.
    text: str
    #: What each letter stands for, in the question's given order (a noul: ``[true, false]``).
    labels: list[str]
    #: A distribution over ``labels``: one-hot for a hard label, the given one for a soft label.
    target: list[float]


@dataclass
class Prepared:
    train: list[dict[str, Any]]
    calibration: list[dict[str, Any]]
    eval: list[dict[str, Any]]
    stats: dict[str, Any]
    warnings: list[str] = field(default_factory=list)


def load_rows(path: Path) -> list[tuple[int, Any]]:
    """``[(line number, parsed row)]`` of a JSONL file; blank lines are skipped, a line that is not JSON
    is a named customer-data failure (the first few are spelled out)."""
    rows: list[tuple[int, Any]] = []
    problems: list[str] = []
    for n, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            rows.append((n, json.loads(line)))
        except ValueError as exc:
            problems.append(f"row {n}: invalid JSON ({exc})")
    if problems:
        raise DecisionDataError(_not_in_format(problems))
    return rows


def _not_in_format(problems: list[str]) -> str:
    shown = "; ".join(problems[:MAX_ERRORS_SHOWN])
    more = len(problems) - MAX_ERRORS_SHOWN
    return "the dataset is not in the decision format: " + shown + (f"; and {more} more row problems" if more > 0 else "")


def _is_number(v: Any) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


def _clean_question(qid: str, spec: Any) -> tuple[dict[str, Any] | None, str | None]:
    """``(stripped question, problem)``."""
    if not isinstance(spec, dict):
        return None, f"question '{qid}' is not an object"
    kind = spec.get("type")
    if kind not in QUESTION_TYPES:
        return None, f"question '{qid}' has type {kind!r}; use one of {', '.join(QUESTION_TYPES)}"
    if kind == "noul":
        statement = spec.get("statement")
        if not isinstance(statement, str) or not statement.strip():
            return None, f"noul question '{qid}' needs a statement"
        return {"type": "noul", "statement": statement.strip()}, None
    key = "options" if kind == "choice" else "levels"
    labels = spec.get(key)
    if not isinstance(labels, list):
        return None, f"question '{qid}': a {kind} question needs a list of {key}"
    if not all(isinstance(x, str) and x.strip() for x in labels):
        return None, f"question '{qid}': {key} must be non-empty strings"
    labels = [x.strip() for x in labels]
    if len(set(labels)) != len(labels):
        return None, f"question '{qid}': {key} are not distinct"
    if not MIN_OPTIONS <= len(labels) <= MAX_OPTIONS:
        return None, (f"question '{qid}': a {kind} question has {len(labels)} {key}; "
                      f"{MIN_OPTIONS} to {MAX_OPTIONS} are supported")
    instructions = spec.get("instructions")
    return {"type": kind, "instructions": instructions.strip() if isinstance(instructions, str) else "",
            key: labels}, None


def _labels_of(question: dict[str, Any]) -> list[str]:
    kind = question["type"]
    return list(NOUL_LABELS) if kind == "noul" else list(question["options" if kind == "choice" else "levels"])


def _target_of(qid: str, question: dict[str, Any], answer: Any) -> tuple[list[float] | None, str | None]:
    """``(target distribution over the question's labels, problem)``."""
    labels = _labels_of(question)
    if question["type"] == "noul":
        if isinstance(answer, bool):
            return ([1.0, 0.0] if answer else [0.0, 1.0]), None
        if _is_number(answer) and 0 <= answer <= 1:  # a soft label: P(true)
            return [float(answer), 1.0 - float(answer)], None
        return None, f"answer to '{qid}' must be true or false, not {answer!r}"
    if isinstance(answer, str):
        answer = answer.strip()
        if answer in labels:
            return [1.0 if lab == answer else 0.0 for lab in labels], None
        return None, f"answer {answer!r} to '{qid}' is not one of its {len(labels)} labels"
    if isinstance(answer, dict):
        dist = {str(k).strip(): v for k, v in answer.items()}
        unknown = [k for k in dist if k not in labels]
        if unknown:
            return None, f"distribution for '{qid}' names {unknown[0]!r}, which is not one of its labels"
        if not dist or not all(_is_number(p) and p >= 0 for p in dist.values()):
            return None, f"distribution for '{qid}' needs non-negative numbers"
        total = float(sum(dist.values()))
        if abs(total - 1) > _DISTRIBUTION_TOLERANCE:
            return None, f"distribution for '{qid}' sums to {total:.3g}, not 1"
        return [dist.get(lab, 0.0) / total for lab in labels], None
    return None, f"answer to '{qid}' must be a label or a distribution over labels, not {answer!r}"


def _clean_answer(answer: Any) -> Any:
    if isinstance(answer, str):
        return answer.strip()
    if isinstance(answer, dict):
        return {str(k).strip(): v for k, v in answer.items()}
    return answer


def _clean_row(row: Any) -> tuple[dict[str, Any] | None, list[str]]:
    """``(stripped row with only its answered questions, problems)``. The row is None when it has problems;
    a row that is valid but has no answered question comes back with empty ``answers``."""
    if not isinstance(row, dict):
        return None, ["expected an object"]
    problems: list[str] = []
    state = row.get("state")
    if isinstance(state, str):
        state = state.strip()
    if state is None or state == "" or state == {} or state == []:
        problems.append("state is missing or empty")
    questions, answers = row.get("questions"), row.get("answers")
    if not isinstance(questions, dict) or not questions:
        problems.append("questions must be a non-empty object")
        questions = {}
    if not isinstance(answers, dict):
        problems.append("answers must be an object")
        answers = {}
    clean_questions: dict[str, dict[str, Any]] = {}
    for qid, spec in questions.items():
        clean, problem = _clean_question(str(qid), spec)
        if problem:
            problems.append(problem)
        elif clean is not None:
            clean_questions[str(qid)] = clean
    clean_answers: dict[str, Any] = {}
    for qid, answer in answers.items():
        if qid not in questions:
            problems.append(f"answers has '{qid}' but there is no such question")
        elif answer is None:
            continue  # left unanswered: not trained or scored for this row
        elif qid in clean_questions:
            _, problem = _target_of(qid, clean_questions[qid], answer)
            if problem:
                problems.append(problem)
            else:
                clean_answers[qid] = _clean_answer(answer)
    if problems:
        return None, problems
    return {"state": state, "questions": clean_questions, "answers": clean_answers}, []


def instances(row: dict[str, Any]) -> list[Instance]:
    """The answered questions of a cleaned row, in the order the row answers them."""
    out: list[Instance] = []
    for qid, answer in row["answers"].items():
        q = row["questions"][qid]
        target, problem = _target_of(qid, q, answer)
        if target is None:
            raise DecisionDataError(problem or f"bad answer to {qid!r}")
        text = q["statement"] if q["type"] == "noul" else q.get("instructions", "")
        out.append(Instance(state=row["state"], qid=qid, kind=q["type"], text=text,
                            labels=_labels_of(q), target=target))
    return out


def _key(value: Any) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, default=str)


def prepare_rows(raw: list[tuple[int, Any]], seed: int) -> Prepared:
    """Validate, strip, drop the unanswered, dedupe on ``(state, questions, answers)``, shuffle with
    ``seed`` and split BY STATE (never by question) into train / calibration / eval, 80 / 10 / 10,
    so no state is in two splits. Raises ``DecisionDataError`` (a named customer-data failure) for a
    malformed row or for fewer than ``MIN_USABLE_ROWS`` usable rows."""
    problems: list[str] = []
    cleaned: list[dict[str, Any]] = []
    unanswered = 0
    for n, row in raw:
        clean, row_problems = _clean_row(row)
        problems.extend(f"row {n}: {p}" for p in row_problems)
        if clean is None:
            continue
        if not clean["answers"]:
            unanswered += 1
            continue
        cleaned.append(clean)
    if problems:
        raise DecisionDataError(_not_in_format(problems))

    seen: set[str] = set()
    rows: list[dict[str, Any]] = []
    for clean in cleaned:
        k = _key([clean["state"], clean["questions"], clean["answers"]])
        if k not in seen:
            seen.add(k)
            rows.append(clean)
    duplicates = len(cleaned) - len(rows)

    warnings: list[str] = []
    if unanswered:
        warnings.append(f"{unanswered} rows have no answered question and were dropped")
    if len(rows) < MIN_USABLE_ROWS:
        raise DecisionDataError(f"too few usable rows ({len(rows)}; at least {MIN_USABLE_ROWS} needed)")
    if len(rows) < WARN_USABLE_ROWS:
        warnings.append(f"only {len(rows)} usable rows; {WARN_USABLE_ROWS} or more make the held-out "
                        "numbers steadier")

    by_state: dict[str, list[dict[str, Any]]] = {}
    for r in rows:
        by_state.setdefault(_key(r["state"]), []).append(r)
    states = sorted(by_state)
    if len(states) < MIN_STATES:
        raise DecisionDataError(f"too few distinct states ({len(states)}; at least {MIN_STATES} needed to fill "
                                "train, calibration and eval, which are split by state)")
    random.Random(seed).shuffle(states)
    n_eval = max(1, round(EVAL_FRACTION * len(states)))
    n_cal = max(1, round(CALIBRATION_FRACTION * len(states)))
    homes = {"eval": states[:n_eval], "calibration": states[n_eval:n_eval + n_cal], "train": states[n_eval + n_cal:]}
    parts = {name: [r for s in keys for r in sorted(by_state[s], key=_key)] for name, keys in homes.items()}

    answers = {name: sum(len(r["answers"]) for r in part) for name, part in parts.items()}
    if answers["eval"] < MIN_EVAL_ANSWERS:
        warnings.append(f"the eval split has only {answers['eval']} answers (< {MIN_EVAL_ANSWERS}): "
                        "its accuracy and calibration error are too noisy to compare runs")
    stats = {"rows_read": len(raw), "usable_rows": len(rows), "dropped_unanswered": unanswered,
             "duplicates": duplicates, "states": len(states),
             "split_states": {name: len(keys) for name, keys in homes.items()},
             "split_rows": {name: len(part) for name, part in parts.items()},
             "split_answers": answers}
    return Prepared(train=parts["train"], calibration=parts["calibration"], eval=parts["eval"],
                    stats=stats, warnings=warnings)


# ------------------------------------------------------- template, option order

#: What the model is shown when a choice / score question carries no instructions of its own.
DEFAULT_INSTRUCTIONS = "Choose the best option."
NOUL_INSTRUCTIONS = "Is the following statement true or false?"


def render_state(state: Any) -> str:
    """The state as text: a string as given, anything else as JSON with sorted keys (so the same
    object is always the same prompt)."""
    return state if isinstance(state, str) else json.dumps(state, sort_keys=True, ensure_ascii=False)


def build_prompt(instance: Instance, order: list[int]) -> tuple[str, str]:
    """``(prefix, suffix)`` of template ``decision/v1``; the model reads the next token after ``Answer:``,
    which is one of the option letters ``" A"``, ``" B"``, ...

    ``prefix`` is the state and depends on nothing else, so every question about one state can reuse
    its cache; ``suffix`` is the question with its options listed as lettered lines. ``order[i]`` is the
    index of the label shown at letter ``i`` (:func:`identity_order` at inference, :func:`shuffled_order`
    in training). Both halves are tokenised on their own and joined as token ids, in training and at
    inference alike, so the model sees the same ids either way.
    """
    prefix = f"### State\n{render_state(instance.state)}\n\n"
    header = instance.text if instance.kind != "noul" else f"{NOUL_INSTRUCTIONS}\n{instance.text}"
    lines = "".join(f"{LETTERS[i]}. {instance.labels[j]}\n" for i, j in enumerate(order))
    return prefix, f"### Question\n{header or DEFAULT_INSTRUCTIONS}\n{lines}Answer:"


def identity_order(k: int) -> list[int]:
    return list(range(k))


def shuffled_order(rng: random.Random, k: int) -> list[int]:
    """A fresh option order, drawn per example per epoch in training so the letter of the right answer
    carries no signal."""
    order = identity_order(k)
    rng.shuffle(order)
    return order


def to_letter_order(values: list[float], order: list[int]) -> list[float]:
    """Per-label ``values`` re-indexed by letter: letter ``i`` shows label ``order[i]``."""
    return [values[j] for j in order]


def to_label_order(values: list[float], order: list[int]) -> list[float]:
    """Inverse of :func:`to_letter_order`: per-letter ``values`` back to the question's label order."""
    out = [0.0] * len(values)
    for i, j in enumerate(order):
        out[j] = values[i]
    return out


# ------------------------------------------------------------------ calibration

#: Probabilities are ``softmax(logits / T)``. A temperature is fitted per answer type on the
#: calibration split, within these bounds.
MIN_TEMPERATURE, MAX_TEMPERATURE = 0.05, 20.0
#: A type with fewer calibration answers than this keeps T = 1 (a fit on a handful of answers is noise).
MIN_CALIBRATION_ANSWERS = 30
ECE_BINS = 15
_EPS = 1e-12


@dataclass(frozen=True)
class Scored:
    """The restricted logits of one answered question, in the question's label order, with its target."""

    qid: str
    kind: str
    logits: list[float]
    target: list[float]


def softmax(logits: list[float], temperature: float = 1.0) -> list[float]:
    scaled = [v / temperature for v in logits]
    top = max(scaled)
    exps = [math.exp(v - top) for v in scaled]
    total = sum(exps)
    return [e / total for e in exps]


def _cross_entropy(target: list[float], logits: list[float], beta: float) -> float:
    scaled = [v * beta for v in logits]
    top = max(scaled)
    log_z = top + math.log(sum(math.exp(v - top) for v in scaled))
    return sum(t * (log_z - v) for t, v in zip(target, scaled) if t > 0)


def fit_temperature(items: list[Scored]) -> float:
    """The ``T`` minimising the negative log-likelihood of ``items`` under ``softmax(logits / T)``.

    The NLL is convex in ``1 / T`` (a log-sum-exp minus a linear term), so a golden-section search over
    ``1 / T`` finds the minimum; the answer is clamped to ``[MIN_TEMPERATURE, MAX_TEMPERATURE]``.
    """
    if not items:
        return 1.0

    def loss(beta: float) -> float:
        return sum(_cross_entropy(i.target, i.logits, beta) for i in items)

    lo, hi = 1.0 / MAX_TEMPERATURE, 1.0 / MIN_TEMPERATURE
    phi = (math.sqrt(5.0) - 1.0) / 2.0
    a, b = lo, hi
    c, d = b - phi * (b - a), a + phi * (b - a)
    fc, fd = loss(c), loss(d)
    for _ in range(80):
        if fc < fd:
            b, d, fd = d, c, fc
            c = b - phi * (b - a)
            fc = loss(c)
        else:
            a, c, fc = c, d, fd
            d = a + phi * (b - a)
            fd = loss(d)
    return min(MAX_TEMPERATURE, max(MIN_TEMPERATURE, 2.0 / (a + b)))


def expected_calibration_error(confidences: list[float], correct: list[bool], bins: int = ECE_BINS) -> float:
    """Top-label ECE: the confidence-vs-accuracy gap per equal-width bin, weighted by the bin's share."""
    n = len(confidences)
    if not n:
        return 0.0
    count = [0] * bins
    conf_sum = [0.0] * bins
    hits = [0] * bins
    for c, ok in zip(confidences, correct):
        b = min(int(c * bins), bins - 1)
        count[b] += 1
        conf_sum[b] += c
        hits[b] += 1 if ok else 0
    return sum(count[b] / n * abs(hits[b] / count[b] - conf_sum[b] / count[b]) for b in range(bins) if count[b])


def _argmax(values: list[float]) -> int:
    return max(range(len(values)), key=lambda i: (values[i], -i))  # ties go to the first


def split_metrics(items: list[Scored], temperatures: dict[str, float]) -> dict[str, Any]:
    """``n``, ``accuracy``, ``log_loss`` and ``ece`` of ``items`` with each type's temperature applied.

    A soft target is scored by its cross-entropy and counts as right when the prediction is its modal label."""
    if not items:
        return {"n": 0}
    loss = acc = 0.0
    confidences: list[float] = []
    correct: list[bool] = []
    for it in items:
        p = softmax(it.logits, temperatures.get(it.kind, 1.0))
        loss += -sum(t * math.log(max(q, _EPS)) for t, q in zip(it.target, p) if t > 0)
        hit = _argmax(p) == _argmax(it.target)
        acc += hit
        confidences.append(max(p))
        correct.append(hit)
    n = len(items)
    return {"n": n, "accuracy": round(acc / n, 6), "log_loss": round(loss / n, 6),
            "ece": round(expected_calibration_error(confidences, correct), 6)}


def fit_temperatures(calibration: list[Scored]) -> tuple[dict[str, float], dict[str, int], list[str]]:
    """``(temperatures, n_calib, warnings)``: one temperature per answer type, fitted on the CALIBRATION
    items only. A type with fewer than ``MIN_CALIBRATION_ANSWERS`` keeps ``T = 1`` and a warning (a type
    that has none at all is simply not in the data)."""
    temps: dict[str, float] = {}
    n_calib: dict[str, int] = {}
    warnings: list[str] = []
    for kind in QUESTION_TYPES:
        of_kind = [i for i in calibration if i.kind == kind]
        n_calib[kind] = len(of_kind)
        if len(of_kind) >= MIN_CALIBRATION_ANSWERS:
            temps[kind] = round(fit_temperature(of_kind), 6)
        else:
            temps[kind] = 1.0
            if of_kind:
                warnings.append(f"{kind}: only {len(of_kind)} calibration answers (< {MIN_CALIBRATION_ANSWERS}); "
                                "temperature kept at 1")
    return temps, n_calib, warnings


#: ``decision_config.json`` schema this engine writes and reads.
CONFIG_SCHEMA = 1


def decision_config(*, temperatures: dict[str, float], base_model: str, base_revision: str | None,
                    base_commit: str | None, max_seq_length: int, precision: str) -> dict[str, Any]:
    """``decision_config.json``: what a reader of the exported model needs to answer as the run did --
    the template version, how the option letters are read, the fitted temperatures and the base."""
    return {"schema": CONFIG_SCHEMA, "template": TEMPLATE_VERSION, "letters": LETTERS,
            "letter_token": "space+letter",  # the token after `Answer:` is " A", " B", ...
            "temperatures": dict(temperatures), "base_model": base_model, "base_revision": base_revision,
            "base_commit": base_commit, "precision": precision, "max_seq_length": max_seq_length}


def read_decision_config(path: Path) -> dict[str, Any]:
    cfg = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(cfg, dict):
        raise ValueError("decision_config must be an object")
    if cfg.get("schema") != CONFIG_SCHEMA:
        raise ValueError(f"decision_config schema {cfg.get('schema')!r} is not supported "
                         f"(this engine reads schema {CONFIG_SCHEMA})")
    if cfg.get("template") != TEMPLATE_VERSION:
        raise ValueError(f"decision_config names prompt template {cfg.get('template')!r}; "
                         f"this engine reads {TEMPLATE_VERSION}")
    return cfg


# ------------------------------------------------------------------ predict

def parse_request(request: Any) -> tuple[Any, list[Instance]]:
    """``(state, one Instance per question)`` of a ``predict`` / ``serve`` request, ``{"state": ...,
    "questions": {...}}`` (the hub posts it inside ``{"inputs": ...}``, which is unwrapped). The questions
    are those of a training row, minus the answers; their options keep the order given."""
    if isinstance(request, dict) and "state" not in request and isinstance(request.get("inputs"), dict):
        request = request["inputs"]
    if not isinstance(request, dict):
        raise DecisionDataError("request must be an object with a state and questions")
    state = request.get("state")
    if isinstance(state, str):
        state = state.strip()
    if state is None or state == "" or state == {} or state == []:
        raise DecisionDataError("state is missing or empty")
    questions = request.get("questions")
    if not isinstance(questions, dict) or not questions:
        raise DecisionDataError("questions must be a non-empty object")
    found: list[Instance] = []
    for qid, spec in questions.items():
        clean, problem = _clean_question(str(qid), spec)
        if problem or clean is None:
            raise DecisionDataError(problem or f"bad question {qid!r}")
        text = clean["statement"] if clean["type"] == "noul" else clean["instructions"]
        found.append(Instance(state=state, qid=str(qid), kind=clean["type"], text=text,
                              labels=_labels_of(clean), target=[]))
    return state, found


def shape_answer(instance: Instance, logits: list[float], temperature: float) -> dict[str, Any]:
    """One typed answer: a choice names its pick and gives every probability, a noul gives P(true), a
    score gives the expected level (indexed from 0 into the ordered levels) and every probability."""
    probs = [round(p, 6) for p in softmax(logits, temperature)]
    by_label = dict(zip(instance.labels, probs))
    if instance.kind == "noul":
        return {"noul": probs[0]}
    if instance.kind == "score":
        return {"score": round(sum(i * p for i, p in enumerate(probs)), 6), "probs": by_label}
    return {"choice": instance.labels[_argmax(probs)], "probs": by_label}


def answer_hit(instance: Instance, shaped: dict[str, Any]) -> bool:
    """Whether a shaped answer (:func:`shape_answer`) picks the target's modal label. A noul is
    read as true from a probability of one half up, as ties go to the first label everywhere."""
    if instance.kind == "noul":
        picked = 0 if shaped["noul"] >= 0.5 else 1
    else:
        picked = _argmax([shaped["probs"][label] for label in instance.labels])
    return picked == _argmax(instance.target)


def answer_request(request: Any, temperatures: dict[str, float],
                   score: Any) -> dict[str, Any]:
    """The ``{"answers": {...}}`` that ``predict`` prints and ``serve`` returns.

    ``score(state, instances) -> list of restricted logits`` (one list per instance, in the instance's
    label order) is the model; it gets every question of the request at once so it can score them off
    one shared state prefix. Each type's fitted temperature is applied here."""
    state, found = parse_request(request)
    logits = score(state, found)
    return {"answers": {i.qid: shape_answer(i, lg, temperatures.get(i.kind, 1.0)) for i, lg in zip(found, logits)}}


def report_blocks(calibration: list[Scored], eval_items: list[Scored], warnings: list[str]) -> dict[str, Any]:
    """The ``metrics``, ``evaluation`` and ``calibration`` blocks of ``report.json`` for a decision run.

    Temperatures are fitted on ``calibration``; ``eval_items`` are only ever scored, before (``T = 1``) and
    after, so ``ece_after`` is measured on rows the fit never saw. ``metrics`` are the calibrated eval numbers."""
    temps, n_calib, fit_warnings = fit_temperatures(calibration)
    before = split_metrics(eval_items, {})
    after = split_metrics(eval_items, temps)
    unfitted = [k for k in QUESTION_TYPES if n_calib[k] == 0 and any(i.kind == k for i in eval_items)]
    fit_warnings += [f"{k}: eval has {k} answers but calibration has none; temperature kept at 1" for k in unfitted]
    eval_block: dict[str, Any] = dict(after)
    eval_block["by_type"] = {k: split_metrics([i for i in eval_items if i.kind == k], temps)
                             for k in QUESTION_TYPES if any(i.kind == k for i in eval_items)}
    qids = sorted({i.qid for i in eval_items})
    eval_block["by_question"] = {q: split_metrics([i for i in eval_items if i.qid == q], temps) for q in qids}
    return {
        "metrics": {k: after.get(k, 0.0) for k in ("accuracy", "log_loss", "ece")},
        "evaluation": {"splits": {"eval": eval_block}, "warnings": [*warnings, *fit_warnings]},
        "calibration": {"method": "temperature", "temperatures": temps, "n_calib": n_calib,
                        "ece_before": before.get("ece", 0.0), "ece_after": after.get("ece", 0.0)},
    }
