"""Recipe options that more than one backend reads from ``hyperparameters``, checked in one place.

``registry.resolve`` calls :func:`check_recipe` on the expanded hyperparameters, so
``python -m sakura.atelier validate`` refuses a combination that makes no sense before anything loads.
The options live in ``hyperparameters`` (a preset, or ``overrides``) and not in a spec block of their own
for the same reason ``base_model`` does: they are part of the training recipe, a preset may fix them, and
``resolved.yaml`` records them with the other values the run used (SAK-0019, CONTRACTS §2.2).

Nothing here imports a training stack.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

#: ``init``: where the starting weights come from. ``pretrained`` (the default) loads the checkpoint
#: ``base_model`` names; ``random`` builds the same architecture from its config with fresh weights.
INITS = ("pretrained", "random")
#: The backends that can build their model from a config (``AutoConfig`` + ``from_config``).
INIT_BACKENDS = ("peft", "hf_image_classification", "hf_audio_classification", "hf_text_classification")
#: Tasks of those backends that still need pretrained weights.
_INIT_REFUSED_TASKS = ("decision",)


def init_of(hp: dict[str, Any]) -> str:
    """``hyperparameters.init``; ``pretrained`` when unset."""
    return str(hp.get("init") or "pretrained")


def init_config_of(hp: dict[str, Any]) -> dict[str, Any]:
    """``hyperparameters.init_config``: config fields to set before the random weights are drawn."""
    return dict(hp.get("init_config") or {})


def _check_init(task: str, hp: dict[str, Any], backend: str) -> None:
    init = init_of(hp)
    if init not in INITS:
        raise ValueError(f"init must be one of {list(INITS)}, got {hp.get('init')!r}")
    config = hp.get("init_config")
    if config is not None:
        if init != "random":
            raise ValueError("init_config only applies with init: random (it sizes the model that is built "
                             "from its config); a pretrained checkpoint keeps its own shape")
        if not isinstance(config, dict) or any(not isinstance(k, str) for k in config):
            raise ValueError("init_config must be a mapping of config field names to values")
        if "vocab_size" in config:
            raise ValueError("init_config.vocab_size cannot be set: the vocabulary size comes from the "
                             "tokenizer of base_model")
        bad = [k for k, v in config.items() if isinstance(v, (dict, type(None)))]
        if bad:
            raise ValueError(f"init_config.{bad[0]} must be a number, a string, a boolean or a list")
    if init != "random":
        return
    if backend not in INIT_BACKENDS or task in _INIT_REFUSED_TASKS:
        raise ValueError(f"init: random is not supported for task {task!r} on backend {backend!r}: it trains "
                         "hf-causal-lm (text_generation) and the Hugging Face classifiers from scratch")
    if str(hp.get("precision", "fp32")) == "4bit":
        raise ValueError("init: random cannot be combined with precision: 4bit: 4-bit loading quantises "
                         "pretrained weights for a LoRA, and a model trained from scratch has neither; "
                         "use precision: fp32 or bf16")


# ------------------------------------------------------------------------------------------ teacher

#: The backends that can train against a teacher's logits: the causal LM (peft) and the classifiers.
TEACHER_BACKENDS = INIT_BACKENDS
DEFAULT_DISTILL_ALPHA = 0.5
DEFAULT_DISTILL_TEMPERATURE = 2.0


@dataclass(frozen=True)
class Teacher:
    """``hyperparameters.teacher`` and its two weights: the loss is ``alpha`` of the temperature-scaled KL
    to the teacher's logits and ``1 - alpha`` of the hard-label (or next-token) loss."""

    ref: str
    alpha: float
    temperature: float


def teacher_of(hp: dict[str, Any]) -> Teacher | None:
    """The teacher a recipe names; None when it names none."""
    ref = hp.get("teacher")
    if ref in (None, ""):
        return None
    return Teacher(ref=str(ref), alpha=float(hp.get("distill_alpha", DEFAULT_DISTILL_ALPHA)),
                   temperature=float(hp.get("distill_temperature", DEFAULT_DISTILL_TEMPERATURE)))


def teacher_source(ref: str) -> tuple[str, str | None]:
    """``(what to load, revision)`` for a teacher reference.

    ``hf:<owner>/<repo>@<40-hex commit>`` is a pinned Hugging Face repo, the same form as an ``hf:`` model
    id. Anything else is a local directory (a path or ``file://``), as ``resume_from`` is: a run directory
    (``OUT``, whose ``checkpoint/`` holds the trained model), a checkpoint directory, or a model directory
    itself. A ``forge://<job>`` reference is for the runner to turn into a local run directory before the
    engine starts; the engine does not fetch it."""
    from sakura.atelier.hf_ref import parse_hf_model

    hf = parse_hf_model(ref)
    if hf is not None:
        return hf.repo, hf.revision
    if "://" in ref and not ref.startswith("file://"):
        raise ValueError(f"teacher {ref} is not a local directory: the runner resolves a forge:// reference to "
                         "the parent job's run directory before the engine starts; give the engine that "
                         "directory or hf:<owner>/<repo>@<commit>")
    path = Path(ref.removeprefix("file://"))
    if not path.exists():
        raise ValueError(f"teacher {ref!r} does not exist locally: {path}")
    for candidate in (path / "checkpoint" / "merged", path / "checkpoint", path / "merged", path):
        if (candidate / "config.json").is_file():
            return str(candidate), None
    raise ValueError(f"teacher {ref!r} holds no model: no config.json in it, its checkpoint/ or its "
                     "checkpoint/merged/")


def _number(value: Any) -> float | None:
    return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else None


def _check_teacher(task: str, hp: dict[str, Any], backend: str) -> None:
    ref = hp.get("teacher")
    if ref is None:
        for key in ("distill_alpha", "distill_temperature"):
            if hp.get(key) is not None:
                raise ValueError(f"{key} only applies with a teacher (hyperparameters.teacher)")
        return
    if not isinstance(ref, str) or not ref.strip():
        raise ValueError("teacher must name a model: hf:<owner>/<repo>@<40-hex commit sha>, or a run or model "
                         "directory")
    if backend not in TEACHER_BACKENDS or task in _INIT_REFUSED_TASKS:
        raise ValueError(f"teacher is not supported for task {task!r} on backend {backend!r}: logit "
                         "distillation trains hf-causal-lm (text_generation) and the Hugging Face classifiers")
    if ref.startswith("hf:"):
        from sakura.atelier.hf_ref import parse_hf_model

        try:
            parse_hf_model(ref)
        except ValueError as exc:
            raise ValueError(f"teacher: {str(exc).removeprefix('model ')}") from None
    if "distill_alpha" in hp:
        alpha = _number(hp["distill_alpha"])
        if alpha is None or not 0.0 <= alpha <= 1.0:
            raise ValueError(f"distill_alpha must be a number from 0 to 1, got {hp['distill_alpha']!r}")
    if "distill_temperature" in hp:
        temperature = _number(hp["distill_temperature"])
        if temperature is None or not temperature > 0.0 or temperature == float("inf"):
            raise ValueError(f"distill_temperature must be a number above 0, got {hp['distill_temperature']!r}")


# ---------------------------------------------------------------------------------------- objective

#: ``objective``: what the peft backend optimises on ``text_generation``. ``sft`` (the default) is the
#: next-token loss on ``{prompt, completion}`` rows; ``dpo`` is Direct Preference Optimization on
#: ``{prompt, chosen, rejected}`` rows.
OBJECTIVES = ("sft", "dpo")
DEFAULT_DPO_BETA = 0.1
DEFAULT_REJECTED_COLUMN = "rejected"


def objective_of(hp: dict[str, Any]) -> str:
    """``hyperparameters.objective``; ``sft`` when unset."""
    return str(hp.get("objective") or "sft")


def rejected_column_of(hp: dict[str, Any]) -> str:
    """The dataset column that holds a DPO row's rejected completion."""
    return str(hp.get("rejected_column") or DEFAULT_REJECTED_COLUMN)


def _check_objective(task: str, hp: dict[str, Any], backend: str) -> None:
    objective = objective_of(hp)
    if objective not in OBJECTIVES:
        raise ValueError(f"objective must be one of {list(OBJECTIVES)}, got {hp.get('objective')!r}")
    if objective != "dpo":
        for key in ("dpo_beta", "rejected_column"):
            if hp.get(key) is not None:
                raise ValueError(f"{key} only applies with objective: dpo")
        return
    if backend != "peft" or task != "text_generation":
        raise ValueError(f"objective: dpo is not supported for task {task!r} on backend {backend!r}: it "
                         "post-trains a causal LM on the peft backend (task text_generation)")
    if init_of(hp) == "random":
        raise ValueError("objective: dpo cannot be combined with init: random: DPO moves a policy away from "
                         "a reference model, which is the pretrained starting weights")
    if hp.get("teacher") is not None:
        raise ValueError("objective: dpo cannot be combined with a teacher: the DPO loss has no hard-label "
                         "term to blend a distillation loss with")
    if "dpo_beta" in hp:
        beta = _number(hp["dpo_beta"])
        if beta is None or not beta > 0.0 or beta == float("inf"):
            raise ValueError(f"dpo_beta must be a number above 0, got {hp['dpo_beta']!r}")
    column = hp.get("rejected_column")
    if column is not None and (not isinstance(column, str) or not column.strip()):
        raise ValueError("rejected_column must name the dataset column that holds the rejected completion")


def _check_train_eval(task: str, hp: dict[str, Any], backend: str) -> None:
    """The schedule, evaluation and metric keys of the causal LM (CONTRACTS §2.2): a wrong value is refused
    here, before a run. A spec that sets none of them is not looked at."""
    if backend != "peft" or task != "text_generation":
        return
    from sakura.atelier.backends import peft  # our module: it imports no training stack at import time

    peft.checkpointing_mode(hp)  # read for every spec, with or without the keys below
    given = [key for key in peft.TRAIN_EVAL_KEYS if key in hp]
    if not given:
        return
    if objective_of(hp) == "dpo":
        raise ValueError(f"{given[0]} does not apply with objective: dpo: the DPO loop has its own evaluation "
                         "(the schedule, evaluation and metric keys belong to the next-token objective)")
    peft.train_eval_options(hp)


def check_recipe(task: str, hp: dict[str, Any], backend: str) -> None:
    """Raise ``ValueError`` naming the first recipe option of ``hp`` that is wrong for this task and backend."""
    _check_init(task, hp, backend)
    _check_teacher(task, hp, backend)
    _check_objective(task, hp, backend)
    _check_train_eval(task, hp, backend)


__all__ = ["INITS", "INIT_BACKENDS", "OBJECTIVES", "TEACHER_BACKENDS", "Teacher", "check_recipe", "init_config_of",
           "init_of", "objective_of", "rejected_column_of", "teacher_of", "teacher_source"]
