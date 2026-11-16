"""PEFT (LoRA) backend: text generation fine-tuning.

Implements the ``Backend`` protocol (CONTRACTS §2.2) for ``text_generation``,
model ids ``lfm2-350m`` (``LiquidAI/LFM2-350M``, ungated, native
``transformers`` support as of 5.x -- no ``trust_remote_code``) and
``hf-causal-lm`` (any Hugging Face causal LM named by ``hp.base_model`` at
``hp.base_revision``). LoRA SFT via ``peft``; merges the adapter before export
so every exported format is a plain dense checkpoint, not an adapter needing
the base model alongside it.

Hyper-parameters (``_training_options``; defaults = the lfm2-350m/lora-fast@1
behaviour): ``base_model``, ``base_revision``, ``epochs``, ``learning_rate``,
``batch_size``, ``grad_accum``, ``lora_r``, ``lora_alpha``, ``lora_dropout``,
``lora_target_modules``, ``max_seq_length`` (``max_length`` deprecated alias),
``precision`` (fp32 | bf16 | 4bit = bitsandbytes NF4 QLoRA, CUDA only),
``prompt_template`` (raw | alpaca = zak-forge ``template.py``'s
``### Instruction:`` format, applied in training, validation, ``runtime_eval``
and ``predict`` via ``artifacts/preprocess.json``), ``gguf_quant`` (a
llama-quantize type or a zak-forge profile: Q8_0 | Q6_K | MXQ4; falls back to
``SAKURA_GGUF_QUANT``, then Q8_0). ``text_generation/qlora-forge@1`` is the
legacy La Forge runner's recipe. ``report.json`` gets ``provenance`` (base
model, requested revision, resolved commit, precision, template).
The learning-rate schedule, the evaluation cadence, early stopping, the best
checkpoint, the generation metrics and the base-model baseline are the keys in
``TRAIN_EVAL_KEYS`` (CONTRACTS §2.2, "Schedule, evaluation and metrics");
``text_generation/lora-tuned@1`` sets them. A spec with none of them is unchanged.

Data convention (Track E1's data layer does not exist yet, same caveat as the
other two backends in this track):
  ``data.train`` / ``data.validation`` = ``list[{"prompt": str, "completion":
  str}]``. ``data.labels`` is unused (``None``).

Exports
-------
- ``safetensors``: the merged HF checkpoint (``model.safetensors`` + config +
  tokenizer), via ``save_pretrained(..., safe_serialization=True)``.
- ``gguf``: via llama.cpp's ``convert_hf_to_gguf.py`` on the merged
  safetensors dir, then ``llama-quantize`` to the format in
  ``SAKURA_GGUF_QUANT`` (default ``Q8_0``). Needs ``LLAMA_CPP_DIR`` pointing
  at a llama.cpp checkout with those two built (docs/atelier/envs/peft.txt).
  Run with ``sys.executable`` (this process' own interpreter), not a bare
  ``python3``: llama.cpp's convert script needs a ``transformers`` new
  enough to read whatever tokenizer backend class this repo's ``transformers``
  saved (found the hard way -- the host's system python3 had an older
  ``transformers`` that could not load LFM2's tokenizer; see this track's
  handoff report). This venv additionally needs the ``gguf`` pip package
  (not a llama.cpp requirement installed by default outside its own repo).
- ``mlx``: ``export_mlx`` -- ``mlx_lm convert`` of the merged safetensors at
  ``SAKURA_MLX_QUANT`` (``fp16`` | ``q8`` | ``q4``, default ``q8``), then a
  re-score on the validation split with the training harness's exact rule
  (raw prompt, greedy, 8 tokens, first line), written next to the artefact as
  ``model.mlx.eval.json`` (score, bytes, ms/example). Apple silicon only
  (macOS arm64 with ``mlx-lm``); anywhere else it raises
  ``MLXUnavailableError`` with the precise reason -- never faked. Also
  callable on its own from a merged dir (``python -m
  sakura.atelier.backends.peft mlx --merged DIR --validation VAL.jsonl
  --quant q8 --out DIR``), which is what an apple-silicon worker runs.

``runtime_eval``/``predict`` re-score the GGUF artifact through
``llama-cpp-python`` (``Llama(model_path, ...)(prompt, ...)``, its raw text
*completion* API, not ``create_chat_completion``) -- no chat template is
applied, so the model sees exactly the training-time prompt string, nothing
wrapped around it. An earlier version of this module shelled out to the
``llama-cli`` binary instead; on this track's the GPU host build, llama-cli still
rendered its interactive REPL/chat framing around a "single-turn" completion
regardless of ``--no-conversation``/``--chat-template ''``, which measured
exact_match 0.0 for reasons that had nothing to do with the GGUF itself (see
this track's handoff report). ``llama-cpp-python`` needs a C/C++ toolchain to
build from source (``CC=gcc CXX=g++ pip install llama-cpp-python`` --
docs/atelier/envs/peft.txt); it is independent of the ``LLAMA_CPP_DIR``
checkout, which is still needed for ``export(..., "gguf")``'s conversion/
quantisation step. safetensors is re-scored through ``transformers`` and
mlx through ``mlx_lm`` (the score ``export_mlx`` measured is reused).

``runtime_eval`` protocol note: ``Backend.runtime_eval(self, artifact, data)``
(CONTRACTS §2.2) is not passed the ``TrainResult``, so ``matches_training`` is
left ``None`` here -- the engine's ``run`` loop (Track E1) holds
``TrainResult.metrics`` and is the right place to apply the §2.3 tolerance.

Licence: LFM2 weights are under the LFM Open License (CONTRACTS §2.4).

Task ``decision`` (model id ``hf-decision-lm``, preset ``decision/lora-forge@1``) is also served here: a LoRA
fine-tune of a small BASE model to answer typed questions about a text or JSON state with a calibrated
probability per option and no generated prose. The data, split, template and calibration rules are
``sakura.atelier.decision`` (no torch); the option-letter readout, the loss and the training loop are
``decision_lm`` (torch only). ``train``, ``runtime_eval`` and ``predict`` dispatch to the ``_decision_*``
methods below; the four model-loading seams (``_decision_load`` / ``_adapt`` / ``_merge`` / ``_load_saved``)
are the only places that touch ``transformers`` and ``peft``, so a test can run the whole pipeline on a fake.
Exports: ``safetensors`` only (a gguf runtime cannot reproduce the letter readout or the temperature).
"""

from __future__ import annotations

import contextlib
import json
import math
import os
import platform
import random
import re
import shutil
import subprocess
import sys
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar

from sakura.atelier import decision as dec
from sakura.atelier import text_metrics
from sakura.atelier.lr_schedule import lr_factor, optimizer_steps, warmup_steps
from sakura.atelier.recipe import (
    DEFAULT_DPO_BETA,
    Teacher,
    init_config_of,
    init_of,
    objective_of,
    teacher_of,
    teacher_source,
)
from sakura.atelier.types import (
    Artifact,
    Checkpoint,
    MaterialisedData,
    MetricsSink,
    ResolvedSpec,
    RunDir,
    RuntimeEval,
    TrainResult,
)
from sakura.atelier.wrap_up import WrapUpWatch, max_steps_of

LICENSE = "LFM Open License v1.0 (LiquidAI/LFM2-350M)"

#: Fallback base when neither the registry entry nor the spec names one (a
#: hand-authored resolved.yaml from before `base_model` existed): `lfm2-350m`'s.
BASE_MODEL = "LiquidAI/LFM2-350M"

# llama-quantize type names: Q8_0, Q4_K_M, IQ2_XXS, F16, BF16, ...
_GGUF_QUANT_RE = re.compile(r"[A-Za-z0-9_]{1,32}")

#: GGUF quantisation profiles: profile -> (extra llama-quantize flags, llama-quantize
#: type). Copied verbatim from zak-forge ``src/forge/runner/adapters/llamacpp.py``
#: (``PROFILE_ARGS``) so a hub job quantised by this engine gets the same tensor
#: mix the legacy runner produced; ``MXQ4`` is not a llama-quantize type name but
#: Q4_K_M with Q6_K output/embedding tensors. Any other bare type name
#: (``Q4_K_M``, ``Q5_K_M``, ``F16``) passes straight through, as before.
GGUF_PROFILES: dict[str, tuple[tuple[str, ...], str]] = {
    "Q6_K": ((), "Q6_K"),
    "Q8_0": ((), "Q8_0"),
    "MXQ4": (("--output-tensor-type", "Q6_K", "--token-embedding-type", "Q6_K"), "Q4_K_M"),
}
DEFAULT_GGUF_QUANT = "Q8_0"

PRECISIONS = ("fp32", "bf16", "4bit")
PROMPT_TEMPLATES = ("raw", "alpaca")
LR_SCHEDULES = ("constant", "linear", "cosine")
#: The generation metrics a run can score (CONTRACTS §2.2). The first one a run computes is its headline.
TEXT_METRICS = text_metrics.METRIC_NAMES
#: Rows an export is re-scored on when the recipe sets no ``runtime_eval_rows``: a GGUF on a CPU costs seconds a row, so
#: re-scoring thousands of validation rows would take longer than the training. An explicit key (a number, or null for
#: every row) is obeyed.
DEFAULT_RUNTIME_EVAL_ROWS = 200
#: The generation length before ``eval_max_new_tokens``; at this length ``exact_match`` reads the first line.
LEGACY_NEW_TOKENS = 8
#: The schedule, evaluation and metric keys added on 2026-11-15 (CONTRACTS §2.2). A spec that sets none of
#: them trains, logs and reports exactly as before; one that sets any of them (to any value) also gets the
#: reporting that goes with them: ``lr`` per optimiser step, ``epoch_metrics`` and ``early_stop``.
TRAIN_EVAL_KEYS = (
    "lr_schedule", "warmup_ratio", "min_lr_ratio", "weight_decay", "max_grad_norm", "group_by_length",
    "lora_use_rslora", "lora_use_dora", "eval_every", "eval_max_rows", "gen_eval_rows", "final_gen_rows",
    "eval_batch_size", "early_stop_patience", "keep_best", "selection_metric", "eval_max_new_tokens",
    "metrics", "baseline", "runtime_eval_rows", "eval_scope", "eval_only_gen_rows",
)
#: ``eval_scope`` values: generation is scored on every validation row, or only on the rows whose
#: ``features.eval.final`` column is true (the last turn of a conversation). The loss always uses its rows.
EVAL_SCOPES = ("all", "final")
#: ``predictions.jsonl`` keeps at most this many characters of each prediction (``prediction_chars`` says how long
#: it really was), so the file stays small whatever ``eval_max_new_tokens`` is.
PREDICTION_CHARS_CAP = 8192
#: The artifact a run with the evaluation keys writes beside its exports: what the final evaluation generated.
PREDICTIONS_FILE = "predictions.jsonl"

# The markers that delimit the alpaca training format: a completion that reaches
# one has left its own answer and started a new turn (zak-forge template.py).
_STOP_MARKERS = ("### Instruction:", "### Response:", "###")


def _gguf_quant_args(quant: str) -> tuple[list[str], str]:
    """``(extra flags, llama-quantize type)`` for a profile or bare type name."""
    if not _GGUF_QUANT_RE.fullmatch(quant):
        raise GGUFExportError(f"gguf_quant={quant!r} is not a llama.cpp quant type name")
    if quant in GGUF_PROFILES:
        opts, qtype = GGUF_PROFILES[quant]
        return list(opts), qtype
    return [], quant


#: ggml's K-quants (Q2_K .. Q6_K, and the Q*_K_S/M/L mixes) pack 256-wide super-blocks:
#: a row that is not a multiple of 256 cannot be quantised to them.
K_QUANT_BLOCK = 256


def _check_k_quant_shape(model_dir: Path, quant: str, types: list[str]) -> None:
    """Refuse a K-quant for a model whose ``hidden_size`` is not a multiple of 256.

    ``llama-quantize`` does not refuse cleanly: it falls back per tensor, then
    aborts on ``token_embd.weight`` with ``GGML_ASSERT(start % blck_size == 0)``
    (SmolLM2-135M, hidden 576, under ``MXQ4``). The message names the mismatch
    and the profiles that do fit; a ``config.json`` without the field is let
    through, llama-quantize's own stderr then says what happened."""
    if not any(_is_k_quant(t) for t in types):
        return
    config = Path(model_dir) / "config.json"
    if not config.is_file():
        return
    try:
        hidden = int(json.loads(config.read_text(encoding="utf-8")).get("hidden_size") or 0)
    except (ValueError, TypeError, json.JSONDecodeError):
        return
    if hidden and hidden % K_QUANT_BLOCK:
        raise GGUFExportError(
            f"gguf_quant={quant!r} uses K-quants, which need hidden_size to be a multiple of "
            f"{K_QUANT_BLOCK}; this model has hidden_size={hidden}. Use Q8_0 (or another non-K "
            "type such as Q5_0 / Q4_0) for this base model.")


def _is_k_quant(qtype: str) -> bool:
    return bool(re.fullmatch(r"I?Q\d_K(_[SML])?", qtype.upper()))


def _format_prompt(template: str, instruction: str) -> str:
    """The string the model sees for an input. ``raw`` = the dataset's own text
    (lfm2-350m / lora-fast@1 behaviour); ``alpaca`` = exactly what zak-forge
    ``src/forge/runner/template.py::format_prompt`` produces, so a model trained
    here answers the prompts the hub's legacy artifacts were trained on."""
    if template == "raw":
        return instruction
    if template == "alpaca":
        return f"### Instruction:\n{instruction}\n\n### Response:\n"
    raise ValueError(f"prompt_template must be one of {PROMPT_TEMPLATES}, got {template!r}")


def _truncate_completion(text: str) -> str:
    """Cut a raw completion at the first point it runs past its own answer
    (zak-forge ``template.py::truncate_completion``): a fine-tune that has not
    learned EOS keeps going into a hallucinated ``### Instruction:`` block."""
    cut = len(text)
    for marker in _STOP_MARKERS:
        i = text.find(marker)
        if i != -1:
            cut = min(cut, i)
    return text[:cut].strip()


def _clean_completion(template: str, text: str) -> str:
    return _truncate_completion(text) if template == "alpaca" else text


def _require_bitsandbytes() -> None:
    try:
        import bitsandbytes  # type: ignore[import-not-found,unused-ignore]  # noqa: F401
    except ImportError as exc:
        raise RuntimeError(
            "precision: 4bit (QLoRA) needs the `bitsandbytes` package, which is CUDA-only "
            f"and not importable here ({type(exc).__name__}: {exc}). Install it on the GPU "
            "worker (docs/atelier/envs/peft.txt) or set precision: fp32|bf16."
        ) from exc


@dataclass(frozen=True)
class _Options:
    """The peft backend's reading of ``spec.hyperparameters``."""

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
    prompt_template: str
    gguf_quant: str | None
    #: ``random`` = train the architecture from fresh weights, in full (no LoRA); see ``_random_causal_lm``.
    init: str = "pretrained"
    init_config: dict[str, Any] | None = None
    #: ``dpo`` = Direct Preference Optimization on {prompt, chosen, rejected} rows (``_train_dpo``).
    objective: str = "sft"
    dpo_beta: float = DEFAULT_DPO_BETA
    # Schedule, evaluation and metrics (``TRAIN_EVAL_KEYS``). Every default is the behaviour before the keys.
    lr_schedule: str = "constant"
    warmup_ratio: float = 0.0
    min_lr_ratio: float = 0.0
    weight_decay: float | None = None  # None = AdamW's own default
    max_grad_norm: float | None = None
    group_by_length: bool = False
    lora_use_rslora: bool = False
    lora_use_dora: bool = False
    eval_every: float | None = None  # share of an epoch; None = once per epoch
    eval_max_rows: int | None = None  # None = every validation row
    gen_eval_rows: int | None = None  # None = the rows the loss uses
    final_gen_rows: int | None = None  # None = every validation row
    eval_batch_size: int = 1
    early_stop_patience: int | None = None
    keep_best: bool = False
    selection_metric: str = "loss"
    eval_max_new_tokens: int | str = 8  # a number of tokens, or "auto"
    metrics: tuple[str, ...] | str = ("exact_match",)  # metric names, or "auto"
    baseline: bool = False
    runtime_eval_rows: int | None = None  # None = every final row
    eval_scope: str = "all"  # "all" | "final": the rows generation is scored on
    eval_only_gen_rows: int | None = None  # evaluation-only rows the final evaluation adds; None = all of them
    #: True when the spec set at least one of ``TRAIN_EVAL_KEYS``: turns on the reporting that goes with them.
    extended: bool = False
    #: ``auto`` | ``true`` | ``false``: recompute activations in the backward pass to save memory (``_checkpointing``).
    gradient_checkpointing: str = "auto"


def _hp_number(hp: dict[str, Any], key: str, low: float, high: float | None, *, above: bool = False) -> float | None:
    """``hp[key]`` as a number in ``[low, high]`` (``(low, high]`` with ``above``); None when unset or null."""
    value = hp.get(key)
    if value is None:
        return None
    ok = isinstance(value, (int, float)) and not isinstance(value, bool) and value == value
    if ok and (value > low if above else value >= low) and (high is None or value <= high):
        return float(value)
    if above:
        span = f"above {low:g}" + (f" and at most {high:g}" if high is not None else "")
    else:
        span = f"from {low:g} to {high:g}" if high is not None else f"of {low:g} or more"
    raise ValueError(f"{key} must be a number {span} (or null), got {value!r}")


def _hp_int(hp: dict[str, Any], key: str, low: int) -> int | None:
    """``hp[key]`` as a whole number of at least ``low``; None when unset or null."""
    value = hp.get(key)
    if value is None:
        return None
    if isinstance(value, int) and not isinstance(value, bool) and value >= low:
        return value
    raise ValueError(f"{key} must be a whole number of at least {low} (or null), got {value!r}")


def _hp_bool(hp: dict[str, Any], key: str) -> bool:
    """``hp[key]`` as a boolean; False when unset or null. A string is refused: ``"false"`` is not false."""
    value = hp.get(key)
    if value is None:
        return False
    if isinstance(value, bool):
        return value
    raise ValueError(f"{key} must be true or false, got {value!r}")


def checkpointing_mode(hp: dict[str, Any]) -> str:
    """``hp.gradient_checkpointing`` as ``auto`` (the default), ``true`` or ``false``; a boolean is accepted."""
    value = hp.get("gradient_checkpointing")
    if value is None:
        return "auto"
    if isinstance(value, bool):
        return "true" if value else "false"
    if value == "auto":
        return "auto"
    raise ValueError(f"gradient_checkpointing must be auto, true or false, got {value!r}")


def train_eval_options(hp: dict[str, Any]) -> dict[str, Any]:
    """The ``TRAIN_EVAL_KEYS`` of ``hp``, checked: the fields of ``_Options`` they set. Raises ``ValueError``
    naming the first key that is wrong. Pure (no model, no tokenizer), so ``resolve`` runs it too."""
    schedule = str(hp.get("lr_schedule") or "constant")
    if schedule not in LR_SCHEDULES:
        raise ValueError(f"lr_schedule must be one of {list(LR_SCHEDULES)}, got {hp.get('lr_schedule')!r}")
    metrics = hp.get("metrics")
    if metrics is None:
        metrics = ("exact_match",)
    elif metrics != "auto":
        if (not isinstance(metrics, (list, tuple)) or not metrics or len(set(metrics)) != len(metrics)
                or any(m not in TEXT_METRICS for m in metrics)):
            raise ValueError(f"metrics must be 'auto' or a list of distinct names from {list(TEXT_METRICS)}, "
                             f"got {metrics!r}")
        metrics = tuple(metrics)
    selection = str(hp.get("selection_metric") or "loss")
    if selection != "loss" and selection not in (TEXT_METRICS if metrics == "auto" else metrics):
        raise ValueError(f"selection_metric must be 'loss' or one of the run's metrics "
                         f"{list(TEXT_METRICS) if metrics == 'auto' else list(metrics)}, got {selection!r}")
    gen_eval_rows = _hp_int(hp, "gen_eval_rows", 0)
    if selection != "loss" and gen_eval_rows == 0:
        raise ValueError(f"selection_metric: {selection} needs generated rows at every evaluation, and "
                         "gen_eval_rows is 0")
    tokens = hp.get("eval_max_new_tokens")
    if tokens is None:
        tokens = 8
    elif tokens != "auto" and (not isinstance(tokens, int) or isinstance(tokens, bool) or tokens < 1):
        raise ValueError(f"eval_max_new_tokens must be a whole number of at least 1 or 'auto', got {tokens!r}")
    scope = hp.get("eval_scope")
    if scope is None:
        scope = "all"
    elif scope not in EVAL_SCOPES:
        raise ValueError(f"eval_scope must be one of {list(EVAL_SCOPES)}, got {scope!r}")
    return {
        "lr_schedule": schedule,
        "warmup_ratio": _hp_number(hp, "warmup_ratio", 0.0, 0.5) or 0.0,
        "min_lr_ratio": _hp_number(hp, "min_lr_ratio", 0.0, 1.0) or 0.0,
        "weight_decay": _hp_number(hp, "weight_decay", 0.0, None),
        "max_grad_norm": _hp_number(hp, "max_grad_norm", 0.0, None, above=True),
        "group_by_length": _hp_bool(hp, "group_by_length"),
        "lora_use_rslora": _hp_bool(hp, "lora_use_rslora"),
        "lora_use_dora": _hp_bool(hp, "lora_use_dora"),
        "eval_every": _hp_number(hp, "eval_every", 0.0, 1.0, above=True),
        "eval_max_rows": _hp_int(hp, "eval_max_rows", 1),
        "gen_eval_rows": gen_eval_rows,
        "final_gen_rows": _hp_int(hp, "final_gen_rows", 1),
        "eval_batch_size": _hp_int(hp, "eval_batch_size", 1) or 1,
        "early_stop_patience": _hp_int(hp, "early_stop_patience", 1),
        "keep_best": _hp_bool(hp, "keep_best"),
        "selection_metric": selection,
        "eval_max_new_tokens": tokens,
        "metrics": metrics,
        "baseline": _hp_bool(hp, "baseline"),
        "runtime_eval_rows": _hp_int(hp, "runtime_eval_rows", 1),
        "eval_scope": scope,
        "eval_only_gen_rows": _hp_int(hp, "eval_only_gen_rows", 0),
        "extended": any(key in hp for key in TRAIN_EVAL_KEYS),
    }


def _training_options(hp: dict[str, Any]) -> _Options:
    """Defaults are the pre-generalisation behaviour of this backend, so an
    `lfm2-350m` + `text_generation/lora-fast@1` run is unchanged; `qlora-forge@1`
    sets every key the legacy La Forge runner fixed."""
    if "max_seq_length" in hp:
        max_seq_length = int(hp["max_seq_length"])
    elif "max_length" in hp:
        warnings.warn("hp.max_length is deprecated for the peft backend; use max_seq_length",
                      DeprecationWarning, stacklevel=2)
        max_seq_length = int(hp["max_length"])
    else:
        max_seq_length = 64
    precision = str(hp.get("precision", "fp32"))
    if precision not in PRECISIONS:
        raise ValueError(f"precision must be one of {PRECISIONS}, got {precision!r}")
    template = str(hp.get("prompt_template", "raw"))
    if template not in PROMPT_TEMPLATES:
        raise ValueError(f"prompt_template must be one of {PROMPT_TEMPLATES}, got {template!r}")
    targets = hp.get("lora_target_modules", "all-linear")
    if not isinstance(targets, str):
        targets = [str(t) for t in targets]
    revision = hp.get("base_revision")
    quant = hp.get("gguf_quant")
    return _Options(
        base_model=str(hp.get("base_model") or BASE_MODEL),
        base_revision=str(revision) if revision not in (None, "") else None,
        epochs=int(hp.get("epochs", 3)),
        learning_rate=float(hp.get("learning_rate", 2e-4)),
        batch_size=int(hp.get("batch_size", 8)),
        grad_accum=max(1, int(hp.get("grad_accum", 1))),
        lora_r=int(hp.get("lora_r", 8)),
        lora_alpha=int(hp.get("lora_alpha", 16)),
        lora_dropout=float(hp.get("lora_dropout", 0.05)),
        lora_target_modules=targets,
        max_seq_length=max_seq_length,
        precision=precision,
        prompt_template=template,
        gguf_quant=str(quant) if quant not in (None, "") else None,
        init=init_of(hp),
        init_config=init_config_of(hp),
        objective=objective_of(hp),
        dpo_beta=float(hp.get("dpo_beta", DEFAULT_DPO_BETA)),
        gradient_checkpointing=checkpointing_mode(hp),
        **train_eval_options(hp),
    )


def _random_causal_lm(opt: _Options, tokenizer: Any, device: str) -> Any:
    """``init: random``: the architecture of ``base_model`` built from its config with fresh weights; no
    checkpoint is read. ``init_config`` sets config fields first (``num_hidden_layers``, ``hidden_size``,
    ``num_attention_heads``, ...), so a small transformer can be trained from scratch. The vocabulary size
    and the special token ids are the tokenizer's."""
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM

    kwargs: dict[str, Any] = {"revision": opt.base_revision} if opt.base_revision else {}
    config = AutoConfig.from_pretrained(opt.base_model, trust_remote_code=False, **kwargs)
    for key, value in (opt.init_config or {}).items():
        if not hasattr(config, key):
            raise ValueError(f"init_config.{key}: {type(config).__name__} has no such field")
        setattr(config, key, value)
    config.vocab_size = len(tokenizer)
    for name in ("pad_token_id", "bos_token_id", "eos_token_id"):
        if getattr(tokenizer, name, None) is not None:
            setattr(config, name, getattr(tokenizer, name))
    model = AutoModelForCausalLM.from_config(config)  # type: ignore[no-untyped-call,unused-ignore]
    return model.to(device=device, dtype=torch.float32 if opt.precision == "fp32" else torch.bfloat16)


def _load_base_model(source: str, revision: str | None, precision: str, device: str) -> Any:
    """The causal LM to put LoRA on. fp32/bf16: dense weights on ``device``.
    4bit: bitsandbytes NF4 (double quant, bf16 compute) + peft's k-bit prep --
    the QLoRA recipe the legacy runner ran through unsloth's ``load_in_4bit``."""
    # The clearest failure first: a 4bit request on a worker without
    # bitsandbytes must say so, not trip over whatever import comes next.
    if precision == "4bit":
        _require_bitsandbytes()
    import torch
    from transformers import AutoModelForCausalLM

    kwargs: dict[str, Any] = {"revision": revision} if revision else {}
    if precision == "4bit":
        from peft import prepare_model_for_kbit_training
        from transformers import BitsAndBytesConfig

        quant = BitsAndBytesConfig(  # type: ignore[no-untyped-call,unused-ignore]
            load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16)
        model = AutoModelForCausalLM.from_pretrained(source, quantization_config=quant,
                                                     device_map={"": device}, **kwargs)
        return prepare_model_for_kbit_training(model)  # type: ignore[no-untyped-call,unused-ignore]
    dtype = torch.float32 if precision == "fp32" else torch.bfloat16
    return AutoModelForCausalLM.from_pretrained(source, dtype=dtype, **kwargs).to(device)


def _load_teacher(ref: str, tokenizer: Any, device: str) -> tuple[Any, str | None]:
    """``(frozen teacher causal LM, its commit)``. The teacher must share the student's vocabulary: the two
    next-token distributions are compared token by token."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from sakura.atelier.backends.distill import check_same_vocabulary, freeze

    source, revision = teacher_source(ref)
    kwargs: dict[str, Any] = {"revision": revision} if revision else {}
    check_same_vocabulary(tokenizer, AutoTokenizer.from_pretrained(source, **kwargs), ref)
    model = AutoModelForCausalLM.from_pretrained(source, dtype=torch.float32, **kwargs).to(device)
    return freeze(model), getattr(getattr(model, "config", None), "_commit_hash", None)


def _distil_step(out_: Any, t_model: Any, batch: dict[str, Any], teacher: Teacher) -> tuple[Any, Any]:
    """``(blended loss, soft term)`` of one batch: the temperature-scaled KL to the teacher's next-token
    distribution on the completion's tokens (the positions the hard loss is taken on), blended with the
    next-token cross-entropy."""
    import torch

    from sakura.atelier.backends.distill import blend, kd_loss

    with torch.no_grad():
        t_logits = t_model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]).logits
    scored = batch["labels"][:, 1:] != -100  # position i predicts token i + 1
    soft = kd_loss(out_.logits[:, :-1][scored], t_logits[:, :-1][scored], teacher.temperature)
    return blend(_reply_loss(out_.logits, batch["labels"]), soft, teacher.alpha), soft


def _token_agreement(model: Any, t_model: Any, batches: list[dict[str, Any]]) -> float:
    """The fraction of completion tokens for which the student's most likely next token is the teacher's."""
    import torch

    same = total = 0
    model.eval()
    with torch.no_grad():
        for batch in batches:
            scored = batch["labels"][:, 1:] != -100
            kwargs = {"input_ids": batch["input_ids"], "attention_mask": batch["attention_mask"]}
            mine = model(**kwargs).logits[:, :-1][scored].argmax(-1)
            theirs = t_model(**kwargs).logits[:, :-1][scored].argmax(-1)
            same += int((mine == theirs).sum())
            total += int(scored.sum())
    return same / total if total else 0.0


def _artifact_settings(artifact: Artifact) -> dict[str, Any]:
    """``preprocess.json`` written by ``train`` next to the artifacts (CONTRACTS
    §2.1): the prompt template and base the run used. Missing (an artifact from
    before this file existed) = raw prompts, the old behaviour."""
    path = artifact.path.parent / "preprocess.json"
    if not path.is_file():
        return {}
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return loaded if isinstance(loaded, dict) else {}


class MLXUnavailableError(RuntimeError):
    """mlx/mlx-lm does not run on this host -- raised with the precise cause,
    never silently skipped without saying why."""


class MLXExportError(RuntimeError):
    """mlx_lm convert failed on a host that does run MLX -- carries its
    stderr. Distinct from MLXUnavailableError (wrong host)."""


class GGUFExportError(RuntimeError):
    """llama.cpp conversion/quantisation failed -- carries the subprocess's
    own stderr."""


@dataclass
class _Fitted:
    model_dir: Path  # merged HF checkpoint (safetensors) on disk
    tokenizer_dir: Path
    eval_examples: list[dict[str, str]]  # validation prompts+expected completions (untemplated)
    prompt_template: str = "raw"
    gguf_quant: str | None = None  # hp.gguf_quant; None = SAKURA_GGUF_QUANT env, then Q8_0
    scoring: dict[str, Any] | None = None  # preprocess.json `eval`; None = 8 tokens, exact match, first line
    decision: dict[str, Any] | None = None  # task decision: the decision_config written beside the model
    predictions: Path | None = None  # predictions.jsonl of an extended text-generation run


def _device() -> str:
    """CUDA on the Linux fleet, Metal (MPS) on Apple silicon -- the only place
    an MLX export can be produced -- else CPU."""
    import torch

    if torch.cuda.is_available():
        return "cuda"
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return "mps"
    return "cpu"


def _hf_load(model_dir: Path) -> Any:
    """``(tokenizer, model, device)`` of a saved HF model dir (safetensors), loaded once per process."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    key = f"hf:{model_dir}"
    if key not in _LOADED:  # once per process (`serve`)
        # On the device the backend trains on: generating on CPU in float32
        # made re-scoring 50 prompts take 28 minutes on an M5 (training: 1).
        # bf16 on CUDA halves the footprint; a GPU another process already
        # fills falls back to CPU rather than failing the re-score.
        device = _device()
        bf16 = device == "cuda" and torch.cuda.is_bf16_supported()
        model = AutoModelForCausalLM.from_pretrained(str(model_dir), dtype=torch.bfloat16 if bf16 else torch.float32)
        try:
            model = model.to(device)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            model, device = model.to(device="cpu", dtype=torch.float32), "cpu"
        _LOADED[key] = (AutoTokenizer.from_pretrained(str(model_dir)), model, device)
    return _LOADED[key]


def _hf_generate(model_dir: Path, prompt: str, max_tokens: int) -> str:
    """Greedy raw completion from a saved HF model dir (safetensors)."""
    tok, model, device = _hf_load(model_dir)
    enc = {k: v.to(device) for k, v in tok(prompt, return_tensors="pt").items()}
    gen = model.generate(**enc, max_new_tokens=max_tokens, do_sample=False, pad_token_id=tok.pad_token_id)
    return str(tok.decode(gen[0][enc["input_ids"].shape[1]:], skip_special_tokens=True))


def _hf_generate_many(model_dir: Path, prompts: list[str], max_tokens: int, batch_size: int,
                      batch_tokens: int | None = None, max_len: int = 0) -> list[str]:
    """Greedy raw completions from a saved HF model dir, at most ``batch_size`` prompts (and ``batch_tokens``
    tokens) at a time, prompts of similar length together, returned in the order given."""
    tok, model, device = _hf_load(model_dir)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    texts = _generate_bounded(model, tok, prompts, max_tokens, device, max_rows=batch_size, max_tokens=batch_tokens,
                              max_len=max_len)
    return [text or "" for text in texts]  # no row is skipped here (no `skip`): every text is a string


def _mlx_generate(model_dir: Path, prompt: str, max_tokens: int) -> str:
    """Greedy raw completion through MLX itself -- what a developer shipping
    the MLX export would run, so its score is that runtime's, not torch's.
    A plain string prompt: mlx_lm applies no chat template to it."""
    from mlx_lm import generate, load
    from mlx_lm.sample_utils import make_sampler

    key = f"mlx:{model_dir}"
    if key not in _LOADED:  # once per process (`serve`)
        _LOADED[key] = load(str(model_dir))
    model, tok = _LOADED[key]
    return str(generate(model, tok, prompt=prompt, max_tokens=max_tokens,
                        sampler=make_sampler(temp=0.0), verbose=False))


def _llama_cpp_dir() -> Path:
    d = os.environ.get("LLAMA_CPP_DIR")
    if not d:
        raise GGUFExportError(
            "LLAMA_CPP_DIR is not set -- point it at a llama.cpp checkout with "
            "convert_hf_to_gguf.py, and build/bin/llama-quantize + llama-cli"
        )
    return Path(d)


def _exact_match(predicted: str, expected: str) -> bool:
    return predicted.strip().split("\n")[0].strip() == expected.strip()


def _target_of(ex: dict[str, Any]) -> str:
    """What a prediction for ``ex`` is compared with: the row's ``reference`` (``features.eval.reference``)
    when it has one, else its completion."""
    reference = (ex.get("eval") or {}).get("reference")
    return ex["completion"] if reference is None else str(reference)


def _eval_columns_report(cols: Any) -> dict[str, Any] | None:
    """``features.eval`` as the report states it: the columns the evaluation read, only those the spec names."""
    if cols is None:
        return None
    named = {"reference": cols.reference, "final": cols.final, "group": cols.group, "keep": cols.keep}
    return {key: list(value) if isinstance(value, list) else value for key, value in named.items() if value is not None}


def _groups_right(examples: list[dict[str, Any]], generated: dict[str, Any], metric: str) -> dict[str, bool]:
    """``{group: every scored row of it scored 1.0 on the headline metric}`` for the rows generated from.
    A row with no group value is in no group."""
    values = generated.get("values", {}).get(metric, [])
    right: dict[str, bool] = {}
    for j, value in zip(generated.get("index", []), values):
        group = (examples[j].get("eval") or {}).get("group")
        if group is not None and value is not None:  # a row with no target is not scored on this metric
            right[group] = right.get(group, True) and value >= 1.0 - 1e-9
    return right


def _group_report(column: str, metric: str, examples: list[dict[str, Any]], tuned: dict[str, Any],
                  base: dict[str, Any] | None) -> dict[str, Any]:
    """The per-group aggregates of the headline metric: ``n`` groups scored and ``all_correct`` of them with every
    scored row fully right (1.0), as a share too; the same for the starting model (``baseline_*``) with
    ``baseline: true``."""
    right = _groups_right(examples, tuned, metric)
    report: dict[str, Any] = {"column": column, "metric": metric, "n": len(right),
                              "all_correct": sum(right.values()),
                              "all_correct_share": sum(right.values()) / len(right) if right else 0.0}
    if base is not None:
        base_right = _groups_right(examples, base, metric)
        report.update(baseline_all_correct=sum(base_right.values()),
                      baseline_all_correct_share=sum(base_right.values()) / len(base_right) if base_right else 0.0)
    return report


def _has_target(ex: dict[str, Any]) -> bool:
    """False for an evaluation-only row: a validation record from a spec with ``features.eval`` (it carries an
    ``eval`` entry) that has an empty completion and no reference. Every other row has a target."""
    if "eval" not in ex:
        return True
    return bool(str(ex["completion"]).strip()) or (ex["eval"] or {}).get("reference") is not None


def _skipped_report(gone: dict[str, set[int]], loss_rows: list[int], gen_rows: list[int]) -> dict[str, int]:
    """``{rows_skipped_memory, loss_rows_skipped_memory}`` of one model's evaluation: the rows that ran out of
    memory on their own and were left out; only the counts that are not zero (a run that skipped nothing reports
    exactly what it did before)."""
    counts = {"rows_skipped_memory": len(gone["gen"] & set(gen_rows)),
              "loss_rows_skipped_memory": len(gone["loss"] & set(loss_rows))}
    return {key: n for key, n in counts.items() if n}


def _scored(generated: dict[str, Any], metric: str) -> int:
    """How many generated rows have a score for ``metric``: a row skipped for memory has no prediction, and a row
    with no target has no score of a metric that needs one (None)."""
    return sum(value is not None for value in generated.get("values", {}).get(metric, []))


def _prediction_lines(examples: list[dict[str, Any]], metric_names: list[str], tuned: dict[str, Any],
                      base: dict[str, Any] | None) -> list[str]:
    """``predictions.jsonl``: one JSON line per row of the final evaluation, in row order. The row's index in the
    validation set, its ``group``, ``final`` flag and ``keep`` fields (``features.eval``), whether it has a target,
    the tuned model's prediction (capped at :data:`PREDICTION_CHARS_CAP` characters) with each metric's score
    (null = not scored: no target), and the same for the starting model with ``baseline``. No prompt and no
    completion text."""
    def side(generated: dict[str, Any]) -> dict[int, tuple[str, dict[str, Any]]]:
        index = generated.get("index", [])
        return {j: (generated["texts"][k], {m: generated["values"][m][k] for m in metric_names})
                for k, j in enumerate(index)}

    tuned_rows, base_rows = side(tuned), side(base) if base is not None else {}
    tuned_skipped, base_skipped = set(tuned.get("skipped", [])), set(base.get("skipped", [])) if base else set()
    lines = []
    for j in sorted({*tuned_rows, *tuned_skipped}):
        fields = examples[j].get("eval") or {}
        row: dict[str, Any] = {"index": j, "group": fields.get("group"), "final": fields.get("final"),
                               "keep": fields.get("keep", {}), "has_target": _has_target(examples[j])}
        if j in tuned_rows:
            text, scores = tuned_rows[j]
            row.update(prediction=text[:PREDICTION_CHARS_CAP], prediction_chars=len(text), scores=scores)
        else:  # the row did not fit in memory on its own: no prediction, no scores
            row["skipped"] = "memory"
        if j in base_skipped:
            row["baseline_skipped"] = "memory"
        if j in base_rows:
            base_text, base_scores = base_rows[j]
            row.update(baseline_prediction=base_text[:PREDICTION_CHARS_CAP], baseline_prediction_chars=len(base_text),
                       baseline_scores=base_scores)
        lines.append(json.dumps(row, ensure_ascii=False))
    return lines


def _eval_order(n: int, seed: int) -> list[int]:
    """The one seeded order of the ``n`` validation rows every evaluation subset is the first rows of (independent of
    the training shuffle)."""
    order = list(range(n))
    random.Random(f"eval:{seed}").shuffle(order)
    return order


def _evaluation_rows(opt: _Options, n: int, seed: int,
                     eligible: list[int] | None = None) -> tuple[list[int], list[int] | None, list[int]]:
    """Which of the ``n`` validation rows the evaluations read: ``(loss rows, generation rows)`` of a periodic
    evaluation, and the final evaluation's generation rows.

    Without ``eval_max_rows``, ``gen_eval_rows`` and ``final_gen_rows`` that is every row, three times. With
    them, one order of the rows is drawn once with the seed (independent of the training shuffle) and each
    subset is its first rows, so the subsets nest and stay the same for the whole run. Generation rows of None
    mean a periodic evaluation generates nothing (``gen_eval_rows: 0``).

    ``eligible`` (``eval_scope: final``) are the rows generation may be scored on: the generation subsets are
    drawn from them, in the same seeded order, so ``gen_eval_rows`` and ``final_gen_rows`` count rows that are
    scored. The loss rows are every row either way."""
    every = list(range(n))
    order = _eval_order(n, seed)

    def first(limit: int | None) -> list[int]:
        return every if limit is None or limit >= n else order[:limit]

    loss_rows = first(opt.eval_max_rows)
    if eligible is not None:
        keep = set(eligible)
        pool = [j for j in order if j in keep]

        def first_eligible(limit: int | None) -> list[int]:
            return sorted(pool) if limit is None or limit >= len(pool) else pool[:limit]

        periodic = [j for j in loss_rows if j in keep] if opt.gen_eval_rows is None else (
            None if opt.gen_eval_rows == 0 else first_eligible(opt.gen_eval_rows))
        return loss_rows, periodic, first_eligible(opt.final_gen_rows)
    if opt.gen_eval_rows is None:
        gen_rows: list[int] | None = loss_rows
    elif opt.gen_eval_rows == 0:
        gen_rows = None
    else:
        gen_rows = first(min(opt.gen_eval_rows, len(loss_rows)))
    return loss_rows, gen_rows, first(opt.final_gen_rows)


def _eval_settings(opt: _Options, tokenizer: Any, targets: list[str],
                   replies: list[str] | None = None) -> dict[str, Any]:
    """What this run's evaluations generate and score: ``{metric (the headline), metrics, max_new_tokens,
    exact_match_rule}``. ``metrics: auto`` and ``eval_max_new_tokens: auto`` are settled here, from the
    validation targets and their length under the model's tokenizer. ``targets`` are what predictions are
    compared with (a row's reference, else its completion); ``replies`` (default: the targets) are the
    completions, whose length sets ``eval_max_new_tokens: auto``: a reference says what a right reply must
    satisfy and is not as long as the reply."""
    counts: list[int] = []
    if opt.metrics == "auto" or opt.eval_max_new_tokens == "auto":
        counts = [len(tokenizer(target, add_special_tokens=False)["input_ids"]) for target in targets]
    names = text_metrics.auto_metrics(targets, counts) if opt.metrics == "auto" else list(opt.metrics)
    if opt.eval_max_new_tokens == "auto":
        if replies is not None:
            counts = [len(tokenizer(reply, add_special_tokens=False)["input_ids"]) for reply in replies]
        new_tokens = text_metrics.auto_max_new_tokens(counts)
    else:
        new_tokens = int(opt.eval_max_new_tokens)
    if opt.selection_metric not in ("loss", *names):
        raise ValueError(f"selection_metric: {opt.selection_metric} is not a metric this run scores: metrics: auto "
                         f"chose {names} from the validation targets. Name one of them, or loss")
    return {"metric": names[0], "metrics": names, "max_new_tokens": new_tokens,
            "exact_match_rule": "first_line" if new_tokens == LEGACY_NEW_TOKENS else "whole"}


def _scoring_of(settings: dict[str, Any]) -> dict[str, Any] | None:
    """The ``eval`` block of a run's ``preprocess.json``: how its exports are re-scored. None for an artifact
    without one (a run from before the block, or one that set none of the evaluation keys): 8 tokens,
    ``exact_match``, first line."""
    scoring = settings.get("eval")
    if (isinstance(scoring, dict) and scoring.get("metric") in TEXT_METRICS
            and isinstance(scoring.get("max_new_tokens"), int) and scoring["max_new_tokens"] > 0):
        return scoring
    return None


def _length_grouped_batches(order: list[int], lengths: list[int], batch_size: int,
                            rng: random.Random) -> list[list[int]]:
    """``group_by_length``: the epoch's shuffled rows as batches of similar token length.

    ``order`` is cut into pools of 50 batches, each pool sorted by length and cut into batches, and the batches
    shuffled with ``rng``. The batch that holds the longest row runs first, so a batch that does not fit in
    memory fails at step 0 and not hours in. As many batches as without grouping."""
    pool = 50 * batch_size
    batches: list[list[int]] = []
    for start in range(0, len(order), pool):
        rows = sorted(order[start:start + pool], key=lambda j: lengths[j], reverse=True)
        batches += [rows[i:i + batch_size] for i in range(0, len(rows), batch_size)]
    rng.shuffle(batches)
    if batches:
        longest = max(range(len(batches)), key=lambda b: max(lengths[j] for j in batches[b]))
        batches.insert(0, batches.pop(longest))
    return batches


def _model_inputs(batch: dict[str, Any]) -> dict[str, Any]:
    """What the model is called with: the tokens and the mask, never the labels. With labels the model builds
    its own loss over every position (rows x tokens x vocabulary, in float32), which is what ran a 10 GiB
    card out of memory at 4,096 tokens; the loss is taken here, on the reply positions only."""
    return {"input_ids": batch["input_ids"], "attention_mask": batch["attention_mask"]}


def _reply_positions(logits: Any, labels: Any) -> tuple[Any, Any, Any]:
    """``(scored, logits, targets)``: the positions that carry a label (position i predicts token i + 1), the
    logits there as ``[n, vocabulary]`` and the labels they predict. Only these rows are made float32."""
    shifted = labels[:, 1:]
    scored = shifted != -100
    return scored, logits[:, :-1][scored].float(), shifted[scored]


def _reply_loss(logits: Any, labels: Any) -> Any:
    """The mean next-token cross-entropy over the reply tokens of the whole batch (what the model's own loss
    was), in float32 on the selected positions only."""
    import torch

    _, picked, targets = _reply_positions(logits, labels)
    return torch.nn.functional.cross_entropy(picked, targets)


def _row_losses(net: Any, batch: dict[str, Any]) -> list[float]:
    """Each row's mean next-token loss on its completion, read off the logits at the reply positions only."""
    import torch

    logits = net(**_model_inputs(batch)).logits
    scored, picked, targets = _reply_positions(logits, batch["labels"])
    del logits
    per_token = torch.nn.functional.cross_entropy(picked, targets, reduction="none")
    rows = scored.nonzero()[:, 0]
    total = torch.zeros(scored.shape[0], dtype=per_token.dtype, device=per_token.device).index_add_(0, rows, per_token)
    return [float(x) for x in total / scored.sum(dim=1)]


def _is_oom(exc: BaseException) -> bool:
    """Whether ``exc`` is the GPU running out of memory: ``torch.OutOfMemoryError``, or the ``RuntimeError``
    whose message starts with "CUDA out of memory" that older torch raised."""
    import torch

    kind = getattr(torch, "OutOfMemoryError", None)
    if kind is not None and isinstance(exc, kind):
        return True
    return isinstance(exc, RuntimeError) and str(exc).startswith("CUDA out of memory")


def _release() -> None:
    """Let go of what a finished or failed batch held, so the next one starts from a clean allocator."""
    import gc

    import torch

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _held() -> str:
    """What this process holds on the card, in words ("" off CUDA): the memory allocated to tensors and the
    memory reserved by the allocator, in GiB. Read before ``_release``, while the failed batch is still held."""
    import torch

    if not torch.cuda.is_available():
        return ""
    return (f"the process held {torch.cuda.memory_allocated() / 2**30:.1f} GiB in tensors and "
            f"{torch.cuda.memory_reserved() / 2**30:.1f} GiB reserved")


def _card() -> str:
    import torch

    if not torch.cuda.is_available():
        return ""
    return f" ({torch.cuda.get_device_properties(0).total_memory / 2**30:.1f} GiB)"


def _same_random_state(device: str) -> Any:
    """A context that puts the random state of the CPU and of ``device`` back as it found it."""
    import torch

    @contextlib.contextmanager
    def kept() -> Any:
        mps = torch.mps.get_rng_state() if device == "mps" else None
        with torch.random.fork_rng(devices=[torch.cuda.current_device()] if torch.cuda.is_available() else []):
            yield
        if mps is not None:
            torch.mps.set_rng_state(mps)

    return kept()


def _no_room(what: str, tokens: int, max_len: int, held: str = "", checkpointing: str = "") -> RuntimeError:
    """The error for a batch of one row that still runs out of memory, in words. ``held`` is ``_held()``;
    ``checkpointing`` says what became of gradient checkpointing, when it was tried or could not be."""
    seen = f" ({held})" if held else ""
    return RuntimeError(
        f"{what} of {tokens} tokens does not fit in the memory of this card{_card()}, even on its own"
        f"{checkpointing}, with max_seq_length {max_len}{seen}. Lower max_seq_length, or run on a card with "
        "more memory.")


class _Resources:
    """What a run learns about the card: the out-of-memory back-offs (the phase, the optimiser step and the
    rows of the batch that failed) and the peak memory of each phase (None off CUDA)."""

    PHASES = ("baseline", "training", "evaluation")

    def __init__(self) -> None:
        self.backoffs: list[dict[str, Any]] = []
        self.peaks: dict[str, int | None] = dict.fromkeys(self.PHASES)
        self.phase: str | None = None
        self.step = 0

    def enter(self, phase: str | None) -> str | None:
        """Close the current phase (keeping its peak) and open ``phase``; returns the one that was open."""
        import torch

        before = self.phase
        if torch.cuda.is_available():
            if before is not None:
                self.peaks[before] = max(self.peaks[before] or 0, int(torch.cuda.max_memory_allocated()))
            torch.cuda.reset_peak_memory_stats()
        self.phase = phase
        return before

    def backoff(self, rows: int, action: str = "halve") -> None:
        self.backoffs.append({"phase": self.phase, "step": self.step, "batch_rows": rows, "action": action})


def _token_batches(items: list[Any], length: Any, max_rows: int, max_tokens: int | None) -> list[list[Any]]:
    """``items`` longest first, cut into batches of at most ``max_rows`` rows and, when ``max_tokens`` is given,
    at most ``max_tokens`` tokens counted as rows x the longest row of the batch (its first). Sorted by
    length, so a batch pads little; one row alone is always a batch, however long."""
    order = sorted(items, key=lambda i: -length(i))
    batches: list[list[Any]] = []
    at = 0
    while at < len(order):
        rows = max_rows if max_tokens is None else min(max_rows, max(1, max_tokens // max(1, length(order[at]))))
        batches.append(order[at:at + rows])
        at += rows
    return batches


def _halving(chunk: list[Any], run: Any, resources: _Resources | None, no_room: Any,
             skip: Any = None) -> list[Any]:
    """``run(chunk)`` (one result per item); when the GPU runs out of memory, the batch is halved and each half
    retried, down to one row, which raises ``no_room(item, held)``. With ``skip``, a row that does not fit on
    its own is passed to ``skip(item, held)`` instead, counted as a back-off with action ``skip``, and its
    result is None."""
    try:
        return list(run(chunk))
    except BaseException as exc:  # noqa: BLE001 - anything else is re-raised at once
        if not _is_oom(exc):
            raise
    held = _held()
    # outside the handler: the failed batch's tensors, held by the traceback, are gone before the cache is emptied
    _release()
    if len(chunk) == 1:
        if skip is None:
            raise no_room(chunk[0], held)
        if resources is not None:
            resources.backoff(1, "skip")
        skip(chunk[0], held)
        return [None]
    if resources is not None:
        resources.backoff(len(chunk))
    if held:
        print(f"a batch of {len(chunk)} rows did not fit in memory ({held}); halving it", file=sys.stderr)
    half = (len(chunk) + 1) // 2
    return (_halving(chunk[:half], run, resources, no_room, skip)
            + _halving(chunk[half:], run, resources, no_room, skip))


def _generate_bounded(net: Any, tokenizer: Any, prompts: list[str], max_new_tokens: int, device: str, *,
                      max_rows: int, max_tokens: int | None = None, max_len: int = 0,
                      resources: _Resources | None = None, skip: Any = None) -> list[str | None]:
    """Greedy completions of ``prompts``, in order, in batches bounded in rows and in tokens (the prompt's tokens
    plus the ones to generate, rows x the longest), longest first; a batch that runs out of memory is halved. With
    ``skip``, a prompt that does not fit on its own is passed to ``skip(index, held)`` and its text is None."""
    lengths = [len(tokenizer(p)["input_ids"]) + max_new_tokens for p in prompts]
    texts: list[str | None] = [""] * len(prompts)

    def run(chunk: list[int]) -> list[str]:
        return _generate(net, tokenizer, [prompts[i] for i in chunk], max_new_tokens, device)

    for chunk in _token_batches(list(range(len(prompts))), lengths.__getitem__, max_rows, max_tokens):
        for i, text in zip(chunk, _halving(chunk, run, resources,
                                           lambda i, held: _no_room("a prompt to generate from", lengths[i], max_len, held),
                                           skip)):
            texts[i] = text
    return texts


def _generate(net: Any, tokenizer: Any, prompts: list[str], max_new_tokens: int, device: str) -> list[str]:
    """Greedy completions of ``prompts``, in order. One prompt is generated as it always was; several go as
    one batch, padded on the left so every row ends where its completion starts (the tokenizer's padding
    side is put back afterwards: training pads on the right, by hand)."""
    if len(prompts) == 1:
        ids = tokenizer(prompts[0], return_tensors="pt")["input_ids"].to(device)
        gen = net.generate(input_ids=ids, max_new_tokens=max_new_tokens, do_sample=False,
                           pad_token_id=tokenizer.pad_token_id)
        return [str(tokenizer.decode(gen[0][ids.shape[1]:], skip_special_tokens=True))]
    side = tokenizer.padding_side
    tokenizer.padding_side = "left"
    try:
        enc = tokenizer(prompts, return_tensors="pt", padding=True)
    finally:
        tokenizer.padding_side = side
    ids, mask = enc["input_ids"].to(device), enc["attention_mask"].to(device)
    gen = net.generate(input_ids=ids, attention_mask=mask, max_new_tokens=max_new_tokens, do_sample=False,
                       pad_token_id=tokenizer.pad_token_id)
    return [str(tokenizer.decode(row[ids.shape[1]:], skip_special_tokens=True)) for row in gen]


def _keep_weights(model: Any, full_training: bool, checkpoint_dir: Path) -> Any:
    """``keep_best``: a copy of the weights being trained. A LoRA's adapter tensors are small and stay in
    memory; a model trained in full (``init: random``) is written to ``checkpoint/best.pt`` instead."""
    import torch

    if full_training:
        path = checkpoint_dir / "best.pt"
        torch.save({k: v.detach().cpu() for k, v in model.state_dict().items()}, path)
        return path
    return {k: v.detach().cpu().clone() for k, v in model.named_parameters() if v.requires_grad}


def _restore_weights(model: Any, kept: Any, load: bool) -> None:
    """Put the weights ``_keep_weights`` kept back into ``model`` when ``load``; the file copy is removed
    either way (it is not part of the checkpoint)."""
    import torch

    if isinstance(kept, Path):
        if load:
            model.load_state_dict(torch.load(kept, weights_only=True))
        kept.unlink(missing_ok=True)
    elif load:
        model.load_state_dict(kept, strict=False)


#: The smallest context a GGUF is loaded with (what every GGUF got before `_gguf_n_ctx`).
_GGUF_MIN_CTX = 256


def _gguf_n_ctx(artifact: Artifact) -> int:
    """The context window a GGUF is loaded with: the longest sequence training fed the
    model (``max_seq_length`` in preprocess.json) and as much again for the reply, or the
    length the run's evaluations generate (``eval.max_new_tokens``) when that is longer; never
    under ``_GGUF_MIN_CTX``. A fixed 256 refused every longer prompt: the GGUF check failed
    the run after training, and so would every such ``/infer`` call."""
    settings = _artifact_settings(artifact)
    seq = settings.get("max_seq_length")
    if isinstance(seq, int) and not isinstance(seq, bool) and seq > 0:
        scoring = _scoring_of(settings)
        reply = max(seq, int(scoring["max_new_tokens"])) if scoring else seq
        return max(_GGUF_MIN_CTX, seq + reply)
    return _GGUF_MIN_CTX


def _load_llama_cpp(model_path: Path, n_ctx: int) -> Any:
    """Load a GGUF for raw-completion inference via llama-cpp-python -- no
    chat template, so the model sees exactly the training-time prompt
    string. Caller handles ImportError/load failures."""
    from llama_cpp import Llama

    key = str(model_path)
    if key not in _LOADED:  # once per process (`serve`)
        _LOADED[key] = Llama(model_path=key, n_ctx=n_ctx, verbose=False)
    return _LOADED[key]


_LOADED: dict[str, Any] = {}


class PEFTBackend:
    name = "peft"
    tasks = frozenset({"text_generation", "decision"})
    licence = LICENSE

    #: Set below, once the methods exist (see NOTEBOOK_CELLS at the end of the module).
    NOTEBOOK_CELLS: ClassVar[tuple[Any, ...]]

    def train(
        self,
        spec: ResolvedSpec,
        data: MaterialisedData,
        out: RunDir,
        metrics: MetricsSink,
        resume: Checkpoint | None,
    ) -> TrainResult:
        if spec.spec.task == "decision":
            return self._train_decision(spec, data, out, metrics, resume)
        if objective_of(spec.hyperparameters) == "dpo":
            return self._train_dpo(spec, data, out, metrics, resume)
        import torch
        from peft import LoraConfig, get_peft_model
        from transformers import AutoTokenizer

        out.ensure()
        opt = _training_options(spec.hyperparameters)
        template = opt.prompt_template
        device = _device()
        tok_kwargs: dict[str, Any] = {"revision": opt.base_revision} if opt.base_revision else {}
        tokenizer = AutoTokenizer.from_pretrained(opt.base_model, **tok_kwargs)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        train_examples = list(data.train)
        val_examples = list(data.validation)
        max_len = opt.max_seq_length
        eval_cols = spec.spec.features.eval if spec.spec.features is not None else None
        # features.eval: a validation row with an empty completion is evaluation-only (the reply it asks for has no
        # reference text): no loss, generated from at the final evaluation and for the baseline. A training row
        # with an empty completion has nothing to learn from: dropped and counted. Without features.eval an
        # empty completion is a row like any other, as before.
        train_rows_in = len(train_examples)
        dropped_empty = 0
        eval_only: set[int] = set()
        if eval_cols is not None:
            kept_rows = [ex for ex in train_examples if str(ex["completion"]).strip()]
            dropped_empty, train_examples = len(train_examples) - len(kept_rows), kept_rows
            eval_only = {j for j, ex in enumerate(val_examples) if not str(ex["completion"]).strip()}

        def encode(ex: dict[str, str]) -> dict[str, torch.Tensor]:
            prompt = _format_prompt(template, ex["prompt"])
            text = prompt + ex["completion"] + tokenizer.eos_token
            enc = tokenizer(text, truncation=True, max_length=max_len, return_tensors="pt")
            prompt_len = len(tokenizer(prompt, truncation=True, max_length=max_len)["input_ids"])
            labels = enc["input_ids"].clone()
            labels[:, :prompt_len] = -100  # loss only on the completion
            labels[enc["attention_mask"] == 0] = -100
            return {"input_ids": enc["input_ids"], "attention_mask": enc["attention_mask"], "labels": labels}

        def measure(ex: dict[str, str]) -> tuple[bool, bool]:
            """``(the reply lost tokens to max_seq_length, no reply token is left)`` for one row."""
            text = _format_prompt(template, ex["prompt"]) + ex["completion"] + tokenizer.eos_token
            cut = len(tokenizer(text, verbose=False)["input_ids"]) > max_len
            return cut, bool(cut and int((encode(ex)["labels"] != -100).sum()) == 0)

        # Replies cut by max_seq_length: counted once, here, for the training and the validation rows. A row with
        # no reply token left has every label at -100 (its loss is 0 / 0), so it stays out of the batches and of
        # the loss evaluation.
        flags = {"train": [measure(ex) for ex in train_examples],
                 "validation": [(False, False) if j in eval_only else measure(ex) for j, ex in enumerate(val_examples)]}
        truncation: dict[str, Any] = {"max_seq_length": max_len}
        for split, rows, rows_in, apart in (("train", train_examples, train_rows_in, dropped_empty),
                                            ("validation", val_examples, len(val_examples), len(eval_only))):
            cuts = flags[split]
            truncation[split] = {"rows": rows_in, "rows_truncated": sum(c for c, _ in cuts),
                                 "rows_without_target": sum(n for _, n in cuts)}
            if eval_cols is not None:  # rows left out for having no completion, apart from those cut by the window
                truncation[split]["rows_dropped_empty" if split == "train" else "rows_eval_only"] = apart
            truncation[split]["rows_used"] = len(rows) - sum(n for _, n in cuts) - (apart if split == "validation" else 0)
        bare = truncation["train"]["rows_without_target"]
        if eval_cols is not None and not train_examples:
            raise ValueError(f"none of the {train_rows_in} training rows has a completion: an empty completion is "
                             "dropped, so there is nothing to train on")
        if 2 * bare > len(train_examples):
            raise ValueError(
                f"{bare} of the {len(train_examples)} training rows have no reply left inside max_seq_length "
                f"{max_len}: their prompts alone fill the window, so there is nothing to learn from them. Raise "
                f"max_seq_length, shorten the prompts, or remove these rows from the data.")
        if val_examples and truncation["validation"]["rows_used"] == 0:
            if len(eval_only) == len(val_examples):
                raise ValueError(
                    f"every validation row is evaluation-only (its completion is empty): the loss, and the choice of a "
                    f"checkpoint, are measured on validation rows that have a completion. Keep at least one such row "
                    f"on the validation side ({len(val_examples)} rows have none).")
            raise ValueError(
                f"none of the {len(val_examples)} validation rows has a reply left inside max_seq_length "
                f"{max_len}, so no loss can be measured. Raise max_seq_length or shorten the prompts.")
        if any(c for c, _ in flags["train"] + flags["validation"]):
            print(f"max_seq_length {max_len} cuts the reply of {truncation['train']['rows_truncated']} training and "
                  f"{truncation['validation']['rows_truncated']} validation rows; {bare} training rows with no reply "
                  f"left are not trained on", file=sys.stderr)
        train_examples = [ex for ex, (_, none) in zip(train_examples, flags["train"]) if not none]
        usable_val = [j for j, (_, none) in enumerate(flags["validation"]) if not none and j not in eval_only]

        # A resume continues from the previous run's merged weights, not the base.
        resume_dir = resume.path / "merged" if resume is not None else None
        resumed = bool(resume_dir and resume_dir.exists())
        source = str(resume_dir) if resumed and resume_dir else opt.base_model
        # init: random is a full training of fresh weights, not a LoRA: there is no pretrained base for an
        # adapter to sit on, so every parameter trains and the dense model is what is saved. A resume of
        # such a run continues its saved weights, in full as well.
        full_training = opt.init == "random"
        if full_training and not resumed:
            model = _random_causal_lm(opt, tokenizer, device)
        else:
            model = _load_base_model(source, None if resumed else opt.base_revision, opt.precision, device)
        # transformers records the Hub commit it resolved a repo id to; a local
        # dir (or a resume) has none. That, not the requested revision, is what
        # pins the bytes a replay must fetch.
        base_commit = getattr(getattr(model, "config", None), "_commit_hash", None)

        # teacher: the loss becomes alpha * T^2 * KL(teacher || student) + (1 - alpha) * next-token loss
        teacher = teacher_of(spec.hyperparameters)
        t_model: Any = None
        teacher_commit: str | None = None
        if teacher is not None:
            t_model, teacher_commit = _load_teacher(teacher.ref, tokenizer, device)

        def collate(batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
            # Pad to the longest example in the batch (not to max_seq_length: at
            # 2048 that is almost all padding for a 30-token prompt); padded
            # positions are masked out of attention and loss alike.
            width = max(int(b["input_ids"].shape[1]) for b in batch)
            pad_id = int(tokenizer.pad_token_id)
            ids = torch.full((len(batch), width), pad_id, dtype=torch.long)
            attn = torch.zeros((len(batch), width), dtype=torch.long)
            labels = torch.full((len(batch), width), -100, dtype=torch.long)
            for row, b in enumerate(batch):
                n = int(b["input_ids"].shape[1])
                ids[row, :n] = b["input_ids"][0]
                attn[row, :n] = b["attention_mask"][0]
                labels[row, :n] = b["labels"][0]
            return {"input_ids": ids.to(device), "attention_mask": attn.to(device), "labels": labels.to(device)}

        # What an evaluation generates and scores, and which validation rows it reads: a fixed subset drawn
        # once with the seed (or every row).
        eligible: list[int] | None = None
        if opt.eval_scope == "final":
            if eval_cols is None or eval_cols.final is None:
                raise ValueError("eval_scope: final needs features.eval.final, the column that flags the rows to "
                                 "score (the last turn of each conversation); the spec names none")
            eligible = [j for j, ex in enumerate(val_examples) if (ex.get("eval") or {}).get("final")]
            if not eligible:
                raise ValueError(f"eval_scope: final, but no validation row has final true in column "
                                 f"{eval_cols.final!r}: there is nothing to generate from and score")
        # `auto` reads the rows that have a target: an evaluation-only row's empty completion is no target and no
        # reply length (its reference, when it has one, is a target)
        with_target = [ex for ex in val_examples if _has_target(ex)]
        scoring = _eval_settings(opt, tokenizer, [_target_of(ex) for ex in with_target],
                                 [ex["completion"] for ex in val_examples if "eval" not in ex or str(ex["completion"]).strip()])
        metric_names, max_new_tokens = list(scoring["metrics"]), int(scoring["max_new_tokens"])
        loss_rows, gen_rows, final_rows = _evaluation_rows(opt, len(val_examples), spec.spec.seed, eligible)
        if eval_only:
            # `final_gen_rows` draws only rows that have a target; every evaluation-only row (up to
            # `eval_only_gen_rows`, in the seeded order) is then added, so sampling never drops them
            targets_only = [j for j in (eligible if eligible is not None else range(len(val_examples)))
                            if j not in eval_only]
            final_rows = _evaluation_rows(opt, len(val_examples), spec.spec.seed, targets_only)[2]
            extra = [j for j in _eval_order(len(val_examples), spec.spec.seed) if j in eval_only]
            final_rows = final_rows + extra[:opt.eval_only_gen_rows]
            if set(final_rows) == set(range(len(val_examples))):
                final_rows = list(range(len(val_examples)))
        if eval_only and gen_rows is not None:
            gen_rows = [j for j in gen_rows if j not in eval_only]  # periodic evaluations skip them: the final one reads them
        every_row = list(range(len(val_examples)))
        usable = set(usable_val)
        loss_rows = [j for j in loss_rows if j in usable]  # rows with no reply token have no loss to measure

        resources = _Resources()
        eval_tokens = opt.batch_size * max_len  # most tokens an evaluation or generation batch holds (rows x longest)

        #: rows that ran out of memory on their own, per model scored: never tried again by that model
        no_fit: dict[str, dict[str, set[int]]] = {
            side: {"loss": set(), "gen": set()} for side in ("base", "tuned", "teacher")}
        last_generated: dict[str, Any] = {}  # the latest generation scored: {index: [...], values: {metric: [...]}}

        @torch.no_grad()
        def score(net: Any, loss_idx: list[int], gen_idx: list[int] | None, side: str = "tuned") -> dict[str, float]:
            """``{loss, <metric>...}`` of a model (the student, its teacher, or the starting weights) on
            validation rows: the mean of each row's own next-token loss over ``loss_idx``, and the run's
            metrics on what it generates for ``gen_idx`` (None = generate nothing). A row that runs out of memory
            on its own is left out of both (and counted); ``side`` names the model whose memory of such rows is kept,
            so a row that does not fit is not tried again by this model."""
            net.eval()
            gone = no_fit[side]
            size = opt.eval_batch_size
            # rows of similar length share a batch (longest first), at most `size` rows and `eval_tokens` tokens
            # (rows x the longest); a batch that does not fit is halved. Each row's loss is its own, so the
            # order the rows ran in changes nothing: the mean is taken in the order given.
            loss_run = [j for j in loss_idx if j not in gone["loss"]]
            encoded = {j: encode(val_examples[j]) for j in loss_run}
            by_row: dict[int, float | None] = {}

            def drop_loss(j: int, held: str) -> None:
                gone["loss"].add(j)
                print(f"a validation row of {int(encoded[j]['input_ids'].shape[1])} tokens does not fit in memory "
                      f"on its own{'; ' + held if held else ''}; its loss is left out ({side})", file=sys.stderr)

            for chunk in _token_batches(loss_run, lambda j: int(encoded[j]["input_ids"].shape[1]), size, eval_tokens):
                for j, value in zip(chunk, _halving(
                        chunk, lambda c: _row_losses(net, collate([encoded[j] for j in c])), resources,
                        lambda j, held: _no_room("a validation row", int(encoded[j]["input_ids"].shape[1]), max_len,
                                                 held), drop_loss)):
                    by_row[j] = value
            losses = [v for j in loss_run if (v := by_row[j]) is not None]
            if loss_idx and not losses:
                raise _no_room("the shortest of the validation rows", min(
                    int(encode(val_examples[j])["input_ids"].shape[1]) for j in loss_idx), max_len)
            del encoded
            scores = {"loss": sum(losses) / len(losses) if losses else 0.0}
            if gen_idx is None:
                return scores
            gen_run = [j for j in gen_idx if j not in gone["gen"]]
            prompts = [_format_prompt(template, val_examples[j]["prompt"]) for j in gen_run]

            def drop_gen(i: int, held: str) -> None:
                gone["gen"].add(gen_run[i])
                print(f"a prompt of {len(tokenizer(prompts[i])['input_ids'])} tokens does not fit in memory on "
                      f"its own{'; ' + held if held else ''}; it is left out of the generation ({side})",
                      file=sys.stderr)

            done = _generate_bounded(net, tokenizer, prompts, max_new_tokens, device, max_rows=size,
                                     max_tokens=eval_tokens, max_len=max_len, resources=resources, skip=drop_gen)
            kept = [j for j, text in zip(gen_run, done) if text is not None]
            if gen_idx and not kept:
                raise _no_room("the shortest of the prompts to generate from", min(
                    len(tokenizer(_format_prompt(template, val_examples[j]["prompt"]))["input_ids"]) + max_new_tokens
                    for j in gen_idx), max_len)
            texts = {j: _clean_completion(template, text) for j, text in zip(gen_run, done) if text is not None}
            gen_idx = kept
            per_row = text_metrics.score_rows(
                metric_names, [texts[j] for j in gen_idx], [_target_of(val_examples[j]) for j in gen_idx],
                first_line=scoring["exact_match_rule"] == "first_line",
                scored=[_has_target(val_examples[j]) for j in gen_idx])
            scores.update({name: text_metrics.mean(values) for name, values in per_row.items()})
            # kept: the final rows' texts and scores, row by row (a None is a row with no target for that metric)
            last_generated.update(index=list(gen_idx), values=per_row, texts=[texts[j] for j in gen_idx],
                                  skipped=sorted(gone["gen"]))
            return scores

        weights_dtype = "nf4" if opt.precision == "4bit" else str(next(model.parameters()).dtype).replace("torch.", "")
        if not full_training:
            # rank-stabilised LoRA and DoRA are named only when asked for, so a peft without them still loads
            variants = {name: True for name, on in (("use_rslora", opt.lora_use_rslora),
                                                    ("use_dora", opt.lora_use_dora)) if on}
            lora_cfg = LoraConfig(
                r=opt.lora_r, lora_alpha=opt.lora_alpha, lora_dropout=opt.lora_dropout,
                target_modules=opt.lora_target_modules, task_type="CAUSAL_LM", **variants,
            )
            model = get_peft_model(model, lora_cfg)

        checkpointing = False  # gradient checkpointing in effect: recompute activations in the backward pass

        def turn_on_checkpointing() -> str | None:
            """Turn gradient checkpointing on, as the decision path does. None when it is on; else why it could not be."""
            nonlocal checkpointing
            try:
                model.gradient_checkpointing_enable()
                # The frozen base gives the checkpointed blocks no input that requires grad; without this the
                # adapter under them would get no gradient.
                model.enable_input_require_grads()
            except Exception as exc:  # noqa: BLE001 - a model class that cannot do it says so in its own way
                return f"{type(model).__name__} does not support gradient checkpointing ({exc})"
            checkpointing = True
            return None

        if opt.gradient_checkpointing == "true":
            failed = turn_on_checkpointing()
            if failed:
                raise RuntimeError(f"gradient_checkpointing is true, but {failed}")

        lengths = [int(encode(ex)["input_ids"].shape[1]) for ex in train_examples]

        def fits(row: int) -> str | None:
            """One training row forward and backward: None when it fits, else what the process held. The random
            state is put back and the gradients dropped, so the run trains exactly as it would have without it."""
            model.train()
            batch = out_ = loss = None
            held: str | None = None
            with _same_random_state(device):
                try:
                    batch = collate([encode(train_examples[row])])
                    out_ = model(**_model_inputs(batch))
                    if teacher is not None:
                        loss = _distil_step(out_, t_model, batch, teacher)[0]
                    else:
                        loss = _reply_loss(out_.logits, batch["labels"])
                    loss.backward()
                except BaseException as exc:  # noqa: BLE001 - anything but running out of memory is re-raised
                    if not _is_oom(exc):
                        raise
                    held = _held()
            batch = out_ = loss = None
            model.zero_grad(set_to_none=True)
            _release()
            return held

        # The longest training row goes forward and backward once, before the baseline and the evaluations (minutes
        # on long rows): a row that cannot train on this card stops the run in seconds, and with `auto` gradient
        # checkpointing is on before the first batch instead of after it fails.
        if train_examples:
            longest = max(range(len(lengths)), key=lengths.__getitem__)
            resources.enter("training")
            held = fits(longest)
            if held is not None:
                note = " and even with gradient checkpointing on" if checkpointing else ""
                if not checkpointing and opt.gradient_checkpointing == "auto":
                    failed = turn_on_checkpointing()
                    if failed is None:
                        resources.backoff(1, "gradient_checkpointing")
                        print(f"the longest training row, {lengths[longest]} tokens, does not fit in memory"
                              f"{_card()}{'; ' + held if held else ''}; turning gradient checkpointing on and "
                              "trying it again", file=sys.stderr)
                        held = fits(longest)
                        note = " and even with gradient checkpointing on"
                    else:
                        note = f", and gradient checkpointing could not be turned on: {failed}"
                if held is not None:
                    raise _no_room("a training batch of one row", lengths[longest], max_len, held, note)
            resources.enter(None)

        # baseline: the starting weights (the adapter is in place but off), on the final evaluation's rows
        baseline: dict[str, Any] | None = None
        baseline_seconds: float | None = None
        baseline_generated: dict[str, Any] = {}  # the starting weights' per-row scores on the final rows
        if opt.baseline:
            started = time.time()
            resources.enter("baseline")
            with model.disable_adapter() if not full_training else contextlib.nullcontext():
                baseline = {**score(model, usable_val, final_rows, "base"), "n": len(final_rows)}
            baseline.update(_skipped_report(no_fit["base"], usable_val, final_rows))
            baseline["n_scored"] = _scored(last_generated, str(scoring["metric"]))
            resources.enter(None)
            _release()  # training starts from a clean allocator
            baseline_generated = {"index": list(last_generated.get("index", [])),
                                  "values": dict(last_generated.get("values", {})),
                                  "texts": list(last_generated.get("texts", [])),
                                  "skipped": list(last_generated.get("skipped", []))}
            baseline_seconds = time.time() - started

        model.train()
        trainable = [p for p in model.parameters() if p.requires_grad]
        adapter_dtype = str(trainable[0].dtype).replace("torch.", "") if trainable else weights_dtype
        decay: dict[str, Any] = {} if opt.weight_decay is None else {"weight_decay": opt.weight_decay}
        optim = torch.optim.AdamW(trainable, lr=opt.learning_rate, **decay)
        rng = random.Random(spec.spec.seed)
        max_steps = max_steps_of(spec)
        scheduler: Any = None
        if opt.lr_schedule != "constant" or opt.warmup_ratio > 0:
            # planned over the steps the run will take, so a run capped by limits.max_steps still decays fully
            total_steps = optimizer_steps(len(train_examples), opt.batch_size, opt.grad_accum, opt.epochs, max_steps)
            warm = warmup_steps(total_steps, opt.warmup_ratio)
            scheduler = torch.optim.lr_scheduler.LambdaLR(
                optim, lambda s: lr_factor(s, total_steps, opt.lr_schedule, warm, opt.min_lr_ratio))

        lower_is_better = opt.selection_metric == "loss"
        evals: list[dict[str, Any]] = []  # one entry per evaluation, in order
        window: list[float] = []  # the training losses since the last evaluation
        best: dict[str, Any] | None = None
        best_weights: Any = None
        stale = 0  # evaluations in a row that did not improve on the best
        eval_seconds = 0.0  # wall time of the periodic evaluations
        train_tokens = 0  # non-padding tokens fed to the model by the loop

        def evaluate(step: int, epoch: float) -> bool:
            """Score the model where training stands, log it and remember the best. True = patience ran out."""
            nonlocal best, best_weights, stale, eval_seconds
            started = time.time()
            resumed_phase = resources.enter("evaluation")
            scores = score(model, loss_rows, gen_rows)
            resources.enter(resumed_phase)
            _release()
            model.train()
            for name, value in scores.items():
                metrics.log(name, value, step=step, split="validation", epoch=epoch)
            evals.append({"epoch": epoch, "step": step, "validation": scores,
                          "train": {"loss": sum(window) / len(window)} if window else {}})
            window.clear()
            value = scores[opt.selection_metric]
            if best is None or (value < best["value"] if lower_is_better else value > best["value"]):
                best, stale = {"epoch": epoch, "step": step, "value": value}, 0
                if opt.keep_best:
                    best_weights = _keep_weights(model, full_training, out.checkpoint)
            else:
                stale += 1
            eval_seconds += time.time() - started
            return opt.early_stop_patience is not None and stale >= opt.early_stop_patience

        t0 = time.time()
        step = 0  # optimizer steps; with grad_accum 1 (the default) one per batch, as before
        patience_out = False
        watch = WrapUpWatch(out.root, max_steps)
        # The batch and the accumulation as they stand: halved and doubled (rows per optimiser step unchanged) when a
        # batch does not fit in memory. The schedule keeps the steps it was planned for.
        cur_bs, cur_ga = opt.batch_size, opt.grad_accum
        position = 0.0
        resources.enter("training")

        def cut(rows: list[int], size: int) -> list[list[int]]:
            if opt.group_by_length:
                return _length_grouped_batches(rows, lengths, size, rng)
            return [rows[i:i + size] for i in range(0, len(rows), size)]

        for epoch in range(opt.epochs):
            order = list(range(len(train_examples)))
            rng.shuffle(order)
            batches = cut(order, cur_bs)
            # eval_every: an evaluation every `interval` optimiser steps inside the epoch, besides the one at its end
            interval = max(1, round(opt.eval_every * -(-len(batches) // cur_ga))) if opt.eval_every else 0
            bi = base = steps_here = 0  # batches done since the batches were last cut, rows before them, steps taken
            group_from, group_rows, group_n, group_tokens = 0, 0, 0, 0  # the accumulation group in progress
            while bi < len(batches):
                batch_idx = batches[bi]
                position = epoch + (base + bi * cur_bs) / len(order)
                if watch.stop(position, step):
                    break  # wrap-up: stop learning; the partial epoch is scored below
                resources.step = step
                batch = out_ = loss = None
                oom = False
                try:
                    batch = collate([encode(train_examples[j]) for j in batch_idx])
                    out_ = model(**_model_inputs(batch))
                    if teacher is not None:
                        loss, soft = _distil_step(out_, t_model, batch, teacher)
                    else:
                        loss = _reply_loss(out_.logits, batch["labels"])
                    loss_value = float(loss.detach())
                    if not math.isfinite(loss_value):
                        raise RuntimeError(
                            f"the training loss is {loss_value}, not a finite number, at optimiser step {step} "
                            f"({position:.2f} epochs into the run, in epoch {epoch + 1} of {opt.epochs}; learning rate "
                            f"{float(optim.param_groups[0]['lr']):g}); the optimiser was not stepped. Lower learning_rate, "
                            f"set max_grad_norm, or use fp32 precision, then run again.")
                    (loss / cur_ga).backward()
                except BaseException as exc:  # noqa: BLE001 - anything but running out of memory is re-raised
                    if not _is_oom(exc):
                        raise
                    oom = True
                if oom:
                    held = _held()  # before the failed batch is let go
                    # outside the handler, so the failed batch's tensors are released before the cache is emptied
                    batch = out_ = loss = None
                    optim.zero_grad(set_to_none=True)
                    _release()
                    new_bs = max(1, cur_bs // 2)
                    if cur_bs == 1:
                        # the last step: gradient checkpointing (auto), else the run stops
                        row_tokens = max((lengths[j] for j in batch_idx), default=0)
                        note = ""
                        if checkpointing:
                            note = " and even with gradient checkpointing on"
                        elif opt.gradient_checkpointing == "auto":
                            failed = turn_on_checkpointing()
                            if failed is None:
                                resources.backoff(1, "gradient_checkpointing")
                                print(f"a training batch of one row of {row_tokens} tokens did not fit in memory"
                                      f"{_card()}{'; ' + held if held else ''}; turning gradient checkpointing on "
                                      "and trying it again", file=sys.stderr)
                            else:
                                note = f", and gradient checkpointing could not be turned on: {failed}"
                        if not checkpointing or note:
                            raise _no_room("a training batch of one row", row_tokens, max_len, held, note)
                    else:
                        resources.backoff(cur_bs)
                        print(f"a training batch of {cur_bs} rows did not fit in memory"
                              f"{'; ' + held if held else ''}; halving it", file=sys.stderr)
                    # the group in progress is dropped (its gradients are gone) and run again from its first row
                    del window[len(window) - group_n:]
                    train_tokens -= group_tokens
                    rows_left = [j for b in batches[group_from:] for j in b]
                    base = group_rows
                    cur_ga, cur_bs = -(-cur_bs * cur_ga // new_bs), new_bs
                    batches, bi = cut(rows_left, cur_bs), 0
                    group_from, group_n, group_tokens = 0, 0, 0
                    continue
                assert batch is not None  # the forward ran: an out-of-memory batch took the branch above
                batch_tokens = int(batch["attention_mask"].sum())
                train_tokens += batch_tokens
                del batch, out_, loss  # the logits of this batch go before the next forward builds its own
                boundary = (bi + 1) % cur_ga == 0 or bi == len(batches) - 1
                if boundary:
                    if opt.max_grad_norm is not None:
                        torch.nn.utils.clip_grad_norm_(trainable, opt.max_grad_norm)
                    if opt.extended:  # the rate this step uses
                        metrics.log("lr", float(optim.param_groups[0]["lr"]), step=step, split="train", epoch=position)
                    optim.step()
                    if scheduler is not None:
                        scheduler.step()
                    optim.zero_grad()
                window.append(loss_value)
                group_n, group_tokens = group_n + 1, group_tokens + batch_tokens
                metrics.log("loss", window[-1], step=step, split="train", epoch=position)
                if teacher is not None:
                    metrics.log("distill_loss", float(soft.detach()), step=step, split="train", epoch=position)
                    del soft
                bi += 1
                if boundary:
                    step += 1
                    steps_here += 1
                    group_from, group_rows, group_n, group_tokens = bi, base + bi * cur_bs, 0, 0
                    # The end of the epoch is scored anyway: an evaluation less than half an interval before it would
                    # score nearly the same weights twice (steps 1,770 and 1,773 of a 1,773-step epoch at 0.1).
                    steps_left = -(-(len(batches) - bi) // cur_ga)
                    if interval and steps_here % interval == 0 and 2 * steps_left >= interval:
                        patience_out = evaluate(step, epoch + (base + bi * cur_bs) / len(order))
                        if patience_out:
                            break
            if watch.triggered:
                optim.zero_grad()  # drop a half-accumulated step
                if not (opt.extended and evals and evals[-1]["step"] == step):  # not scored at this step yet
                    evaluate(step, position)
                break
            if patience_out or evaluate(step, float(epoch + 1)):
                patience_out = True
                break

        resources.enter(None)
        loop_seconds = time.time() - t0 - eval_seconds
        # keep_best: the weights of the best evaluation, unless the last step is that evaluation
        restored = opt.keep_best and best is not None and best["step"] != step
        if best_weights is not None:
            _restore_weights(model, best_weights, restored)
        # The final evaluation: the loss on every validation row, generation on the final rows. The last
        # evaluation is reused when it already measured these weights on these rows (always, without the keys).
        final_eval_seconds = 0.0  # a reused evaluation costs nothing more: its time is in eval_seconds
        if evals and not restored and (loss_rows, gen_rows) == (usable_val, final_rows):
            final = evals[-1]["validation"]
        else:
            started = time.time()
            resources.enter("evaluation")
            final = score(model, usable_val, final_rows)
            resources.enter(None)
            _release()
            final_eval_seconds = time.time() - started
        final_generated: dict[str, Any] = {"index": list(last_generated.get("index", [])),
                           "values": dict(last_generated.get("values", {})),
                           "texts": list(last_generated.get("texts", [])),
                           "skipped": list(last_generated.get("skipped", []))}  # what `final` was scored on
        val_loss, headline = final["loss"], str(scoring["metric"])
        train_seconds = time.time() - t0

        epoch_metrics: list[dict[str, Any]] | None = None
        early_stop: dict[str, Any] | None = None
        if opt.extended and best is not None:
            epoch_metrics = [{"epoch": round(e["epoch"], 6), "step": e["step"],
                              **{split: {k: round(v, 6) for k, v in e[split].items()} for split in ("train", "validation")}}
                             for e in evals]
            reason = (watch.stopped_early or {}).get("reason") or ("early_stop" if patience_out else "max_epochs")
            early_stop = {
                "metric": opt.selection_metric, "field": "completion", "mode": "min" if lower_is_better else "max",
                "patience": opt.early_stop_patience, "enabled": opt.early_stop_patience is not None,
                "best_epoch": round(best["epoch"], 6), "best_step": best["step"], "best_value": round(best["value"], 6),
                "stopped_epoch": round(evals[-1]["epoch"], 6), "stopped_step": step, "max_epochs": opt.epochs,
                "reason": reason, "evaluations": len(evals), "kept": "best" if opt.keep_best else "last",
                "kept_step": best["step"] if opt.keep_best else step}

        distillation = None
        if teacher is not None:
            # the student and the teacher on the same validation rows, and how often they pick the same token
            resources.enter("evaluation")
            t_scores = score(t_model, usable_val, final_rows, "teacher")
            resources.enter(None)
            _release()
            distillation = {
                "teacher": teacher.ref, "teacher_commit": teacher_commit, "alpha": teacher.alpha,
                "temperature": teacher.temperature, "metric": headline, "n": len(final_rows),
                "student_score": final[headline], "teacher_score": t_scores[headline], "student_loss": val_loss,
                "teacher_loss": t_scores["loss"],
                "agreement": _token_agreement(model, t_model, [collate([encode(val_examples[j])]) for j in usable_val])}
            del t_model

        ckpt_dir = out.checkpoint
        merged_dir = ckpt_dir / "merged"
        merged_dir.mkdir(parents=True, exist_ok=True)
        if full_training:
            merged = model  # no adapter to merge: the trained weights are the model
        elif opt.precision == "4bit":
            # Merging into NF4 weights is lossy and unsupported by peft; like the
            # legacy runner's `merged_16bit`, save the adapter, reload the base
            # dense in bf16 and merge there.
            from peft import PeftModel

            adapter_dir = ckpt_dir / "adapter"
            model.save_pretrained(str(adapter_dir))
            del model
            base = _load_base_model(source, None if resumed else opt.base_revision, "bf16", "cpu")
            merged = PeftModel.from_pretrained(base, str(adapter_dir)).merge_and_unload()
        else:
            merged = model.merge_and_unload()
        merged.save_pretrained(str(merged_dir), safe_serialization=True)
        tokenizer.save_pretrained(str(merged_dir))
        provenance = {
            "base_model": opt.base_model, "base_revision": opt.base_revision, "base_commit": base_commit,
            "precision": opt.precision, "prompt_template": template, "resumed": resumed,
        }
        if teacher is not None:
            provenance["teacher"] = teacher.ref
        if full_training:
            # what the report says about the weights: drawn at random and trained in full, no adapter
            provenance.update(init="random", init_config=opt.init_config or {}, training="full",
                              parameters=sum(p.numel() for p in merged.parameters()))
        (ckpt_dir / "meta.json").write_text(json.dumps({"model": spec.spec.model, "task": "text_generation",
                                                        **provenance}))
        # What a runtime needs to feed the exported model (CONTRACTS §2.1):
        # runtime_eval/predict read the template from here, in any later process.
        preprocess: dict[str, Any] = {
            "prompt_template": template, "base_model": opt.base_model, "base_revision": opt.base_revision,
            "base_commit": base_commit, "max_seq_length": max_len, "eos_token": tokenizer.eos_token,
        }
        predictions_path: Path | None = None  # the final evaluation's predictions (extended runs)
        blocks: dict[str, Any] = {"distillation": distillation} if distillation else {}
        blocks["truncation"] = truncation
        # What the hub sizes and prices a job from (CONTRACTS §2.2); seconds as measured, tokens as fed
        blocks["training"] = {
            "train_tokens": train_tokens, "train_loop_seconds": round(loop_seconds, 3),
            "eval_seconds": round(eval_seconds, 3), "final_eval_seconds": round(final_eval_seconds, 3),
            "baseline_seconds": None if baseline_seconds is None else round(baseline_seconds, 3),
            "steps_done": step,
            "steps_planned": optimizer_steps(len(train_examples), opt.batch_size, opt.grad_accum, opt.epochs, max_steps),
            "precision": opt.precision, "weights_dtype": weights_dtype, "adapter_dtype": adapter_dtype,
            "autocast": False, "device": device, "batch_size": cur_bs, "grad_accum": cur_ga,
            "effective_batch_size": cur_bs * cur_ga,
            "oom_backoffs": len(resources.backoffs), "oom_backoff_steps": resources.backoffs,
            "gradient_checkpointing": checkpointing,
            "eval_batch_tokens": eval_tokens, "peak_memory_bytes": resources.peaks}
        if baseline is not None:
            blocks["baseline"] = baseline
        # With no runtime_eval_rows at all (not even null), a validation set over DEFAULT_RUNTIME_EVAL_ROWS is re-scored
        # on the first rows of the seeded order only: the same check on fewer rows, compared like for like.
        by_default = ("runtime_eval_rows" not in spec.hyperparameters
                      and len(final_rows) > DEFAULT_RUNTIME_EVAL_ROWS)
        if opt.extended or by_default:
            # how runtime_eval re-scores the exports: the same rows, length and headline metric as here. A row the
            # tuned model could not score (out of memory) is not re-scored either: the two sides compare like for like.
            tuned_skipped = set(final_generated.get("skipped", []))
            runtime_rows = [j for j in final_rows if j not in tuned_skipped]
            preprocess["eval"] = {**scoring, "batch_size": opt.eval_batch_size,
                                  "rows": None if runtime_rows == every_row else sorted(runtime_rows)}
            cap: int | None
            if by_default:
                final_set = set(runtime_rows)
                subset = [j for j in _eval_order(len(val_examples), spec.spec.seed) if j in final_set][
                    :DEFAULT_RUNTIME_EVAL_ROWS]
                capped, cap = True, DEFAULT_RUNTIME_EVAL_ROWS
            else:
                capped, cap = (opt.runtime_eval_rows is not None and opt.runtime_eval_rows < len(runtime_rows),
                               opt.runtime_eval_rows)
                subset = list(runtime_rows[:cap]) if capped else list(runtime_rows)
            rescored = [j for j in subset if _has_target(val_examples[j])]  # runtime_eval skips the rows with no target
            if capped or len(rescored) < len(subset):
                # The exports are re-scored on the first rows of the final evaluation's order (seeded, so the subset
                # is stable and nested), without the rows that have no target. The training-side score on those same
                # rows comes from the per-row scores of the final evaluation: no generation is repeated.
                seen = dict(zip(final_generated["index"], range(len(final_generated["index"]))))
                headline_rows = final_generated["values"].get(headline, [])
                if all(j in seen for j in rescored) and headline_rows:
                    if capped:
                        preprocess["eval"].update(rows=sorted(subset), runtime_eval_rows=cap,
                                                  runtime_eval_total=len(runtime_rows))
                    preprocess["eval"].update(
                        train_n=len(rescored), train_value=text_metrics.mean([headline_rows[seen[j]] for j in rescored]))
                elif by_default:  # still capped: a check on every row is the cost the cap is there to avoid
                    preprocess["eval"].update(
                        rows=sorted(subset), runtime_eval_rows=cap, runtime_eval_total=len(runtime_rows),
                        train_unavailable="the final evaluation did not score these rows")
        if opt.extended:
            blocks["generation_eval"] = {"metric": headline, "metrics": metric_names, "max_new_tokens": max_new_tokens,
                                         "rows": len(final_rows), "loss_rows": len(usable_val)}
            blocks["generation_eval"].update(_skipped_report(no_fit["tuned"], usable_val, final_rows))
            blocks["generation_eval"]["rows_scored"] = _scored(final_generated, headline)
            if eval_only:  # generated from at the final evaluation, scored only by the metrics that need no target
                blocks["generation_eval"]["rows_eval_only"] = sum(j in eval_only for j in final_rows)
            lines = _prediction_lines(val_examples, metric_names, final_generated,
                                      baseline_generated if baseline is not None else None)
            predictions_path = out.artifacts / PREDICTIONS_FILE
            predictions_path.write_text("".join(line + "\n" for line in lines), encoding="utf-8")
            if eval_cols is not None or "eval_scope" in spec.hyperparameters:
                # which columns and which scope the scores above used (CONTRACTS §2.2)
                blocks["generation_eval"].update(scope=opt.eval_scope, columns=_eval_columns_report(eval_cols))
                if eval_cols is not None:  # final rows that only json_valid could score
                    blocks["generation_eval"]["rows_without_target"] = sum(
                        not _has_target(val_examples[j]) for j in final_rows)
                if eval_cols is not None and eval_cols.group is not None:
                    blocks["generation_eval"]["groups"] = _group_report(
                        eval_cols.group, headline, val_examples, final_generated,
                        baseline_generated if baseline is not None else None)
        (out.artifacts / "preprocess.json").write_text(json.dumps(preprocess, indent=2) + "\n", encoding="utf-8")

        fitted = _Fitted(model_dir=merged_dir, tokenizer_dir=merged_dir, eval_examples=val_examples,
                         prompt_template=template, gguf_quant=opt.gguf_quant, scoring=preprocess.get("eval"),
                         predictions=predictions_path)
        return TrainResult(
            model=fitted,
            metrics={"loss": val_loss, **{name: final[name] for name in metric_names}},
            train_seconds=train_seconds,
            checkpoint=Checkpoint(path=ckpt_dir, model=spec.spec.model, task="text_generation"),
            provenance=provenance,
            stopped_early=watch.stopped_early,
            early_stop=early_stop,
            epoch_metrics=epoch_metrics,
            report=blocks or None,
        )

    def _train_dpo(self, spec: ResolvedSpec, data: MaterialisedData, out: RunDir, metrics: MetricsSink,
                   resume: Checkpoint | None) -> TrainResult:
        """``objective: dpo``: Direct Preference Optimization on ``{prompt, completion (chosen), rejected}``
        rows. A LoRA on the policy; the reference is the starting weights, frozen. The reference's two
        log-probabilities per row never change, so they are computed once before the LoRA is attached.

        Reported, for the policy and for the reference on the same validation pairs: the DPO loss and
        ``preference_accuracy``, the fraction of pairs where the model gives ``chosen`` a higher
        log-probability than ``rejected``."""
        import torch
        from peft import LoraConfig, get_peft_model
        from transformers import AutoTokenizer

        from sakura.atelier.backends import dpo

        out.ensure()
        opt = _training_options(spec.hyperparameters)
        template, beta, max_len = opt.prompt_template, opt.dpo_beta, opt.max_seq_length
        device = _device()
        tok_kwargs: dict[str, Any] = {"revision": opt.base_revision} if opt.base_revision else {}
        tokenizer = AutoTokenizer.from_pretrained(opt.base_model, **tok_kwargs)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        resume_dir = resume.path / "merged" if resume is not None else None
        resumed = bool(resume_dir and resume_dir.exists())
        source = str(resume_dir) if resumed and resume_dir else opt.base_model
        model = _load_base_model(source, None if resumed else opt.base_revision, opt.precision, device)
        base_commit = getattr(getattr(model, "config", None), "_commit_hash", None)

        train_rows, val_rows = list(data.train), list(data.validation)

        def encoded(rows: list[dict[str, str]], key: str) -> list[tuple[list[int], int]]:
            return [dpo.encode_pair(tokenizer, _format_prompt(template, r["prompt"]), r[key], max_len) for r in rows]

        train_c, train_r = encoded(train_rows, "completion"), encoded(train_rows, "rejected")
        val_c, val_r = encoded(val_rows, "completion"), encoded(val_rows, "rejected")

        @torch.no_grad()
        def logprobs(net: Any, pairs: list[tuple[list[int], int]]) -> list[float]:
            net.eval()
            values: list[float] = []
            for i in range(0, len(pairs), 16):
                values += dpo.completion_logprob(net, tokenizer, pairs[i:i + 16], device).tolist()
            return values

        # the frozen reference: the starting weights, scored once on every pair
        ref_train_c, ref_train_r = logprobs(model, train_c), logprobs(model, train_r)
        ref_val_c, ref_val_r = logprobs(model, val_c), logprobs(model, val_r)
        reference = dpo.preference_stats(ref_val_c, ref_val_r, ref_val_c, ref_val_r, beta)

        model = get_peft_model(model, LoraConfig(
            r=opt.lora_r, lora_alpha=opt.lora_alpha, lora_dropout=opt.lora_dropout,
            target_modules=opt.lora_target_modules, task_type="CAUSAL_LM"))
        model.train()
        optim = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=opt.learning_rate)
        rng = random.Random(spec.spec.seed)

        def validate(step: int, epoch: float) -> dict[str, float]:
            stats = dpo.preference_stats(logprobs(model, val_c), logprobs(model, val_r), ref_val_c, ref_val_r, beta)
            model.train()
            metrics.log("loss", stats["loss"], step=step, split="validation", epoch=epoch)
            metrics.log("preference_accuracy", stats["preference_accuracy"], step=step, split="validation", epoch=epoch)
            return stats

        t0 = time.time()
        step = 0
        policy = dict(reference)
        watch = WrapUpWatch(out.root, max_steps_of(spec))
        for epoch in range(opt.epochs):
            order = list(range(len(train_rows)))
            rng.shuffle(order)
            batches = [order[i:i + opt.batch_size] for i in range(0, len(order), opt.batch_size)]
            for bi, idx in enumerate(batches):
                if watch.stop(epoch + bi * opt.batch_size / len(order), step):
                    break  # wrap-up: stop learning; the partial epoch is scored below
                loss = dpo.dpo_loss(
                    dpo.completion_logprob(model, tokenizer, [train_c[j] for j in idx], device),
                    dpo.completion_logprob(model, tokenizer, [train_r[j] for j in idx], device),
                    torch.tensor([ref_train_c[j] for j in idx], device=device),
                    torch.tensor([ref_train_r[j] for j in idx], device=device), beta)
                (loss / opt.grad_accum).backward()
                boundary = (bi + 1) % opt.grad_accum == 0 or bi == len(batches) - 1
                if boundary:
                    optim.step()
                    optim.zero_grad()
                metrics.log("loss", float(loss.detach()), step=step, split="train",
                            epoch=epoch + bi * opt.batch_size / len(order))
                if boundary:
                    step += 1
            if watch.triggered:
                optim.zero_grad()  # drop a half-accumulated step
                policy = validate(step, epoch + bi * opt.batch_size / len(order))
                break
            policy = validate(step, float(epoch + 1))
        train_seconds = time.time() - t0

        ckpt_dir = out.checkpoint
        merged_dir = ckpt_dir / "merged"
        merged_dir.mkdir(parents=True, exist_ok=True)
        if opt.precision == "4bit":
            # as in the next-token path: save the adapter, reload the base dense in bf16 and merge there
            from peft import PeftModel

            adapter_dir = ckpt_dir / "adapter"
            model.save_pretrained(str(adapter_dir))
            del model
            base = _load_base_model(source, None if resumed else opt.base_revision, "bf16", "cpu")
            merged = PeftModel.from_pretrained(base, str(adapter_dir)).merge_and_unload()
        else:
            merged = model.merge_and_unload()
        merged.save_pretrained(str(merged_dir), safe_serialization=True)
        tokenizer.save_pretrained(str(merged_dir))
        provenance = {
            "base_model": opt.base_model, "base_revision": opt.base_revision, "base_commit": base_commit,
            "precision": opt.precision, "prompt_template": template, "resumed": resumed, "objective": "dpo",
        }
        (ckpt_dir / "meta.json").write_text(json.dumps({"model": spec.spec.model, "task": "text_generation",
                                                        **provenance}))
        (out.artifacts / "preprocess.json").write_text(json.dumps({
            "prompt_template": template, "base_model": opt.base_model, "base_revision": opt.base_revision,
            "base_commit": base_commit, "max_seq_length": max_len, "eos_token": tokenizer.eos_token,
            "objective": "dpo", "dpo_beta": beta,
        }, indent=2) + "\n", encoding="utf-8")
        fitted = _Fitted(model_dir=merged_dir, tokenizer_dir=merged_dir, eval_examples=val_rows,
                         prompt_template=template, gguf_quant=opt.gguf_quant)
        return TrainResult(
            model=fitted,
            metrics={"loss": policy["loss"], "preference_accuracy": policy["preference_accuracy"]},
            train_seconds=train_seconds,
            checkpoint=Checkpoint(path=ckpt_dir, model=spec.spec.model, task="text_generation"),
            provenance=provenance,
            stopped_early=watch.stopped_early,
            report={"preference": {"objective": "dpo", "beta": beta, "n": len(val_rows),
                                   "policy": policy, "reference": reference}},
        )

    def export(self, result: TrainResult, fmt: str, out: RunDir) -> Artifact:
        out.ensure()
        fitted: _Fitted = result.model
        if fitted.decision is not None and fmt != "safetensors":
            raise dec.DecisionDataError(
                f"the decision task exports safetensors only, not {fmt!r}: a {fmt} runtime cannot reproduce the "
                "option-letter readout or the fitted calibration temperature")
        if fmt == "safetensors":
            dest = out.artifacts / "model.safetensors"
            shutil.copy(fitted.model_dir / "model.safetensors", dest)
            # Every OTHER file next to the merged weights (config.json,
            # tokenizer.json, tokenizer_config.json, special_tokens_map.json,
            # generation_config.json, ...) -- an earlier version of this
            # method copied only config.json, so AutoTokenizer.from_pretrained
            # on the exported artifact's own directory failed outright
            # (no tokenizer.json to find); the exported safetensors artifact
            # must be loadable standalone, same as the merged checkpoint is.
            for extra in fitted.model_dir.iterdir():
                if extra.is_file() and extra.name != "model.safetensors":
                    shutil.copy(extra, out.artifacts / extra.name)
            return _artifact(dest, "safetensors")
        if fmt == "gguf":
            return self._export_gguf(fitted, out)
        if fmt == "mlx":
            return self._export_mlx(fitted, out)
        raise ValueError(f"peft backend exports gguf|safetensors|mlx, got {fmt!r}")

    def _export_gguf(self, fitted: _Fitted, out: RunDir) -> Artifact:
        llama_dir = _llama_cpp_dir()
        convert_script = llama_dir / "convert_hf_to_gguf.py"
        quantize_bin = llama_dir / "build" / "bin" / "llama-quantize"
        if not convert_script.exists():
            raise GGUFExportError(f"{convert_script} not found")
        if not quantize_bin.exists():
            raise GGUFExportError(f"{quantize_bin} not found -- build llama.cpp first")
        # hp.gguf_quant first; the SAKURA_GGUF_QUANT env is a fallback only when
        # the spec does not set it (an operator default, never an override).
        quant = fitted.gguf_quant or os.environ.get("SAKURA_GGUF_QUANT", DEFAULT_GGUF_QUANT)
        if not _GGUF_QUANT_RE.fullmatch(quant):
            where = "gguf_quant" if fitted.gguf_quant else "SAKURA_GGUF_QUANT"
            raise GGUFExportError(f"{where}={quant!r} is not a llama.cpp quant type name")
        quant_opts, quant_type = _gguf_quant_args(quant)
        _check_k_quant_shape(fitted.model_dir, quant, [quant_type, *quant_opts])

        fp16_path = out.artifacts / "model.fp16.gguf"
        proc = subprocess.run(
            [sys.executable, str(convert_script), str(fitted.model_dir), "--outfile", str(fp16_path), "--outtype", "f16"],
            capture_output=True, text=True, errors="replace",
        )
        if proc.returncode != 0:
            raise GGUFExportError(f"convert_hf_to_gguf.py failed: {proc.stderr[-4000:]}")

        dest = out.artifacts / "model.gguf"
        # The quant value is one argv element (no shell) and is validated above
        # as a bare quant name (e.g. Q8_0, Q4_K_M), so it cannot inject. No
        # importance matrix: an imatrix needs a calibration corpus and a
        # llama-imatrix pass per model (not supported here; see CONTRACTS §2.2).
        proc = subprocess.run(
            # nosemgrep: python.lang.security.audit.dangerous-subprocess-use-tainted-env-args.dangerous-subprocess-use-tainted-env-args
            [str(quantize_bin), *quant_opts, str(fp16_path), str(dest), quant_type],
            capture_output=True, text=True, errors="replace",
        )
        if proc.returncode != 0:
            raise GGUFExportError(f"llama-quantize failed: {proc.stderr[-4000:]}")
        fp16_path.unlink(missing_ok=True)
        return _artifact(dest, "gguf")

    def _export_mlx(self, fitted: _Fitted, out: RunDir) -> Artifact:
        quant = os.environ.get("SAKURA_MLX_QUANT", DEFAULT_MLX_QUANT)
        return export_mlx(fitted.model_dir, fitted.eval_examples, out.artifacts / "model.mlx", quant,
                          template=fitted.prompt_template, scoring=fitted.scoring).artifact

    def _rescore(self, artifact: Artifact, data: MaterialisedData, runtime: str, complete: Any, template: str,
                 scoring: dict[str, Any] | None = None, complete_many: Any = None) -> RuntimeEval:
        """The run's headline metric over the validation split, generated by `complete`."""
        return _rescore(artifact, list(data.validation), runtime, complete, template=template, scoring=scoring,
                        complete_many=complete_many)

    def release(self) -> None:
        """Let go of the exports this process loaded to re-score them (``_LOADED``: the safetensors model on the
        card, the GGUF's llama.cpp context) and of the allocator's cache. Called once the run has nothing left to
        score: the notebook check re-scores the exports from another process on the same card."""
        for key in list(_LOADED):
            entry = _LOADED.pop(key)
            close = getattr(entry, "close", None)  # llama.cpp frees its context on close
            if callable(close):
                with contextlib.suppress(Exception):
                    close()
            del entry
        _release()

    def runtime_eval(self, artifact: Artifact, data: MaterialisedData) -> RuntimeEval:
        if _is_decision(artifact):
            return self._decision_runtime_eval(artifact, data)
        # The same prompt template training used (artifacts/preprocess.json),
        # so every runtime is scored on the strings the model was trained on.
        settings = _artifact_settings(artifact)
        template = str(settings.get("prompt_template", "raw"))
        if settings.get("objective") == "dpo" and artifact.format == "safetensors":
            # a DPO run is re-scored on what it optimised: which completion of each pair the model prefers
            return _preference_rescore(artifact, list(data.validation), settings)
        # The generation length, the headline metric and the rows training scored (preprocess.json `eval`);
        # an artifact without the block is re-scored as it always was: 8 tokens, exact match, first line.
        scoring = _scoring_of(settings)
        metric = str(scoring["metric"]) if scoring else "exact_match"
        new_tokens = int(scoring["max_new_tokens"]) if scoring else LEGACY_NEW_TOKENS
        if artifact.format == "mlx":
            cached = _MLX_SCORES.get(str(artifact.path))
            if cached is not None and cached[0] == list(data.validation):
                return cached[1]  # scored by export_mlx on these exact examples
            return self._rescore(artifact, data, "mlx-lm",
                                 lambda prompt: _mlx_generate(artifact.path, prompt, new_tokens), template, scoring)
        if artifact.format == "safetensors":
            batch = int(scoring.get("batch_size") or 1) if scoring else 1
            window = int(settings.get("max_seq_length") or 0)  # the bound training used: rows x the training length
            return self._rescore(
                artifact, data, "transformers",
                lambda prompt: _hf_generate(artifact.path.parent, prompt, new_tokens), template, scoring,
                (lambda prompts: _hf_generate_many(artifact.path.parent, prompts, new_tokens, batch,
                                                   *((batch * window, window) if window else ()))) if batch > 1 else None)
        if artifact.format != "gguf":
            return RuntimeEval(
                format=artifact.format, runtime="none", metric=metric,
                value=None, n=0, matches_training=None,
                reason=f"this backend only re-scores gguf through llama-cpp-python; {artifact.format} has no runtime scorer in phase 1",
            )
        try:
            llm = _load_llama_cpp(artifact.path, _gguf_n_ctx(artifact))
        except ImportError as exc:
            return RuntimeEval(
                format="gguf", runtime="llama-cpp-python", metric=metric,
                value=None, n=0, matches_training=None,
                reason=f"llama_cpp (llama-cpp-python) is not importable: {type(exc).__name__}: {exc}",
            )
        except Exception as exc:  # noqa: BLE001
            return RuntimeEval(
                format="gguf", runtime="llama-cpp-python", metric=metric,
                value=None, n=0, matches_training=None,
                reason=f"llama_cpp.Llama failed to load {artifact.path}: {type(exc).__name__}: {exc}",
            )

        # Raw completion, exactly the training prompt string: no chat template, no
        # system/role wrapping -- only the training template. Scored like the other
        # formats, so a prompt llama.cpp refuses (longer than the context) is a
        # reason on the report, not a failed run after training. matches_training
        # is the engine's call (Track E1's `run` loop holds TrainResult.metrics).
        # The first-line rule stops at the end of the line; a whole completion (JSON, several lines) runs to
        # the model's own end-of-sequence token or the length.
        stop: dict[str, Any] = {"stop": ["\n"]} if not scoring or scoring.get("exact_match_rule") == "first_line" else {}
        return self._rescore(artifact, data, "llama-cpp-python", lambda prompt: llm(
            prompt, max_tokens=new_tokens, temperature=0.0, **stop)["choices"][0]["text"], template, scoring)

    def predict(self, artifact: Artifact, inputs: Any) -> Any:
        if _is_decision(artifact):
            cfg = dec.read_decision_config(artifact.path.parent / DECISION_CONFIG)
            return dec.answer_request(inputs, cfg["temperatures"], self._decision_scorer(artifact.path.parent, cfg))
        # `prompt` is the user's input text; the training template (alpaca:
        # "### Instruction:\n...\n\n### Response:\n") wraps it here, as the
        # legacy runner's llama-cli path did, so /infer never needs to know it.
        instruction = inputs["prompt"] if isinstance(inputs, dict) else str(inputs)
        max_tokens = inputs.get("max_tokens", 64) if isinstance(inputs, dict) else 64
        template = str(_artifact_settings(artifact).get("prompt_template", "raw"))
        prompt = _format_prompt(template, instruction)
        if artifact.format == "gguf":
            llm = _load_llama_cpp(artifact.path, _gguf_n_ctx(artifact))
            out_ = llm(prompt, max_tokens=max_tokens, temperature=0.0)
            text = out_["choices"][0]["text"]
        elif artifact.format == "safetensors":
            text = _hf_generate(artifact.path.parent, prompt, max_tokens)
        elif artifact.format == "mlx":
            text = _mlx_generate(artifact.path, prompt, max_tokens)
        else:
            raise ValueError(f"cannot predict from artifact format {artifact.format!r}")
        return {"text": _clean_completion(template, text).strip()}

    # ---- task decision ---------------------------------------------------------------------

    def _decision_load(self, opt: dec.DecisionOptions, source: str, revision: str | None,
                       device: str) -> tuple[Any, Any, str | None]:
        """``(tokenizer, base model, resolved Hub commit)``. A seam: the one place a decision run loads weights."""
        from transformers import AutoTokenizer

        kwargs: dict[str, Any] = {"revision": revision} if revision else {}
        tokenizer = AutoTokenizer.from_pretrained(source, **kwargs)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = _load_base_model(source, revision, opt.precision, device)
        return tokenizer, model, getattr(getattr(model, "config", None), "_commit_hash", None)

    def _decision_adapt(self, model: Any, opt: dec.DecisionOptions) -> Any:
        """The base with a LoRA on its attention / MLP projections; the base weights stay frozen. A seam."""
        from peft import LoraConfig, get_peft_model

        if opt.gradient_checkpointing:
            model.gradient_checkpointing_enable()
            # The frozen base gives the checkpointed blocks no input that requires grad; without this the
            # LoRA under them would get no gradient.
            model.enable_input_require_grads()
        return get_peft_model(model, LoraConfig(
            r=opt.lora_r, lora_alpha=opt.lora_alpha, lora_dropout=opt.lora_dropout,
            target_modules=opt.lora_target_modules, task_type="CAUSAL_LM"))

    def _decision_merge(self, model: Any, opt: dec.DecisionOptions, merged_dir: Path, tokenizer: Any,
                        source: str, revision: str | None) -> Any:
        """Merge the adapter into the base and save the dense model to ``merged_dir``. A seam."""
        if opt.precision == "4bit":
            # Merging into NF4 weights is lossy and unsupported by peft: save the adapter, reload the base
            # dense in bf16 and merge there (as the text_generation path does).
            from peft import PeftModel

            adapter_dir = merged_dir.parent / "adapter"
            model.save_pretrained(str(adapter_dir))
            merged = PeftModel.from_pretrained(_load_base_model(source, revision, "bf16", "cpu"),
                                               str(adapter_dir)).merge_and_unload()
        else:
            merged = model.merge_and_unload()
        merged_dir.mkdir(parents=True, exist_ok=True)
        # One file whatever its size: export() ships model.safetensors, and a sharded save has none.
        merged.save_pretrained(str(merged_dir), safe_serialization=True, max_shard_size="64GB")
        tokenizer.save_pretrained(str(merged_dir))
        return merged

    def _decision_load_saved(self, model_dir: Path, cfg: dict[str, Any], device: str) -> tuple[Any, Any]:
        """``(tokenizer, model)`` of a delivered decision model. A seam."""
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        dtype = torch.bfloat16 if cfg.get("precision") == "bf16" and device == "cuda" else torch.float32
        return (AutoTokenizer.from_pretrained(str(model_dir)),
                AutoModelForCausalLM.from_pretrained(str(model_dir), dtype=dtype))

    def _decision_scorer(self, model_dir: Path, cfg: dict[str, Any]) -> Any:
        """``(state, instances) -> restricted logits``, the model loaded once per process (``serve``)."""
        from sakura.atelier.backends import decision_lm as lm

        key = f"decision:{model_dir}"
        if key not in _LOADED:
            device = _device()
            tokenizer, model = self._decision_load_saved(model_dir, cfg, device)
            model.to(device)
            model.eval()
            _LOADED[key] = (tokenizer, model, lm.make_readout(tokenizer), device)
        tokenizer, model, readout, device = _LOADED[key]
        max_len = int(cfg["max_seq_length"])
        return lambda state, instances: lm.score_state(model, tokenizer, readout, state, instances, max_len, device)

    def _train_decision(self, spec: ResolvedSpec, data: MaterialisedData, out: RunDir, metrics: MetricsSink,
                        resume: Checkpoint | None) -> TrainResult:
        """Prepare is done (``materialise``: the rows are validated, deduplicated and split by state); this trains
        on ``data.train``, scores the calibration split (``data.extra["calibration"]``) to fit one temperature per
        answer type, and measures everything it reports on ``data.validation`` (the EVAL split), which neither
        training nor the fit has seen."""
        import torch

        from sakura.atelier.backends import decision_lm as lm

        out.ensure()
        opt = dec.decision_options(spec.hyperparameters)
        device = _device()
        # A resume continues from the previous run's merged weights, not the base.
        resume_dir = resume.path / "merged" if resume is not None else None
        resumed = bool(resume_dir and resume_dir.exists())
        source = str(resume_dir) if resumed and resume_dir else opt.base_model
        revision = None if resumed else opt.base_revision
        tokenizer, model, base_commit = self._decision_load(opt, source, revision, device)
        readout = lm.make_readout(tokenizer)
        model = self._decision_adapt(model, opt)

        train = [i for row in data.train for i in dec.instances(row)]
        calibration = [i for row in data.extra["calibration"] for i in dec.instances(row)]
        held_out = [i for row in data.validation for i in dec.instances(row)]
        cfg = lm.TrainConfig(epochs=opt.epochs, batch_size=opt.batch_size, grad_accum=opt.grad_accum,
                             learning_rate=opt.learning_rate, max_len=opt.max_seq_length, seed=spec.spec.seed)
        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=opt.learning_rate)
        t0 = time.time()
        loop = lm.train_loop(model, tokenizer, readout, train, calibration, cfg=cfg, device=device,
                             optimizer=optimizer, metrics=metrics, watch=WrapUpWatch(out.root, max_steps_of(spec)))
        train_seconds = time.time() - t0

        ckpt_dir = out.checkpoint
        merged = self._decision_merge(model, opt, ckpt_dir / "merged", tokenizer, source, revision)
        # What is reported is what the delivered model does: score with the merged weights, in the given option order.
        where = next(merged.parameters()).device
        scored_calibration = lm.score_instances(merged, tokenizer, readout, calibration, opt.max_seq_length, where,
                                                opt.batch_size)
        scored_eval = lm.score_instances(merged, tokenizer, readout, held_out, opt.max_seq_length, where, opt.batch_size)
        info = data.extra.get("decision", {})
        warnings = list(info.get("warnings", []))
        every = [*train, *calibration, *held_out]
        cut = lm.count_truncated(tokenizer, readout, every, opt.max_seq_length)
        if cut:
            warnings.append(f"{cut} of {len(every)} answers had their state cut to fit max_seq_length="
                            f"{opt.max_seq_length} tokens")
        blocks = dec.report_blocks(scored_calibration, scored_eval, warnings)
        blocks["evaluation"]["data"] = info.get("stats", {})

        provenance = {"base_model": opt.base_model, "base_revision": opt.base_revision, "base_commit": base_commit,
                      "precision": opt.precision, "prompt_template": dec.TEMPLATE_VERSION, "resumed": resumed}
        (ckpt_dir / "meta.json").write_text(json.dumps({"model": spec.spec.model, "task": "decision", **provenance}))
        config = dec.decision_config(temperatures=blocks["calibration"]["temperatures"], base_model=opt.base_model,
                                     base_revision=opt.base_revision, base_commit=base_commit,
                                     max_seq_length=opt.max_seq_length, precision=opt.precision)
        (out.artifacts / DECISION_CONFIG).write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
        fitted = _Fitted(model_dir=ckpt_dir / "merged", tokenizer_dir=ckpt_dir / "merged", eval_examples=[],
                         prompt_template=dec.TEMPLATE_VERSION, decision=config)
        return TrainResult(
            model=fitted, metrics=blocks["metrics"], train_seconds=train_seconds,
            checkpoint=Checkpoint(path=ckpt_dir, model=spec.spec.model, task="decision"), provenance=provenance,
            calibration=blocks["calibration"], evaluation=blocks["evaluation"],
            stopped_early=loop.get("stopped_early"))

    def _decision_runtime_eval(self, artifact: Artifact, data: MaterialisedData) -> RuntimeEval:
        """Re-score the delivered model on the EVAL rows through the same path ``predict`` answers by."""
        rows = list(data.validation)
        if not rows:
            return RuntimeEval(format="safetensors", runtime="transformers", metric="accuracy", value=None, n=0,
                               matches_training=None, reason="no eval rows to re-score on")
        try:
            cfg = dec.read_decision_config(artifact.path.parent / DECISION_CONFIG)
            score = self._decision_scorer(artifact.path.parent, cfg)
            hits = total = 0
            for row in rows:
                asked = {qid: row["questions"][qid] for qid in row["answers"]}
                answers = dec.answer_request({"state": row["state"], "questions": asked}, cfg["temperatures"],
                                             score)["answers"]
                for inst in dec.instances(row):
                    hits += dec.answer_hit(inst, answers[inst.qid])
                    total += 1
        except ImportError as exc:
            return RuntimeEval(format="safetensors", runtime="transformers", metric="accuracy", value=None, n=0,
                               matches_training=None,
                               reason=f"transformers is not importable here: {type(exc).__name__}: {exc}")
        except Exception as exc:  # noqa: BLE001
            return RuntimeEval(format="safetensors", runtime="transformers", metric="accuracy", value=None, n=0,
                               matches_training=None,
                               reason=f"transformers could not re-score {artifact.path.name}: {type(exc).__name__}: {exc}")
        return RuntimeEval(format="safetensors", runtime="transformers", metric="accuracy", value=hits / total,
                           n=total, matches_training=None, reason=None)

    def extra_artifacts(self, result: TrainResult, out: RunDir) -> list[Artifact]:
        """Files delivered beside the exports: a decision run's ``decision_config.json``, and the
        ``predictions.jsonl`` of a text-generation run that sets the evaluation keys."""
        extras: list[Artifact] = []
        if getattr(result.model, "decision", None) is not None:
            extras.append(_artifact(out.artifacts / DECISION_CONFIG, "decision_config"))
        predictions = getattr(result.model, "predictions", None)
        if predictions is not None and predictions.is_file():
            extras.append(_artifact(predictions, "predictions"))
        return extras


#: Written beside the model by a decision run; its presence is what makes `runtime_eval` / `predict` answer typed questions.
DECISION_CONFIG = "decision_config.json"


def _is_decision(artifact: Artifact) -> bool:
    return artifact.format == "safetensors" and (artifact.path.parent / DECISION_CONFIG).is_file()


def _artifact(path: Path, fmt: str) -> Artifact:
    import hashlib

    data = path.read_bytes()
    return Artifact(path=path, format=fmt, sha256=hashlib.sha256(data).hexdigest(), bytes=len(data))


# --------------------------------------------------------------------------
# MLX export + re-score (Apple silicon only). Callable on its own -- a
# scheduler that places it on an `apple-silicon` worker runs exactly this, from
# the merged safetensors dir a Linux GPU run produced -- and from export("mlx").
# The engine never places it itself (ADR SAK-0008).

#: Quantisation level -> mlx_lm.convert flags. Non-quantised tensors are
#: fp16 in every level. Measured on an Apple M5 (phase-1 check, LFM2-350M
#: LoRA, 50 held-out prompts, PyTorch exact_match 0.98): fp16 0.98 / 1.42 GB /
#: 78 ms/ex; q8 0.98 / 404 MB / 19 ms/ex; q4 0.84 / 227 MB / 20 ms/ex. The
#: level is the user's choice; the report carries its measured cost.
MLX_QUANTS: dict[str, tuple[str, ...]] = {
    "fp16": ("--dtype", "float16"),
    "q8": ("--dtype", "float16", "-q", "--q-bits", "8"),
    "q4": ("--dtype", "float16", "-q", "--q-bits", "4"),
}
#: Same accuracy as fp16 at ~3.5x smaller and ~4x faster (phase-1 check); mirrors the
#: GGUF default (Q8_0). Override per run with SAKURA_MLX_QUANT.
DEFAULT_MLX_QUANT = "q8"

#: artifact path -> (validation examples, score) from the last export_mlx,
#: so the engine's runtime_eval right after export does not re-run 50 prompts.
_MLX_SCORES: dict[str, tuple[list[dict[str, str]], RuntimeEval]] = {}


@dataclass(frozen=True)
class MLXExport:
    """An MLX export and its re-score with the training harness's rule."""

    artifact: Artifact
    quant: str
    score: RuntimeEval
    ms_per_example: float | None
    report_path: Path

    def to_json(self) -> dict[str, Any]:
        return {
            "format": "mlx", "path": str(self.artifact.path), "quant": self.quant,
            "bytes": self.artifact.bytes, "sha256": self.artifact.sha256,
            "runtime": self.score.runtime, "metric": self.score.metric,
            "value": self.score.value, "n": self.score.n, "reason": self.score.reason,
            "ms_per_example": self.ms_per_example,
        }


def _is_apple_silicon() -> bool:
    return sys.platform == "darwin" and platform.machine() == "arm64"


def _require_mlx() -> None:
    """Raise MLXUnavailableError unless this host is Apple silicon with mlx-lm.

    Gated on the platform first, not just on the import: mlx now ships Linux
    (CPU/CUDA) wheels, but the MLX artefact targets Metal on Apple devices and
    its score must be measured there."""
    if not _is_apple_silicon():
        raise MLXUnavailableError(
            f"MLX export needs Apple Silicon (macOS arm64, Metal); this host is "
            f"{sys.platform}/{platform.machine()}. Run it on an apple-silicon worker "
            "from the merged safetensors dir (python -m sakura.atelier.backends.peft mlx ...)"
        )
    try:
        import mlx_lm  # noqa: F401
    except ImportError as exc:
        raise MLXUnavailableError(
            f"mlx_lm does not import on this host ({type(exc).__name__}: {exc}); "
            "install mlx-lm (Apple Silicon only) -- docs/atelier/envs/peft.txt"
        ) from exc


def _rescore(artifact: Artifact, examples: list[dict[str, str]], runtime: str, complete: Any, template: str = "raw",
             scoring: dict[str, Any] | None = None, complete_many: Any = None) -> RuntimeEval:
    """The run's headline metric over `examples`, generated by `complete` on the templated prompt (or by
    `complete_many` on all of them at once) -- reported with a reason (value None), never raised: an export
    that cannot be re-scored is still delivered.

    `scoring` is the run's `preprocess.json` `eval` block: the metric, whether exact match reads the first
    line only, and the validation rows training scored (indices past the rows given are skipped; none left =
    every row). Without it: exact match on the first line, over every example, as before the block."""
    metric = str(scoring["metric"]) if scoring else "exact_match"

    def unscored(reason: str) -> RuntimeEval:
        return RuntimeEval(format=artifact.format, runtime=runtime, metric=metric, value=None, n=0,
                           matches_training=None, reason=reason)

    if not examples:
        return unscored("no validation examples to re-score on")
    try:
        rows = examples
        if scoring and scoring.get("rows") is not None:
            rows = [examples[i] for i in scoring["rows"] if 0 <= i < len(examples)] or examples
        expected = len(scoring["rows"]) if scoring and scoring.get("rows") is not None else len(examples)
        # an evaluation-only row (no completion, no reference) has no target to re-score against: skipped, counted
        scored_rows = [ex for ex in rows if _has_target(ex)]
        skipped = len(rows) - len(scored_rows)
        runtime_skipped = 0
        if not scored_rows:
            return unscored("every row to re-score is evaluation-only (no completion and no reference)")
        prompts = [_format_prompt(template, ex["prompt"]) for ex in scored_rows]
        if complete_many is not None:
            texts = complete_many(prompts)
        else:
            # a row the runtime cannot take on its own (out of memory, a context it cannot hold) is left out and counted;
            # when it takes none, the first failure is the answer, as it always was
            texts, kept, failures = [], [], []
            for ex, prompt in zip(scored_rows, prompts):
                try:
                    texts.append(complete(prompt))
                    kept.append(ex)
                except ImportError:
                    raise
                except Exception as exc:  # noqa: BLE001 - reported per row below
                    failures.append(exc)
            if not kept:
                raise failures[0]
            scored_rows, runtime_skipped = kept, len(failures)
        value = text_metrics.score(
            [metric], [_clean_completion(template, text) for text in texts], [_target_of(ex) for ex in scored_rows],
            first_line=not scoring or scoring.get("exact_match_rule") == "first_line")[metric]
    except ImportError as exc:
        return unscored(f"{runtime} is not importable here: {type(exc).__name__}: {exc}")
    except Exception as exc:  # noqa: BLE001
        return unscored(f"{runtime} could not re-score {artifact.path.name}: {type(exc).__name__}: {exc}")
    # fewer rows than the final evaluation: the training-side score on those very rows (the notebook's capped
    # replay may hold fewer rows than were scored in training, then there is no like-for-like number)
    like_for_like = (scoring is not None and scoring.get("train_value") is not None and len(rows) == expected
                     and not runtime_skipped)
    # capped from this many rows (the run's cap, honoured: every row it named was here to re-score)
    total = scoring.get("runtime_eval_total") if scoring and len(rows) == expected else None
    reason = None
    if total is not None and scoring and scoring.get("train_unavailable") and not like_for_like:
        reason = (f"not compared with training: {scoring['train_unavailable']}, so the training score on these "
                  f"{len(scored_rows)} rows is not known")
    if runtime_skipped:
        reason = (f"{runtime_skipped} rows the runtime could not take were left out; not compared with training, "
                  f"which scored them")
    return RuntimeEval(format=artifact.format, runtime=runtime, metric=metric, value=value, n=len(scored_rows),
                       matches_training=None, reason=reason, rows_eval_only=skipped, rows_skipped=runtime_skipped,
                       n_total=int(total) if total is not None and int(total) > len(scored_rows) else None,
                       train_value=float(scoring["train_value"]) if like_for_like and scoring else None,
                       train_n=int(scoring["train_n"]) if like_for_like and scoring else None)


def _preference_rescore(artifact: Artifact, rows: list[dict[str, str]], settings: dict[str, Any]) -> RuntimeEval:
    """``preference_accuracy`` of a delivered DPO model, reloaded by transformers: the fraction of validation
    pairs where it gives ``chosen`` a higher log-probability than ``rejected``. Reported with a reason (value
    None), never raised."""
    def unscored(reason: str) -> RuntimeEval:
        return RuntimeEval(format=artifact.format, runtime="transformers", metric="preference_accuracy",
                           value=None, n=0, matches_training=None, reason=reason)

    pairs = [r for r in rows if "rejected" in r]
    if not pairs:
        return unscored("no validation pairs to re-score on")
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        from sakura.atelier.backends import dpo

        model_dir = artifact.path.parent
        tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        device = _device()
        model = AutoModelForCausalLM.from_pretrained(str(model_dir), dtype=torch.float32).to(device).eval()
        template = str(settings.get("prompt_template", "raw"))
        max_len = int(settings.get("max_seq_length") or 512)
        wins = 0
        with torch.no_grad():
            for i in range(0, len(pairs), 16):
                chunk = pairs[i:i + 16]
                chosen, rejected = (dpo.completion_logprob(model, tokenizer, [
                    dpo.encode_pair(tokenizer, _format_prompt(template, r["prompt"]), r[key], max_len)
                    for r in chunk], device) for key in ("completion", "rejected"))
                wins += int((chosen > rejected).sum())
    except ImportError as exc:
        return unscored(f"transformers is not importable here: {type(exc).__name__}: {exc}")
    except Exception as exc:  # noqa: BLE001
        return unscored(f"transformers could not re-score {artifact.path.name}: {type(exc).__name__}: {exc}")
    return RuntimeEval(format=artifact.format, runtime="transformers", metric="preference_accuracy",
                       value=wins / len(pairs), n=len(pairs), matches_training=None, reason=None)


def _dir_artifact(path: Path, fmt: str) -> Artifact:
    """Artifact for a directory export: sha256 over (relative path, bytes) of
    every file in sorted order, streamed -- an fp16 MLX dir is ~1.4 GB."""
    import hashlib

    h = hashlib.sha256()
    total = 0
    for f in sorted(p for p in path.rglob("*") if p.is_file()):
        h.update(f.relative_to(path).as_posix().encode() + b"\0")
        with f.open("rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
                total += len(chunk)
    return Artifact(path=path, format=fmt, sha256=h.hexdigest(), bytes=total)


def export_mlx(merged_dir: Path | str, validation: list[dict[str, str]], dest_dir: Path | str,
               quant: str | None = None, template: str = "raw", scoring: dict[str, Any] | None = None) -> MLXExport:
    """Convert a merged HF safetensors checkpoint to MLX and re-score it.

    ``mlx_lm.convert`` at ``quant`` (fp16 | q8 | q4; None = ``DEFAULT_MLX_QUANT``)
    into ``dest_dir``, then exact match on ``validation`` (``[{"prompt",
    "completion"}]``) with the training harness's rule: the training prompt
    template (``raw`` | ``alpaca``, no chat template), greedy, 8 new tokens,
    first line compared stripped; with ``scoring`` (the run's ``preprocess.json``
    ``eval`` block) the run's own length, headline metric and rows instead.
    Writes ``<dest_dir>.eval.json`` next to the artefact. Raises
    MLXUnavailableError off Apple silicon."""
    # Resolved in the body, not as a default argument: the notebook bundle
    # ships this function as a cell on its own (NOTEBOOK_CELLS), where a
    # def-time reference to the module constant is a NameError.
    if quant is None:
        quant = DEFAULT_MLX_QUANT
    if quant not in MLX_QUANTS:
        raise ValueError(f"MLX quant must be one of {sorted(MLX_QUANTS)}, got {quant!r}")
    if template not in PROMPT_TEMPLATES:
        raise ValueError(f"prompt template must be one of {PROMPT_TEMPLATES}, got {template!r}")
    _require_mlx()
    merged_dir, dest_dir = Path(merged_dir), Path(dest_dir)
    if not (merged_dir / "config.json").is_file():
        raise FileNotFoundError(f"{merged_dir} is not a merged HF checkpoint (no config.json)")
    if dest_dir.exists():
        shutil.rmtree(dest_dir)  # mlx_lm.convert refuses an existing path
    dest_dir.parent.mkdir(parents=True, exist_ok=True)
    proc = subprocess.run(
        [sys.executable, "-m", "mlx_lm", "convert", "--hf-path", str(merged_dir),
         "--mlx-path", str(dest_dir), *MLX_QUANTS[quant]],
        capture_output=True, text=True, errors="replace",
    )
    if proc.returncode != 0:
        raise MLXExportError(f"mlx_lm convert failed: {proc.stderr[-4000:]}")
    artifact = _dir_artifact(dest_dir, "mlx")

    examples = list(validation)
    _LOADED.pop(f"mlx:{dest_dir}", None)  # a re-export must not score the old weights
    t0 = time.perf_counter()
    new_tokens = int(scoring["max_new_tokens"]) if scoring else LEGACY_NEW_TOKENS
    score = _rescore(artifact, examples, "mlx-lm", lambda prompt: _mlx_generate(dest_dir, prompt, new_tokens),
                     template=template, scoring=scoring)
    elapsed = time.perf_counter() - t0
    ms = round(1000 * elapsed / score.n, 1) if score.n else None
    report_path = dest_dir.parent / f"{dest_dir.name}.eval.json"
    result = MLXExport(artifact=artifact, quant=quant, score=score, ms_per_example=ms,
                       report_path=report_path)
    report_path.write_text(json.dumps(result.to_json(), indent=2) + "\n")
    _MLX_SCORES[str(dest_dir)] = (examples, score)
    return result


def _read_validation(path: Path) -> list[dict[str, str]]:
    """A JSONL of ``{"prompt", "completion"}`` records (the materialised
    text_generation validation split)."""
    rows = []
    for i, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        rec = json.loads(line)
        if not isinstance(rec, dict) or "prompt" not in rec or "completion" not in rec:
            raise ValueError(f"{path}:{i}: expected an object with 'prompt' and 'completion'")
        rows.append({"prompt": str(rec["prompt"]), "completion": str(rec["completion"])})
    return rows


def main(argv: list[str] | None = None) -> int:
    """``python -m sakura.atelier.backends.peft mlx --merged DIR --validation
    VAL.jsonl --quant q8 --out DIR``: prints the MLXExport JSON. Exit 3 when
    the host cannot run MLX (MLXUnavailableError), so a placer can tell
    "wrong host" from a failed conversion."""
    import argparse

    p = argparse.ArgumentParser(prog="python -m sakura.atelier.backends.peft")
    sub = p.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("mlx", help="convert merged safetensors to MLX and re-score on the validation split")
    m.add_argument("--merged", required=True, type=Path, help="merged HF safetensors checkpoint dir")
    m.add_argument("--validation", required=True, type=Path, help="JSONL of {prompt, completion}")
    m.add_argument("--quant", default=DEFAULT_MLX_QUANT, choices=sorted(MLX_QUANTS))
    m.add_argument("--out", required=True, type=Path, help="MLX artefact dir to write (replaced if present)")
    m.add_argument("--template", default="raw", choices=PROMPT_TEMPLATES,
                   help="prompt template the run trained with (hp.prompt_template)")
    m.add_argument("--preprocess", type=Path, default=None,
                   help="the run's artifacts/preprocess.json: re-score with its `eval` block (length, metric, rows)")
    args = p.parse_args(argv)
    scoring = _scoring_of(json.loads(args.preprocess.read_text(encoding="utf-8"))) if args.preprocess else None
    try:
        result = export_mlx(args.merged, _read_validation(args.validation), args.out, args.quant,
                            template=args.template, scoring=scoring)
    except MLXUnavailableError as exc:
        print(f"MLXUnavailableError: {exc}", file=sys.stderr)
        return 3
    print(json.dumps(result.to_json(), indent=2))
    return 0


#: Functions the notebook shows by source, in pipeline order (CONTRACTS §2.5): helpers before
#: `train` are the Model section, those after `train`, `export` and `runtime_eval` follow that step.
NOTEBOOK_CELLS = (
    _training_options, train_eval_options, _format_prompt, _load_base_model,
    PEFTBackend.train, _eval_settings, _evaluation_rows, _length_grouped_batches, _row_losses, _generate,
    _keep_weights, _restore_weights, _exact_match, _truncate_completion,
    PEFTBackend.export, PEFTBackend._export_gguf, _gguf_quant_args, export_mlx,
    PEFTBackend.runtime_eval, _rescore, _scoring_of, _artifact_settings, _hf_load, _hf_generate,
    _hf_generate_many, _gguf_n_ctx, _load_llama_cpp,
)
PEFTBackend.NOTEBOOK_CELLS = NOTEBOOK_CELLS

BACKEND = PEFTBackend()


if __name__ == "__main__":
    raise SystemExit(main())
