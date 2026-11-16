"""AtelierSpec v1: the replayable description of one training job.

The spec, plus the bytes its ``data.sha256`` pins, must be enough to replay a
run. That is the whole design constraint, and it is why hyperparameters are an
*id* (``hp: image_classification/fast@1``) rather than a block of numbers: a
preset is immutable once released, so a light spec still means exactly one
set of values. ``resolved.yaml`` is the same spec with the preset expanded,
for a reader who wants the numbers without the registry.

The contract this implements is ``docs/atelier/CONTRACTS.md`` section 1.
"""

from __future__ import annotations

import hashlib
import re
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_serializer, model_validator

Task = Literal[
    "image_classification",
    "image_regression",
    "object_detection",
    "tabular_classification",
    "tabular_regression",
    "audio_classification",
    "text_generation",
    "speech_recognition",
    "time_series_forecasting",
    "text_classification",
    "decision",
    "risk_simulation",
    "reinforcement_learning",
]
#: Tasks whose data carries its own structure: no ``features`` block and no ``split`` (``decision`` splits
#: its rows by state itself; ``risk_simulation`` reads three named tables and fits nothing to hold out).
SELF_STRUCTURED_TASKS = ("decision", "risk_simulation")
Format = Literal["onnx", "gguf", "mlx", "torchscript", "safetensors", "ggml", "joblib", "json",
                 "onnx_int8"]  # joblib: classic_ml's pickled sklearn pipeline; json: risk_simulation's results
InputType = Literal["image", "category", "number", "binary", "text", "audio", "timeseries"]
OutputType = Literal["category", "binary", "number", "boxes", "text", "timeseries"]
DataFormat = Literal[
    "hf", "image_folder", "yolo", "csv", "parquet", "audio_folder", "jsonl", "asr_manifest"
]
Strategy = Literal["best_price", "best_availability", "best_latency"]
#: How a ``validation_fraction`` holdout picks its rows (CONTRACTS §1.3).
SplitStrategy = Literal["stratified", "random"]

#: ``<task>/<name>@<version>``. The version is what makes a preset immutable:
#: changing its values means releasing ``@2``, never editing ``@1``.
PRESET_ID = re.compile(r"^[a-z_]+/[a-z0-9_.-]+@[0-9]+$")


class _Strict(BaseModel):
    # Unknown keys are an error. A typo in a replayable spec that is silently
    # ignored is a run that does not replay.
    model_config = ConfigDict(extra="forbid", frozen=True)


#: A value of a split column as a spec spells it. Compared as text (CONTRACTS §1.3).
SplitValue = str | int | float | bool
#: Formats whose rows come from one table, where a split column / stratification applies.
_TABLE_FORMATS = ("csv", "parquet", "jsonl")


class Split(_Strict):
    train: str | SplitValue | list[SplitValue] | None = None
    validation: str | SplitValue | list[SplitValue] | None = None
    validation_fraction: float | None = Field(default=None, gt=0, lt=1)
    #: Name of a table column that says which rows are train and which are
    #: validation. ``train`` / ``validation`` then hold the VALUE(S) of that
    #: column instead of split expressions (CONTRACTS §1.3).
    column: str | None = None
    #: Column values whose rows are dropped (e.g. a ``test`` partition).
    exclude: list[SplitValue] | None = None
    #: Columns a ``validation_fraction`` holdout is stratified by (§1.3).
    stratify_by: list[str] | None = None
    #: ``stratified`` draws the holdout per class (per output label), ``random``
    #: draws it from the whole set. Unset means the default: stratified for a
    #: classification output, random otherwise (:func:`split_strategy`).
    strategy: SplitStrategy | None = None
    #: Column whose distinct values are the groups (a conversation, a patient, a customer) a
    #: ``validation_fraction`` holdout keeps whole: every row of a group lands on one side (§1.3).
    group_by: str | None = None

    @model_validator(mode="after")
    def _one_way(self) -> Split:
        explicit = self.train is not None or self.validation is not None
        if explicit and self.validation_fraction is not None:
            raise ValueError("give explicit splits or validation_fraction, not both")
        if not explicit and self.validation_fraction is None:
            raise ValueError("a split needs train/validation or validation_fraction")
        if self.column is not None:
            if not self.column.strip():
                raise ValueError("split.column must name a column")
            if self.train is None or self.validation is None:
                raise ValueError("split.column needs both train and validation: the column values for each side")
            if self.validation_fraction is not None:
                raise ValueError("split.column and validation_fraction are two different ways to split; give one")
            for role in ("train", "validation"):
                vals = getattr(self, role)
                if isinstance(vals, list) and not vals:
                    raise ValueError(f"split.{role} lists no values")
            clash = {str(v) for v in _listed(self.train)} & {str(v) for v in _listed(self.validation)}
            if clash:
                raise ValueError(f"split.train and split.validation both claim {sorted(clash)} of column {self.column!r}")
            bad = {str(v) for v in _listed(self.exclude)} & {str(v) for v in _listed(self.train) + _listed(self.validation)}
            if bad:
                raise ValueError(f"split.exclude lists {sorted(bad)}, which train/validation use")
        else:
            if self.exclude is not None:
                raise ValueError("split.exclude only applies with split.column")
            for role in ("train", "validation"):
                v = getattr(self, role)
                if v is not None and not isinstance(v, str):
                    raise ValueError(f"split.{role} must be a split expression string unless split.column is set")
        if self.stratify_by is not None:
            if self.validation_fraction is None:
                raise ValueError("stratify_by needs validation_fraction: it stratifies a random holdout "
                                 "(with split.column the data already says who is where)")
            if not self.stratify_by or len(set(self.stratify_by)) != len(self.stratify_by)                     or any(not c or not c.strip() for c in self.stratify_by):
                raise ValueError("stratify_by must list one or more distinct column names")
        if self.strategy is not None and self.validation_fraction is None:
            raise ValueError("split.strategy needs validation_fraction: explicit splits and split.column "
                             "already say which rows are which")
        if self.strategy == "random" and self.stratify_by:
            raise ValueError("split.strategy random contradicts stratify_by")
        if self.group_by is not None:
            if not self.group_by.strip():
                raise ValueError("split.group_by must name a column")
            if self.validation_fraction is None:
                raise ValueError("split.group_by needs validation_fraction: it draws the held-out share over "
                                 "distinct group values (with split.column the data already says who is where)")
        return self

    @property
    def train_expr(self) -> str | None:
        """``train`` as a split expression; None when it holds column values
        instead (``column`` set -- those specs only reach table splitting)."""
        return self.train if isinstance(self.train, str) else None

    @property
    def validation_expr(self) -> str | None:
        """``validation`` as a split expression; see :attr:`train_expr`."""
        return self.validation if isinstance(self.validation, str) else None

    @model_serializer(mode="wrap")
    def _omit_unset_split_fields(self, handler: Any) -> dict[str, Any]:
        # The four fields above are newer than every spec already in the
        # wild. Dumping them as nulls would change `resolved.yaml`, and with
        # it the spec hash, of runs that never used them.
        out: dict[str, Any] = handler(self)
        for key in ("column", "exclude", "stratify_by", "strategy", "group_by"):
            if out.get(key) is None:
                out.pop(key, None)
        return out


def _listed(v: Any) -> list[Any]:
    return [] if v is None else (v if isinstance(v, list) else [v])


class Data(_Strict):
    uri: str
    #: sha256 of the materialised dataset manifest. Empty means "fill it in":
    #: the engine records it in resolved.yaml, and a replay then pins it.
    sha256: str = ""
    format: DataFormat
    #: Required for every task but ``decision``, whose rows are split by state inside the
    #: engine (80/10/10), and ``risk_simulation``, which fits nothing to hold out (checked in
    #: :class:`AtelierSpec`).
    split: Split | None = None

    @model_serializer(mode="wrap")
    def _omit_absent_split(self, handler: Any) -> dict[str, Any]:
        # A decision spec has no split; dumping `split: null` would put a key in its
        # resolved.yaml that the spec never had. Every other task always has one.
        out: dict[str, Any] = handler(self)
        if out.get("split") is None:
            out.pop("split", None)
        return out

    @model_validator(mode="before")
    @classmethod
    def _hub_fraction(cls, v: Any) -> Any:
        # The hub writes `data.validation_fraction` (and `data.stratify_by`)
        # beside `split` (its own schema's shape); the engine keeps every
        # split rule in `Split`. Same spec, two spellings -- a spec.yaml
        # downloaded from the hub must replay here unchanged.
        if isinstance(v, dict) and ("validation_fraction" in v or "stratify_by" in v):
            v = dict(v)
            for key in ("validation_fraction", "stratify_by"):
                if key not in v:
                    continue
                val = v.pop(key)
                if val is not None:
                    if key == "stratify_by":
                        if v.get("split") is not None and not isinstance(v["split"], dict):
                            raise ValueError("data.split must be a mapping")
                        v["split"] = {**(v.get("split") or {}), "stratify_by": val}
                    elif v.get("split") and any(v["split"].get(k) is not None for k in ("train", "validation", "column")):
                        raise ValueError("data: set either split or validation_fraction, not both")
                    else:
                        v["split"] = {**(v.get("split") or {}), "validation_fraction": val}
        return v

    @model_validator(mode="after")
    def _table_only_splits(self) -> Data:
        if self.split is None:
            return self
        if (self.split.column is not None or self.split.stratify_by or self.split.group_by is not None) \
                and self.format not in _TABLE_FORMATS:
            what = ("split.column" if self.split.column is not None
                    else "stratify_by" if self.split.stratify_by else "split.group_by")
            raise ValueError(f"data.{what} needs a csv, parquet or jsonl table, not data.format {self.format!r}")
        return self

    @field_validator("sha256")
    @classmethod
    def _hex(cls, v: str) -> str:
        if v and not re.fullmatch(r"[0-9a-f]{64}", v):
            raise ValueError("data.sha256 must be 64 lowercase hex characters or empty")
        return v

    @model_validator(mode="after")
    def _synthetic_source(self) -> Data:
        # A generated dataset (SAK-0019): the kind, its parameters and the format it is written as are
        # checked here, so `validate` refuses a bad uri without running anything.
        if self.uri.startswith("synthetic://"):
            from sakura.atelier import synthetic

            synthetic.check(self.uri, self.format)
        return self


class InputFeature(_Strict):
    name: str
    type: InputType


class OutputFeature(_Strict):
    name: str
    type: OutputType


class EvalColumns(_Strict):
    """Columns of a text-generation table that ride along on validation records and are never fed to the model
    (§1.3): ``reference`` (text or JSON the prediction is compared with instead of the completion), ``final``
    (a bool column: the rows ``eval_scope: final`` scores), ``group`` (the column per-group aggregates use) and
    ``keep`` (columns carried untouched, into ``predictions.jsonl``)."""

    reference: str | None = None
    final: str | None = None
    group: str | None = None
    keep: list[str] | None = None

    @model_validator(mode="after")
    def _something_named(self) -> EvalColumns:
        named = [c for c in (self.reference, self.final, self.group, *(self.keep or [])) if c is not None]
        if not named:
            raise ValueError("features.eval must set at least one of reference, final, group, keep")
        if any(not c.strip() for c in named):
            raise ValueError("features.eval names columns: a name cannot be empty")
        if self.keep is not None and len(set(self.keep)) != len(self.keep):
            raise ValueError("features.eval.keep must list distinct columns")
        return self

    @property
    def columns(self) -> list[str]:
        """Every column named, each once."""
        named = [self.reference, self.final, self.group, *(self.keep or [])]
        return list(dict.fromkeys(c for c in named if c is not None))

    @model_serializer(mode="wrap")
    def _omit_unset(self, handler: Any) -> dict[str, Any]:
        out: dict[str, Any] = handler(self)
        return {k: v for k, v in out.items() if v is not None}


class Features(_Strict):
    inputs: list[InputFeature] = Field(min_length=1)
    output: OutputFeature
    #: Text generation only: the evaluation columns. Left out of the dump when unset, so a spec without it
    #: hashes as before.
    eval: EvalColumns | None = None

    @model_serializer(mode="wrap")
    def _omit_unset_eval(self, handler: Any) -> dict[str, Any]:
        out: dict[str, Any] = handler(self)
        if out.get("eval") is None:
            out.pop("eval", None)
        return out


class Artifact(_Strict):
    uri: str
    sha256: str = ""

    @field_validator("sha256", mode="before")
    @classmethod
    def _none_is_unpinned(cls, v: Any) -> Any:
        return "" if v is None else v  # the hub writes null for "not pinned"


class Placement(_Strict):
    """Read by the hub and zc, ignored by the engine: where a job runs is not
    part of what it computes."""

    strategy: Strategy = "best_price"
    gpus: int = Field(default=1, ge=0)
    min_vram_gb: int = Field(default=0, ge=0)
    #: The hub's: a GPU job the customer also allowed on a CPU at its longer ETA.
    cpu_fallback: bool = False

    @field_validator("min_vram_gb", mode="before")
    @classmethod
    def _none_is_any(cls, v: Any) -> Any:
        return 0 if v is None else v  # the hub writes null for "any GPU"


AUGMENTATION_OPS = ("horizontal_flip", "vertical_flip", "random_crop", "rotate", "brightness", "contrast", "blur")


class Augmentation(_Strict):
    """Image augmentation for the TRAINING split only (validation and test images are never touched).

    ``ops`` default to a horizontal flip and a random crop. Only the ``image_classification`` task on the Ludwig
    backend applies it; the compiled config (and ``report.json`` ``evaluation.augmentation``) records the ops."""

    enabled: bool = False
    ops: list[str] = Field(default_factory=lambda: ["horizontal_flip", "random_crop"])

    @field_validator("ops")
    @classmethod
    def _known_ops(cls, v: list[str]) -> list[str]:
        bad = [o for o in v if o not in AUGMENTATION_OPS]
        if bad:
            raise ValueError(f"unknown augmentation op(s) {bad}; known: {list(AUGMENTATION_OPS)}")
        return list(dict.fromkeys(v))


class Limits(_Strict):
    """Caps for a run that must finish in minutes (a smoke run, SAK-0019). Every field is optional; a
    limit that is not set changes nothing.

    Rows are capped once, when the dataset is materialised, after the split; ``max_steps`` bounds every
    backend's training loop; ``time_budget_s`` raises the wrap-up signal that many seconds after training
    starts. ``report.json`` says what the limits changed (``limits_applied``, CONTRACTS §2.1)."""

    max_steps: int | None = Field(default=None, gt=0, strict=True)
    max_train_rows: int | None = Field(default=None, gt=0, strict=True)
    max_eval_rows: int | None = Field(default=None, gt=0, strict=True)
    time_budget_s: float | None = Field(default=None, gt=0, allow_inf_nan=False)

    @field_validator("time_budget_s", mode="before")
    @classmethod
    def _a_number_of_seconds(cls, v: Any) -> Any:
        if isinstance(v, bool) or (v is not None and not isinstance(v, (int, float))):
            raise ValueError("limits.time_budget_s must be a number of seconds")
        return v

    @model_serializer(mode="wrap")
    def _omit_unset_limits(self, handler: Any) -> dict[str, Any]:
        out: dict[str, Any] = handler(self)
        return {k: v for k, v in out.items() if v is not None}


class CurationNearDuplicates(_Strict):
    """Near-duplicate text removal: rows whose normalised text is at least ``threshold`` similar (Jaccard
    over character 5-grams) to an earlier kept row are dropped (``sakura/atelier/curation.py``)."""

    #: The text column compared; unset = the spec's first text input.
    column: str | None = None
    threshold: float = Field(default=0.9, gt=0, le=1, allow_inf_nan=False)

    @model_serializer(mode="wrap")
    def _omit_unset(self, handler: Any) -> dict[str, Any]:
        out: dict[str, Any] = handler(self)
        return {k: v for k, v in out.items() if v is not None}


class CurationTextLength(_Strict):
    """Drop rows whose text is shorter than ``min`` or longer than ``max`` characters."""

    column: str | None = None
    min: int | None = Field(default=None, ge=0, strict=True)
    max: int | None = Field(default=None, gt=0, strict=True)

    @model_validator(mode="after")
    def _a_bound(self) -> CurationTextLength:
        if self.min is None and self.max is None:
            raise ValueError("curation.text_length needs min or max")
        if self.min is not None and self.max is not None and self.min > self.max:
            raise ValueError("curation.text_length.min is above max")
        return self

    @model_serializer(mode="wrap")
    def _omit_unset(self, handler: Any) -> dict[str, Any]:
        out: dict[str, Any] = handler(self)
        return {k: v for k, v in out.items() if v is not None}


class CurationClasses(_Strict):
    """Keep only the classes in ``allow``, or drop the classes in ``deny``. Compared as text."""

    allow: list[SplitValue] | None = None
    deny: list[SplitValue] | None = None

    @model_validator(mode="after")
    def _one_list(self) -> CurationClasses:
        if self.allow is None and self.deny is None:
            raise ValueError("curation.classes needs allow or deny")
        if self.allow is not None and self.deny is not None:
            raise ValueError("curation.classes takes allow or deny, not both")
        for name in ("allow", "deny"):
            if getattr(self, name) is not None and not getattr(self, name):
                raise ValueError(f"curation.classes.{name} lists no class")
        return self

    @model_serializer(mode="wrap")
    def _omit_unset(self, handler: Any) -> dict[str, Any]:
        out: dict[str, Any] = handler(self)
        return {k: v for k, v in out.items() if v is not None}


class Curation(_Strict):
    """Data curation (SAK-0019, issue 202), applied once when the dataset is materialised, before
    the split and before ``limits``. Every rule is optional; a rule that is not set does nothing. The rules
    run in the order of the fields below and a row is attributed to the first rule that drops it.
    ``report.json`` gets a ``curation`` block and the run delivers a curated manifest (CONTRACTS §1.6)."""

    #: Sample ids to drop: a table row (``17``, ``row:17``, or a report's ``validation:17``) or a file's
    #: path relative to the dataset (``airplane/022.png``). An id that names nothing is reported, not fatal.
    exclude_samples: list[str | int] | None = None
    drop_missing_target: bool = False
    classes: CurationClasses | None = None
    text_length: CurationTextLength | None = None
    exact_duplicates: bool = False
    near_duplicates: CurationNearDuplicates | None = None

    @field_validator("exclude_samples", mode="before")
    @classmethod
    def _ids_are_text_or_row_numbers(cls, v: Any) -> Any:
        if isinstance(v, list) and any(isinstance(x, bool) or not isinstance(x, (str, int)) for x in v):
            raise ValueError("curation.exclude_samples lists sample ids: row numbers or strings")
        return v

    @field_validator("near_duplicates", mode="before")
    @classmethod
    def _true_is_the_default_threshold(cls, v: Any) -> Any:
        if v is True:
            return {}
        return None if v is False else v

    @model_serializer(mode="wrap")
    def _omit_unset(self, handler: Any) -> dict[str, Any]:
        out: dict[str, Any] = handler(self)
        return {k: v for k, v in out.items() if v is not None and v is not False}


#: What a ``curation`` block can be applied to: one table read whole, or a class-per-folder tree.
_CURATION_FOLDER_FORMATS = ("image_folder", "audio_folder")
# reinforcement_learning: its triples table is read by kg.materialise_triples, which applies no curation
_CURATION_REFUSED_TASKS = ("decision", "risk_simulation", "image_regression", "time_series_forecasting",
                           "reinforcement_learning")


class AtelierSpec(_Strict):
    #: The hub's discriminator between its llm / asr / atelier job specs.
    #: Optional here (a hand-written spec has none) but accepted, so the
    #: spec.yaml a customer downloads from the hub replays as-is.
    model_type: Literal["atelier"] | None = None
    atelier: Literal[1]
    name: str
    task: Task
    data: Data
    #: Required for every task but ``decision``: a decision row carries its own typed
    #: questions and answers, so there are no columns to name (checked below).
    features: Features | None = None
    #: Dotted paths the agent inferred rather than the user stating them. The
    #: hub refuses to accept a spec whose guesses are not confirmed.
    guessed: list[str] = Field(default_factory=list)
    confirmed: bool = False
    model: str
    hp: str
    overrides: dict[str, Any] = Field(default_factory=dict)
    #: Preset expanded + overrides applied. Empty on a hand-authored spec;
    #: set on resolved.yaml (registry.resolve fills this in). A spec that
    #: already carries this is treated as already-resolved -- the preset
    #: file is not re-read, which is what makes resolved.yaml replay exactly
    #: even if the preset it came from later changes (it should not, but the
    #: spec itself must not depend on that).
    hyperparameters: dict[str, Any] | None = None
    #: Backend-native keys merged last -- the "deep dive at will" escape hatch
    #: (e.g. raw Ludwig config fragments). Recorded in resolved.yaml verbatim.
    backend_config: dict[str, Any] = Field(default_factory=dict)
    seed: int = 42
    #: Formats to deliver; must be a subset of the model's registry ``exports``
    #: (checked in registry.resolve). Empty only for an evaluation-only model.
    export: list[Format] = Field(default_factory=list)
    resume_from: Artifact | None = None
    placement: Placement = Field(default_factory=Placement)
    #: Training-split image augmentation. ``None`` (absent) = the run behaves exactly as before; the key is
    #: left out of the dump, so `resolved.yaml` and the spec hash of every older spec do not change.
    augmentation: Augmentation | None = None
    #: Caps for a smoke run (SAK-0019). ``None`` (absent) = no cap; the key is left out of the dump, so
    #: `resolved.yaml` and the spec hash of every spec without it do not change. When set it stays in
    #: `resolved.yaml`, so a replay is capped the same way.
    limits: Limits | None = None
    #: Data curation (SAK-0019). ``None`` (absent) = the dataset is used as it is; the key is left out of
    #: the dump, so a spec without it hashes as before. When set it stays in `resolved.yaml`.
    curation: Curation | None = None

    @model_serializer(mode="wrap")
    def _omit_unset_augmentation(self, handler: Any) -> dict[str, Any]:
        out: dict[str, Any] = handler(self)
        if out.get("augmentation") is None:
            out.pop("augmentation", None)
        if not out.get("limits"):  # absent, or a block that sets nothing
            out.pop("limits", None)
        if not out.get("curation"):  # absent, or a block that sets nothing
            out.pop("curation", None)
        if out.get("features") is None:  # a decision spec has none; every other task always does
            out.pop("features", None)
        return out

    @model_validator(mode="after")
    def _features_and_split_follow_the_task(self) -> AtelierSpec:
        # Runs before the validators below, which read features / split and may assume them.
        if self.task == "decision":
            if self.features is not None:
                raise ValueError("features does not apply to task decision: its rows carry their own typed "
                                 "questions and answers")
            if self.data.split is not None:
                raise ValueError("data.split does not apply to task decision: the engine splits by state "
                                 "itself (80% train, 10% calibration, 10% eval)")
        elif self.task == "risk_simulation":
            if self.features is not None:
                raise ValueError("features does not apply to task risk_simulation: its dataset is three named "
                                 "tables (losses, factors, exposure)")
            if self.data.split is not None:
                raise ValueError("data.split does not apply to task risk_simulation: nothing is fitted to hold "
                                 "a split out against")
        else:
            if self.features is None:
                raise ValueError(f"features is required for task {self.task!r}")
            if self.data.split is None:
                raise ValueError(f"data.split is required for task {self.task!r}")
            if self.features.eval is not None:
                if self.task != "text_generation":
                    raise ValueError(f"features.eval only applies to task text_generation, not {self.task!r}")
                if self.data.format not in _TABLE_FORMATS:
                    raise ValueError(f"features.eval needs a csv, parquet or jsonl table, not data.format "
                                     f"{self.data.format!r}")
        return self

    @model_validator(mode="after")
    def _limits_follow_the_task(self) -> AtelierSpec:
        if self.limits is None or self.task != "risk_simulation":
            return self
        rows = [k for k in ("max_train_rows", "max_eval_rows") if getattr(self.limits, k) is not None]
        if rows:
            raise ValueError(f"limits.{rows[0]} does not apply to task risk_simulation: its three tables are "
                             "read whole (limits.max_steps caps the Monte Carlo iterations)")
        return self

    @model_validator(mode="after")
    def _int8_beside_the_float_graph(self) -> AtelierSpec:
        # onnx_int8 is the float ONNX graph with its weights quantised (SAK-0019): it is delivered and
        # re-scored next to the float one, so a report always shows both.
        if "onnx_int8" in self.export and "onnx" not in self.export:
            raise ValueError("export onnx_int8 is quantised from the float ONNX graph and delivered next to it: "
                             "list onnx as well (export: [onnx, onnx_int8])")
        return self

    @model_validator(mode="after")
    def _curation_follows_the_data(self) -> AtelierSpec:
        c = self.curation
        if c is None or self.features is None:
            if c is not None:
                raise ValueError(f"curation is not supported for task {self.task!r}: it applies to csv, parquet "
                                 "and jsonl tables and to image and audio folders")
            return self
        folder = self.data.format in _CURATION_FOLDER_FORMATS
        if self.task in _CURATION_REFUSED_TASKS or not (folder or self.data.format in _TABLE_FORMATS):
            raise ValueError(f"curation is not supported for task {self.task!r} with data.format "
                             f"{self.data.format!r}: it applies to csv, parquet and jsonl tables and to image "
                             "and audio folders")
        if folder:
            for rule, is_set in (("drop_missing_target", c.drop_missing_target), ("text_length", c.text_length),
                                 ("near_duplicates", c.near_duplicates)):
                if is_set:
                    raise ValueError(f"curation.{rule} does not apply to a folder dataset (image_folder, "
                                     "audio_folder): exclude_samples, exact_duplicates (by file hash) and "
                                     "classes do")
        if c.classes is not None and self.features.output.type not in ("category", "binary"):
            raise ValueError("curation.classes needs a category or binary output "
                             f"(output type {self.features.output.type!r})")
        from sakura.atelier.curation import text_column  # no training stack: pandas-free until it runs

        for rule in ("text_length", "near_duplicates"):
            if getattr(c, rule) is not None:
                text_column(self, getattr(c, rule).column, rule)
        return self

    @model_validator(mode="after")
    def _triples_features(self) -> AtelierSpec:
        # reinforcement_learning walks a knowledge graph: `features` names the three columns of its triples
        # table (CONTRACTS §2.2, backend rl).
        if self.task != "reinforcement_learning" or self.features is None:
            return self
        types = [f.type for f in self.features.inputs] + [self.features.output.type]
        if types != ["category"] * 3:
            raise ValueError("task reinforcement_learning reads a triples table: features needs exactly two "
                             "category inputs (the head column, then the relation column) and a category "
                             "output (the tail column)")
        if self.data.format not in ("csv", "parquet"):
            raise ValueError(f"task reinforcement_learning reads a csv or parquet triples table, not "
                             f"data.format {self.data.format!r}")
        return self

    @field_validator("hp")
    @classmethod
    def _preset_id(cls, v: str) -> str:
        if not PRESET_ID.match(v):
            raise ValueError(f"hp must look like '<task>/<name>@<version>', got {v!r}")
        return v

    @model_validator(mode="after")
    def _preset_matches_task(self) -> AtelierSpec:
        # A tabular preset on an image job is not a configuration choice, it is
        # a mistake -- catch it before anything downloads.
        family = self.hp.split("/", 1)[0]
        if not self.task.startswith(family) and family != self.task:
            raise ValueError(f"preset {self.hp!r} is for {family!r}, not {self.task!r}")
        return self

    @model_validator(mode="after")
    def _no_stratified_forecast(self) -> AtelierSpec:
        split = self.data.split
        if split is None or self.features is None:  # decision: no holdout of its own
            return self
        if self.task == "time_series_forecasting" and split.stratify_by:
            raise ValueError("data.stratify_by does not apply to time_series_forecasting: "
                             "its holdout is chronological")
        if split.strategy == "stratified" and not split.stratify_by and not _per_class_split(self):
            raise ValueError(f"data.split.strategy stratified needs a category or binary output to stratify by "
                             f"(task {self.task!r}, output type {self.features.output.type!r}); "
                             "name the columns in stratify_by instead")
        return self

    @property
    def unconfirmed_guesses(self) -> list[str]:
        return [] if self.confirmed else list(self.guessed)


#: Tasks whose holdout is never stratified: chronological, or not a label.
_UNSTRATIFIED_TASKS = ("time_series_forecasting", "text_generation", "speech_recognition", "object_detection",
                       "reinforcement_learning")  # its output is the tail entity: nothing to draw per class


def _per_class_split(spec: AtelierSpec) -> bool:
    """A holdout of this spec can be drawn per class of its output label."""
    if spec.features is None:
        return False
    return spec.features.output.type in ("category", "binary") and spec.task not in _UNSTRATIFIED_TASKS


def split_strategy(spec: AtelierSpec) -> SplitStrategy | None:
    """How ``spec``'s ``validation_fraction`` holdout is drawn: the spec's own
    ``split.strategy``, else stratified for a classification output (so a
    small class keeps its share of validation instead of a lucky handful,
    BE-18) and random for everything else. None when there is no holdout to
    draw (explicit splits, a split column)."""
    split = spec.data.split
    if split is None or split.validation_fraction is None:
        return None
    if split.strategy is not None:
        return split.strategy
    if split.stratify_by:
        return "stratified"
    if split.group_by is not None:
        return "random"  # whole groups are drawn at random unless stratify_by says otherwise
    return "stratified" if _per_class_split(spec) else "random"


def features_of(spec: AtelierSpec) -> Features:
    """The ``features`` of a table task. A ``decision`` spec has none (its rows carry their own
    typed questions), so a table path that reaches one is a bug, and it says so by name."""
    if spec.features is None:
        raise ValueError(f"task {spec.task!r} has no features block")
    return spec.features


def split_of(data: Data) -> Split:
    """The ``split`` of a table task's data. A ``decision`` spec has none (the engine splits by
    state itself), so a table path that reaches one is a bug, and it says so by name."""
    if data.split is None:
        raise ValueError("this data block has no split (task decision splits by state itself)")
    return data.split


def load_spec(text: str) -> AtelierSpec:
    """Parse and validate. Raises ``ValueError`` with every problem at once."""
    raw = yaml.safe_load(text)
    if not isinstance(raw, dict):
        raise ValueError("an AtelierSpec is a YAML mapping")
    return AtelierSpec.model_validate(raw)


def dump_spec(spec: AtelierSpec | dict[str, Any]) -> str:
    """Canonical YAML: stable key order, so the same spec always hashes the same."""
    data = spec.model_dump(mode="json") if isinstance(spec, AtelierSpec) else spec
    return str(yaml.safe_dump(data, sort_keys=True, default_flow_style=False, allow_unicode=True))


def spec_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()
