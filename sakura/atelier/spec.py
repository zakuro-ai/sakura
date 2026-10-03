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
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

Task = Literal[
    "image_classification",
    "object_detection",
    "tabular_classification",
    "tabular_regression",
    "audio_classification",
    "text_generation",
    "speech_recognition",
]
Format = Literal["onnx", "gguf", "mlx", "torchscript", "safetensors"]
InputType = Literal["image", "category", "number", "binary", "text", "audio"]
OutputType = Literal["category", "binary", "number", "boxes", "text"]
DataFormat = Literal[
    "hf", "image_folder", "yolo", "csv", "parquet", "audio_folder", "jsonl", "asr_manifest"
]
Strategy = Literal["best_price", "best_availability", "best_latency"]

#: ``<task>/<name>@<version>``. The version is what makes a preset immutable:
#: changing its values means releasing ``@2``, never editing ``@1``.
PRESET_ID = re.compile(r"^[a-z_]+/[a-z0-9_.-]+@[0-9]+$")


class _Strict(BaseModel):
    # Unknown keys are an error. A typo in a replayable spec that is silently
    # ignored is a run that does not replay.
    model_config = ConfigDict(extra="forbid", frozen=True)


class Split(_Strict):
    train: str | None = None
    validation: str | None = None
    validation_fraction: float | None = Field(default=None, gt=0, lt=1)

    @model_validator(mode="after")
    def _one_way(self) -> Split:
        explicit = self.train is not None or self.validation is not None
        if explicit and self.validation_fraction is not None:
            raise ValueError("give explicit splits or validation_fraction, not both")
        if not explicit and self.validation_fraction is None:
            raise ValueError("a split needs train/validation or validation_fraction")
        return self


class Data(_Strict):
    uri: str
    #: sha256 of the materialised dataset manifest. Empty means "fill it in":
    #: the engine records it in resolved.yaml, and a replay then pins it.
    sha256: str = ""
    format: DataFormat
    split: Split

    @model_validator(mode="before")
    @classmethod
    def _hub_fraction(cls, v: Any) -> Any:
        # The hub writes `data.validation_fraction` beside `split` (its own
        # schema's shape); the engine keeps every split rule in `Split`.
        # Same spec, two spellings -- a spec.yaml downloaded from the hub
        # must replay here unchanged.
        if isinstance(v, dict) and "validation_fraction" in v:
            v = dict(v)
            frac = v.pop("validation_fraction")
            if frac is not None:
                if v.get("split"):
                    raise ValueError("data: set either split or validation_fraction, not both")
                v["split"] = {"validation_fraction": frac}
        return v

    @field_validator("sha256")
    @classmethod
    def _hex(cls, v: str) -> str:
        if v and not re.fullmatch(r"[0-9a-f]{64}", v):
            raise ValueError("data.sha256 must be 64 lowercase hex characters or empty")
        return v


class InputFeature(_Strict):
    name: str
    type: InputType


class OutputFeature(_Strict):
    name: str
    type: OutputType


class Features(_Strict):
    inputs: list[InputFeature] = Field(min_length=1)
    output: OutputFeature


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

    @field_validator("min_vram_gb", mode="before")
    @classmethod
    def _none_is_any(cls, v: Any) -> Any:
        return 0 if v is None else v  # the hub writes null for "any GPU"


class AtelierSpec(_Strict):
    #: The hub's discriminator between its llm / asr / atelier job specs.
    #: Optional here (a hand-written spec has none) but accepted, so the
    #: spec.yaml a customer downloads from the hub replays as-is.
    model_type: Literal["atelier"] | None = None
    atelier: Literal[1]
    name: str
    task: Task
    data: Data
    features: Features
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

    @property
    def unconfirmed_guesses(self) -> list[str]:
        return [] if self.confirmed else list(self.guessed)


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
