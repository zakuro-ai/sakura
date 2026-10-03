"""Runtime objects passed between the engine and its backends.

Kept free of any training stack: ``import sakura.atelier`` must work in an
environment with no Ludwig, ultralytics, TabPFN or peft installed, because the
runner validates specs and serves metrics without them, and each backend lives
in its own venv. A backend imports its stack inside its own methods.

Contract: ``docs/atelier/CONTRACTS.md`` section 2.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Protocol, runtime_checkable

from sakura.atelier.spec import AtelierSpec


@dataclass(frozen=True)
class ResolvedSpec:
    """The spec with its preset expanded. ``hyperparameters`` is the preset's
    values with ``overrides`` applied; ``spec`` keeps ``hp`` for provenance."""

    spec: AtelierSpec
    hyperparameters: dict[str, Any]
    backend: str


@dataclass(frozen=True)
class MaterialisedData:
    """The dataset on local disk, split, with the manifest that pins it."""

    root: Path
    train: Any
    validation: Any
    manifest_sha256: str
    #: Ordered class names for category / boxes outputs; None for regression
    #: and text generation.
    labels: list[str] | None = None
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RunDir:
    root: Path

    @property
    def artifacts(self) -> Path:
        return self.root / "artifacts"

    @property
    def checkpoint(self) -> Path:
        return self.root / "checkpoint"

    @property
    def metrics_path(self) -> Path:
        return self.root / "metrics.jsonl"

    def ensure(self) -> RunDir:
        for p in (self.root, self.artifacts, self.checkpoint):
            p.mkdir(parents=True, exist_ok=True)
        return self


@dataclass(frozen=True)
class Checkpoint:
    path: Path
    #: What produced it, so a resume can refuse a checkpoint from another model.
    model: str
    task: str


@dataclass
class TrainResult:
    #: Backend-owned handle to the trained model (a LudwigModel, a YOLO, …).
    model: Any
    metrics: dict[str, float]
    train_seconds: float
    checkpoint: Checkpoint | None = None


@dataclass(frozen=True)
class Artifact:
    path: Path
    format: str
    sha256: str
    bytes: int


@dataclass(frozen=True)
class RuntimeEval:
    format: str
    runtime: str
    metric: str
    #: None when the format could not be re-scored -- ``reason`` then says why.
    #: An export is never dropped from the report for being unscorable.
    value: float | None
    n: int
    matches_training: bool | None
    reason: str | None = None


class MetricsSink:
    """Append-only ``metrics.jsonl``, flushed per line.

    Flushed per line because the runner tails this file while training runs:
    a buffered point is a point the live chart does not have yet.
    """

    SPLITS = frozenset({"train", "validation"})

    def __init__(self, path: Path, clock: Callable[[], float] = time.time) -> None:
        self.path = path
        self._clock = clock
        path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = open(path, "a", encoding="utf-8")

    def log(self, name: str, value: float, *, step: int, split: str = "train",
            epoch: float | None = None) -> None:
        if split not in self.SPLITS:
            raise ValueError(f"split must be one of {sorted(self.SPLITS)}, got {split!r}")
        value = float(value)
        if value != value:  # NaN: a diverged run says so in the log, not as a hole in the curve
            raise ValueError(f"metric {name!r} is NaN at step {step}")
        point = {"t": round(self._clock(), 3), "step": int(step),
                 "epoch": None if epoch is None else float(epoch),
                 "split": split, "name": name, "value": value}
        self._fh.write(json.dumps(point, separators=(",", ":")) + "\n")
        self._fh.flush()
        if os.environ.get("SAKURA_METRICS_FSYNC"):
            os.fsync(self._fh.fileno())

    def close(self) -> None:
        self._fh.close()

    def __enter__(self) -> MetricsSink:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


def read_metrics(path: Path, since: int = 0) -> tuple[list[dict[str, Any]], int]:
    """Points from line ``since`` on, and the next line number to ask for.

    A trailing line without its newline is a write in progress and is left for
    the next call rather than parsed half-written.
    """
    if not path.exists():
        return [], since
    points: list[dict[str, Any]] = []
    n = 0
    with open(path, encoding="utf-8") as fh:
        for n, line in enumerate(fh, start=1):
            if n <= since:
                continue
            if not line.endswith("\n"):
                return points, n - 1
            points.append(json.loads(line))
    return points, max(n, since)


@runtime_checkable
class Backend(Protocol):
    name: str
    tasks: frozenset[str]

    def train(self, spec: ResolvedSpec, data: MaterialisedData, out: RunDir,
              metrics: MetricsSink, resume: Checkpoint | None) -> TrainResult: ...

    def export(self, result: TrainResult, fmt: str, out: RunDir) -> Artifact: ...

    def runtime_eval(self, artifact: Artifact, data: MaterialisedData) -> RuntimeEval: ...

    def predict(self, artifact: Artifact, inputs: Any) -> Any: ...
