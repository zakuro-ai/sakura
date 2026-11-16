"""A backend that trains instantly and exports one file, for CLI report tests."""

from __future__ import annotations

import hashlib
from typing import Any

from sakura.atelier.types import Artifact, RuntimeEval, TrainResult


class FakeTrain:
    name = "ludwig"
    tasks = frozenset({"tabular_classification"})

    def train(self, spec, data, out, metrics, resume) -> TrainResult:
        return TrainResult(model=None, metrics={"accuracy": 1.0}, train_seconds=0.01)

    def export(self, result, fmt, out) -> Artifact:
        out.artifacts.mkdir(parents=True, exist_ok=True)
        p = out.artifacts / f"model.{fmt}"
        p.write_bytes(b"model")
        return Artifact(path=p, format=fmt, sha256=hashlib.sha256(b"model").hexdigest(), bytes=5)

    def runtime_eval(self, artifact, data) -> RuntimeEval:
        return RuntimeEval(format=artifact.format, runtime="none", metric=None, value=None, n=0,
                           matches_training=None, reason="fake")

    def predict(self, artifact: Any, inputs: Any) -> Any:  # pragma: no cover
        return None


BACKEND = FakeTrain()
