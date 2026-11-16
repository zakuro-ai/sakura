"""A backend whose `predict` reports what the process saw, for the `serve` tests."""

from __future__ import annotations

import os
from typing import Any

from sakura.atelier.types import Artifact


class EchoBackend:
    name = "peft"
    tasks = frozenset({"text_generation", "decision"})

    def train(self, *a: Any, **k: Any) -> Any:  # pragma: no cover -- never trained
        raise NotImplementedError

    def export(self, *a: Any, **k: Any) -> Any:  # pragma: no cover
        raise NotImplementedError

    def runtime_eval(self, *a: Any, **k: Any) -> Any:  # pragma: no cover
        raise NotImplementedError

    def predict(self, artifact: Artifact, inputs: Any) -> Any:
        return {"cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"), "inputs": inputs}


BACKEND = EchoBackend()
