"""A backend whose training loop honours ``limits`` the way the real ones do (a ``WrapUpWatch`` built with the
spec's ``max_steps``), for the engine's ``limits_applied`` tests. Loaded by name through
SAKURA_ATELIER_BACKEND_MODULE_PEFT.

Env ``FAKE_LIMITS_STEP_SECONDS`` makes every step take that long (a time budget then has something to stop);
``FAKE_LIMITS_NO_LOOP`` makes it a backend with no loop, which reports its own step cap."""

from __future__ import annotations

import os
import time
from typing import Any

from sakura.atelier import wrap_up

STEPS = 200
STEPS_PER_EPOCH = 4


class StepBackend:
    name = "peft"
    tasks = frozenset({"text_generation"})
    seen: dict[str, Any] = {}

    def train(self, spec, data, out, metrics, resume):
        from sakura.atelier.types import Checkpoint, TrainResult

        cap = wrap_up.max_steps_of(spec)
        checkpoint = Checkpoint(path=out.checkpoint, model=spec.spec.model, task=spec.spec.task)
        if os.environ.get("FAKE_LIMITS_NO_LOOP"):
            return TrainResult(model=None, metrics={"loss": 0.5}, train_seconds=0.0, checkpoint=checkpoint,
                               limits_applied={"max_steps": cap} if cap is not None and cap < STEPS else None)
        pause = float(os.environ.get("FAKE_LIMITS_STEP_SECONDS", "0"))
        watch = wrap_up.WrapUpWatch(out.root, cap)
        done = 0
        for step in range(STEPS):
            if watch.stop(step / STEPS_PER_EPOCH, step):
                break
            metrics.log("loss", 1.0 / (step + 1), step=step)
            time.sleep(pause)
            done += 1
        type(self).seen = {"train_rows": len(data.train), "validation_rows": len(data.validation), "steps": done}
        return TrainResult(model=None, metrics={"loss": 0.5}, train_seconds=0.1, checkpoint=checkpoint,
                           stopped_early=watch.stopped_early)

    def export(self, result, fmt, out):
        from sakura.atelier.types import Artifact

        p = out.artifacts / "model.safetensors"
        p.write_bytes(b"w")
        return Artifact(path=p, format=fmt, sha256="0" * 64, bytes=1)

    def runtime_eval(self, artifact, data):
        from sakura.atelier.types import RuntimeEval

        return RuntimeEval(format="safetensors", runtime="fake", metric="loss", value=0.5, n=1, matches_training=True)

    def predict(self, artifact, inputs):  # pragma: no cover
        raise NotImplementedError


BACKEND = StepBackend()
