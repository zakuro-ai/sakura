"""A backend whose runtime check crashes for one export format and works for the others, for the engine's
guard around ``runtime_eval``. Loaded by name through SAKURA_ATELIER_BACKEND_MODULE_PEFT.

Env ``FAKE_RUNTIME_EVAL_RAISES`` names the formats whose check raises (comma separated);
``FAKE_RUNTIME_EVAL_TRAIN_VALUE`` makes the check score 0.9 on 3 rows against that training-side value."""

from __future__ import annotations

import os


class CrashingCheckBackend:
    name = "peft"
    tasks = frozenset({"text_generation"})

    def train(self, spec, data, out, metrics, resume):
        from sakura.atelier.types import Checkpoint, TrainResult

        metrics.log("loss", 0.5, step=0)
        return TrainResult(model=None, metrics={"loss": 0.5}, train_seconds=0.1,
                           checkpoint=Checkpoint(path=out.checkpoint, model=spec.spec.model, task=spec.spec.task))

    def export(self, result, fmt, out):
        from sakura.atelier.types import Artifact

        p = out.artifacts / f"model.{fmt}"
        p.write_bytes(b"w")
        return Artifact(path=p, format=fmt, sha256="0" * 64, bytes=1)

    def runtime_eval(self, artifact, data):
        from sakura.atelier.types import RuntimeEval

        if artifact.format in os.environ.get("FAKE_RUNTIME_EVAL_RAISES", "").split(","):
            raise RuntimeError("the runtime ran out of memory\nsecond line of the message " + "x" * 400)
        if os.environ.get("FAKE_RUNTIME_EVAL_TRAIN_VALUE"):  # fewer rows than the training evaluation used
            return RuntimeEval(format=artifact.format, runtime="fake", metric="loss", value=0.9, n=3,
                               matches_training=None, train_value=float(os.environ["FAKE_RUNTIME_EVAL_TRAIN_VALUE"]),
                               train_n=3)
        return RuntimeEval(format=artifact.format, runtime="fake", metric="loss", value=0.5, n=3, matches_training=True)

    def predict(self, artifact, inputs):  # pragma: no cover
        raise NotImplementedError


BACKEND = CrashingCheckBackend()
