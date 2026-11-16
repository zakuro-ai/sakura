"""A backend with a training loop that honours the wrap-up signal, for the engine's wrap-up tests.
Loaded by name through SAKURA_ATELIER_BACKEND_MODULE_PEFT."""

from __future__ import annotations

from sakura.atelier import wrap_up


class LoopBackend:
    """A backend with a training loop: 10 steps, checks the watch at each, then exports and re-scores."""

    name = "peft"
    tasks = frozenset({"text_generation"})
    steps_run = 0

    def train(self, spec, data, out, metrics, resume):
        from sakura.atelier.types import Checkpoint, TrainResult

        watch = wrap_up.WrapUpWatch(out.root)
        n = 0
        for step in range(10):
            if watch.stop(step / 4, step):
                break
            metrics.log("loss", 1.0 / (step + 1), step=step)
            n += 1
            if step == 2:  # the runner writes the file mid-run
                (out.root / "wrap-up-requested").write_text('{"reason": "budget"}')
        type(self).steps_run = n
        return TrainResult(model=None, metrics={"loss": 0.5}, train_seconds=0.1,
                           checkpoint=Checkpoint(path=out.checkpoint, model=spec.spec.model, task=spec.spec.task),
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


BACKEND = LoopBackend()
