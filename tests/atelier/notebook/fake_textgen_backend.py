"""The fake backend for a `text_generation` run: the same trivial training as
`fake_backend`, exporting `safetensors`. The CLI and the notebook's kernel
resolve it through `SAKURA_ATELIER_BACKEND_MODULE_PEFT`."""

from __future__ import annotations

from sakura.atelier.types import Artifact, MaterialisedData, ResolvedSpec, RunDir, RuntimeEval, TrainResult
from tests.atelier.notebook.fake_backend import FakeBackend


class FakeTextGenBackend(FakeBackend):
    name = "fake-textgen"
    tasks = frozenset({"text_generation"})

    def export(self, result: TrainResult, fmt: str, out: RunDir) -> Artifact:
        p = out.artifacts / "model.safetensors"
        p.write_bytes(b"safe")
        return Artifact(p, "safetensors", "b" * 64, 4)

    def runtime_eval(self, artifact: Artifact, data: MaterialisedData) -> RuntimeEval:
        return RuntimeEval("safetensors", "transformers", "loss", 0.34, 2, None)

    def predictions(self, result: TrainResult, resolved: ResolvedSpec, data: MaterialisedData) -> None:
        return None


BACKEND = FakeTextGenBackend()
