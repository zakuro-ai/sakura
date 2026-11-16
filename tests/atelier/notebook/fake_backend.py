"""A backend that trains nothing and exports four bytes, importable by module
path so the notebook agent's kernel (a fresh process) can resolve it through
``SAKURA_ATELIER_BACKEND_MODULE_<NAME>``. Logs a three-point loss curve so the
Training section has something to plot."""

from __future__ import annotations

from typing import Any

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


class FakeBackend:
    name = "fake"
    tasks = frozenset({"tabular_classification"})
    NOTEBOOK_CELLS = ()

    def train(self, spec: ResolvedSpec, data: MaterialisedData, out: RunDir, metrics: MetricsSink,
              resume: Checkpoint | None) -> TrainResult:
        for s in range(3):
            metrics.log("loss", 1.0 / (s + 1), step=s)
            metrics.log("loss", 1.1 / (s + 1), step=s, split="validation")
        (out.checkpoint / "w.bin").write_bytes(b"w")
        return TrainResult(model=object(), metrics={"loss": 0.33}, train_seconds=0.1,
                           checkpoint=Checkpoint(out.checkpoint, "ludwig_ecd", "tabular_classification"))

    def export(self, result: TrainResult, fmt: str, out: RunDir) -> Artifact:
        p = out.artifacts / "model.onnx"
        p.write_bytes(b"onnx")
        return Artifact(p, "onnx", "a" * 64, 4)

    def runtime_eval(self, artifact: Artifact, data: MaterialisedData) -> RuntimeEval:
        return RuntimeEval("onnx", "onnxruntime", "loss", 0.34, 2, None, provider="CPUExecutionProvider")

    def predictions(self, result: TrainResult, resolved: ResolvedSpec, data: MaterialisedData) -> Any:
        from sakura.atelier.evaluation import ModelPredictions, SplitPredictions

        proba = [[0.8, 0.2], [0.3, 0.7], [0.6, 0.4], [0.1, 0.9]]
        return ModelPredictions(
            kind="classification", classes=["a", "b"],
            splits={"validation": SplitPredictions(
                y_true=["a", "b", "a", "b"], y_pred=["a", "b", "b", "b"], proba=proba,
                inputs=[{"x": float(i)} for i in range(4)])})

    def predict(self, artifact: Artifact, inputs: Any) -> Any:
        return {}


BACKEND = FakeBackend()
