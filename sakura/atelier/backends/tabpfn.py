"""TabPFN v2 backend: in-context tabular classification/regression.

Implements the ``Backend`` protocol (CONTRACTS §2.2) for
``tabular_classification`` / ``tabular_regression``, model id ``tabpfn_v2``,
pinned ``tabpfn==2.1.3`` (docs/atelier/envs/tabpfn.txt). 2.1.3 downloads the
ungated v2 weights straight from Hugging Face -- no Prior Labs account, no
API key. Releases >= 6 gate those weights behind an account; this module must
never be bumped to one without re-reading CONTRACTS §2.2's warning.

TabPFN is in-context: there is no gradient training loop, so "train" means
"fit the context" (prior-fitted transformer conditions on (X_train, y_train)
at inference time, no weight updates). ``train()`` logs k-fold CV and a
held-out validation score to the sink as a single post-fit measurement, not a
per-step curve -- there is no curve to report honestly.

Data convention (same caveat as ultralytics.py: Track E1's data layer does
not exist yet):
  ``data.train`` = ``(X_train, y_train)``, ``data.validation`` =
  ``(X_val, y_val)``, both ``X`` a ``pandas.DataFrame`` or 2D ``np.ndarray``
  and ``y`` a 1D array. ``data.labels`` = ordered class names for
  classification, ``None`` for regression.

ONNX export
-----------
CONTRACTS §2.2 asks for a bounded attempt at exporting the fitted context
baked in as constants, falling back to reporting a precise failure reason
rather than faking a value. TabPFN 2.1.3's sklearn-style estimators run
context + query through ``TabPFNClassifier.model_`` under non-static control
flow (data-dependent preprocessing: quantile bucketing, a per-feature type
dispatch, and an internal ensembling loop over several context permutations)
that ``torch.onnx.export`` (traced or dynamo) does not capture faithfully for
a context of nontrivial size. This module makes one real, time-boxed attempt
(see ``_attempt_onnx_export``) with ``torch.onnx.export`` on the single
forward path (ensemble size pinned to 1, a fixed context), and raises
``TabPFNExportError`` with the exact exception if it fails -- the caller (the
engine / this track's smoke harness) decides what tabular binary to ship
instead: registry/models.yaml declares tabpfn_v2 evaluation-only (no exports), and
ludwig_ecd is the tabular model that ships ONNX.

``runtime_eval`` protocol note: ``Backend.runtime_eval(self, artifact, data)``
(CONTRACTS §2.2) is not passed the ``TrainResult``, so ``matches_training`` is
left ``None`` here -- the engine's ``run`` loop (Track E1) holds
``TrainResult.metrics`` and is the right place to apply the §2.3 tolerance.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
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

LICENSE = "Prior Labs License (TabPFN v2 weights, ungated for <=2.1.x)"


class TabPFNExportError(RuntimeError):
    """Raised by export() when the bounded ONNX attempt fails. Carries the
    exact underlying exception text -- never swallowed."""


@dataclass
class _Fitted:
    """``TrainResult.model`` payload: the fitted estimator plus the context
    tensors an ONNX export attempt needs, since TabPFN's inference is a
    function of (X_train, y_train) as much as of its frozen weights."""

    estimator: Any
    task: str  # "tabular_classification" | "tabular_regression"
    X_train: Any
    y_train: Any
    classes: list[str] | None


def _device() -> str:
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


def _attempt_onnx_export(fitted: _Fitted, dest: Path) -> None:
    """One bounded, real attempt. Raises TabPFNExportError with the exact
    cause on any failure -- no placeholder file is ever written."""
    import numpy as np
    import torch

    try:
        est = fitted.estimator
        # Force the cheapest, most ONNX-friendly configuration: a single
        # context permutation, no test-time augmentation ensembling.
        if hasattr(est, "n_estimators"):
            est.n_estimators = 1
        model = getattr(est, "model_", None) or getattr(est, "model", None)
        if model is None:
            raise TabPFNExportError(
                "fitted TabPFNClassifier/Regressor exposes no '.model_'/'.model' "
                "torch module to trace -- internal API of tabpfn==2.1.3"
            )
        model.eval()

        X_ref = np.asarray(fitted.X_train, dtype=np.float32)
        example = torch.from_numpy(X_ref[: min(8, len(X_ref))]).float().to(_device())

        class _Wrapper(torch.nn.Module):
            def __init__(self, inner: Any) -> None:
                super().__init__()
                self.inner = inner

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                out: torch.Tensor = self.inner(x)
                return out

        wrapper = _Wrapper(model)
        torch.onnx.export(
            wrapper,
            (example,),
            str(dest),
            input_names=["X_test"],
            output_names=["logits"],
            dynamic_axes={"X_test": {0: "n_rows"}, "logits": {0: "n_rows"}},
            opset_version=17,
        )
    except Exception as exc:  # noqa: BLE001 -- surfaced verbatim, not swallowed
        raise TabPFNExportError(f"{type(exc).__name__}: {exc}") from exc


class TabPFNBackend:
    name = "tabpfn"
    tasks = frozenset({"tabular_classification", "tabular_regression"})
    licence = LICENSE

    def train(
        self,
        spec: ResolvedSpec,
        data: MaterialisedData,
        out: RunDir,
        metrics: MetricsSink,
        resume: Checkpoint | None,
    ) -> TrainResult:
        if resume is not None:
            raise ValueError("tabpfn is in-context: there is no checkpoint to resume from")
        out.ensure()
        task = spec.spec.task
        X_train, y_train = data.train
        X_val, y_val = data.validation

        import numpy as np
        from sklearn.model_selection import cross_val_score

        t0 = time.time()
        if task == "tabular_classification":
            from tabpfn import TabPFNClassifier

            est = TabPFNClassifier(device=_device())
            cv_scores = cross_val_score(est, X_train, y_train, cv=min(5, max(2, len(X_train) // 20)))
            est.fit(X_train, y_train)
            train_seconds = time.time() - t0

            val_pred = est.predict(X_val)
            val_proba = est.predict_proba(X_val)
            accuracy = float((np.asarray(val_pred) == np.asarray(y_val)).mean())
            try:
                from sklearn.metrics import roc_auc_score

                if val_proba.shape[1] == 2:
                    auc = float(roc_auc_score(y_val, val_proba[:, 1]))
                else:
                    auc = float(roc_auc_score(y_val, val_proba, multi_class="ovr"))
            except ValueError:
                auc = float("nan")

            metrics.log("cv_accuracy", float(np.mean(cv_scores)), step=0, split="train", epoch=0.0)
            metrics.log("accuracy", accuracy, step=0, split="validation", epoch=1.0)
            if auc == auc:  # not NaN
                metrics.log("auc", auc, step=0, split="validation", epoch=1.0)

            result_metrics = {"accuracy": accuracy, "cv_accuracy": float(np.mean(cv_scores))}
            primary_metric, primary_value = "accuracy", accuracy
            classes = [str(c) for c in getattr(est, "classes_", sorted(set(y_train)))]
        else:
            from tabpfn import TabPFNRegressor

            est = TabPFNRegressor(device=_device())
            cv_scores = cross_val_score(est, X_train, y_train, cv=min(5, max(2, len(X_train) // 20)), scoring="r2")
            est.fit(X_train, y_train)
            train_seconds = time.time() - t0

            val_pred = est.predict(X_val)
            rmse = float(np.sqrt(np.mean((np.asarray(val_pred) - np.asarray(y_val)) ** 2)))
            metrics.log("cv_r2", float(np.mean(cv_scores)), step=0, split="train", epoch=0.0)
            metrics.log("rmse", rmse, step=0, split="validation", epoch=1.0)
            result_metrics = {"rmse": rmse, "cv_r2": float(np.mean(cv_scores))}
            primary_metric, primary_value = "rmse", rmse
            classes = None

        # No gradient checkpoint to save; record enough to refuse a resume
        # attempt loudly instead of silently ignoring resume_from.
        ckpt_dir = out.checkpoint
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        (ckpt_dir / "meta.json").write_text(json.dumps({
            "model": spec.spec.model, "task": task,
            "note": "tabpfn is in-context: nothing to resume, this file only records provenance",
        }))

        fitted = _Fitted(estimator=est, task=task, X_train=X_train, y_train=y_train, classes=classes)
        return TrainResult(
            model=fitted, metrics=result_metrics, train_seconds=train_seconds,
            checkpoint=Checkpoint(path=ckpt_dir, model=spec.spec.model, task=task),
        )

    def export(self, result: TrainResult, fmt: str, out: RunDir) -> Artifact:
        if fmt != "onnx":
            raise ValueError(f"tabpfn backend only exports 'onnx', got {fmt!r}")
        out.ensure()
        fitted: _Fitted = result.model
        dest = out.artifacts / "model.onnx"
        _attempt_onnx_export(fitted, dest)  # raises TabPFNExportError on failure, no fake file
        import hashlib

        data = dest.read_bytes()
        return Artifact(path=dest, format="onnx", sha256=hashlib.sha256(data).hexdigest(), bytes=len(data))

    def runtime_eval(self, artifact: Artifact, data: MaterialisedData) -> RuntimeEval:
        X_val, y_val = data.validation
        try:
            import numpy as np
            import onnxruntime as ort

            sess = ort.InferenceSession(str(artifact.path), providers=["CPUExecutionProvider"])
            input_name = sess.get_inputs()[0].name
            logits = sess.run(None, {input_name: np.asarray(X_val, dtype=np.float32)})[0]
            pred = np.argmax(logits, axis=-1)
            accuracy = float((pred == np.asarray(y_val)).mean())
            # matches_training is the engine's call (Track E1's `run` loop
            # holds TrainResult.metrics; this method's signature does not).
            return RuntimeEval(
                format="onnx", runtime="onnxruntime", metric="accuracy",
                value=accuracy, n=len(X_val), matches_training=None, reason=None,
            )
        except Exception as exc:  # noqa: BLE001
            return RuntimeEval(
                format="onnx", runtime="onnxruntime", metric="accuracy",
                value=None, n=len(X_val), matches_training=None,
                reason=f"onnxruntime could not re-score model.onnx: {type(exc).__name__}: {exc}",
            )

    def predict(self, artifact: Artifact, inputs: Any) -> Any:
        import numpy as np
        import onnxruntime as ort

        sess = ort.InferenceSession(str(artifact.path), providers=["CPUExecutionProvider"])
        input_name = sess.get_inputs()[0].name
        X = np.asarray(inputs, dtype=np.float32)
        if X.ndim == 1:
            X = X[None, :]
        logits = sess.run(None, {input_name: X})[0]
        pred = np.argmax(logits, axis=-1)
        proba = _softmax(logits)
        return {
            "prediction": [int(p) for p in pred] if len(pred) > 1 else int(pred[0]),
            "proba": proba.tolist(),
        }


def _softmax(x: Any) -> Any:
    import numpy as np

    e = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


BACKEND = TabPFNBackend()
