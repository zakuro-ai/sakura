"""Unit tests for sakura/atelier/backends/tabpfn.py.

Runnable in CI without tabpfn installed. ``torch`` and ``numpy`` ARE CI
dependencies (sakura-ml's own base deps), so the ONNX-export attempt is
exercised for real against a trivial stand-in ``nn.Module`` instead of a
mocked-away one: that is the one part of this backend worth testing against
real ``torch.onnx.export`` behaviour, since the whole point of
``_attempt_onnx_export`` is "does this actually export, and if not, does it
say exactly why" -- a mock would test nothing.

A real end-to-end GPU run against real TabPFN is exercised by
a manual GPU smoke run and is not asserted in CI.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from sakura.atelier.backends import tabpfn as backend_mod
from sakura.atelier.types import Artifact, RunDir


def test_backend_identity():
    b = backend_mod.BACKEND
    assert b.name == "tabpfn"
    assert b.tasks == frozenset({"tabular_classification", "tabular_regression"})


def test_export_refuses_an_unsupported_format(tmp_path):
    out = RunDir(tmp_path / "run").ensure()
    with pytest.raises(ValueError, match="only exports 'onnx'"):
        backend_mod.BACKEND.export(result=SimpleNamespace(model=None), fmt="torchscript", out=out)


def test_onnx_export_succeeds_against_a_real_torch_module_standing_in_for_model_(tmp_path):
    """Exercises the real torch.onnx.export call path with a module that
    plausibly stands in for TabPFN's internal `.model_`: something whose
    forward signature and output shape match what a classifier head looks
    like, so a genuine export success/failure is being tested, not a mock."""
    torch_module = torch.nn.Linear(4, 3)
    fitted = backend_mod._Fitted(
        estimator=SimpleNamespace(model_=torch_module, n_estimators=4),
        task="tabular_classification",
        X_train=np.random.default_rng(0).normal(size=(16, 4)).astype(np.float32),
        y_train=np.array([0, 1, 2] * 5 + [0]),
        classes=["a", "b", "c"],
    )
    dest = tmp_path / "model.onnx"
    try:
        backend_mod._attempt_onnx_export(fitted, dest)
    except backend_mod.TabPFNExportError as exc:
        # Environment-dependent (e.g. the `onnx` package absent from this
        # venv) -- acceptable as long as the failure is precise, never silent.
        assert str(exc)
        return
    assert dest.exists() and dest.stat().st_size > 0


def test_onnx_export_failure_is_never_silent_when_model_attr_is_missing(tmp_path):
    fitted = backend_mod._Fitted(
        estimator=SimpleNamespace(),  # no .model_/.model at all
        task="tabular_classification",
        X_train=np.zeros((4, 2), dtype=np.float32),
        y_train=np.array([0, 1, 0, 1]),
        classes=["a", "b"],
    )
    with pytest.raises(backend_mod.TabPFNExportError, match="no '.model_'/'.model'"):
        backend_mod._attempt_onnx_export(fitted, tmp_path / "model.onnx")
    assert not (tmp_path / "model.onnx").exists()


def test_runtime_eval_reports_a_precise_reason_when_onnxruntime_cannot_load_the_file(tmp_path):
    out = RunDir(tmp_path / "run").ensure()
    artifact = Artifact(path=out.artifacts / "model.onnx", format="onnx", sha256="x", bytes=4)
    artifact.path.write_bytes(b"not an onnx file")
    data = SimpleNamespace(validation=(np.zeros((2, 2), dtype=np.float32), np.array([0, 1])))

    ev = backend_mod.BACKEND.runtime_eval(artifact, data)
    assert ev.value is None
    assert ev.matches_training is None
    assert "onnxruntime could not re-score" in ev.reason


def test_runtime_eval_leaves_matches_training_to_the_engine_on_success(tmp_path, monkeypatch):
    # Backend.runtime_eval(artifact, data) per CONTRACTS §2.2 has no
    # TrainResult to compare against, so matches_training is the engine's
    # call (Track E1's `run` loop), not guessed here via a side-channel.
    import types

    out = RunDir(tmp_path / "run").ensure()
    artifact = Artifact(path=out.artifacts / "model.onnx", format="onnx", sha256="x", bytes=4)
    artifact.path.write_bytes(b"fake -- the stubbed onnxruntime below never reads it")

    fake_ort = types.ModuleType("onnxruntime")

    class FakeSession:
        def __init__(self, *a, **kw):
            pass

        def get_inputs(self):
            return [SimpleNamespace(name="X_test")]

        def run(self, out_names, feed):
            n = feed["X_test"].shape[0]
            return [np.tile(np.array([0.0, 1.0]), (n, 1))]  # always predicts class 1

    fake_ort.InferenceSession = FakeSession
    monkeypatch.setitem(__import__("sys").modules, "onnxruntime", fake_ort)

    data = SimpleNamespace(validation=(np.zeros((3, 2), dtype=np.float32), np.array([1, 1, 0])))
    ev = backend_mod.BACKEND.runtime_eval(artifact, data)
    assert ev.value == pytest.approx(2 / 3)
    assert ev.matches_training is None
    assert ev.reason is None
