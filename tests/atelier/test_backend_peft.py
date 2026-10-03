"""Unit tests for sakura/atelier/backends/peft.py.

Runnable in CI without peft/transformers/llama.cpp/llama-cpp-python
installed: the gguf/mlx export paths both fail fast on an unmet
precondition (LLAMA_CPP_DIR unset, mlx_lm not importable) before any heavy
dependency is touched, and runtime_eval's gguf path fails fast on
llama_cpp not being importable -- these failure paths, the ones CONTRACTS
§2.2 cares most about getting right ("never silently omitted"), are
exercised for real in CI, not mocked.

A real end-to-end GPU run (LoRA SFT -> merge -> gguf/safetensors export ->
llama-cpp-python runtime_eval) is exercised by a manual smoke run
on a GPU host and is not asserted in CI.
"""

from __future__ import annotations

import pytest

from sakura.atelier.backends import peft as backend_mod
from sakura.atelier.types import Artifact, RunDir


def test_backend_identity():
    b = backend_mod.BACKEND
    assert b.name == "peft"
    assert b.tasks == frozenset({"text_generation"})
    assert b.licence == "LFM Open License v1.0 (LiquidAI/LFM2-350M)"


def test_export_refuses_an_unsupported_format(tmp_path):
    out = RunDir(tmp_path / "run").ensure()
    fitted = backend_mod._Fitted(model_dir=tmp_path, tokenizer_dir=tmp_path, eval_examples=[])
    with pytest.raises(ValueError, match="exports gguf"):
        backend_mod.BACKEND.export(result=type("R", (), {"model": fitted})(), fmt="torchscript", out=out)


def test_gguf_export_fails_precisely_when_llama_cpp_dir_is_unset(tmp_path, monkeypatch):
    monkeypatch.delenv("LLAMA_CPP_DIR", raising=False)
    out = RunDir(tmp_path / "run").ensure()
    fitted = backend_mod._Fitted(model_dir=tmp_path / "merged", tokenizer_dir=tmp_path / "merged", eval_examples=[])
    with pytest.raises(backend_mod.GGUFExportError, match="LLAMA_CPP_DIR is not set"):
        backend_mod.BACKEND.export(result=type("R", (), {"model": fitted})(), fmt="gguf", out=out)


def test_gguf_export_fails_precisely_when_the_checkout_is_missing_pieces(tmp_path, monkeypatch):
    monkeypatch.setenv("LLAMA_CPP_DIR", str(tmp_path / "no-such-llama-cpp"))
    out = RunDir(tmp_path / "run").ensure()
    fitted = backend_mod._Fitted(model_dir=tmp_path / "merged", tokenizer_dir=tmp_path / "merged", eval_examples=[])
    with pytest.raises(backend_mod.GGUFExportError, match="not found"):
        backend_mod.BACKEND.export(result=type("R", (), {"model": fitted})(), fmt="gguf", out=out)


def test_mlx_export_fails_precisely_on_a_host_without_mlx(tmp_path):
    # mlx/mlx-lm genuinely are not installed in this CI venv (Linux x86_64) --
    # this is the real failure path CONTRACTS asks us to report, not a mock.
    out = RunDir(tmp_path / "run").ensure()
    fitted = backend_mod._Fitted(model_dir=tmp_path, tokenizer_dir=tmp_path, eval_examples=[])
    with pytest.raises(backend_mod.MLXUnavailableError, match="Apple Silicon"):
        backend_mod.BACKEND.export(result=type("R", (), {"model": fitted})(), fmt="mlx", out=out)


def test_safetensors_is_rescored_through_transformers(tmp_path, monkeypatch):
    out = RunDir(tmp_path / "run").ensure()
    artifact = Artifact(path=out.artifacts / "model.safetensors", format="safetensors", sha256="x", bytes=1)
    artifact.path.write_bytes(b"x")
    answers = {"1+1=": "2", "2+2=": "5"}
    monkeypatch.setattr(backend_mod, "_hf_generate", lambda d, prompt, n: answers[prompt])
    data = type("D", (), {"validation": [{"prompt": "1+1=", "completion": "2"},
                                         {"prompt": "2+2=", "completion": "4"}]})()
    ev = backend_mod.BACKEND.runtime_eval(artifact, data=data)
    assert (ev.runtime, ev.value, ev.n) == ("transformers", 0.5, 2)


def test_mlx_off_apple_silicon_reports_why(tmp_path):
    # mlx-lm genuinely does not install on this Linux CI venv: the real
    # ImportError path.
    out = RunDir(tmp_path / "run").ensure()
    artifact = Artifact(path=out.artifacts / "model.mlx", format="mlx", sha256="x", bytes=1)
    data = type("D", (), {"validation": [{"prompt": "1+1=", "completion": "2"}]})()
    ev = backend_mod.BACKEND.runtime_eval(artifact, data=data)
    assert ev.value is None and ev.runtime == "mlx-lm" and "not importable" in ev.reason


def test_no_validation_examples_is_no_score_not_zero(tmp_path):
    out = RunDir(tmp_path / "run").ensure()
    artifact = Artifact(path=out.artifacts / "model.safetensors", format="safetensors", sha256="x", bytes=1)
    ev = backend_mod.BACKEND.runtime_eval(artifact, data=type("D", (), {"validation": []})())
    assert ev.value is None and "no validation examples" in ev.reason


def test_runtime_eval_gguf_without_llama_cpp_python_reports_why(tmp_path):
    # llama-cpp-python genuinely is not installed in this CI venv -- this is
    # the real ImportError path, not a mock.
    out = RunDir(tmp_path / "run").ensure()
    artifact = Artifact(path=out.artifacts / "model.gguf", format="gguf", sha256="x", bytes=1)
    artifact.path.write_bytes(b"x")
    ev = backend_mod.BACKEND.runtime_eval(artifact, data=type("D", (), {"validation": []})())
    assert ev.value is None and ev.matches_training is None
    assert "llama_cpp" in ev.reason and "not importable" in ev.reason


def test_exact_match_ignores_surrounding_whitespace_and_trailing_text():
    assert backend_mod._exact_match("  42\nextra tokens after", "42")
    assert not backend_mod._exact_match("43", "42")
