"""Unit tests for sakura/atelier/backends/peft.py.

Runnable in CI without peft/transformers/llama.cpp/llama-cpp-python
installed: the gguf/mlx export paths both fail fast on an unmet
precondition (LLAMA_CPP_DIR unset, mlx_lm not importable) before any heavy
dependency is touched, and runtime_eval's gguf path fails fast on
llama_cpp not being importable -- these failure paths, the ones CONTRACTS
§2.2 cares most about getting right ("never silently omitted"), are
exercised for real in CI, not mocked.

A real end-to-end GPU run (LoRA SFT -> merge -> gguf/safetensors export ->
llama-cpp-python runtime_eval) is exercised by scripts/atelier/smoke_e2.py
on the GPU host and reported in this track's handoff.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from sakura.atelier.backends import peft as backend_mod
from sakura.atelier.types import Artifact, RunDir


def _mlx_runs_here() -> bool:
    if not backend_mod._is_apple_silicon():
        return False
    try:
        import mlx_lm  # noqa: F401
    except ImportError:
        return False
    return True


def test_backend_identity():
    b = backend_mod.BACKEND
    assert b.name == "peft"
    assert b.tasks == frozenset({"text_generation", "decision"})  # decision: sakura.atelier.decision / decision_lm
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


@pytest.mark.skipif(_mlx_runs_here(), reason="this host does run mlx; the real path is tested below")
@pytest.mark.parametrize("quant", ["Q8_0; rm -rf ~", "--help", "Q4 K M", ""])
def test_gguf_export_rejects_a_quant_that_is_not_a_type_name(tmp_path, monkeypatch, quant):
    llama = tmp_path / "llama.cpp"
    (llama / "build" / "bin").mkdir(parents=True)
    (llama / "convert_hf_to_gguf.py").write_text("")
    (llama / "build" / "bin" / "llama-quantize").write_text("")
    monkeypatch.setenv("LLAMA_CPP_DIR", str(llama))
    monkeypatch.setenv("SAKURA_GGUF_QUANT", quant)
    calls: list[list[str]] = []
    monkeypatch.setattr(backend_mod.subprocess, "run", lambda cmd, **kw: calls.append(cmd))
    out = RunDir(tmp_path / "run").ensure()
    fitted = backend_mod._Fitted(model_dir=tmp_path / "merged", tokenizer_dir=tmp_path / "merged", eval_examples=[])
    with pytest.raises(backend_mod.GGUFExportError, match="not a llama.cpp quant type name"):
        backend_mod.BACKEND.export(result=type("R", (), {"model": fitted})(), fmt="gguf", out=out)
    assert calls == []  # rejected before any subprocess runs


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


def _fake_hf(monkeypatch, *, bf16_ok=True, oom=False):
    """Fake torch + transformers: records where the model went, at what dtype."""
    import sys
    import types

    torch = types.ModuleType("torch")
    torch.bfloat16, torch.float32 = "bf16", "fp32"
    oom_error = type("OutOfMemoryError", (RuntimeError,), {})
    torch.cuda = types.SimpleNamespace(OutOfMemoryError=oom_error, is_bf16_supported=lambda: bf16_ok,
                                       empty_cache=lambda: None)

    class Model:
        def __init__(self, dtype):
            self.dtype, self.device = dtype, None

        def to(self, device=None, dtype=None):
            if device == "cuda" and oom:
                raise oom_error("CUDA out of memory")
            self.device, self.dtype = device, dtype or self.dtype
            return self

        def generate(self, **kw):
            return [[1, 2, 3]]

    class Ids:
        shape = (1, 1)

        def to(self, device):
            return self

    class Tok:
        pad_token_id = 0

        def __call__(self, prompt, return_tensors):
            return {"input_ids": Ids()}

        def decode(self, ids, skip_special_tokens):
            return "out"

    models = []

    def from_pretrained(path, dtype):
        models.append(Model(dtype))
        return models[-1]

    transformers = types.ModuleType("transformers")
    transformers.AutoModelForCausalLM = types.SimpleNamespace(from_pretrained=from_pretrained)
    transformers.AutoTokenizer = types.SimpleNamespace(from_pretrained=lambda path: Tok())
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "transformers", transformers)
    monkeypatch.setattr(backend_mod, "_LOADED", {})
    monkeypatch.setattr(backend_mod, "_device", lambda: "cuda")
    return models


@pytest.mark.parametrize("bf16_ok, oom, expected", [
    (True, False, ("cuda", "bf16")),   # bf16 halves the footprint on a GPU that has it
    (False, False, ("cuda", "fp32")),  # pre-Ampere: no bf16, stay exact
    (True, True, ("cpu", "fp32")),     # a GPU another process fills: slow, not a crash
])
def test_hf_generate_loads_bf16_on_cuda_and_falls_back_to_cpu_when_the_gpu_is_full(
        tmp_path, monkeypatch, bf16_ok, oom, expected):
    models = _fake_hf(monkeypatch, bf16_ok=bf16_ok, oom=oom)
    assert backend_mod._hf_generate(tmp_path, "1+1=", 4) == "out"
    assert (models[0].device, models[0].dtype) == expected
    assert backend_mod._LOADED[f"hf:{tmp_path}"][2] == expected[0]


@pytest.mark.skipif(_mlx_runs_here(), reason="this host does run mlx; the real path is tested below")
def test_release_lets_go_of_every_loaded_export_and_closes_the_llama_context(monkeypatch):
    # what runtime_eval left loaded: the safetensors model (a tuple) and the GGUF's llama.cpp context (has close)
    closed, released = [], []

    class Llama:
        def close(self):
            closed.append("gguf")

    class Stubborn:
        def close(self):
            raise RuntimeError("already closed")

    loaded = {"hf:/run/model": ("tokenizer", "model", "cuda"), "/run/model.gguf": Llama(), "/run/other.gguf": Stubborn()}
    monkeypatch.setattr(backend_mod, "_LOADED", loaded)
    monkeypatch.setattr(backend_mod, "_release", lambda: released.append(dict(loaded)))
    backend_mod.PEFTBackend().release()
    assert loaded == {} and closed == ["gguf"]
    assert released == [{}], "the allocator's cache is emptied once, after the exports are gone"
    backend_mod.PEFTBackend().release()  # nothing loaded: nothing to do, no error
    assert released == [{}, {}]


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


# --- MLX export + re-score ----------------------------------------------------

_VAL = [{"prompt": "1+1=", "completion": "2"}, {"prompt": "2+2=", "completion": "4"}]


def _merged_dir(tmp_path):
    d = tmp_path / "merged"
    d.mkdir()
    (d / "config.json").write_text("{}")
    return d


def test_mlx_export_raises_off_apple_silicon_even_if_mlx_imports(tmp_path, monkeypatch):
    # mlx now has Linux wheels; the gate is the platform, not just the import.
    monkeypatch.setattr(backend_mod.sys, "platform", "linux")
    with pytest.raises(backend_mod.MLXUnavailableError, match="Apple Silicon"):
        backend_mod.export_mlx(_merged_dir(tmp_path), _VAL, tmp_path / "out" / "model.mlx", "q8")


def test_mlx_export_rejects_an_unknown_quant_level(tmp_path):
    with pytest.raises(ValueError, match="fp16"):
        backend_mod.export_mlx(_merged_dir(tmp_path), _VAL, tmp_path / "model.mlx", "q3")


def test_mlx_cli_exits_3_on_a_host_that_cannot_run_mlx(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(backend_mod.sys, "platform", "linux")
    val = tmp_path / "val.jsonl"
    val.write_text("".join(json.dumps(r) + "\n" for r in _VAL))
    rc = backend_mod.main(["mlx", "--merged", str(_merged_dir(tmp_path)), "--validation", str(val),
                           "--quant", "q4", "--out", str(tmp_path / "model.mlx")])
    assert rc == 3 and "MLXUnavailableError" in capsys.readouterr().err


@pytest.mark.parametrize("quant,flags", [
    ("fp16", ["--dtype", "float16"]),
    ("q8", ["--dtype", "float16", "-q", "--q-bits", "8"]),
    ("q4", ["--dtype", "float16", "-q", "--q-bits", "4"]),
])
def test_export_converts_at_the_chosen_quant_and_scores_with_the_harness_rule(tmp_path, monkeypatch, quant, flags):
    # The conversion and generation are stubbed (they need Metal); everything
    # around them -- flags, scoring rule, report, reuse by runtime_eval -- is real.
    calls = []

    def fake_run(cmd, **kw):
        calls.append(cmd)
        dest = Path(cmd[cmd.index("--mlx-path") + 1])
        dest.mkdir(parents=True)
        (dest / "model.safetensors").write_bytes(b"w" * 10)
        (dest / "config.json").write_text("{}")
        return subprocess.CompletedProcess(cmd, 0, "", "")

    generated = []

    def fake_generate(model_dir, prompt, max_tokens):
        generated.append((prompt, max_tokens))
        return {"1+1=": " 2\n3", "2+2=": "5"}[prompt]  # first line, stripped: 1 of 2

    monkeypatch.setattr(backend_mod, "_require_mlx", lambda: None)
    monkeypatch.setattr(backend_mod.subprocess, "run", fake_run)
    monkeypatch.setattr(backend_mod, "_mlx_generate", fake_generate)
    monkeypatch.setenv("SAKURA_MLX_QUANT", quant)
    out = RunDir(tmp_path / "run").ensure()
    fitted = backend_mod._Fitted(model_dir=_merged_dir(tmp_path), tokenizer_dir=tmp_path, eval_examples=list(_VAL))

    artifact = backend_mod.BACKEND.export(result=type("R", (), {"model": fitted})(), fmt="mlx", out=out)

    assert calls[0][-len(flags):] == flags and calls[0][1:4] == ["-m", "mlx_lm", "convert"]
    assert artifact.format == "mlx" and artifact.path == out.artifacts / "model.mlx" and artifact.bytes == 12
    assert generated == [("1+1=", 8), ("2+2=", 8)]
    report = json.loads((out.artifacts / "model.mlx.eval.json").read_text())
    assert (report["quant"], report["value"], report["n"], report["bytes"]) == (quant, 0.5, 2, 12)
    assert report["ms_per_example"] is not None
    # The engine's runtime_eval right after export reuses that score.
    ev = backend_mod.BACKEND.runtime_eval(artifact, data=type("D", (), {"validation": list(_VAL)})())
    assert (ev.runtime, ev.value, ev.n) == ("mlx-lm", 0.5, 2) and len(generated) == 2


def test_mlx_convert_failure_is_an_export_error_not_wrong_host(tmp_path, monkeypatch):
    monkeypatch.setattr(backend_mod, "_require_mlx", lambda: None)
    monkeypatch.setattr(backend_mod.subprocess, "run",
                        lambda cmd, **kw: subprocess.CompletedProcess(cmd, 1, "", "boom"))
    with pytest.raises(backend_mod.MLXExportError, match="boom"):
        backend_mod.export_mlx(_merged_dir(tmp_path), _VAL, tmp_path / "model.mlx", "q8")


def _tiny_llama(d):
    """A random 2-layer Llama in HF safetensors layout with a word-level
    tokenizer -- a few hundred KB, built offline."""
    import numpy as np
    from safetensors.numpy import save_file
    from tokenizers import Tokenizer, models, pre_tokenizers

    vocab = {t: i for i, t in enumerate(["<unk>", "<s>", "</s>", "1", "2", "4", "+", "="])}
    tok = Tokenizer(models.WordLevel(vocab=vocab, unk_token="<unk>"))
    tok.pre_tokenizer = pre_tokenizers.Split("", behavior="isolated")
    d.mkdir()
    tok.save(str(d / "tokenizer.json"))
    (d / "tokenizer_config.json").write_text(json.dumps({
        "tokenizer_class": "PreTrainedTokenizerFast", "bos_token": "<s>", "eos_token": "</s>",
        "unk_token": "<unk>"}))
    h, inter, layers, v = 64, 128, 2, len(vocab)
    (d / "config.json").write_text(json.dumps({
        "architectures": ["LlamaForCausalLM"], "model_type": "llama", "hidden_size": h,
        "intermediate_size": inter, "num_hidden_layers": layers, "num_attention_heads": 2,
        "num_key_value_heads": 2, "vocab_size": v, "rms_norm_eps": 1e-6, "rope_theta": 10000.0,
        "max_position_embeddings": 64, "tie_word_embeddings": False, "bos_token_id": 1,
        "eos_token_id": 2, "torch_dtype": "float32"}))
    rng = np.random.default_rng(0)

    def w(*shape):
        return (rng.standard_normal(shape) * 0.02).astype(np.float32)

    t = {"model.embed_tokens.weight": w(v, h), "model.norm.weight": np.ones(h, np.float32),
         "lm_head.weight": w(v, h)}
    for i in range(layers):
        p = f"model.layers.{i}."
        for n in ("q", "k", "v", "o"):
            t[f"{p}self_attn.{n}_proj.weight"] = w(h, h)
        t[f"{p}mlp.gate_proj.weight"] = w(inter, h)
        t[f"{p}mlp.up_proj.weight"] = w(inter, h)
        t[f"{p}mlp.down_proj.weight"] = w(h, inter)
        t[f"{p}input_layernorm.weight"] = np.ones(h, np.float32)
        t[f"{p}post_attention_layernorm.weight"] = np.ones(h, np.float32)
    save_file(t, str(d / "model.safetensors"))
    return d


@pytest.mark.skipif(not _mlx_runs_here(), reason="needs Apple silicon with mlx-lm installed")
@pytest.mark.parametrize("quant", ["fp16", "q8", "q4"])
def test_mlx_export_for_real_on_apple_silicon(tmp_path, quant):
    merged = _tiny_llama(tmp_path / "merged")
    res = backend_mod.export_mlx(merged, _VAL, tmp_path / "artifacts" / "model.mlx", quant)
    assert (res.artifact.path / "config.json").is_file() and res.artifact.bytes > 0
    assert res.score.runtime == "mlx-lm" and res.score.n == 2 and res.score.reason is None
    assert res.score.value is not None and 0.0 <= res.score.value <= 1.0
    assert json.loads(res.report_path.read_text())["quant"] == quant
    cfg = json.loads((res.artifact.path / "config.json").read_text())
    assert ("quantization" in cfg) == (quant != "fp16")


# --- generalised hyper-parameters (2026-10-10: La Forge LLM jobs on this backend) ---

def test_training_options_defaults_are_the_lfm2_lora_fast_behaviour():
    o = backend_mod._training_options({})
    assert (o.base_model, o.base_revision) == ("LiquidAI/LFM2-350M", None)
    assert (o.epochs, o.learning_rate, o.batch_size, o.grad_accum) == (3, 2e-4, 8, 1)
    assert (o.lora_r, o.lora_alpha, o.lora_dropout, o.lora_target_modules) == (8, 16, 0.05, "all-linear")
    assert (o.max_seq_length, o.precision, o.prompt_template, o.gguf_quant) == (64, "fp32", "raw", None)


def test_training_options_read_the_preset_keys():
    from sakura.atelier.registry import load_preset

    o = backend_mod._training_options(load_preset("text_generation/lora-fast@1"))
    assert o.max_seq_length == 512  # was silently ignored (the backend read `max_length`)
    q = backend_mod._training_options({**load_preset("text_generation/qlora-forge@1"),
                                       "base_model": "Qwen/Qwen3-0.6B", "base_revision": "deadbeef"})
    assert (q.precision, q.prompt_template, q.gguf_quant, q.grad_accum, q.max_seq_length) == ("4bit", "alpaca", "Q4_K_M", 16, 2048)
    assert (q.lora_r, q.lora_alpha, q.lora_dropout, q.epochs, q.batch_size, q.learning_rate) == (16, 32, 0.0, 3, 1, 1e-4)
    assert q.lora_target_modules == ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    assert (q.base_model, q.base_revision) == ("Qwen/Qwen3-0.6B", "deadbeef")


def test_max_length_is_a_deprecated_alias_of_max_seq_length():
    with pytest.warns(DeprecationWarning, match="max_length is deprecated"):
        assert backend_mod._training_options({"max_length": 96}).max_seq_length == 96
    # when both are present the new key wins, without a warning
    import warnings as _w
    with _w.catch_warnings():
        _w.simplefilter("error")
        assert backend_mod._training_options({"max_seq_length": 128, "max_length": 96}).max_seq_length == 128


@pytest.mark.parametrize("hp,match", [({"precision": "fp16"}, "precision must be one of"),
                                      ({"prompt_template": "chatml"}, "prompt_template must be one of")])
def test_training_options_refuse_unknown_choices(hp, match):
    with pytest.raises(ValueError, match=match):
        backend_mod._training_options(hp)


# --- schedule, evaluation and metric keys (2026-11-15, CONTRACTS §2.2) ---

def test_schedule_and_evaluation_defaults_are_the_behaviour_before_the_keys():
    o = backend_mod._training_options({})
    assert (o.lr_schedule, o.warmup_ratio, o.min_lr_ratio, o.weight_decay, o.max_grad_norm) == ("constant", 0.0, 0.0, None, None)
    assert (o.group_by_length, o.lora_use_rslora, o.lora_use_dora) == (False, False, False)
    assert (o.eval_every, o.eval_max_rows, o.gen_eval_rows, o.final_gen_rows, o.eval_batch_size) == (None, None, None, None, 1)
    assert (o.early_stop_patience, o.keep_best, o.selection_metric) == (None, False, "loss")
    assert (o.eval_max_new_tokens, o.metrics, o.baseline, o.extended) == (8, ("exact_match",), False, False)


def test_schedule_and_evaluation_keys_are_read():
    o = backend_mod._training_options({
        "lr_schedule": "linear", "warmup_ratio": 0.5, "min_lr_ratio": 1, "weight_decay": 0, "max_grad_norm": 0.5,
        "group_by_length": True, "lora_use_rslora": True, "lora_use_dora": True, "eval_every": 1, "eval_max_rows": 64,
        "gen_eval_rows": 0, "final_gen_rows": 10, "eval_batch_size": 4, "early_stop_patience": 2, "keep_best": True,
        "selection_metric": "loss", "eval_max_new_tokens": 300, "metrics": ["json_field_f1", "json_valid"],
        "baseline": True})
    assert (o.lr_schedule, o.warmup_ratio, o.min_lr_ratio, o.weight_decay, o.max_grad_norm) == ("linear", 0.5, 1.0, 0.0, 0.5)
    assert (o.group_by_length, o.lora_use_rslora, o.lora_use_dora) == (True, True, True)
    assert (o.eval_every, o.eval_max_rows, o.gen_eval_rows, o.final_gen_rows, o.eval_batch_size) == (1.0, 64, 0, 10, 4)
    assert (o.early_stop_patience, o.keep_best, o.selection_metric) == (2, True, "loss")
    assert (o.eval_max_new_tokens, o.metrics, o.baseline, o.extended) == (300, ("json_field_f1", "json_valid"), True, True)
    auto = backend_mod._training_options({"metrics": "auto", "eval_max_new_tokens": "auto",
                                          "selection_metric": "json_field_f1"})
    assert (auto.metrics, auto.eval_max_new_tokens, auto.selection_metric) == ("auto", "auto", "json_field_f1")


def test_a_key_set_to_null_or_to_its_default_is_the_default_and_still_turns_the_reporting_on():
    o = backend_mod._training_options({"eval_every": None, "max_grad_norm": None, "eval_max_new_tokens": None,
                                       "metrics": None, "keep_best": None})
    assert (o.eval_every, o.max_grad_norm, o.eval_max_new_tokens, o.metrics, o.keep_best) == (None, None, 8, ("exact_match",), False)
    assert o.extended is True
    assert backend_mod._training_options({"lr_schedule": "constant"}).extended is True


@pytest.mark.parametrize("hp, match", [
    ({"lr_schedule": "exponential"}, "lr_schedule must be one of"),
    ({"warmup_ratio": 0.6}, "warmup_ratio must be a number from 0 to 0.5"),
    ({"warmup_ratio": "0.1"}, "warmup_ratio must be a number"),
    ({"min_lr_ratio": -0.1}, "min_lr_ratio must be a number from 0 to 1"),
    ({"weight_decay": -1}, "weight_decay must be a number of 0 or more"),
    ({"max_grad_norm": 0}, "max_grad_norm must be a number above 0"),
    ({"max_grad_norm": True}, "max_grad_norm must be a number"),
    ({"group_by_length": "false"}, "group_by_length must be true or false"),
    ({"lora_use_dora": 1}, "lora_use_dora must be true or false"),
    ({"eval_every": 0}, "eval_every must be a number above 0 and at most 1"),
    ({"eval_every": 2}, "eval_every must be a number above 0 and at most 1"),
    ({"eval_max_rows": 0}, "eval_max_rows must be a whole number of at least 1"),
    ({"gen_eval_rows": -1}, "gen_eval_rows must be a whole number of at least 0"),
    ({"final_gen_rows": 2.5}, "final_gen_rows must be a whole number"),
    ({"eval_batch_size": 0}, "eval_batch_size must be a whole number of at least 1"),
    ({"early_stop_patience": 0}, "early_stop_patience must be a whole number of at least 1"),
    ({"keep_best": "yes"}, "keep_best must be true or false"),
    ({"eval_max_new_tokens": 0}, "eval_max_new_tokens must be a whole number of at least 1 or 'auto'"),
    ({"eval_max_new_tokens": "long"}, "eval_max_new_tokens must be"),
    ({"metrics": []}, "metrics must be 'auto' or a list"),
    ({"metrics": "exact_match"}, "metrics must be 'auto' or a list"),
    ({"metrics": ["bleu"]}, "metrics must be 'auto' or a list"),
    ({"metrics": ["token_f1", "token_f1"]}, "metrics must be 'auto' or a list of distinct names"),
    ({"selection_metric": "accuracy"}, "selection_metric must be 'loss' or one of the run's metrics"),
    ({"selection_metric": "token_f1"}, "selection_metric must be 'loss' or one of the run's metrics"),  # not scored
    ({"metrics": ["token_f1"], "selection_metric": "token_f1", "gen_eval_rows": 0}, "gen_eval_rows is 0"),
    ({"baseline": "true"}, "baseline must be true or false"),
])
def test_schedule_and_evaluation_keys_refuse_a_wrong_value_by_name(hp, match):
    with pytest.raises(ValueError, match=match):
        backend_mod._training_options(hp)


def test_evaluation_rows_are_nested_subsets_drawn_once_with_the_seed():
    def rows(n=100, seed=7, **hp):
        return backend_mod._evaluation_rows(backend_mod._training_options(hp), n, seed)

    every = list(range(100))
    assert rows() == (every, every, every)  # without the keys: every row, in the order of the file
    loss, gen, final = rows(eval_max_rows=40, gen_eval_rows=8, final_gen_rows=60)
    assert (len(loss), len(gen), len(final)) == (40, 8, 60) and len(set(final)) == 60
    assert gen == loss[:8] and loss == final[:40]  # one seeded order: each subset is the start of the next
    assert loss != every[:40]  # a draw, not the first rows of the file
    assert rows(eval_max_rows=40, gen_eval_rows=8, final_gen_rows=60) == (loss, gen, final)  # the same on a replay
    assert rows(seed=8, eval_max_rows=40)[0] != loss
    assert rows(eval_max_rows=40)[1] == loss  # gen_eval_rows unset: generate from the loss rows
    assert rows(gen_eval_rows=0)[:2] == (every, None)  # 0: a periodic evaluation generates nothing
    assert rows(eval_max_rows=40, gen_eval_rows=64)[1] == loss  # never more rows than the loss reads
    assert rows(n=10, eval_max_rows=40, final_gen_rows=500) == (every[:10], every[:10], every[:10])
    assert rows(n=0, eval_max_rows=4) == ([], [], [])


def test_group_by_length_batches_rows_of_similar_length_and_keeps_every_row_once():
    import random

    order = list(range(1003))
    random.Random(1).shuffle(order)
    lengths = [(37 * j) % 211 + 1 for j in range(1003)]
    batches = backend_mod._length_grouped_batches(order, lengths, 8, random.Random(3))
    assert sorted(j for b in batches for j in b) == list(range(1003))
    assert len(batches) == 126 and sorted(len(b) for b in batches)[:2] == [3, 8]  # as many batches as ungrouped
    spread = [max(lengths[j] for j in b) - min(lengths[j] for j in b) for b in batches if len(b) == 8]
    assert sum(spread) / len(spread) < 12  # lengths go from 1 to 211: an ungrouped batch spans most of that
    assert max(lengths[j] for j in batches[0]) == max(lengths)  # the longest row is in the first batch
    assert batches == backend_mod._length_grouped_batches(order, lengths, 8, random.Random(3))
    assert batches != backend_mod._length_grouped_batches(order, lengths, 8, random.Random(4))
    assert backend_mod._length_grouped_batches([], [], 8, random.Random(3)) == []


def test_alpaca_template_is_byte_identical_to_zak_forge_template_py():
    # Literal copy of zak-forge src/forge/runner/template.py (the legacy runner's
    # training format); the hub's existing artifacts were trained on exactly this.
    def format_prompt(instruction: str) -> str:
        return f"### Instruction:\n{instruction}\n\n### Response:\n"

    def format_example(instruction: str, response: str) -> str:
        return format_prompt(instruction) + response

    _STOP_MARKERS = ("### Instruction:", "### Response:", "###")

    def truncate_completion(text: str) -> str:
        cut = len(text)
        for marker in _STOP_MARKERS:
            i = text.find(marker)
            if i != -1:
                cut = min(cut, i)
        return text[:cut].strip()

    for instruction in ("Classify: great movie", "multi\nline\n", ""):
        assert backend_mod._format_prompt("alpaca", instruction) == format_prompt(instruction)
        assert backend_mod._format_prompt("alpaca", instruction) + "pos" == format_example(instruction, "pos")
    assert backend_mod._format_prompt("raw", "1+1=") == "1+1="
    for raw in ("positive\n### Instruction:\nnext", " neg ", "x ### Response: y", "a\n###"):
        assert backend_mod._clean_completion("alpaca", raw) == truncate_completion(raw)
    assert backend_mod._clean_completion("raw", "a\n### b") == "a\n### b"  # raw never truncates


def test_gguf_quant_profiles_match_zak_forge_llamacpp_py():
    # Literal copy of PROFILE_ARGS in zak-forge src/forge/runner/adapters/llamacpp.py.
    PROFILE_ARGS = {
        "Q6_K": ([], "Q6_K"),
        "Q8_0": ([], "Q8_0"),
        "MXQ4": (["--output-tensor-type", "Q6_K", "--token-embedding-type", "Q6_K"], "Q4_K_M"),
    }
    for profile, (opts, qtype) in PROFILE_ARGS.items():
        assert backend_mod._gguf_quant_args(profile) == (opts, qtype)
    assert backend_mod._gguf_quant_args("Q4_K_M") == ([], "Q4_K_M")  # bare type names pass through
    with pytest.raises(backend_mod.GGUFExportError, match="not a llama.cpp quant type name"):
        backend_mod._gguf_quant_args("Q4 K M")


def _fake_llama_cpp(tmp_path, monkeypatch):
    llama = tmp_path / "llama.cpp"
    (llama / "build" / "bin").mkdir(parents=True)
    (llama / "convert_hf_to_gguf.py").write_text("")
    (llama / "build" / "bin" / "llama-quantize").write_text("")
    monkeypatch.setenv("LLAMA_CPP_DIR", str(llama))
    calls: list[list[str]] = []

    def fake_run(cmd, **kw):
        calls.append([str(c) for c in cmd])
        Path(cmd[cmd.index("--outfile") + 1] if "--outfile" in cmd else cmd[-2]).write_bytes(b"gguf")
        return subprocess.CompletedProcess(cmd, 0, "", "")

    monkeypatch.setattr(backend_mod.subprocess, "run", fake_run)
    return calls


@pytest.mark.parametrize("hp_quant,env,expected_tail", [
    (None, None, ["Q8_0"]),  # nothing set: the old default
    (None, "Q6_K", ["Q6_K"]),  # env is the fallback when hp does not set it
    ("Q4_K_M", "Q6_K", ["Q4_K_M"]),  # hp wins over env
    ("MXQ4", None, ["--output-tensor-type", "Q6_K", "--token-embedding-type", "Q6_K", "<f16>", "<out>", "Q4_K_M"]),
])
def test_gguf_export_quantises_with_the_profile_from_hp_then_env(tmp_path, monkeypatch, hp_quant, env, expected_tail):
    calls = _fake_llama_cpp(tmp_path, monkeypatch)
    if env is None:
        monkeypatch.delenv("SAKURA_GGUF_QUANT", raising=False)
    else:
        monkeypatch.setenv("SAKURA_GGUF_QUANT", env)
    out = RunDir(tmp_path / "run").ensure()
    fitted = backend_mod._Fitted(model_dir=tmp_path / "merged", tokenizer_dir=tmp_path / "merged",
                                 eval_examples=[], gguf_quant=hp_quant)
    artifact = backend_mod.BACKEND.export(result=type("R", (), {"model": fitted})(), fmt="gguf", out=out)
    assert artifact.format == "gguf" and artifact.path == out.artifacts / "model.gguf"
    convert, quantize = calls
    assert convert[1].endswith("convert_hf_to_gguf.py") and convert[-1] == "f16"
    f16, dest = str(out.artifacts / "model.fp16.gguf"), str(out.artifacts / "model.gguf")
    expected = [{"<f16>": f16, "<out>": dest}.get(t, t) for t in expected_tail]
    if len(expected) == 1:
        expected = [f16, dest, *expected]
    assert quantize[1:] == expected


@pytest.mark.parametrize("quant,hidden,ok", [
    ("MXQ4", 576, False),  # SmolLM2-135M under the hub's MXQ4 profile: llama-quantize aborted (staging 9f21e4a6)
    ("Q4_K_M", 576, False),
    ("Q6_K", 1000, False),
    ("MXQ4", 1024, True),  # Qwen3-0.6B
    ("Q8_0", 576, True),  # non-K types have no 256 super-block
    ("Q5_0", 576, True),
    ("MXQ4", None, True),  # no hidden_size in config.json: let llama-quantize speak
])
def test_gguf_export_refuses_k_quants_for_a_hidden_size_that_is_not_a_multiple_of_256(
        tmp_path, monkeypatch, quant, hidden, ok):
    import json

    calls = _fake_llama_cpp(tmp_path, monkeypatch)
    merged = tmp_path / "merged"
    merged.mkdir(exist_ok=True)
    (merged / "config.json").write_text(json.dumps({"hidden_size": hidden} if hidden else {"model_type": "x"}))
    out = RunDir(tmp_path / "run").ensure()
    fitted = backend_mod._Fitted(model_dir=merged, tokenizer_dir=merged, eval_examples=[], gguf_quant=quant)
    result = type("R", (), {"model": fitted})()
    if ok:
        backend_mod.BACKEND.export(result=result, fmt="gguf", out=out)
        assert len(calls) == 2
    else:
        with pytest.raises(backend_mod.GGUFExportError, match=f"hidden_size={hidden}"):
            backend_mod.BACKEND.export(result=result, fmt="gguf", out=out)
        assert calls == []


def test_export_mlx_resolves_its_default_quant_at_call_time():
    # The notebook ships export_mlx as a cell of its own: a default argument bound to
    # DEFAULT_MLX_QUANT is evaluated when the cell defines the function -> NameError
    # (staging job 83cafca4 needed an LLM repair for exactly this).
    import inspect

    assert inspect.signature(backend_mod.export_mlx).parameters["quant"].default is None


def test_gguf_export_rejects_a_bad_hp_quant_before_any_subprocess(tmp_path, monkeypatch):
    calls = _fake_llama_cpp(tmp_path, monkeypatch)
    out = RunDir(tmp_path / "run").ensure()
    fitted = backend_mod._Fitted(model_dir=tmp_path / "merged", tokenizer_dir=tmp_path / "merged",
                                 eval_examples=[], gguf_quant="Q8_0; rm -rf ~")
    with pytest.raises(backend_mod.GGUFExportError, match="gguf_quant="):
        backend_mod.BACKEND.export(result=type("R", (), {"model": fitted})(), fmt="gguf", out=out)
    assert calls == []


def test_4bit_without_bitsandbytes_fails_with_a_clear_message(monkeypatch):
    import sys

    monkeypatch.setitem(sys.modules, "bitsandbytes", None)  # import raises ImportError
    with pytest.raises(RuntimeError, match="precision: 4bit .*bitsandbytes.*CUDA-only"):
        backend_mod._load_base_model("x", None, "4bit", "cpu")


def test_runtime_eval_and_predict_apply_the_template_recorded_next_to_the_artifact(tmp_path, monkeypatch):
    out = RunDir(tmp_path / "run").ensure()
    (out.artifacts / "preprocess.json").write_text(json.dumps({"prompt_template": "alpaca"}))
    artifact = Artifact(path=out.artifacts / "model.safetensors", format="safetensors", sha256="x", bytes=1)
    artifact.path.write_bytes(b"x")
    seen = []

    def fake_generate(d, prompt, n):
        seen.append(prompt)
        return {"### Instruction:\n1+1=\n\n### Response:\n": "2\n### Instruction:\nnext",
                "### Instruction:\n2+2=\n\n### Response:\n": "5"}[prompt]

    monkeypatch.setattr(backend_mod, "_hf_generate", fake_generate)
    data = type("D", (), {"validation": [{"prompt": "1+1=", "completion": "2"},
                                         {"prompt": "2+2=", "completion": "4"}]})()
    ev = backend_mod.BACKEND.runtime_eval(artifact, data=data)
    assert (ev.runtime, ev.value, ev.n) == ("transformers", 0.5, 2)
    assert seen == ["### Instruction:\n1+1=\n\n### Response:\n", "### Instruction:\n2+2=\n\n### Response:\n"]
    assert backend_mod.BACKEND.predict(artifact, {"prompt": "1+1="}) == {"text": "2"}
    # No preprocess.json (an artifact from before it existed): raw prompts, as before.
    (out.artifacts / "preprocess.json").unlink()
    monkeypatch.setattr(backend_mod, "_hf_generate", lambda d, p, n: {"1+1=": "2"}[p])
    assert backend_mod.BACKEND.predict(artifact, {"prompt": "1+1="}) == {"text": "2"}


# --- runtime_eval with the evaluation recorded next to the artifact (preprocess.json `eval`) ---

_JSON_VAL = [{"prompt": "a?", "completion": '{"k": "a", "n": 1}'}, {"prompt": "b?", "completion": '{"k": "b", "n": 2}'},
             {"prompt": "c?", "completion": '{"k": "c", "n": 3}'}]
_JSON_ANSWERS = {"a?": '{"n": 1, "k": "a"}', "b?": '{"k": "b", "n": 9}', "c?": "not json"}  # field F1: 1, 0.5, 0


def _scored_artifact(tmp_path, fmt="safetensors", **eval_block):
    out = RunDir(tmp_path / "run").ensure()
    artifact = Artifact(path=out.artifacts / f"model.{fmt}", format=fmt, sha256="x", bytes=1)
    artifact.path.write_bytes(b"x")
    block = {"metric": "json_field_f1", "metrics": ["json_field_f1", "json_match", "json_valid"], "max_new_tokens": 40,
             "exact_match_rule": "whole", "batch_size": 1, "rows": None, **eval_block}
    (out.artifacts / "preprocess.json").write_text(json.dumps({"prompt_template": "raw", "max_seq_length": 64,
                                                               "eval": block}))
    return artifact


def test_runtime_eval_uses_the_length_the_metric_and_the_rows_recorded_with_the_artifact(tmp_path, monkeypatch):
    seen = []
    monkeypatch.setattr(backend_mod, "_hf_generate", lambda d, prompt, n: seen.append((prompt, n)) or _JSON_ANSWERS[prompt])
    data = type("D", (), {"validation": _JSON_VAL})()
    ev = backend_mod.BACKEND.runtime_eval(_scored_artifact(tmp_path), data=data)
    assert (ev.runtime, ev.metric, ev.n, ev.reason) == ("transformers", "json_field_f1", 3, None)
    assert ev.value == pytest.approx((1.0 + 0.5 + 0.0) / 3)
    assert seen == [("a?", 40), ("b?", 40), ("c?", 40)]  # 40 new tokens, not 8
    # the rows training generated from; an index past the rows given (a capped replay) is skipped
    ev = backend_mod.BACKEND.runtime_eval(_scored_artifact(tmp_path / "rows", rows=[0, 2, 7]), data=data)
    assert (ev.n, ev.value) == (2, pytest.approx(0.5))
    ev = backend_mod.BACKEND.runtime_eval(_scored_artifact(tmp_path / "none-left", rows=[7, 8]), data=data)
    assert ev.n == 3  # none of the recorded rows is there: every row given is scored


def test_runtime_eval_generates_in_batches_when_training_did(tmp_path, monkeypatch):
    calls = []

    def many(model_dir, prompts, max_tokens, batch_size, *bounds):  # bounds: the token bound and the window
        calls.append((list(prompts), max_tokens, batch_size))
        return [_JSON_ANSWERS[p] for p in prompts]

    monkeypatch.setattr(backend_mod, "_hf_generate_many", many)
    monkeypatch.setattr(backend_mod, "_hf_generate", lambda *a: pytest.fail("one row at a time with batch_size 4"))
    ev = backend_mod.BACKEND.runtime_eval(_scored_artifact(tmp_path, batch_size=4, metric="json_valid"),
                                          data=type("D", (), {"validation": _JSON_VAL})())
    assert (ev.metric, ev.value, ev.n) == ("json_valid", pytest.approx(2 / 3), 3)
    assert calls == [(["a?", "b?", "c?"], 40, 4)]


def test_an_eval_block_this_engine_cannot_read_is_rescored_as_before(tmp_path, monkeypatch):
    seen = []
    monkeypatch.setattr(backend_mod, "_hf_generate", lambda d, prompt, n: seen.append(n) or "2\nmore")
    data = type("D", (), {"validation": [{"prompt": "1+1=", "completion": "2"}]})()
    for bad in ({"metric": "bleu"}, {"max_new_tokens": "long"}, {"max_new_tokens": 0}):
        ev = backend_mod.BACKEND.runtime_eval(_scored_artifact(tmp_path / str(len(seen)), **bad), data=data)
        assert (ev.metric, ev.value) == ("exact_match", 1.0)  # 8 tokens, first line
    assert seen == [8, 8, 8]


def test_a_gguf_is_rescored_at_the_recorded_length_and_stops_at_a_newline_only_under_the_first_line_rule(tmp_path, monkeypatch):
    import sys
    import types

    calls = []

    class Llama:
        def __init__(self, model_path, n_ctx, verbose):
            pass

        def __call__(self, prompt, max_tokens, temperature, **kw):
            calls.append((max_tokens, kw))
            return {"choices": [{"text": '{"k": "a",\n "n": 1}'}]}

    monkeypatch.setitem(sys.modules, "llama_cpp", types.SimpleNamespace(Llama=Llama))
    monkeypatch.setattr(backend_mod, "_LOADED", {})
    data = type("D", (), {"validation": _JSON_VAL[:1]})()
    ev = backend_mod.BACKEND.runtime_eval(_scored_artifact(tmp_path, "gguf"), data=data)
    assert (ev.runtime, ev.metric, ev.value, ev.n) == ("llama-cpp-python", "json_field_f1", 1.0, 1)
    assert calls == [(40, {})]  # a JSON answer may span lines: no stop at the first newline
    short = _scored_artifact(tmp_path / "short", "gguf", metric="exact_match", metrics=["exact_match"], max_new_tokens=8,
                             exact_match_rule="first_line")
    ev = backend_mod.BACKEND.runtime_eval(short, data=data)
    assert calls[-1] == (8, {"stop": ["\n"]}) and (ev.metric, ev.value) == ("exact_match", 0.0)


def test_an_mlx_export_is_scored_with_the_runs_own_evaluation(tmp_path, monkeypatch):
    def fake_run(cmd, **kw):
        dest = Path(cmd[cmd.index("--mlx-path") + 1])
        dest.mkdir(parents=True)
        (dest / "config.json").write_text("{}")
        return subprocess.CompletedProcess(cmd, 0, "", "")

    generated = []
    monkeypatch.setattr(backend_mod, "_require_mlx", lambda: None)
    monkeypatch.setattr(backend_mod.subprocess, "run", fake_run)
    monkeypatch.setattr(backend_mod, "_mlx_generate", lambda d, prompt, n: generated.append((prompt, n)) or _JSON_ANSWERS[prompt])
    out = RunDir(tmp_path / "run").ensure()
    scoring = {"metric": "json_field_f1", "max_new_tokens": 40, "exact_match_rule": "whole", "rows": [1]}
    fitted = backend_mod._Fitted(model_dir=_merged_dir(tmp_path), tokenizer_dir=tmp_path, eval_examples=list(_JSON_VAL),
                                 scoring=scoring)
    backend_mod.BACKEND.export(result=type("R", (), {"model": fitted})(), fmt="mlx", out=out)
    report = json.loads((out.artifacts / "model.mlx.eval.json").read_text())
    assert (report["metric"], report["value"], report["n"]) == ("json_field_f1", 0.5, 1) and generated == [("b?", 40)]
    # the standalone command an apple-silicon worker runs reads the same block from preprocess.json
    val, pre = tmp_path / "val.jsonl", tmp_path / "preprocess.json"
    val.write_text("".join(json.dumps(r) + "\n" for r in _JSON_VAL))
    pre.write_text(json.dumps({"prompt_template": "raw", "eval": scoring}))
    assert backend_mod.main(["mlx", "--merged", str(tmp_path / "merged"), "--validation", str(val), "--quant", "q8",
                             "--out", str(tmp_path / "cli" / "model.mlx"), "--preprocess", str(pre)]) == 0
    assert generated[-1] == ("b?", 40)


# --- a real LoRA run on CPU, from the tiny offline Llama --------------------------

def _hp(**over):
    return {"epochs": 1, "batch_size": 2, "learning_rate": 1e-3, "lora_r": 4, "lora_alpha": 8,
            "max_seq_length": 32, **over}


def _resolved(tmp_path, model_id, hp):
    from sakura.atelier.registry import resolve
    from sakura.atelier.spec import load_spec

    spec = load_spec(f"""
atelier: 1
name: tiny
task: text_generation
data: {{uri: "file://{tmp_path}/x.jsonl", format: jsonl, split: {{validation_fraction: 0.5}}}}
features:
  inputs: [{{name: prompt, type: text}}]
  output: {{name: completion, type: text}}
model: {model_id}
hp: text_generation/lora-fast@1
export: [safetensors]
""").model_copy(update={"hyperparameters": hp})
    return resolve(spec)


class _Metrics:
    def __init__(self):
        self.rows = []

    def log(self, name, value, *, step, split, epoch):
        self.rows.append((name, split, step, epoch, value))


_TINY_DATA = [{"prompt": "1+1=", "completion": "2"}, {"prompt": "2+2=", "completion": "4"},
              {"prompt": "1+2=", "completion": "4"}, {"prompt": "2+1=", "completion": "2"}]


@pytest.mark.parametrize("precision,template,grad_accum", [("fp32", "raw", 1), ("bf16", "alpaca", 2)])
def test_train_runs_on_cpu_from_a_local_base_model(tmp_path, precision, template, grad_accum):
    pytest.importorskip("transformers")
    pytest.importorskip("peft")
    base = _tiny_llama(tmp_path / "base")
    hp = _hp(base_model=str(base), precision=precision, prompt_template=template, grad_accum=grad_accum,
             lora_target_modules=["q_proj", "v_proj"], lora_dropout=0.0, gguf_quant="MXQ4",
             max_seq_length=256)  # the alpaca preamble alone is longer than 32 tokens: no reply would be left to learn
    resolved = _resolved(tmp_path, "hf-causal-lm", hp)
    data = type("D", (), {"train": _TINY_DATA, "validation": _TINY_DATA[:2]})()
    out = RunDir(tmp_path / "run").ensure()
    metrics = _Metrics()

    result = backend_mod.BACKEND.train(resolved, data, out, metrics, None)

    assert result.provenance == {"base_model": str(base), "base_revision": None, "base_commit": None,
                                 "precision": precision, "prompt_template": template, "resumed": False}
    assert set(result.metrics) == {"loss", "exact_match"}
    train_steps = [r[2] for r in metrics.rows if r[1] == "train"]
    assert len(train_steps) == 2  # 4 examples / batch 2 = 2 micro-batches
    assert train_steps == ([0, 1] if grad_accum == 1 else [0, 0])  # one optimizer step per grad_accum
    assert next(r for r in metrics.rows if r[1] == "validation")[2] == (2 if grad_accum == 1 else 1)
    merged = out.checkpoint / "merged"
    assert (merged / "model.safetensors").is_file() and (merged / "tokenizer.json").is_file()
    assert json.loads((out.checkpoint / "meta.json").read_text())["base_model"] == str(base)
    pre = json.loads((out.artifacts / "preprocess.json").read_text())
    assert pre["prompt_template"] == template and pre["max_seq_length"] == 256
    assert result.model.gguf_quant == "MXQ4" and result.model.prompt_template == template

    artifact = backend_mod.BACKEND.export(result, "safetensors", out)
    assert artifact.format == "safetensors" and (out.artifacts / "config.json").is_file()
    backend_mod._LOADED.clear()
    ev = backend_mod.BACKEND.runtime_eval(artifact, data)
    assert ev.runtime == "transformers" and ev.n == 2 and ev.value is not None
    pred = backend_mod.BACKEND.predict(artifact, {"prompt": "1+1=", "max_tokens": 4})
    assert isinstance(pred["text"], str)


def test_train_fails_on_a_real_run_with_4bit_and_no_bitsandbytes(tmp_path, monkeypatch):
    import sys

    pytest.importorskip("transformers")
    pytest.importorskip("peft")
    monkeypatch.setitem(sys.modules, "bitsandbytes", None)
    base = _tiny_llama(tmp_path / "base")
    resolved = _resolved(tmp_path, "hf-causal-lm", _hp(base_model=str(base), precision="4bit"))
    data = type("D", (), {"train": _TINY_DATA, "validation": _TINY_DATA[:1]})()
    with pytest.raises(RuntimeError, match="bitsandbytes"):
        backend_mod.BACKEND.train(resolved, data, RunDir(tmp_path / "run").ensure(), _Metrics(), None)


def _fake_llama(monkeypatch, n_ctx_seen: list):
    """Fake llama_cpp: a Llama that refuses a prompt over its context the way
    llama-cpp-python does, and answers "2" otherwise."""
    import sys
    import types

    class Llama:
        def __init__(self, model_path, n_ctx, verbose):
            self.n_ctx = n_ctx
            n_ctx_seen.append(n_ctx)

        def __call__(self, prompt, max_tokens, temperature, stop=None):
            if len(prompt.split()) >= self.n_ctx:
                raise ValueError(f"Requested tokens ({len(prompt.split())}) exceed context window of {self.n_ctx}")
            return {"choices": [{"text": "2"}]}

    monkeypatch.setitem(sys.modules, "llama_cpp", types.SimpleNamespace(Llama=Llama))
    monkeypatch.setattr(backend_mod, "_LOADED", {})


def _gguf(tmp_path, max_seq_length=None):
    out = RunDir(tmp_path / "run").ensure()
    artifact = Artifact(path=out.artifacts / "model.gguf", format="gguf", sha256="x", bytes=1)
    artifact.path.write_bytes(b"x")
    if max_seq_length is not None:
        (out.artifacts / "preprocess.json").write_text(
            json.dumps({"prompt_template": "raw", "max_seq_length": max_seq_length}))
    return artifact


@pytest.mark.parametrize("max_seq_length, n_ctx", [(None, 256), (64, 256), (1024, 2048)])
def test_a_gguf_is_loaded_with_room_for_its_training_length(tmp_path, monkeypatch, max_seq_length, n_ctx):
    seen: list = []
    _fake_llama(monkeypatch, seen)
    data = type("D", (), {"validation": [{"prompt": "1+1=", "completion": "2"}]})()
    ev = backend_mod.BACKEND.runtime_eval(_gguf(tmp_path, max_seq_length), data=data)
    assert seen == [n_ctx]
    assert (ev.runtime, ev.value, ev.n, ev.reason) == ("llama-cpp-python", 1.0, 1, None)


def test_a_gguf_has_room_for_the_answers_the_run_generates(tmp_path, monkeypatch):
    # eval_max_new_tokens: auto may ask for 1,024 tokens after a prompt of up to 512: twice the
    # training length (1,024) would refuse the longest rows.
    seen: list = []
    _fake_llama(monkeypatch, seen)
    artifact = _gguf(tmp_path, 512)
    settings = artifact.path.parent / "preprocess.json"
    settings.write_text(json.dumps({**json.loads(settings.read_text()), "eval": {
        "metric": "token_f1", "metrics": ["token_f1"], "max_new_tokens": 1024, "exact_match_rule": "whole"}}))
    data = type("D", (), {"validation": [{"prompt": "1+1=", "completion": "2"}]})()
    ev = backend_mod.BACKEND.runtime_eval(artifact, data=data)
    assert seen == [1536] and ev.reason is None


def test_a_prompt_over_the_gguf_context_is_a_reason_not_a_failed_run(tmp_path, monkeypatch):
    # A run trained on 1,024-token chats loaded its GGUF with a 256-token context, and the
    # first 433-token validation prompt raised out of runtime_eval: the run failed after training.
    _fake_llama(monkeypatch, [])
    data = type("D", (), {"validation": [{"prompt": "word " * 3000, "completion": "2"}]})()
    ev = backend_mod.BACKEND.runtime_eval(_gguf(tmp_path, 1024), data=data)
    assert ev.value is None and ev.n == 0
    assert "exceed context window of 2048" in ev.reason
