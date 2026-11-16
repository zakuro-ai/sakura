"""The decision task through the REAL peft backend (LoRA, merge, safetensors, the Hugging Face KV cache) on a
tiny random Llama built offline. Skipped where `transformers` / `peft` are not installed (CI has neither; the
fake-LM tests in test_decision_run.py cover the same pipeline there).

A random tiny model learns nothing useful, so nothing here is about accuracy: it is about the plumbing that a
fake cannot exercise -- that the adapter merges to the same function, that the delivered safetensors reloads
and answers as the run scored it, and that a cached state prefix equals a full forward pass."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

pytest.importorskip("transformers")
pytest.importorskip("peft")
torch = pytest.importorskip("torch")

from sakura.atelier import registry  # noqa: E402
from sakura.atelier.backends import decision_lm as lm  # noqa: E402
from sakura.atelier.backends import peft as backend_mod  # noqa: E402
from sakura.atelier.data import materialise  # noqa: E402
from sakura.atelier.decision import Instance  # noqa: E402
from sakura.atelier.spec import load_spec  # noqa: E402
from sakura.atelier.types import MetricsSink, RunDir  # noqa: E402
from tests.atelier.decision_fakes import TONE, TOPIC, URGENT, write_dataset  # noqa: E402


def tiny_decision_llama(d: Path) -> Path:
    """A random 2-layer Llama in HF layout whose tokenizer is character-level except that ` A` ... ` Z` are
    single tokens, as they are in the bases the task allows."""
    import numpy as np
    from safetensors.numpy import save_file
    from tokenizers import Regex, Tokenizer, models, pre_tokenizers

    tokens = ["<unk>", "<s>", "</s>"] + [chr(c) for c in range(32, 127)] + ["\n"] + [f" {chr(c)}" for c in range(65, 91)]
    vocab = {t: i for i, t in enumerate(dict.fromkeys(tokens))}
    tok = Tokenizer(models.WordLevel(vocab=vocab, unk_token="<unk>"))
    tok.pre_tokenizer = pre_tokenizers.Split(Regex(r" [A-Z]|[\s\S]"), behavior="isolated")
    d.mkdir()
    tok.save(str(d / "tokenizer.json"))
    (d / "tokenizer_config.json").write_text(json.dumps({
        "tokenizer_class": "PreTrainedTokenizerFast", "eos_token": "</s>", "unk_token": "<unk>", "pad_token": "<unk>"}))
    h, inter, layers, v = 64, 128, 2, len(vocab)
    (d / "config.json").write_text(json.dumps({
        "architectures": ["LlamaForCausalLM"], "model_type": "llama", "hidden_size": h, "intermediate_size": inter,
        "num_hidden_layers": layers, "num_attention_heads": 2, "num_key_value_heads": 2, "vocab_size": v,
        "rms_norm_eps": 1e-6, "rope_theta": 10000.0, "max_position_embeddings": 1024, "tie_word_embeddings": False,
        "bos_token_id": 1, "eos_token_id": 2, "torch_dtype": "float32"}))
    rng = np.random.default_rng(0)

    def w(*shape):
        return (rng.standard_normal(shape) * 0.05).astype(np.float32)

    t = {"model.embed_tokens.weight": w(v, h), "model.norm.weight": np.ones(h, np.float32), "lm_head.weight": w(v, h)}
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


def inst(state="refund please", qid="topic", labels=("billing", "bug", "other"), kind="choice", text="Topic?"):
    return Instance(state=state, qid=qid, kind=kind, text=text, labels=list(labels),
                    target=[1.0] + [0.0] * (len(labels) - 1))


@pytest.fixture
def base(tmp_path):
    return tiny_decision_llama(tmp_path / "base")


def loaded(base):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(str(base))
    model = AutoModelForCausalLM.from_pretrained(str(base), dtype=torch.float32)
    return tok, model


def test_the_letters_are_single_tokens_of_the_real_tokenizer_class(base):
    tok, _ = loaded(base)
    r = lm.make_readout(tok)
    assert len(set(r.letters)) == 26 and r.start == []


def test_a_cached_state_prefix_scores_like_a_full_forward_pass_on_a_real_hf_model(base):
    tok, model = loaded(base)
    r = lm.make_readout(tok)
    state = "Hi, I was charged twice for March and the app keeps crashing."
    insts = [inst(state), inst(state, "urgent", ("true", "false"), "noul", "Needs a reply today?"),
             inst(state, "tone", ("calm", "annoyed", "angry"), "score", "How upset?")]
    shared = lm.score_state(model, tok, r, state, insts, 512, "cpu")
    for got, i in zip(shared, insts):
        alone = lm.score_instances(model, tok, r, [i], 512, "cpu")[0].logits
        assert got == pytest.approx(alone, abs=1e-4)


def test_left_padding_does_not_change_the_score_on_a_real_hf_model(base):
    tok, model = loaded(base)
    r = lm.make_readout(tok)
    items = [inst("short"), inst("a rather longer state, " * 6, labels=("a", "b")), inst("mid length state")]
    together = lm.score_instances(model, tok, r, items, 512, "cpu", batch_size=3)
    for t, i in zip(together, items):
        assert t.logits == pytest.approx(lm.score_instances(model, tok, r, [i], 512, "cpu")[0].logits, abs=1e-4)


def test_the_adapted_model_trains_with_gradient_checkpointing(base):
    """Without it a 2B base does not fit an 11 GB card at the preset's 8 x 1024 tokens (the GPU host 2080 Ti: 1 x 1024
    already ran out of memory; with it 8 x 1024 peaked at 7.5 GiB). Inputs must require grad, or a frozen
    base under checkpointing hands the LoRA no gradient at all."""
    from sakura.atelier.decision import decision_options

    backend = backend_mod.PEFTBackend()
    hp = {"base_model": str(base), "base_model_allowlist": {str(base): None}, "precision": "fp32",
          "lora_target_modules": ["q_proj", "v_proj"], "lora_r": 4, "lora_alpha": 8}
    tok, model = loaded(base)
    peft_model = backend._decision_adapt(model, decision_options(hp))
    assert peft_model.base_model.model.is_gradient_checkpointing
    peft_model.train()
    r = lm.make_readout(tok)
    loss, _ = lm.batch_loss(peft_model, tok, r, [(inst("refund please"), [0, 1, 2])], 512, "cpu")
    loss.backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for n, p in peft_model.named_parameters() if "lora_" in n)

    _, plain = loaded(base)
    off = backend._decision_adapt(plain, decision_options({**hp, "gradient_checkpointing": False}))
    assert not off.base_model.model.is_gradient_checkpointing


def test_the_merged_model_computes_what_the_adapter_did(base, tmp_path):
    from sakura.atelier.decision import decision_options

    backend = backend_mod.PEFTBackend()
    tok, model = loaded(base)
    r = lm.make_readout(tok)
    opt = decision_options({"base_model": str(base), "base_model_allowlist": {str(base): None}, "precision": "fp32",
                            "lora_target_modules": ["q_proj", "v_proj"], "lora_r": 4, "lora_alpha": 8})
    peft_model = backend._decision_adapt(model, opt)
    with torch.no_grad():  # LoRA's B starts at zero; make the adapter do something
        for n, p in peft_model.named_parameters():
            if "lora_B" in n:
                p.add_(torch.randn_like(p) * 0.2)
    items = [inst("refund please"), inst("the app crashed", labels=("a", "b", "c", "d"))]
    before = [s.logits for s in lm.score_instances(peft_model, tok, r, items, 512, "cpu")]
    merged = backend._decision_merge(peft_model, opt, tmp_path / "merged", tok, str(base), None)
    after = [s.logits for s in lm.score_instances(merged, tok, r, items, 512, "cpu")]
    for b, a in zip(before, after):
        assert b == pytest.approx(a, abs=1e-4)
    assert (tmp_path / "merged" / "model.safetensors").is_file()  # a single file, not shards
    assert (tmp_path / "merged" / "tokenizer.json").is_file()


# ----------------------------------------------------------- the whole backend

SPEC = """
atelier: 1
name: tiny
task: decision
data: {uri: "data.jsonl", format: jsonl}
model: hf-decision-lm
hp: decision/lora-forge@1
export: [safetensors]
seed: 3
"""


def resolved_for(base, tmp_path):
    hp = {**registry.load_preset("decision/lora-forge@1"), "base_model": str(base), "base_revision": None,
          "epochs": 2, "batch_size": 8, "grad_accum": 1, "learning_rate": 0.002, "lora_r": 4, "lora_alpha": 8,
          "lora_dropout": 0.0, "lora_target_modules": ["q_proj", "v_proj"], "max_seq_length": 256, "precision": "fp32"}
    hp["base_model_allowlist"] = {str(base): None}
    spec = load_spec(SPEC).model_copy(update={"hyperparameters": hp})
    from sakura.atelier.types import ResolvedSpec

    return ResolvedSpec(spec=spec, hyperparameters=hp, backend="peft"), spec


def test_the_real_backend_trains_exports_and_answers(base, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    write_dataset(tmp_path / "data.jsonl", n=80)
    resolved, spec = resolved_for(base, tmp_path)
    data = materialise(spec, tmp_path / "data")
    out = RunDir(tmp_path / "run").ensure()
    backend = backend_mod.PEFTBackend()
    with MetricsSink(out.metrics_path) as metrics:
        result = backend.train(resolved, data, out, metrics, None)

    assert set(result.metrics) == {"accuracy", "log_loss", "ece"}
    assert result.provenance["prompt_template"] == "decision/v1" and result.provenance["base_model"] == str(base)
    assert result.calibration["method"] == "temperature"
    artifact = backend.export(result, "safetensors", out)
    assert artifact.format == "safetensors" and (out.artifacts / "config.json").is_file()
    (cfg_art,) = backend.extra_artifacts(result, out)
    assert cfg_art.format == "decision_config"

    backend_mod._LOADED.clear()
    ev = backend.runtime_eval(artifact, data)
    assert ev.runtime == "transformers" and ev.value == pytest.approx(result.metrics["accuracy"], abs=1e-6)
    request = {"state": "refund for the invoice", "questions": {"topic": TOPIC, "urgent": URGENT, "tone": TONE}}
    answers = backend.predict(artifact, request)["answers"]
    assert set(answers) == {"topic", "urgent", "tone"} and sum(answers["topic"]["probs"].values()) == pytest.approx(1.0, abs=1e-4)
    one = backend.predict(artifact, {**request, "questions": {"topic": TOPIC}})["answers"]["topic"]
    assert one["probs"] == pytest.approx(answers["topic"]["probs"], abs=1e-4)  # shared prefix == alone
