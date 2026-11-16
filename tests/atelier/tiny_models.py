"""Tiny Hugging Face model directories built offline for the end-to-end tests: nothing is downloaded.

``tiny_bert_dir`` holds a config and a word-level tokenizer and NO weights (enough for ``init: random``);
the random two-layer Llama with weights is ``test_smoke_runs._tiny_causal_lm``."""

from __future__ import annotations

import json
from pathlib import Path


def tiny_bert_dir(d: Path, words: list[str], *, layers: int = 4, hidden: int = 32) -> Path:
    """A BERT sequence-classification config and a whitespace word-level tokenizer over ``words``."""
    from tokenizers import Tokenizer, models, pre_tokenizers, processors

    vocab = {t: i for i, t in enumerate(["[PAD]", "[UNK]", "[CLS]", "[SEP]", *sorted(set(words))])}
    tok = Tokenizer(models.WordLevel(vocab=vocab, unk_token="[UNK]"))
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    tok.post_processor = processors.TemplateProcessing(
        single="[CLS] $A [SEP]", special_tokens=[("[CLS]", vocab["[CLS]"]), ("[SEP]", vocab["[SEP]"])])
    d.mkdir(parents=True)
    tok.save(str(d / "tokenizer.json"))
    (d / "tokenizer_config.json").write_text(json.dumps({
        "tokenizer_class": "PreTrainedTokenizerFast", "pad_token": "[PAD]", "unk_token": "[UNK]",
        "cls_token": "[CLS]", "sep_token": "[SEP]"}))
    (d / "config.json").write_text(json.dumps({
        "architectures": ["BertForSequenceClassification"], "model_type": "bert", "hidden_size": hidden,
        "num_hidden_layers": layers, "num_attention_heads": 2, "intermediate_size": 2 * hidden,
        "max_position_embeddings": 512, "vocab_size": 30522, "type_vocab_size": 2, "pad_token_id": 0}))
    return d


def tiny_llama_dir(d: Path, alphabet: str, *, hidden: int = 64, layers: int = 2, positions: int = 256) -> Path:
    """A random Llama with weights and a character tokenizer over ``alphabet`` that decodes back to plain text:
    a few hundred KB. One token per character, so a completion's length in tokens is its length in characters."""
    import numpy as np
    from safetensors.numpy import save_file
    from tokenizers import Tokenizer, models, pre_tokenizers

    vocab = {t: i for i, t in enumerate(["<unk>", "<s>", "</s>", *dict.fromkeys(alphabet)])}
    tok = Tokenizer(models.WordLevel(vocab=vocab, unk_token="<unk>"))
    tok.pre_tokenizer = pre_tokenizers.Split("", behavior="isolated")
    d.mkdir(parents=True)
    tok.save(str(d / "tokenizer.json"))
    (d / "tokenizer_config.json").write_text(json.dumps({
        "tokenizer_class": "PreTrainedTokenizerFast", "bos_token": "<s>", "eos_token": "</s>", "unk_token": "<unk>"}))
    inter, v = 2 * hidden, len(vocab)
    (d / "config.json").write_text(json.dumps({
        "architectures": ["LlamaForCausalLM"], "model_type": "llama", "hidden_size": hidden,
        "intermediate_size": inter, "num_hidden_layers": layers, "num_attention_heads": 2,
        "num_key_value_heads": 2, "vocab_size": v, "rms_norm_eps": 1e-6, "rope_theta": 10000.0,
        "max_position_embeddings": positions, "tie_word_embeddings": False, "bos_token_id": 1,
        "eos_token_id": 2, "torch_dtype": "float32"}))
    rng = np.random.default_rng(0)

    def w(*shape: int):
        return (rng.standard_normal(shape) * 0.02).astype(np.float32)

    t = {"model.embed_tokens.weight": w(v, hidden), "model.norm.weight": np.ones(hidden, np.float32),
         "lm_head.weight": w(v, hidden)}
    for i in range(layers):
        prefix = f"model.layers.{i}."
        for n in ("q", "k", "v", "o"):
            t[f"{prefix}self_attn.{n}_proj.weight"] = w(hidden, hidden)
        t[f"{prefix}mlp.gate_proj.weight"], t[f"{prefix}mlp.up_proj.weight"] = w(inter, hidden), w(inter, hidden)
        t[f"{prefix}mlp.down_proj.weight"] = w(hidden, inter)
        t[f"{prefix}input_layernorm.weight"] = np.ones(hidden, np.float32)
        t[f"{prefix}post_attention_layernorm.weight"] = np.ones(hidden, np.float32)
    save_file(t, str(d / "model.safetensors"))
    return fuse_decoder(d)


def fuse_decoder(model_dir: Path) -> Path:
    """Make a character-level tokenizer decode ``f``, ``u``, ``k`` back to ``fuk`` (not ``f u k``), so a
    generated completion can equal the expected text and ``exact_match`` means something."""
    from tokenizers import Tokenizer, decoders

    tok = Tokenizer.from_file(str(model_dir / "tokenizer.json"))
    tok.decoder = decoders.Fuse()
    tok.save(str(model_dir / "tokenizer.json"))
    return model_dir


def synthetic_words(uri: str, tmp: Path) -> list[str]:
    """Every word of a ``synthetic://text`` dataset's text column (to build a tokenizer that covers it)."""
    import csv

    from sakura.atelier import synthetic

    path = synthetic.generate(uri, tmp, "csv")
    with open(path, newline="", encoding="utf-8") as fh:
        return sorted({w for row in csv.DictReader(fh) for w in row["text"].split()})
