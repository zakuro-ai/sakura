"""A tiny causal LM and tokenizer for the decision tests: they run without a GPU or `transformers`.

`CharTokenizer` is BPE-like in the one way the readout cares about: `" A"` ... `" Z"` are single tokens.
`FakeLM` is a real causal model (a running mean of the token embeddings, then a linear head), with
`attention_mask` and a KV cache like a Hugging Face model's, so left padding, cached prefixes and
gradients through the head behave as they do on the real thing."""

from __future__ import annotations

import json
import random
import re
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn

VOCAB = 3100
LETTER_BASE = 3000
_TOKEN = re.compile(r" [A-Z]|[\s\S]")


class CharTokenizer:
    pad_token_id = 0
    eos_token_id = 1
    bos_token_id = None
    pad_token = "<pad>"
    eos_token = "<eos>"

    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
        return [self._id(t) for t in _TOKEN.findall(text)]

    @staticmethod
    def _id(token: str) -> int:
        if len(token) == 2:  # " A" .. " Z"
            return LETTER_BASE + ord(token[1]) - ord("A")
        return 2 + ord(token) % 2000

    def decode(self, ids: list[int]) -> str:
        return "".join(chr((i - 2)) if i < LETTER_BASE else f" {chr(ord('A') + i - LETTER_BASE)}" for i in ids)


class SplitLetterTokenizer(CharTokenizer):
    """A tokenizer whose option letters are NOT single tokens (a space and a letter)."""

    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
        return [2 + ord(c) % 2000 for c in text]


class FakeLM(nn.Module):
    def __init__(self, dim: int = 16, seed: int = 0) -> None:
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.emb = nn.Embedding(VOCAB, dim)
        self.head = nn.Linear(dim, VOCAB, bias=False)
        with torch.no_grad():
            self.emb.weight.copy_(torch.randn(VOCAB, dim, generator=g) * 0.5)
            self.head.weight.copy_(torch.randn(VOCAB, dim, generator=g) * 0.1)
        self.scale = 1.0  # a test can make the model over- or under-confident
        self.calls = 0

    def forward(self, input_ids, attention_mask=None, past_key_values=None, use_cache=False,
                logits_to_keep=None, **_):
        self.calls += 1
        m = torch.ones_like(input_ids, dtype=torch.float32) if attention_mask is None else attention_mask.float()
        if past_key_values is not None and attention_mask is not None and attention_mask.shape[1] != input_ids.shape[1]:
            m = m[:, -input_ids.shape[1]:]
        m = m.unsqueeze(-1)
        e = self.emb(input_ids) * m
        b, _, d = e.shape
        prev_sum, prev_n = past_key_values if past_key_values is not None else (e.new_zeros(b, d), e.new_zeros(b, 1))
        cum = prev_sum.unsqueeze(1) + torch.cumsum(e, dim=1)
        cnt = prev_n.unsqueeze(1) + torch.cumsum(m, dim=1)
        logits = self.head(cum / cnt.clamp(min=1.0)) * self.scale
        cache = (cum[:, -1], cnt[:, -1]) if use_cache else None
        return SimpleNamespace(logits=logits, past_key_values=cache)


TOPIC = {"type": "choice", "instructions": "What is this about?", "options": ["billing", "bug", "other"]}
URGENT = {"type": "noul", "statement": "The customer needs a reply today."}
TONE = {"type": "score", "instructions": "How upset is the customer?", "levels": ["calm", "annoyed", "angry"]}


def write_dataset(path: Path, n: int = 90, seed: int = 0) -> list[dict]:
    rng = random.Random(seed)
    words = {"billing": "refund charged invoice payment", "bug": "crash error broken login",
             "other": "hello thanks question hours"}
    rows = []
    for i in range(n):
        topic = rng.choice(list(words))
        rows.append({"state": f"{' '.join(rng.sample(words[topic].split(), 2))} ticket {i}",
                     "questions": {"topic": TOPIC, "urgent": URGENT, "tone": TONE},
                     "answers": {"topic": topic, "urgent": topic == "bug", "tone": rng.choice(["calm", "annoyed", "angry"])}})
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    return rows
