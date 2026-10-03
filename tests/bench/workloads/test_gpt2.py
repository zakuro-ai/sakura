"""GPT-2 workload tests — shape contract, CE path, factory fields, train step."""
from __future__ import annotations

import math
import os

import pytest

torch = pytest.importorskip("torch")

import torch.nn.functional as F

from sakura.bench.harness import Workload
from sakura.bench.workloads.gpt2 import (
    GPT, GPTConfig, GPTLMWrapper, Block, BLOCK_TYPES, make_workload,
)


def _tiny_config(seq_len=16):
    return GPTConfig(block_size=seq_len, vocab_size=128, n_layer=2, n_head=2, n_embd=64)


def _tiny_model(seq_len=16):
    torch.manual_seed(0)
    return GPTLMWrapper(GPT(_tiny_config(seq_len)))


def test_forward_returns_b_v_t():
    """Wrapper forward(idx) must return (B, V, T) for the harness CE contract."""
    seq_len, vocab, B = 16, 128, 4
    model = _tiny_model(seq_len)
    x = torch.randint(0, vocab, (B, seq_len))
    logits = model(x)
    assert logits.shape == (B, vocab, seq_len)


def test_cross_entropy_runs_and_is_finite():
    """F.cross_entropy(model(x), y) — y (B, T) int64 — is a finite scalar."""
    seq_len, vocab, B = 16, 128, 4
    model = _tiny_model(seq_len)
    x = torch.randint(0, vocab, (B, seq_len))
    y = torch.randint(0, vocab, (B, seq_len), dtype=torch.int64)
    loss = F.cross_entropy(model(x), y)
    assert loss.ndim == 0
    loss_val = loss.item()
    assert math.isfinite(loss_val)
    # Random init over a uniform vocab -> loss near ln(vocab).
    assert 0.0 < loss_val < math.log(vocab) + 1.0


def test_one_train_step_decreases_loss_cpu():
    """A single optimizer step lowers the loss on the same batch (CPU, tiny)."""
    seq_len, vocab, B = 16, 128, 4
    model = _tiny_model(seq_len)
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    x = torch.randint(0, vocab, (B, seq_len))
    y = torch.randint(0, vocab, (B, seq_len), dtype=torch.int64)

    loss0 = F.cross_entropy(model(x), y)
    opt.zero_grad()
    loss0.backward()
    opt.step()
    loss1 = F.cross_entropy(model(x), y)

    assert loss1.item() < loss0.item()


def test_make_workload_fields():
    """make_workload wires tokens_per_sample and block_types per the contract."""
    wl = make_workload(size="tiny", seq_len=16, batch_size=2,
                       n_train_batches=2, n_val_batches=1)
    assert isinstance(wl, Workload)
    assert wl.name == "gpt2-tiny"
    assert wl.tier == "perf"
    assert wl.tokens_per_sample == 16
    assert wl.block_types == (Block,)
    assert wl.block_types == BLOCK_TYPES
    # Loaders yield (x, y) both (B, T) int64 with y == x shifted by one.
    xb, yb = next(iter(wl.make_train_loader()))
    assert xb.shape == (2, 16) and yb.shape == (2, 16)
    assert xb.dtype == torch.int64 and yb.dtype == torch.int64
    assert torch.equal(xb[:, 1:], yb[:, :-1])


def test_default_workload_is_gpt2_124m(monkeypatch):
    """Default factory matches the registry contract (name + 512 seq_len)."""
    # Clear BOTH the architecture vars AND the sizing vars: tokens_per_sample
    # maps to SAKURA_GPT2_SEQ_LEN, so a stale sizing var (e.g. left over from
    # the ablation driver running earlier in the same session) would otherwise
    # make this test flaky.
    for var in ["SAKURA_GPT2_N_LAYER", "SAKURA_GPT2_N_HEAD", "SAKURA_GPT2_N_EMBD",
                "SAKURA_GPT2_BLOCK_SIZE", "SAKURA_GPT2_VOCAB_SIZE",
                "SAKURA_GPT2_SEQ_LEN", "SAKURA_GPT2_BATCH_SIZE", "SAKURA_GPT2_EPOCHS",
                "SAKURA_GPT2_N_TRAIN_BATCHES", "SAKURA_GPT2_N_VAL_BATCHES"]:
        monkeypatch.delenv(var, raising=False)
    wl = make_workload(n_train_batches=1, n_val_batches=1, batch_size=1)
    assert wl.name == "gpt2-124m"
    assert wl.tokens_per_sample == 512


def test_eval_fn_returns_val_loss_and_perplexity():
    """eval_fn(model, loader) -> {"val_loss", "perplexity"} on a tiny workload."""
    wl = make_workload(size="tiny", seq_len=16, batch_size=2,
                       n_train_batches=2, n_val_batches=1)
    out = wl.eval_fn(wl.make_model(), wl.make_val_loader())
    assert set(out) == {"val_loss", "perplexity"}
    assert math.isfinite(out["val_loss"]) and out["val_loss"] > 0.0
    # perplexity == exp(val_loss) within float tolerance.
    assert abs(out["perplexity"] - math.exp(out["val_loss"])) < 1e-3


def test_mmap_loader_shapes(tmp_path):
    """synthetic=False reads uint16 train.bin/val.bin into (B, T) int64 batches."""
    import numpy as np

    seq_len = 16
    rng = np.random.default_rng(0)
    for split in ("train", "val"):
        arr = rng.integers(0, 128, size=2048, dtype=np.uint16)
        arr.tofile(tmp_path / f"{split}.bin")

    wl = make_workload(size="tiny", seq_len=seq_len, batch_size=2, synthetic=False,
                       data_dir=str(tmp_path), n_train_batches=3, n_val_batches=1)
    batches = list(wl.make_train_loader())
    assert len(batches) == 3
    xb, yb = batches[0]
    assert xb.shape == (2, seq_len) and yb.shape == (2, seq_len)
    assert xb.dtype == torch.int64 and yb.dtype == torch.int64
    # y is x shifted by one token (next-token target).
    assert torch.equal(xb[:, 1:], yb[:, :-1])


def test_registry_resolves_gpt2_124m():
    """The CLI registry maps gpt2-124m to the factory (choices auto-update)."""
    from sakura.bench.__main__ import _WORKLOAD_REGISTRY, _resolve_workload

    assert "gpt2-124m" in _WORKLOAD_REGISTRY
    wl = _resolve_workload("gpt2-124m")
    assert wl.name == "gpt2-124m"
    assert wl.tokens_per_sample == 512
    assert wl.block_types == (Block,)


def test_env_sizing_overrides_defaults(monkeypatch):
    """Env vars SAKURA_GPT2_* override make_workload defaults when called arg-less."""
    monkeypatch.setenv("SAKURA_GPT2_SEQ_LEN", "32")
    monkeypatch.setenv("SAKURA_GPT2_BATCH_SIZE", "3")
    monkeypatch.setenv("SAKURA_GPT2_EPOCHS", "3")
    monkeypatch.setenv("SAKURA_GPT2_N_TRAIN_BATCHES", "2")
    monkeypatch.setenv("SAKURA_GPT2_N_VAL_BATCHES", "1")
    # Also override to "tiny" size via a direct kwarg so we don't instantiate 124M
    wl = make_workload(size="tiny")
    assert wl.tokens_per_sample == 32
    assert wl.epochs == 3
    xb, yb = next(iter(wl.make_train_loader()))
    assert xb.shape == (3, 32)
    assert yb.shape == (3, 32)


def test_explicit_kwarg_wins_over_env(monkeypatch):
    """An explicitly-passed kwarg must override the env var (precedence contract)."""
    monkeypatch.setenv("SAKURA_GPT2_SEQ_LEN", "32")
    monkeypatch.setenv("SAKURA_GPT2_BATCH_SIZE", "3")
    # Explicit seq_len/batch_size kwargs must win over the env vars above.
    wl = make_workload(size="tiny", seq_len=16, batch_size=2,
                       n_train_batches=1, n_val_batches=1)
    assert wl.tokens_per_sample == 16
    xb, yb = next(iter(wl.make_train_loader()))
    assert xb.shape == (2, 16)
    assert yb.shape == (2, 16)
