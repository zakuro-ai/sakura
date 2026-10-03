"""GPT-2 (nanoGPT) pure-torch workload (perf tier).

A self-contained nanoGPT-124M implementation — no huggingface, no tiktoken.
Used by the GPU ablation harness to prove amp / compile / activation_checkpoint
on a realistic causal-LM workload.

The model wrapper's forward(idx) returns logits shaped (B, V, T) so the
harness's hardcoded ``F.cross_entropy(logits, y)`` (with y shaped (B, T) int64)
computes per-token causal-LM cross-entropy directly — no reshape in the loop.

Loaders yield (x, y) where both are (B, T) int64 and ``y`` is ``x`` shifted by
one token (contiguous-stream next-token sampling). Synthetic data is the P0
default; an mmap uint16-corpus path (nanoGPT-style random-offset get_batch) is
provided for the P3 WikiText-2 time-to-loss run.
"""
from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import Any, cast

import torch
import torch.nn as nn
import torch.nn.functional as F

from sakura.bench.harness import Workload


# --------------------------------------------------------------------------
# Model — nanoGPT.
# --------------------------------------------------------------------------
@dataclass
class GPTConfig:
    block_size: int = 512
    vocab_size: int = 50304  # GPT-2 50257 padded up to a multiple of 64
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    bias: bool = True


class LayerNorm(nn.Module):
    """LayerNorm with an optional bias (nanoGPT ships its own for parity)."""

    def __init__(self, ndim: int, bias: bool):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.layer_norm(x, self.weight.shape, self.weight, self.bias, 1e-5)


class CausalSelfAttention(nn.Module):
    def __init__(self, config: "GPTConfig"):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.size()
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        # Turing (cap 7.5) has no flash kernel; SDPA falls back to the
        # memory-efficient backend. is_causal=True applies the causal mask.
        y = F.scaled_dot_product_attention(
            q, k, v, attn_mask=None,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=True,
        )
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        out: torch.Tensor = self.c_proj(y)
        return out


class MLP(nn.Module):
    def __init__(self, config: "GPTConfig"):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out: torch.Tensor = self.c_proj(self.gelu(self.c_fc(x)))
        return out


class Block(nn.Module):
    def __init__(self, config: "GPTConfig"):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class GPT(nn.Module):
    """nanoGPT decoder. forward(idx: (B, T)) -> logits (B, T, V)."""

    def __init__(self, config: "GPTConfig"):
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict(dict(
            wte=nn.Embedding(config.vocab_size, config.n_embd),
            wpe=nn.Embedding(config.block_size, config.n_embd),
            drop=nn.Dropout(config.dropout),
            h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f=LayerNorm(config.n_embd, bias=config.bias),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        # Weight tying: token embedding and output projection share one matrix
        # (nanoGPT / "Using the Output Embedding to Improve Language Models").
        cast(nn.Embedding, self.transformer.wte).weight = self.lm_head.weight
        self.apply(self._init_weights)
        # Scaled init for residual projections (GPT-2 paper, section 2.3).
        for pn, p in self.named_parameters():
            if pn.endswith("c_proj.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        b, t = idx.size()
        assert t <= self.config.block_size, \
            f"sequence length {t} exceeds block_size {self.config.block_size}"
        pos = torch.arange(0, t, dtype=torch.long, device=idx.device)
        wte = cast(nn.Embedding, self.transformer.wte)
        wpe = cast(nn.Embedding, self.transformer.wpe)
        drop = cast(nn.Dropout, self.transformer.drop)
        blocks = cast(nn.ModuleList, self.transformer.h)
        ln_f = cast(nn.Module, self.transformer.ln_f)
        tok_emb = wte(idx)                          # (B, T, n_embd)
        pos_emb = wpe(pos)                          # (T, n_embd)
        x = drop(tok_emb + pos_emb)
        for block in blocks:
            x = block(x)
        x = ln_f(x)
        logits: torch.Tensor = self.lm_head(x)      # (B, T, V)
        return logits


class GPTLMWrapper(nn.Module):
    """Adapts GPT's (B, T, V) logits to the harness's (B, V, T) contract so
    ``F.cross_entropy(model(x), y)`` (y is (B, T) int64) is per-token causal CE."""

    def __init__(self, gpt: "GPT"):
        super().__init__()
        self.gpt = gpt

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        logits: torch.Tensor = self.gpt(idx)  # (B, T, V)
        return logits.transpose(1, 2)          # (B, V, T)


# The transformer block classes activation_checkpoint wraps. Surfaced at module
# level and via Workload.block_types so the CLI factory can pass real targets.
BLOCK_TYPES = (Block,)

# Named model presets. block_size is set from make_workload's seq_len.
GPT_SIZES: dict[str, dict[str, Any]] = {
    "124m": dict(n_layer=12, n_head=12, n_embd=768, vocab_size=50304),
    "tiny": dict(n_layer=2, n_head=2, n_embd=64, vocab_size=128),
}


# --------------------------------------------------------------------------
# Data.
# --------------------------------------------------------------------------
class _StreamDataset(torch.utils.data.Dataset[Any]):
    """Contiguous-stream next-token windows over a 1-D token tensor.

    Sample i is the window ``data[i*T : i*T + T + 1]`` split into
    x = window[:-1], y = window[1:] — both length T, int64.
    """

    def __init__(self, data: torch.Tensor, block_size: int, n_samples: int):
        self.data = data
        self.block_size = block_size
        self.n_samples = n_samples

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, i: int) -> tuple[torch.Tensor, torch.Tensor]:
        start = i * self.block_size
        chunk = self.data[start:start + self.block_size + 1]
        return chunk[:-1].to(torch.int64), chunk[1:].to(torch.int64)


def _make_synthetic_loaders(
    vocab_size: int, seq_len: int, batch_size: int,
    n_train_batches: int, n_val_batches: int,
) -> tuple[Any, Any]:
    """Deterministic synthetic token streams. Same seed every build, shuffle off,
    so baseline and sakura rows see an identical sample sequence (equal work)."""
    g = torch.Generator().manual_seed(1337)

    def _stream(n_batches: int) -> Any:
        n_samples = n_batches * batch_size
        n_tokens = n_samples * seq_len + 1
        data = torch.randint(0, vocab_size, (n_tokens,), generator=g, dtype=torch.int64)
        return torch.utils.data.DataLoader(
            _StreamDataset(data, seq_len, n_samples),
            batch_size=batch_size, shuffle=False, drop_last=True,
        )

    return _stream(n_train_batches), _stream(n_val_batches)


def _make_mmap_loaders(
    data_dir: str | None, seq_len: int, batch_size: int,
    n_train_batches: int, n_val_batches: int,
) -> tuple[Any, Any]:
    """nanoGPT-style random-offset batches over uint16 ``train.bin``/``val.bin``.

    The corpus is GPT-2-BPE-tokenized offline (no runtime tokenizer). Batches
    are materialized once into lists so the iterable is fixed and reproducible.
    """
    import numpy as np

    if not data_dir:
        raise ValueError("make_workload(synthetic=False) requires data_dir=<dir with train.bin/val.bin>")

    def _split(split: str, n_batches: int, seed: int) -> list[tuple[torch.Tensor, torch.Tensor]]:
        path = os.path.join(data_dir, f"{split}.bin")
        if not os.path.exists(path):
            raise FileNotFoundError(f"corpus file not found: {path}")
        arr = np.memmap(path, dtype=np.uint16, mode="r")
        g = torch.Generator().manual_seed(seed)
        hi = len(arr) - seq_len - 1
        if hi <= 0:
            raise ValueError(f"corpus {path} too short for seq_len={seq_len}")
        batches: list[tuple[torch.Tensor, torch.Tensor]] = []
        for _ in range(n_batches):
            ix = torch.randint(hi, (batch_size,), generator=g)
            x = torch.stack([
                torch.from_numpy(arr[int(i):int(i) + seq_len].astype(np.int64)) for i in ix
            ])
            y = torch.stack([
                torch.from_numpy(arr[int(i) + 1:int(i) + 1 + seq_len].astype(np.int64)) for i in ix
            ])
            batches.append((x, y))
        return batches

    return _split("train", n_train_batches, 1337), _split("val", n_val_batches, 2337)


def _eval_fn(model: Any, loader: Any) -> dict[str, float]:
    """Mean per-token val loss + perplexity. loader yields (x, y) (B, T) int64."""
    model.eval()
    device = next(model.parameters()).device
    loss_sum = 0.0
    n_tokens = 0
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            logits = model(x)  # (B, V, T)
            loss = F.cross_entropy(logits, y, reduction="sum")
            loss_sum += float(loss)
            n_tokens += int(y.numel())
    val_loss = loss_sum / max(n_tokens, 1)
    return {"val_loss": val_loss, "perplexity": math.exp(min(val_loss, 80.0))}


_UNSET: Any = object()


def make_workload(
    *, size: str = "124m", seq_len: Any = _UNSET, batch_size: Any = _UNSET,
    epochs: Any = _UNSET, synthetic: bool = True, data_dir: str | None = None,
    n_train_batches: Any = _UNSET, n_val_batches: Any = _UNSET,
    metric_target: Any = None,
) -> Workload:
    """Build the GPT-2 causal-LM workload.

    size: a key into GPT_SIZES ("124m" headline, "tiny" for CI/tests).
    seq_len: tokens per sample (= block_size); sets tokens_per_sample.
    synthetic: True (default) uses random token streams; False reads
    ``data_dir``/train.bin,val.bin (uint16 GPT-2 BPE, nanoGPT mmap batches).

    Sizing precedence (per the cross-task contract): an explicitly-passed
    kwarg wins; else the environment variable; else the documented default.
    The CLI registry calls ``make_workload()`` arg-less, so the env vars are
    how Task 7 sizes a run:
      SAKURA_GPT2_SEQ_LEN (512), SAKURA_GPT2_BATCH_SIZE (8),
      SAKURA_GPT2_EPOCHS (1), SAKURA_GPT2_N_TRAIN_BATCHES (64),
      SAKURA_GPT2_N_VAL_BATCHES (8). The _UNSET sentinel distinguishes
      "caller passed nothing" from "caller passed a falsy/zero value".
    """
    if seq_len is _UNSET:
        seq_len = int(os.environ.get("SAKURA_GPT2_SEQ_LEN", 512))
    if batch_size is _UNSET:
        batch_size = int(os.environ.get("SAKURA_GPT2_BATCH_SIZE", 8))
    if epochs is _UNSET:
        epochs = int(os.environ.get("SAKURA_GPT2_EPOCHS", 1))
    if n_train_batches is _UNSET:
        n_train_batches = int(os.environ.get("SAKURA_GPT2_N_TRAIN_BATCHES", 64))
    if n_val_batches is _UNSET:
        n_val_batches = int(os.environ.get("SAKURA_GPT2_N_VAL_BATCHES", 8))

    if size not in GPT_SIZES:
        raise ValueError(f"unknown gpt2 size {size!r}; available: {sorted(GPT_SIZES)}")
    config = GPTConfig(block_size=seq_len, **GPT_SIZES[size])

    def make_model() -> GPTLMWrapper:
        return GPTLMWrapper(GPT(config))

    if synthetic:
        train_loader, val_loader = _make_synthetic_loaders(
            config.vocab_size, seq_len, batch_size, n_train_batches, n_val_batches,
        )
    else:
        train_loader, val_loader = _make_mmap_loaders(
            data_dir, seq_len, batch_size, n_train_batches, n_val_batches,
        )

    return Workload(
        name=f"gpt2-{size}",
        tier="perf",
        make_model=make_model,
        make_train_loader=lambda: train_loader,
        make_val_loader=lambda: val_loader,
        eval_fn=_eval_fn,
        epochs=epochs,
        metric_target=metric_target,
        tokens_per_sample=seq_len,
        block_types=BLOCK_TYPES,
    )


__all__ = [
    "GPTConfig", "LayerNorm", "CausalSelfAttention", "MLP", "Block", "GPT",
    "GPTLMWrapper", "BLOCK_TYPES", "GPT_SIZES", "make_workload",
]
