"""Task 3 — measurement honesty (CUDA sync + reset-peak) + tokens/sec wiring.

CPU-runnable: the CUDA helpers are gated on _cuda_available(), and the
runner-wiring tests monkeypatch the gated helpers to plain counters, so no
GPU is required. The tiny GPT-2-shaped workload mirrors the (B, V, T) logits
/ (B, T) int64-target contract of sakura/bench/workloads/gpt2.py (Task 2) but
uses a 2-param toy model so the tokens-math assertions run in well under a
second on CPU.
"""
from __future__ import annotations

import math

import pytest

torch = pytest.importorskip("torch")

import sakura.bench.harness as harness


def test_cuda_helpers_call_torch_under_fake_cuda(monkeypatch):
    """Under a faked _cuda_available()=True, the helpers call through to
    torch.cuda.{synchronize,reset_peak_memory_stats} exactly once each."""
    monkeypatch.setattr(harness, "_cuda_available", lambda: True)
    counts = {"sync": 0, "reset": 0}
    monkeypatch.setattr(
        torch.cuda, "synchronize",
        lambda *a, **k: counts.__setitem__("sync", counts["sync"] + 1),
    )
    monkeypatch.setattr(
        torch.cuda, "reset_peak_memory_stats",
        lambda *a, **k: counts.__setitem__("reset", counts["reset"] + 1),
    )
    harness._cuda_reset_peak()
    harness._cuda_sync()
    assert counts == {"sync": 1, "reset": 1}


def test_cuda_helpers_are_noop_without_cuda(monkeypatch):
    """When CUDA is unavailable the helpers must not touch torch.cuda at all."""
    monkeypatch.setattr(harness, "_cuda_available", lambda: False)

    def _boom(*a, **k):
        raise AssertionError("torch.cuda must not be called without CUDA")

    monkeypatch.setattr(torch.cuda, "synchronize", _boom)
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", _boom)
    harness._cuda_reset_peak()  # no-op
    harness._cuda_sync()        # no-op


from sakura.bench.harness import BaselineRunner, Workload


class _TinyGPT(torch.nn.Module):
    """Toy causal-LM mirroring the gpt2.py contract: forward(idx:(B,T)) returns
    logits (B, V, T) so the harness's hardcoded F.cross_entropy(logits, y) with
    y:(B,T) int64 computes per-token CE directly."""

    def __init__(self, vocab: int, dim: int = 16):
        super().__init__()
        self.tok = torch.nn.Embedding(vocab, dim)
        self.head = torch.nn.Linear(dim, vocab)

    def forward(self, idx):
        h = self.tok(idx)               # (B, T, dim)
        logits = self.head(h)           # (B, T, V)
        return logits.transpose(1, 2)   # (B, V, T)


def _tiny_gpt2_workload(*, vocab=32, seq_len=8, batch_size=4, n_batches=3, epochs=1):
    def make_model():
        torch.manual_seed(0)
        return _TinyGPT(vocab)

    def make_loader():
        torch.manual_seed(0)
        n = n_batches * batch_size
        xs = torch.randint(0, vocab, (n, seq_len))
        ys = torch.randint(0, vocab, (n, seq_len))
        ds = torch.utils.data.TensorDataset(xs, ys)
        return torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=False)

    def eval_fn(model, loader):
        model.eval()
        total = 0.0
        n = 0
        with torch.no_grad():
            for x, y in loader:
                total += float(torch.nn.functional.cross_entropy(model(x), y))
                n += 1
        val_loss = total / max(n, 1)
        return {"val_loss": val_loss, "perplexity": math.exp(val_loss)}

    return Workload(
        name="gpt2-tiny-test",
        tier="smoke",
        make_model=make_model,
        make_train_loader=make_loader,
        make_val_loader=make_loader,
        eval_fn=eval_fn,
        epochs=epochs,
        tokens_per_sample=seq_len,  # Task 1 field — enables tokens/sec
    )


def test_baseline_runner_tokens_per_sec_and_total_tokens():
    seq_len, batch_size, n_batches, epochs = 8, 4, 3, 1
    wl = _tiny_gpt2_workload(seq_len=seq_len, batch_size=batch_size,
                             n_batches=n_batches, epochs=epochs)
    report = BaselineRunner(framework="pytorch-ddp").run(wl)
    expected_samples = n_batches * batch_size * epochs   # 12
    assert report.total_tokens == expected_samples * seq_len   # 12 * 8 == 96
    assert report.tokens_per_sec > 0
    assert report.elapsed_secs > 0


def test_baseline_runner_resets_peak_then_syncs(monkeypatch):
    calls = {"reset": 0, "sync": 0}
    monkeypatch.setattr(harness, "_cuda_reset_peak",
                        lambda: calls.__setitem__("reset", calls["reset"] + 1))
    monkeypatch.setattr(harness, "_cuda_sync",
                        lambda: calls.__setitem__("sync", calls["sync"] + 1))
    # warmup_steps=0 so we get the canonical pre-device-move reset and end-of-loop
    # sync without the extra post-warmup calls (those are tested in test_harness_warmup).
    BaselineRunner(framework="pytorch-ddp", warmup_steps=0).run(_tiny_gpt2_workload())
    assert calls == {"reset": 1, "sync": 1}


from sakura.bench.harness import SakuraRunner


def test_sakura_runner_tokens_per_sec_and_total_tokens():
    from sakura.services.telemetry import Telemetry
    seq_len, batch_size, n_batches = 8, 4, 3
    wl = _tiny_gpt2_workload(seq_len=seq_len, batch_size=batch_size, n_batches=n_batches)
    runner = SakuraRunner(framework="pytorch-ddp",
                          services=[Telemetry(output=lambda _r: None)])
    report = runner.run(wl)
    assert report.total_tokens == n_batches * batch_size * seq_len   # 96
    assert report.tokens_per_sec > 0


def test_sakura_runner_resets_peak_then_syncs(monkeypatch):
    from sakura.services.telemetry import Telemetry
    calls = {"reset": 0, "sync": 0}
    monkeypatch.setattr(harness, "_cuda_reset_peak",
                        lambda: calls.__setitem__("reset", calls["reset"] + 1))
    monkeypatch.setattr(harness, "_cuda_sync",
                        lambda: calls.__setitem__("sync", calls["sync"] + 1))
    # warmup_steps=0 so we get the canonical pre-device-move reset and end-of-loop
    # sync without the extra post-warmup calls (those are tested in test_harness_warmup).
    runner = SakuraRunner(framework="pytorch-ddp",
                          services=[Telemetry(output=lambda _r: None)],
                          warmup_steps=0)
    runner.run(_tiny_gpt2_workload())
    assert calls == {"reset": 1, "sync": 1}
