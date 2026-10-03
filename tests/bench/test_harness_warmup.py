"""Tests for warmup_steps in BaselineRunner / SakuraRunner.

CPU-only: no torch.cuda or GPU required.  The warmup logic is conditioned on
``_cuda_available()`` only for the sync/reset calls — the training loop itself
runs on CPU without any special handling.

Three assertion groups:
  (a) Token / sample accounting: warmup steps are excluded from total_tokens.
  (b) Forward-call counting: warmup_steps=W adds exactly W extra forward()
      calls before the timed region; warmup_steps=0 adds none.
  (c) _cuda_reset_peak call count: called twice when warmup_steps>0 (once
      before model.to(device), once after warmup), once when warmup_steps=0.
"""
from __future__ import annotations

import math

import pytest

torch = pytest.importorskip("torch")

import sakura.bench.harness as harness
from sakura.bench.harness import BaselineRunner, Workload
from sakura.service import BaseService


# ──────────────────────────────────────────────────────────────────────────────
# Tiny workloads
# ──────────────────────────────────────────────────────────────────────────────

class _ForwardCounter(torch.nn.Module):
    """Tiny linear model that counts how many times forward() is called.

    A single shared instance is returned by make_model() so that both the
    warmup path and the timed path increment the same counter.
    """

    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(4, 2)
        self.forward_count = 0

    def forward(self, x):
        self.forward_count += 1
        return self.linear(x)


def _make_counting_workload(n_batches=3, batch_size=4, epochs=1):
    """Return (workload, counter) where counter.forward_count tracks every
    forward() call (warmup + train + eval) on the shared model instance."""
    counter = _ForwardCounter()

    def make_model():
        return counter  # same instance; harness never replaces it

    def make_loader():
        torch.manual_seed(0)
        n = n_batches * batch_size
        xs = torch.randn(n, 4)
        ys = torch.randint(0, 2, (n,))
        ds = torch.utils.data.TensorDataset(xs, ys)
        return torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=False)

    def eval_fn(model, loader):
        with torch.no_grad():
            for x, _y in loader:
                model(x)  # counted here too
        return {"val_acc": 0.5}

    wl = Workload(
        name="counting-workload",
        tier="smoke",
        make_model=make_model,
        make_train_loader=make_loader,
        make_val_loader=make_loader,
        eval_fn=eval_fn,
        epochs=epochs,
    )
    return wl, counter


def _make_token_workload(seq_len=8, batch_size=4, n_batches=3, epochs=1):
    """Tiny GPT-shaped workload for total_tokens accounting assertions."""
    vocab = 32

    class _TinyGPT(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.tok = torch.nn.Embedding(vocab, 16)
            self.head = torch.nn.Linear(16, vocab)

        def forward(self, idx):
            h = self.tok(idx)
            logits = self.head(h)
            return logits.transpose(1, 2)  # (B, V, T)

    def make_model():
        torch.manual_seed(0)
        return _TinyGPT()

    def make_loader():
        torch.manual_seed(0)
        n = n_batches * batch_size
        xs = torch.randint(0, vocab, (n, seq_len))
        ys = torch.randint(0, vocab, (n, seq_len))
        ds = torch.utils.data.TensorDataset(xs, ys)
        return torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=False)

    def eval_fn(model, loader):
        model.eval()
        loss_sum, n = 0.0, 0
        with torch.no_grad():
            for x, y in loader:
                loss_sum += float(torch.nn.functional.cross_entropy(model(x), y))
                n += 1
        val_loss = loss_sum / max(n, 1)
        return {"val_loss": val_loss, "perplexity": math.exp(val_loss)}

    return Workload(
        name="gpt2-warmup-test",
        tier="smoke",
        make_model=make_model,
        make_train_loader=make_loader,
        make_val_loader=make_loader,
        eval_fn=eval_fn,
        epochs=epochs,
        tokens_per_sample=seq_len,
    )


# ──────────────────────────────────────────────────────────────────────────────
# (a) total_tokens accounting: warmup excluded
# ──────────────────────────────────────────────────────────────────────────────

def test_total_tokens_excludes_warmup_steps():
    """With warmup_steps>0, total_tokens == seq×batch×batches×epochs (no warmup inflation)."""
    seq_len, batch_size, n_batches, epochs = 8, 4, 3, 2
    wl = _make_token_workload(seq_len=seq_len, batch_size=batch_size,
                               n_batches=n_batches, epochs=epochs)
    runner = BaselineRunner(framework="pytorch-ddp", warmup_steps=3)
    report = runner.run(wl)
    expected = seq_len * batch_size * n_batches * epochs
    assert report.total_tokens == expected, (
        f"total_tokens {report.total_tokens} != expected {expected} — "
        "warmup steps must not be counted in total_tokens"
    )


def test_total_tokens_zero_warmup_unchanged():
    """warmup_steps=0 gives identical token accounting to old behaviour."""
    seq_len, batch_size, n_batches, epochs = 8, 4, 3, 1
    wl = _make_token_workload(seq_len=seq_len, batch_size=batch_size,
                               n_batches=n_batches, epochs=epochs)
    runner = BaselineRunner(framework="pytorch-ddp", warmup_steps=0)
    report = runner.run(wl)
    expected = seq_len * batch_size * n_batches * epochs
    assert report.total_tokens == expected


def test_samples_per_sec_positive_with_warmup():
    """samples_per_sec is computed from the timed region only and must be > 0."""
    wl = _make_token_workload()
    runner = BaselineRunner(framework="pytorch-ddp", warmup_steps=2)
    report = runner.run(wl)
    assert report.samples_per_sec > 0
    assert report.elapsed_secs > 0


# ──────────────────────────────────────────────────────────────────────────────
# (b) forward-call counting: warmup adds exactly W extra calls
# ──────────────────────────────────────────────────────────────────────────────

def test_warmup_steps_adds_extra_forward_calls():
    """With warmup_steps=W, forward() is called W more times than with warmup_steps=0."""
    n_batches, batch_size, epochs, warmup_steps = 3, 4, 1, 2
    wl, counter = _make_counting_workload(n_batches=n_batches,
                                           batch_size=batch_size, epochs=epochs)

    runner_w = BaselineRunner(framework="pytorch-ddp", warmup_steps=warmup_steps)
    runner_w.run(wl)
    count_with = counter.forward_count

    # Reset counter and run without warmup on the same model instance.
    counter.forward_count = 0
    runner_0 = BaselineRunner(framework="pytorch-ddp", warmup_steps=0)
    runner_0.run(wl)
    count_without = counter.forward_count

    assert count_with == count_without + warmup_steps, (
        f"expected {count_without + warmup_steps} total forwards with warmup_steps={warmup_steps}, "
        f"got {count_with} (without-warmup baseline was {count_without})"
    )


def test_warmup_steps_zero_no_extra_forwards():
    """warmup_steps=0 does not add any extra forward() calls (old behaviour)."""
    n_batches, batch_size, epochs = 3, 4, 1
    wl, counter = _make_counting_workload(n_batches=n_batches,
                                           batch_size=batch_size, epochs=epochs)
    BaselineRunner(framework="pytorch-ddp", warmup_steps=0).run(wl)
    # Training: n_batches×epochs forwards; eval: n_batches×epochs forwards (val_loader has same size).
    expected = (n_batches * epochs) + (n_batches * epochs)
    assert counter.forward_count == expected, (
        f"warmup_steps=0 should produce {expected} forward calls, got {counter.forward_count}"
    )


def test_warmup_accounting_matches_between_runs():
    """total_tokens is identical whether warmup_steps is 0 or >0."""
    seq_len, batch_size, n_batches, epochs = 8, 4, 3, 1
    wl0 = _make_token_workload(seq_len=seq_len, batch_size=batch_size,
                                n_batches=n_batches, epochs=epochs)
    wl2 = _make_token_workload(seq_len=seq_len, batch_size=batch_size,
                                n_batches=n_batches, epochs=epochs)
    r0 = BaselineRunner(framework="pytorch-ddp", warmup_steps=0).run(wl0)
    r2 = BaselineRunner(framework="pytorch-ddp", warmup_steps=2).run(wl2)
    assert r0.total_tokens == r2.total_tokens


# ──────────────────────────────────────────────────────────────────────────────
# (c) _cuda_reset_peak call count: 1 without warmup, 2 with warmup
# ──────────────────────────────────────────────────────────────────────────────

def test_cuda_reset_called_twice_with_warmup(monkeypatch):
    """With warmup_steps>0, _cuda_reset_peak is called twice: pre-device and post-warmup."""
    reset_count = {"n": 0}
    monkeypatch.setattr(harness, "_cuda_reset_peak",
                        lambda: reset_count.__setitem__("n", reset_count["n"] + 1))
    monkeypatch.setattr(harness, "_cuda_sync", lambda: None)

    BaselineRunner(framework="pytorch-ddp", warmup_steps=2).run(_make_token_workload())
    assert reset_count["n"] == 2, (
        f"expected 2 _cuda_reset_peak calls (pre-device-move + post-warmup), "
        f"got {reset_count['n']}"
    )


def test_cuda_reset_called_once_without_warmup(monkeypatch):
    """With warmup_steps=0, _cuda_reset_peak is called once (pre-device-move only)."""
    reset_count = {"n": 0}
    monkeypatch.setattr(harness, "_cuda_reset_peak",
                        lambda: reset_count.__setitem__("n", reset_count["n"] + 1))
    monkeypatch.setattr(harness, "_cuda_sync", lambda: None)

    BaselineRunner(framework="pytorch-ddp", warmup_steps=0).run(_make_token_workload())
    assert reset_count["n"] == 1, (
        f"expected 1 _cuda_reset_peak call with warmup_steps=0, got {reset_count['n']}"
    )


def test_cuda_sync_called_twice_with_warmup(monkeypatch):
    """With warmup_steps>0, _cuda_sync is called twice: post-warmup and end-of-timed-loop."""
    sync_count = {"n": 0}
    monkeypatch.setattr(harness, "_cuda_sync",
                        lambda: sync_count.__setitem__("n", sync_count["n"] + 1))
    monkeypatch.setattr(harness, "_cuda_reset_peak", lambda: None)

    BaselineRunner(framework="pytorch-ddp", warmup_steps=2).run(_make_token_workload())
    assert sync_count["n"] == 2, (
        f"expected 2 _cuda_sync calls with warmup_steps=2, got {sync_count['n']}"
    )


# ──────────────────────────────────────────────────────────────────────────────
# env override: SAKURA_BENCH_WARMUP_STEPS
# ──────────────────────────────────────────────────────────────────────────────

def test_env_var_sets_warmup_steps(monkeypatch):
    """SAKURA_BENCH_WARMUP_STEPS env var controls warmup_steps when kwarg is omitted."""
    monkeypatch.setenv("SAKURA_BENCH_WARMUP_STEPS", "0")
    runner = BaselineRunner(framework="pytorch-ddp")
    assert runner.warmup_steps == 0


def test_env_var_nonzero(monkeypatch):
    """SAKURA_BENCH_WARMUP_STEPS=5 is parsed correctly."""
    monkeypatch.setenv("SAKURA_BENCH_WARMUP_STEPS", "5")
    runner = BaselineRunner(framework="pytorch-ddp")
    assert runner.warmup_steps == 5


def test_kwarg_wins_over_env(monkeypatch):
    """Explicit warmup_steps kwarg overrides SAKURA_BENCH_WARMUP_STEPS env var."""
    monkeypatch.setenv("SAKURA_BENCH_WARMUP_STEPS", "99")
    runner = BaselineRunner(framework="pytorch-ddp", warmup_steps=1)
    assert runner.warmup_steps == 1


def test_default_warmup_steps_is_2(monkeypatch):
    """Default warmup_steps is 2 when neither kwarg nor env var is set."""
    monkeypatch.delenv("SAKURA_BENCH_WARMUP_STEPS", raising=False)
    runner = BaselineRunner(framework="pytorch-ddp")
    assert runner.warmup_steps == 2


# ──────────────────────────────────────────────────────────────────────────────
# SakuraRunner inherits warmup_steps
# ──────────────────────────────────────────────────────────────────────────────

def test_sakura_runner_inherits_warmup_steps():
    """SakuraRunner passes warmup_steps through to BaselineRunner.__init__."""
    from sakura.bench.harness import SakuraRunner
    runner = SakuraRunner(framework="pytorch-ddp", warmup_steps=7)
    assert runner.warmup_steps == 7


def test_sakura_runner_default_warmup_steps(monkeypatch):
    """SakuraRunner defaults to the same env-driven warmup_steps as BaselineRunner."""
    from sakura.bench.harness import SakuraRunner
    monkeypatch.delenv("SAKURA_BENCH_WARMUP_STEPS", raising=False)
    runner = SakuraRunner(framework="pytorch-ddp")
    assert runner.warmup_steps == 2


# ──────────────────────────────────────────────────────────────────────────────
# SakuraRunner warmup routes through the SERVICE path (critical invariant)
# ──────────────────────────────────────────────────────────────────────────────

class _LossSpyService(BaseService):
    """Counts every rt.scale_loss()/rt.optimizer_step() dispatch.

    Installed on the SakuraRuntime, its wrap_loss/optimizer_step are invoked by
    the runtime coordinators (SakuraRuntime.scale_loss / .optimizer_step). The
    warmup loop in _run_raw_pytorch_with_adapter calls those same coordinators,
    so a non-zero warmup count proves warmup exercises the real service path
    (AMP/compile-wrapped), not a stripped plain-pytorch path.
    """
    name = "loss-spy"
    priority = 100  # run after real services; never claims the step

    def __init__(self):
        super().__init__()
        self.wrap_loss_calls = 0
        self.optimizer_step_calls = 0

    def wrap_loss(self, loss):
        self.wrap_loss_calls += 1
        return loss

    def optimizer_step(self, optimizer) -> bool:
        self.optimizer_step_calls += 1
        return False  # don't claim the step; let the loop call opt.step()


def test_sakura_warmup_routes_through_service_path():
    """warmup steps invoke rt.scale_loss (service wrap_loss) — total == warmup + timed steps."""
    from sakura.bench.harness import SakuraRunner
    n_batches, batch_size, epochs, warmup_steps = 3, 4, 1, 2
    wl = _make_token_workload(batch_size=batch_size, n_batches=n_batches, epochs=epochs)

    spy = _LossSpyService()
    runner = SakuraRunner(framework="pytorch-ddp", services=[spy],
                          warmup_steps=warmup_steps)
    runner.run(wl)

    timed_steps = n_batches * epochs
    assert spy.wrap_loss_calls == warmup_steps + timed_steps, (
        f"expected {warmup_steps} warmup + {timed_steps} timed = "
        f"{warmup_steps + timed_steps} wrap_loss dispatches, got {spy.wrap_loss_calls}"
    )
    assert spy.optimizer_step_calls == warmup_steps + timed_steps, (
        f"expected {warmup_steps + timed_steps} optimizer_step dispatches, "
        f"got {spy.optimizer_step_calls}"
    )


def test_sakura_warmup_delta_equals_warmup_steps():
    """The service path runs exactly warmup_steps more times with warmup vs without."""
    from sakura.bench.harness import SakuraRunner
    n_batches, batch_size, epochs, warmup_steps = 3, 4, 1, 2

    spy0 = _LossSpyService()
    SakuraRunner(framework="pytorch-ddp", services=[spy0],
                 warmup_steps=0).run(
        _make_token_workload(batch_size=batch_size, n_batches=n_batches, epochs=epochs))

    spyW = _LossSpyService()
    SakuraRunner(framework="pytorch-ddp", services=[spyW],
                 warmup_steps=warmup_steps).run(
        _make_token_workload(batch_size=batch_size, n_batches=n_batches, epochs=epochs))

    assert spyW.wrap_loss_calls - spy0.wrap_loss_calls == warmup_steps, (
        f"warmup should add exactly {warmup_steps} service-path dispatches "
        f"(got {spyW.wrap_loss_calls} vs {spy0.wrap_loss_calls})"
    )
