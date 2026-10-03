"""Benchmark: AsyncEval overlaps eval with next-epoch training.

Demonstrates the structural win sakura's `AsyncEval` provides: when the
per-epoch validation step has non-trivial cost, baseline serializes
(`train → eval → train → eval → ...`) while sakura overlaps
(`train_n` runs concurrently with `eval_{n-1}`). Total wallclock drops
because (N-1) of the N evals finish during training time that would have
happened anyway.

Run on CPU; uses ThreadDispatcher so torch's GIL-releasing C++ kernels
provide real parallelism without subprocess pickle overhead.

Usage:
    python scripts/bench_async_eval.py
    python scripts/bench_async_eval.py --epochs 8 --eval-batches 64
"""
from __future__ import annotations

import argparse
import statistics
import time
from dataclasses import dataclass

import torch

from sakura.adapters.ddp import DDPAdapter
from sakura.dispatch import ThreadDispatcher
from sakura.events import OnEpochEnd
from sakura.runtime import SakuraRuntime
from sakura.services.async_eval import AsyncEval


# ---------------------------------------------------------------- workload

def make_model() -> torch.nn.Module:
    """Small but non-trivial MLP. Big enough that linear ops dominate the
    Python overhead; small enough that 5 epochs run in a few seconds on CPU."""
    return torch.nn.Sequential(
        torch.nn.Linear(128, 256),
        torch.nn.ReLU(),
        torch.nn.Linear(256, 256),
        torch.nn.ReLU(),
        torch.nn.Linear(256, 10),
    )


def make_train_loader(n_samples: int = 2048, batch_size: int = 64) -> torch.utils.data.DataLoader:
    torch.manual_seed(0)
    ds = torch.utils.data.TensorDataset(
        torch.randn(n_samples, 128),
        torch.randint(0, 10, (n_samples,)),
    )
    return torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=False)


def make_val_data(n_eval_batches: int, batch_size: int = 64) -> tuple[torch.Tensor, torch.Tensor]:
    """A larger val set than train — eval is the heavy step we want to overlap."""
    torch.manual_seed(1)
    n = n_eval_batches * batch_size
    return torch.randn(n, 128), torch.randint(0, 10, (n,))


def heavy_eval(state_dict: dict, val_x: torch.Tensor, val_y: torch.Tensor) -> dict:
    """Run a full forward pass over the val set. CPU torch matmul releases
    the GIL inside the C++ kernel, so this runs in true parallel with
    training when invoked on a thread."""
    model = make_model()
    model.load_state_dict(state_dict)
    model.eval()
    correct = total = 0
    loss_sum = 0.0
    with torch.no_grad():
        bs = 64
        for i in range(0, val_x.shape[0], bs):
            x = val_x[i:i+bs]
            y = val_y[i:i+bs]
            logits = model(x)
            loss_sum += float(torch.nn.functional.cross_entropy(logits, y, reduction="sum"))
            correct += int((logits.argmax(dim=-1) == y).sum())
            total += int(y.numel())
    return {"val_loss": loss_sum / max(total, 1), "val_acc": correct / max(total, 1)}


# ---------------------------------------------------------------- runners

@dataclass
class Run:
    label: str
    elapsed_secs: float
    final_metrics: dict


def train_baseline(epochs: int, n_eval_batches: int) -> Run:
    """Vanilla loop: train epoch → blocking eval → next epoch."""
    model = make_model()
    train_loader = make_train_loader()
    val_x, val_y = make_val_data(n_eval_batches)
    opt = torch.optim.SGD(model.parameters(), lr=0.01)

    t0 = time.perf_counter()
    metrics = {}
    for epoch in range(epochs):
        model.train()
        for x, y in train_loader:
            opt.zero_grad()
            loss = torch.nn.functional.cross_entropy(model(x), y)
            loss.backward()
            opt.step()
        # Synchronous eval after each epoch.
        metrics = heavy_eval({k: v.detach().clone() for k, v in model.state_dict().items()},
                              val_x, val_y)
    elapsed = time.perf_counter() - t0
    return Run(label="baseline", elapsed_secs=elapsed, final_metrics=metrics)


def train_sakura_async_eval(epochs: int, n_eval_batches: int) -> Run:
    """Sakura+AsyncEval: eval submitted to a thread, runs in parallel with next epoch."""
    model = make_model()
    train_loader = make_train_loader()
    val_x, val_y = make_val_data(n_eval_batches)
    opt = torch.optim.SGD(model.parameters(), lr=0.01)

    # Snapshot of the model at the most recent epoch_end. AsyncEval reads
    # this via the closure each time the dispatcher invokes the eval_fn;
    # backpressure="block" with max_pending=1 ensures consecutive epochs
    # see consecutive snapshots (no reordering).
    snapshot: dict = {"state_dict": None}

    def epoch_eval_fn(epoch: int, payload):
        sd = snapshot["state_dict"]
        if sd is None:
            return {"epoch": epoch, "skipped": True}
        return heavy_eval(sd, *payload)

    dispatcher = ThreadDispatcher(max_workers=1)
    rt = SakuraRuntime()
    async_eval = AsyncEval(
        eval_fn=epoch_eval_fn,
        eval_payload=(val_x, val_y),
        dispatcher=dispatcher,
        max_pending=1,
        on_backpressure="block",
    )
    rt.install(async_eval)
    adapter = DDPAdapter(rt, rank=0, world_size=1)

    with rt:
        adapter.on_train_begin(model, opt, train_loader)
        t0 = time.perf_counter()
        for epoch in range(epochs):
            adapter.on_epoch_begin(epoch)
            model.train()
            for step, (x, y) in enumerate(train_loader):
                adapter.on_train_step_begin(model, (x, y), step)
                opt.zero_grad()
                loss = torch.nn.functional.cross_entropy(model(x), y)
                loss.backward()
                adapter.on_optimizer_step(opt)
                if not rt.optimizer_step(opt):
                    opt.step()
            # Capture the post-epoch state and hand it to AsyncEval via the
            # snapshot closure. The dispatcher invokes epoch_eval_fn on a
            # background thread; training continues on the main thread for
            # the next epoch, overlapping with this evaluation.
            snapshot["state_dict"] = {k: v.detach().clone()
                                       for k, v in model.state_dict().items()}
            adapter.on_epoch_end(epoch, model, opt, metrics={})
        # Drain pending evals — AsyncEval blocks on remaining futures in
        # on_train_end, so this captures the final wallclock honestly.
        adapter.on_train_end(model)
        elapsed = time.perf_counter() - t0

    dispatcher.shutdown()
    metrics = async_eval.history[-1] if async_eval.history else {}
    return Run(label="sakura+async_eval", elapsed_secs=elapsed, final_metrics=metrics)


# ---------------------------------------------------------------- driver

def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--eval-batches", type=int, default=32)
    p.add_argument("--trials", type=int, default=3)
    args = p.parse_args()

    # Warm up torch ops so first-run JIT cost doesn't skew either side. We
    # rely on OMP_NUM_THREADS / MKL_NUM_THREADS in the environment to bound
    # the thread pool; the AsyncEval win is most visible when training and
    # eval don't fight for every available thread (typical 4-16 cores).
    _ = train_baseline(epochs=1, n_eval_batches=args.eval_batches)

    print(f"epochs={args.epochs} eval_batches={args.eval_batches} trials={args.trials}")
    print(f"torch.get_num_threads()={torch.get_num_threads()}")

    base_times = [train_baseline(args.epochs, args.eval_batches).elapsed_secs
                  for _ in range(args.trials)]
    saku_times = [train_sakura_async_eval(args.epochs, args.eval_batches).elapsed_secs
                  for _ in range(args.trials)]

    base_med = statistics.median(base_times)
    saku_med = statistics.median(saku_times)
    speedup = base_med / saku_med if saku_med > 0 else float("nan")

    print()
    print(f"baseline:           median={base_med*1000:.0f}ms  trials={[f'{t*1000:.0f}ms' for t in sorted(base_times)]}")
    print(f"sakura+async_eval:  median={saku_med*1000:.0f}ms  trials={[f'{t*1000:.0f}ms' for t in sorted(saku_times)]}")
    print(f"speedup (baseline/sakura): {speedup:.2f}x  ({(1-saku_med/base_med)*100:+.1f}% wallclock)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
