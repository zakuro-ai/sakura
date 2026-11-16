#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "sakura-ml>=1.1.0,<2",
#     "torch",
#     "torchvision",
# ]
#
# # On Linux, pull the CPU build of PyTorch (~200 MB) instead of the default
# # CUDA build (several GB). The macOS and Windows wheels are small already.
# [[tool.uv.index]]
# name = "pytorch-cpu"
# url = "https://download.pytorch.org/whl/cpu"
# explicit = true
#
# [tool.uv.sources]
# torch = [{ index = "pytorch-cpu", marker = "sys_platform == 'linux'" }]
# torchvision = [{ index = "pytorch-cpu", marker = "sys_platform == 'linux'" }]
# ///
"""Sakura in one command: the same training job, with and without Sakura.

    uv run demo.py              # from a fresh clone; installs sakura-ml from PyPI
    uv run demo.py --trials 5   # more trials, tighter numbers

Trains a small classifier on CPU (synthetic data, no download, no GPU needed),
evaluating on a held-out set after every epoch -- once as a plain PyTorch loop,
once with Sakura's `AsyncEval`. Both runs use the same model, data, seed and
CPU thread count. The only difference: in the plain loop each evaluation
blocks the next epoch; with Sakura it runs on a background thread while the
next epoch trains. Everything printed is measured on your machine.
"""
from __future__ import annotations

import argparse
import platform
import statistics
import time
from importlib.metadata import version

import torch
from torch import nn

from sakura import SakuraRuntime
from sakura.adapters import DDPAdapter
from sakura.dispatch import ThreadDispatcher
from sakura.services import AsyncEval

REPO = "https://github.com/zakuro-ai/sakura"
FEATURES, CLASSES = 128, 10
EPOCHS, BATCH = 6, 64
N_TRAIN, N_VAL = 4096, 32768  # a big held-out set: eval costs about as much as an epoch


# --------------------------------------------------------------------------- data + model
def make_data(seed: int = 0) -> tuple[torch.Tensor, ...]:
    """Inputs labelled by a fixed random 'teacher' network, so there is real signal to learn."""
    torch.manual_seed(seed)
    teacher = nn.Sequential(nn.Linear(FEATURES, 64), nn.Tanh(), nn.Linear(64, CLASSES))
    with torch.no_grad():
        x = torch.randn(N_TRAIN + N_VAL, FEATURES)
        y = teacher(x).argmax(dim=-1)
    return x[:N_TRAIN], y[:N_TRAIN], x[N_TRAIN:], y[N_TRAIN:]


def make_model() -> nn.Module:
    return nn.Sequential(
        nn.Linear(FEATURES, 512), nn.ReLU(),
        nn.Linear(512, 512), nn.ReLU(),
        nn.Linear(512, CLASSES),
    )


def train_one_epoch(model, opt, x, y, gen) -> None:
    model.train()
    for idx in torch.randperm(len(x), generator=gen).split(BATCH):
        opt.zero_grad(set_to_none=True)
        nn.functional.cross_entropy(model(x[idx]), y[idx]).backward()
        opt.step()


@torch.no_grad()
def evaluate(model, x, y) -> float:
    model.eval()
    correct = sum(int((model(xb).argmax(-1) == yb).sum())
                  for xb, yb in zip(x.split(256), y.split(256)))
    return correct / len(x)


# --------------------------------------------------------------------------- the two loops
def run_plain(data) -> tuple[float, float]:
    xt, yt, xv, yv = data
    torch.manual_seed(1)
    model, gen = make_model(), torch.Generator().manual_seed(2)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    t0 = time.perf_counter()
    for _ in range(EPOCHS):
        train_one_epoch(model, opt, xt, yt, gen)
        acc = evaluate(model, xv, yv)            # training waits for this
    return time.perf_counter() - t0, acc


def run_sakura(data) -> tuple[float, float]:
    xt, yt, xv, yv = data
    torch.manual_seed(1)
    model, gen = make_model(), torch.Generator().manual_seed(2)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    snapshots: dict[int, dict] = {}              # epoch -> weights to evaluate

    def eval_epoch(epoch: int, snaps: dict) -> dict:
        m = make_model()
        m.load_state_dict(snaps.pop(epoch))
        return {"val_acc": evaluate(m, xv, yv)}

    async_eval = AsyncEval(eval_fn=eval_epoch, eval_payload=snapshots,
                           dispatcher=ThreadDispatcher(), adaptive=False,
                           on_backpressure="block")  # never skip an epoch's eval
    t0 = time.perf_counter()
    with SakuraRuntime() as rt:
        rt.install(async_eval)
        loop = DDPAdapter(rt, rank=0, world_size=1)
        loop.on_train_begin(model, opt, None)
        for epoch in range(EPOCHS):
            loop.on_epoch_begin(epoch)
            train_one_epoch(model, opt, xt, yt, gen)
            snapshots[epoch] = {k: v.detach().clone() for k, v in model.state_dict().items()}
            loop.on_epoch_end(epoch, model, opt, metrics={})  # eval starts here, in the background
        loop.on_train_end(model)                  # waits for the last evaluation
    elapsed = time.perf_counter() - t0
    accs = [h["val_acc"] for h in async_eval.history if "val_acc" in h]
    assert len(accs) == EPOCHS, f"expected {EPOCHS} evaluations, got {async_eval.history}"
    return elapsed, accs[-1]


# --------------------------------------------------------------------------- main
def fastest_threads(fn, data, candidates: list[int], repeats: int = 2) -> int:
    """Time each candidate thread count (best of `repeats` runs); keep the fastest.

    Best-of-N, not a single run: one noisy run must not hand plain PyTorch a
    slow thread count and make Sakura look better than it is.
    """
    timed = []
    for t in candidates:
        torch.set_num_threads(t)
        timed.append((min(fn(data)[0] for _ in range(repeats)), t))
    return min(timed)[1]


def main() -> int:
    ap = argparse.ArgumentParser(description="Same training job, with and without Sakura.")
    ap.add_argument("--trials", type=int, default=3,
                    help="timed runs per setup; the median is reported (default 3)")
    ap.add_argument("--threads", type=int, default=None,
                    help="force this CPU thread count for both setups (default: probe "
                         "and give each setup its own fastest count)")
    args = ap.parse_args()

    default_threads = torch.get_num_threads()
    print(f"\nSakura {version('sakura-ml')} | PyTorch {version('torch')} | "
          f"Python {platform.python_version()} | {platform.system()} {platform.machine()}")
    print(f"Training a {EPOCHS}-epoch classifier, evaluating on {N_VAL:,} held-out samples "
          f"after each epoch.\n")

    data = make_data()
    setups = {"plain PyTorch": run_plain, "with Sakura": run_sakura}
    run_plain(data)  # warm-up, not timed: first-touch allocations, kernel selection

    if args.threads:
        threads = {name: args.threads for name in setups}
    else:
        # Fair fight: plain PyTorch gets its own best thread count too, so the
        # comparison is best-vs-best, not Sakura-vs-a-handicapped-baseline.
        candidates = sorted({1, 2, max(1, default_threads // 2), default_threads})
        print(f"  probing CPU threads {candidates} for each setup ...")
        threads = {name: fastest_threads(fn, data, candidates) for name, fn in setups.items()}
        print("  fastest: " + ", ".join(f"{n} {t}" for n, t in threads.items()) + "\n")

    times: dict[str, list[float]] = {k: [] for k in setups}
    accs: dict[str, float] = {}
    for trial in range(1, args.trials + 1):
        order = list(setups) if trial % 2 else list(reversed(setups))  # no setup always goes first
        for name in order:
            torch.set_num_threads(threads[name])
            secs, acc = setups[name](data)
            times[name].append(secs)
            accs[name] = acc
            print(f"  trial {trial}/{args.trials}   {name:<14} {secs:6.2f} s   val acc {acc:.3f}")

    base, fast = (statistics.median(times[k]) for k in setups)
    print(f"\n  {'setup':<15} {'threads':>7} {'median time':>12}   {'final val acc':>13}")
    for name in setups:
        print(f"  {name:<15} {threads[name]:>7} {statistics.median(times[name]):>10.2f} s"
              f"   {accs[name]:>13.3f}")
    print()
    speedup = base / fast
    if speedup >= 1.05:
        print(f"  {speedup:.2f}x faster on this machine ({(1 - fast / base) * 100:.0f}% less "
              f"wall-clock), same model, same accuracy:")
        print("  evaluation now overlaps the next epoch instead of blocking it.")
    else:
        print(f"  {speedup:.2f}x on this machine -- no real gain here: training already keeps "
              f"every core busy,")
        print("  so there is no idle compute for the evaluation to overlap with. AsyncEval pays "
              "off when")
        print("  training leaves resources idle -- e.g. a GPU run, where evaluation moves to the "
              "CPU.")

    print("\nThe Sakura part of run_sakura() in demo.py:\n")
    print("    with SakuraRuntime() as rt:")
    print("        rt.install(AsyncEval(eval_fn=eval_epoch, eval_payload=snapshots,")
    print("                             dispatcher=ThreadDispatcher(), adaptive=False,")
    print("                             on_backpressure=\"block\"))")
    print("        loop = DDPAdapter(rt, rank=0, world_size=1)")
    print("        ...  # your training loop, plus a hook call at the start and end of each epoch\n")
    print(f"If this was useful, a star on {REPO} helps others find it,")
    print("and it's the easiest way to follow what we ship next.\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
