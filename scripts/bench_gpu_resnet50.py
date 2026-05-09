"""GPU benchmark: ResNet-50 + CIFAR-10 at meaningful scale.

Compares vanilla PyTorch DDP loop vs sakura with realistic service stacks.
Designed for a single RTX 4090-class GPU (24 GB VRAM); set
`CUDA_VISIBLE_DEVICES` to target a specific device.

Each config runs in its own subprocess to isolate torch global state
(compile cache, autocast context, GradScaler state). The driver collects
elapsed_secs / samples_per_sec / peak_gpu_mem_mb / val_loss / val_acc and
prints a markdown table.

Usage:
    CUDA_VISIBLE_DEVICES=0 python scripts/bench_gpu_resnet50.py
    CUDA_VISIBLE_DEVICES=0 python scripts/bench_gpu_resnet50.py --epochs 3 --batch 128
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import asdict, dataclass


@dataclass
class Result:
    config: str
    elapsed_secs: float
    samples_per_sec: float
    peak_gpu_mem_mb: float
    val_loss: float
    val_acc: float


# --------------------------------------------------------- subprocess entry

def _run_one(config: str, epochs: int, batch_size: int, n_train: int, n_val: int) -> Result:
    """Run a single config in this process. Called by the driver via subprocess."""
    import torch
    from sakura.bench.harness import BaselineRunner, RunReport, SakuraRunner
    from sakura.bench.workloads.cifar import make_workload_imagenet_shape

    wl = make_workload_imagenet_shape(batch_size=batch_size, epochs=epochs,
                                       n_train=n_train, n_val=n_val)

    services: list = []
    if config == "baseline":
        runner = BaselineRunner(framework="pytorch-ddp")
    else:
        # Build sakura's service stack from the config name.
        if "bf16" in config:
            from sakura.services.mixed_precision import MixedPrecision
            services.append(MixedPrecision(dtype="bf16"))
        if "compile" in config:
            from sakura.services.compile import Compile
            services.append(Compile(mode="default"))
        if "async_eval" in config:
            from sakura.dispatch import ThreadDispatcher
            from sakura.services.async_eval import AsyncEval

            # Bridge AsyncEval to the workload's eval_fn — same pattern as
            # sakura/bench/__main__._make_async_eval but inlined to avoid
            # CLI factory dependencies.
            snapshot: dict = {"state_dict": None, "val_loader": None}

            def evf(epoch, payload):
                sd = snapshot["state_dict"]; loader = snapshot["val_loader"]
                if sd is None or loader is None:
                    return {"epoch": epoch, "skipped": True}
                m = wl.make_model()
                m.load_state_dict(sd)
                # Eval on CPU so it doesn't fight training for the GPU —
                # this is the entire point of async overlap.
                m = m.cpu()
                from sakura.bench.harness import _DeviceLoader
                cpu_loader = _DeviceLoader(loader, "cpu")
                return wl.eval_fn(m, cpu_loader)

            svc = AsyncEval(eval_fn=evf, eval_payload=None,
                            dispatcher=ThreadDispatcher(max_workers=1),
                            max_pending=1, on_backpressure="block")
            svc._bench_snapshot = snapshot
            services.append(svc)
        runner = SakuraRunner(framework="pytorch-ddp", services=services)

    report = runner.run(wl)
    print(json.dumps({
        "config": config,
        "elapsed_secs": report.elapsed_secs,
        "samples_per_sec": report.samples_per_sec,
        "peak_gpu_mem_mb": report.peak_gpu_mem_mb,
        "val_loss": float(report.final_metrics.get("val_loss", float("nan"))),
        "val_acc": float(report.final_metrics.get("val_acc", float("nan"))),
    }))
    return None


def _spawn_one(config: str, args, repo_root: str) -> Result:
    """Run `_run_one` in a fresh subprocess, parse the JSON line it prints.

    Subprocess isolation is necessary because torch.compile mutates module
    forward methods (and a stale compile-cache survives in process state),
    GradScaler holds CUDA state, and the harness caches CUDA-availability.
    """
    env = os.environ.copy()
    env["PYTHONPATH"] = repo_root + os.pathsep + env.get("PYTHONPATH", "")
    cmd = [
        sys.executable, "-c",
        f"""
import sys, os
sys.path.insert(0, {repo_root!r})
from scripts.bench_gpu_resnet50 import _run_one
_run_one({config!r}, {args.epochs}, {args.batch}, {args.n_train}, {args.n_val})
"""
    ]
    out = subprocess.check_output(cmd, env=env, text=True)
    # Extract the last JSON line (subprocess may print torch warnings before).
    for line in reversed(out.strip().splitlines()):
        line = line.strip()
        if line.startswith("{") and line.endswith("}"):
            d = json.loads(line)
            return Result(**d)
    raise RuntimeError(f"no JSON output from {config}: {out[-500:]}")


# ---------------------------------------------------------------- driver

def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--n-train", type=int, default=8192,
                    help="Training samples (subset of CIFAR-10's 50k).")
    p.add_argument("--n-val", type=int, default=2048,
                    help="Validation samples (subset of CIFAR-10's 10k).")
    p.add_argument("--configs", nargs="+", default=[
        "baseline",
        "sakura+bf16",
    ])
    p.add_argument("--trials", type=int, default=1,
                    help="Repeat each config N times; report median of elapsed_secs.")
    args = p.parse_args()

    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    print(f"# GPU bench — ResNet-50 + CIFAR-10")
    import torch  # noqa: PLC0415
    if torch.cuda.is_available():
        print(f"# device: {torch.cuda.get_device_name(0)}  "
              f"compute={torch.cuda.get_device_capability()}  "
              f"vram={torch.cuda.get_device_properties(0).total_memory/(1024**3):.1f}GB")
    print(f"# epochs={args.epochs} batch={args.batch} "
          f"n_train={args.n_train} n_val={args.n_val}")
    print()

    results: list[Result] = []
    for cfg in args.configs:
        trials_data = []
        for trial in range(args.trials):
            t0 = time.perf_counter()
            r = _spawn_one(cfg, args, repo_root)
            wall = time.perf_counter() - t0
            trials_data.append(r)
            print(f"  {cfg:35s}  trial {trial+1}/{args.trials}: "
                  f"{r.elapsed_secs*1000:7.0f}ms  "
                  f"{r.samples_per_sec:7.0f} samples/s  "
                  f"{r.peak_gpu_mem_mb:6.0f}MB  "
                  f"val_loss={r.val_loss:.3f}  val_acc={r.val_acc:.3f}  "
                  f"(wall {wall:.0f}s)")
        # Median by elapsed_secs.
        trials_data.sort(key=lambda r: r.elapsed_secs)
        med = trials_data[len(trials_data) // 2]
        results.append(med)

    # Markdown table.
    base = next((r for r in results if r.config == "baseline"), None)
    if base is None:
        return 0
    print()
    print("| Config | elapsed | samples/sec | peak GPU mem | val_acc | speedup vs baseline |")
    print("|---|---|---|---|---|---|")
    for r in results:
        speedup = base.elapsed_secs / r.elapsed_secs if r.elapsed_secs > 0 else 0.0
        print(f"| `{r.config}` | {r.elapsed_secs:.2f}s | {r.samples_per_sec:.0f} | "
              f"{r.peak_gpu_mem_mb:.0f} MB | {r.val_acc:.3f} | "
              f"{speedup:.2f}× |")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
