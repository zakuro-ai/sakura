"""GPU benchmark: DistilBERT + SST-2 (HF-shaped path) on a real transformer.

DistilBERT (66M params) is the regime where mixed precision pays off the
most — most operations are GEMM-heavy and tensor-cores at bf16 are roughly
2x throughput on Ada (RTX 4090, compute capability 8.9).

HF Trainer owns the training loop including opt.step() and (when configured)
its own GradScaler — sakura's MixedPrecision service can't intercept that.
So this benchmark compares:
  - HF Trainer fp32 (baseline)
  - HF Trainer bf16 (Trainer's native precision config)
  - HF Trainer bf16 + sakura HFAdapter + Telemetry (same precision, plus
    sakura's observability pipeline; verifies adapter overhead is small)

Usage:
    CUDA_VISIBLE_DEVICES=0 python scripts/bench_gpu_distilbert.py
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass


@dataclass
class Result:
    config: str
    elapsed_secs: float
    samples_per_sec: float
    peak_gpu_mem_mb: float


def _run_one(config: str, epochs: int, batch_size: int, n_train: int):
    """Run a single config in this process. Called via subprocess."""
    import tempfile
    import torch
    from transformers import Trainer, TrainingArguments

    from sakura.bench.workloads.distilbert_hf import make_workload

    wl = make_workload(batch_size=batch_size, epochs=epochs, n_train=n_train, n_val=64,
                        max_length=128)
    model = wl.make_model()
    train_loader = wl.make_train_loader()
    train_dataset = train_loader.dataset
    collate_fn = train_loader.collate_fn

    fp16 = (config == "hf+fp16")
    bf16 = (config in ("hf+bf16", "hf+bf16+sakura"))

    callbacks = []
    if config == "hf+bf16+sakura":
        from sakura.adapters.huggingface import HFAdapter
        from sakura.runtime import SakuraRuntime
        from sakura.services.telemetry import Telemetry
        rt = SakuraRuntime()
        rt.install(Telemetry(output=lambda _r: None))
        callbacks.append(HFAdapter(rt, rank=0, world_size=1))

    with tempfile.TemporaryDirectory() as out_dir:
        args = TrainingArguments(
            output_dir=out_dir,
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            logging_strategy="no",
            save_strategy="no",
            eval_strategy="no",
            report_to=[],
            disable_tqdm=True,
            use_cpu=False,
            dataloader_num_workers=0,
            fp16=fp16,
            bf16=bf16,
        )
        trainer = Trainer(
            model=model,
            args=args,
            data_collator=collate_fn,
            train_dataset=train_dataset,
            callbacks=callbacks or None,
        )

        torch.cuda.reset_peak_memory_stats()
        t0 = time.perf_counter()
        trainer.train()
        elapsed = time.perf_counter() - t0
        peak_mb = torch.cuda.max_memory_allocated() / (1024**2)

    n_samples = len(train_dataset) * epochs
    print(json.dumps({
        "config": config,
        "elapsed_secs": elapsed,
        "samples_per_sec": n_samples / max(elapsed, 1e-9),
        "peak_gpu_mem_mb": peak_mb,
    }))


def _spawn_one(config: str, args, repo_root: str) -> Result:
    env = os.environ.copy()
    env["PYTHONPATH"] = repo_root + os.pathsep + env.get("PYTHONPATH", "")
    cmd = [
        sys.executable, "-c",
        f"""
import sys
sys.path.insert(0, {repo_root!r})
from scripts.bench_gpu_distilbert import _run_one
_run_one({config!r}, {args.epochs}, {args.batch}, {args.n_train})
"""
    ]
    out = subprocess.check_output(cmd, env=env, text=True)
    for line in reversed(out.strip().splitlines()):
        line = line.strip()
        if line.startswith("{") and line.endswith("}"):
            return Result(**json.loads(line))
    raise RuntimeError(f"no JSON output from {config}: {out[-500:]}")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--n-train", type=int, default=2048)
    p.add_argument("--configs", nargs="+", default=[
        "hf+fp32",
        "hf+bf16",
        "hf+bf16+sakura",
    ])
    args = p.parse_args()

    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    print("# GPU bench — DistilBERT + SST-2 (HF Trainer path)")
    import torch
    if torch.cuda.is_available():
        print(f"# device: {torch.cuda.get_device_name(0)}  "
              f"compute={torch.cuda.get_device_capability()}  "
              f"vram={torch.cuda.get_device_properties(0).total_memory/(1024**3):.1f}GB")
    print(f"# epochs={args.epochs} batch={args.batch} n_train={args.n_train}")
    print()

    results = []
    for cfg in args.configs:
        t0 = time.perf_counter()
        r = _spawn_one(cfg, args, repo_root)
        wall = time.perf_counter() - t0
        results.append(r)
        print(f"  {cfg:25s}  {r.elapsed_secs*1000:7.0f}ms  "
              f"{r.samples_per_sec:7.0f} samples/s  "
              f"{r.peak_gpu_mem_mb:6.0f}MB  (wall {wall:.0f}s)")

    base = next((r for r in results if r.config == "hf+fp32"), None)
    if base is None:
        return 0
    print()
    print("| Config | elapsed | samples/sec | peak GPU mem | speedup |")
    print("|---|---|---|---|---|")
    for r in results:
        sp = base.elapsed_secs / r.elapsed_secs if r.elapsed_secs > 0 else 0.0
        print(f"| `{r.config}` | {r.elapsed_secs:.2f}s | {r.samples_per_sec:.0f} | "
              f"{r.peak_gpu_mem_mb:.0f} MB | {sp:.2f}× |")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
