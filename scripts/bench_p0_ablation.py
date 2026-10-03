"""P0 smoke ablation driver — baseline / +amp / +compile / +activation_ckpt.

Runs the four P0 configs on the `gpt2-124m` workload via the `sakura.bench`
CLI (one isolated subprocess per config, so each gets a clean CUDA context,
its own ``reset_peak_memory_stats``, and no cross-config peak-memory
contamination), then renders the comparison table and runs the P0 *sanity
gate* (this driver's acceptance check).

This is a GPU job; run it on a CUDA host from the repository root::

    CUDA_VISIBLE_DEVICES=0 python scripts/bench_p0_ablation.py --out-dir ~/p0-reports

``--dry-run`` prints the exact per-config commands without executing (no GPU).
"""
from __future__ import annotations

import os
import subprocess
import sys

# The four P0 ablation rows, in progressive-composition order. Each entry is
# (label, extra CLI args appended after `... run --workload gpt2-124m ...`).
P0_CONFIGS = [
    ("baseline", ["--runner", "baseline"]),
    ("amp", ["--runner", "sakura", "--service", "mixed_precision:fp16"]),
    ("compile", ["--runner", "sakura",
                 "--service", "mixed_precision:fp16",
                 "--service", "compile"]),
    ("activation_ckpt", ["--runner", "sakura",
                         "--service", "mixed_precision:fp16",
                         "--service", "compile",
                         "--service", "activation_checkpoint"]),
]


def build_run_cmd(python_exe, label, extra_args, out_dir,
                  workload="gpt2-124m", framework="pytorch-ddp"):
    """Build the argv for one `sakura.bench run` subprocess + its output path."""
    out_path = os.path.join(out_dir, f"gpt2-p0-{label}.json")
    cmd = [python_exe, "-m", "sakura.bench", "run",
           "--workload", workload, "--framework", framework,
           *extra_args, "--output", out_path]
    return cmd, out_path


def sizing_env(seq_len, batch_size, epochs, n_train_batches, n_val_batches, base=None):
    """Return an env mapping that sizes the gpt2-124m workload for a subprocess.

    The CLI calls ``make_workload()`` with no args, so headline sizing is passed
    via ``SAKURA_GPT2_*`` env vars (the gpt2 workload factory reads them as
    defaults). If the factory does not honor them, runs fall back to the
    workload's own defaults — the sanity gate is qualitative, so it stays valid.
    """
    env = dict(base if base is not None else os.environ)
    env["SAKURA_GPT2_SEQ_LEN"] = str(seq_len)
    env["SAKURA_GPT2_BATCH_SIZE"] = str(batch_size)
    env["SAKURA_GPT2_EPOCHS"] = str(epochs)
    env["SAKURA_GPT2_N_TRAIN_BATCHES"] = str(n_train_batches)
    env["SAKURA_GPT2_N_VAL_BATCHES"] = str(n_val_batches)
    return env


def count_orphan_workers(pattern=r"python.* -m sakura\.worker"):
    """Count surviving worker subprocesses (``python -m sakura.worker ...``).

    Uses an argv-list pgrep (no shell) with a pattern anchored on the module
    invocation flag so the search cannot match source files, config keys, or
    other strings that happen to contain "sakura.worker".  Returns 0 when
    pgrep is unavailable.
    """
    try:
        out = subprocess.run(["pgrep", "-f", pattern], capture_output=True, text=True)
    except FileNotFoundError:
        return 0
    return len([ln for ln in out.stdout.splitlines() if ln.strip()])


def check_sanity(reports, orphan_count):
    """The P0 acceptance check. Returns a list of failure messages (empty = pass).

    Assertions:
      1. +amp tokens_per_sec > baseline tokens_per_sec   (Turing fp16 throughput)
      2. +activation_ckpt peak_gpu_mem_mb < +compile peak_gpu_mem_mb (mem unlock)
      3. every row reports gpu_util_mean_pct > 0          (sampler actually ran)
      4. no orphaned sakura.worker processes survived the run
    """
    failures = []
    base, amp = reports.get("baseline"), reports.get("amp")
    comp, ac = reports.get("compile"), reports.get("activation_ckpt")

    if base is not None and amp is not None:
        if not (amp.tokens_per_sec > base.tokens_per_sec):
            failures.append(
                f"[amp] tokens_per_sec {amp.tokens_per_sec:.0f} "
                f"!> baseline {base.tokens_per_sec:.0f}")
    else:
        failures.append("[amp] missing baseline or amp report")

    if comp is not None and ac is not None:
        if not (ac.peak_gpu_mem_mb < comp.peak_gpu_mem_mb):
            failures.append(
                f"[activation_ckpt] peak_gpu_mem_mb {ac.peak_gpu_mem_mb:.0f} "
                f"!< compile {comp.peak_gpu_mem_mb:.0f}")
    else:
        failures.append("[activation_ckpt] missing compile or activation_ckpt report")

    for label, r in reports.items():
        if not (r.gpu_util_mean_pct > 0):
            failures.append(
                f"[{label}] gpu_util_mean_pct {r.gpu_util_mean_pct:.1f} !> 0")

    if orphan_count > 0:
        failures.append(
            f"[orphans] {orphan_count} sakura.worker process(es) survived the run")

    return failures


def main(argv=None) -> int:
    import argparse

    p = argparse.ArgumentParser(
        prog="bench_p0_ablation",
        description="P0 smoke ablation: baseline / +amp / +compile / +activation_ckpt on gpt2-124m.")
    p.add_argument("--python", default=sys.executable,
                   help="python for each per-config subprocess (default: this interpreter)")
    p.add_argument("--out-dir", default="p0-reports", help="directory for RunReport JSONs")
    p.add_argument("--seq-len", type=int, default=512)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--n-train-batches", type=int, default=128)
    p.add_argument("--n-val-batches", type=int, default=8)
    p.add_argument("--dry-run", action="store_true",
                   help="print commands without executing (no GPU needed)")
    args = p.parse_args(argv)

    os.makedirs(args.out_dir, exist_ok=True)
    env = sizing_env(args.seq_len, args.batch_size, args.epochs,
                     args.n_train_batches, args.n_val_batches)

    # The harness now does warmup-step exclusion via the warmup_steps kwarg:
    # each per-config subprocess runs its own warmup steps inside the runner
    # before the timed region starts, so no cross-process pre-warm is needed
    # here.  (The old driver-level pre-warm only warmed the heaviest config and
    # left the lighter configs cold — harness-level warmup is strictly better.)
    runs = []
    for label, extra in P0_CONFIGS:
        cmd, out_path = build_run_cmd(args.python, label, extra, args.out_dir)
        runs.append((label, cmd, out_path))
        print(f"[{label}]", " ".join(cmd))
        if not args.dry_run:
            subprocess.run(cmd, env=env, check=True)

    if args.dry_run:
        return 0

    from sakura.bench.compare import render_markdown_table, speedup_summary
    from sakura.bench.harness import RunReport

    reports = {}
    for label, _cmd, out_path in runs:
        with open(out_path, "r", encoding="utf-8") as f:
            reports[label] = RunReport.from_json(f.read())

    print()
    print(render_markdown_table([reports[label] for label, _, _ in runs]))
    print()
    base = reports["baseline"]
    for label in ("amp", "compile", "activation_ckpt"):
        print(speedup_summary(base, reports[label]))

    orphans = count_orphan_workers()
    failures = check_sanity(reports, orphans)
    print()
    if failures:
        print("SANITY: FAIL")
        for msg in failures:
            print("  -", msg)
        return 1
    print("SANITY: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
