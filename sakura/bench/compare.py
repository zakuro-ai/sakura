"""Comparison utilities for RunReport JSON files."""
from __future__ import annotations

import json
from typing import Iterable, List

from sakura.bench.harness import RunReport


def load_reports(paths: Iterable[str]) -> List[RunReport]:
    out: list[RunReport] = []
    for p in paths:
        with open(p, "r", encoding="utf-8") as f:
            out.append(RunReport.from_json(f.read()))
    return out


def render_markdown_table(reports: list[RunReport]) -> str:
    """Render a side-by-side markdown table of reports."""
    if not reports:
        return "(no reports)"
    headers = ["workload", "framework", "services", "elapsed_secs",
               "samples_per_sec", "tokens_per_sec",
               "gpu_util_mean_pct", "peak_gpu_mem_mb", "gpu_mem_used_peak_mb",
               "reached_target"]
    metric_names = sorted({k for r in reports for k in r.final_metrics.keys()})
    headers.extend(metric_names)
    rows = []
    rows.append("| " + " | ".join(headers) + " |")
    rows.append("|" + "|".join(["---"] * len(headers)) + "|")
    for r in reports:
        services = ",".join(r.sakura_services or []) or "(baseline)"
        row = [
            r.workload, r.framework, services,
            f"{r.elapsed_secs:.2f}",
            f"{r.samples_per_sec:.0f}",
            f"{r.tokens_per_sec:.0f}",
            f"{r.gpu_util_mean_pct:.1f}",
            f"{r.peak_gpu_mem_mb:.0f}",
            f"{r.gpu_mem_used_peak_mb:.0f}",
            "" if r.reached_target is None else str(r.reached_target),
        ]
        for m in metric_names:
            v = r.final_metrics.get(m)
            row.append(f"{v:.4f}" if isinstance(v, float) else str(v) if v is not None else "")
        rows.append("| " + " | ".join(row) + " |")
    return "\n".join(rows)


def speedup_summary(baseline: RunReport, sakura: RunReport) -> str:
    """One-line summary comparing baseline vs sakura on the same workload."""
    if baseline.workload != sakura.workload:
        return f"(workloads differ: {baseline.workload} vs {sakura.workload})"
    delta = baseline.elapsed_secs - sakura.elapsed_secs
    pct = (delta / baseline.elapsed_secs) * 100 if baseline.elapsed_secs else 0
    factor = baseline.elapsed_secs / sakura.elapsed_secs if sakura.elapsed_secs else float("inf")
    faster = "sakura" if delta > 0 else "baseline"
    if baseline.tokens_per_sec:
        tok_part = (
            f"tokens/sec {baseline.tokens_per_sec:.0f}->{sakura.tokens_per_sec:.0f} "
            f"({sakura.tokens_per_sec / baseline.tokens_per_sec:.2f}x)"
        )
    else:
        tok_part = "tokens/sec n/a"
    return (
        f"{baseline.workload}: {faster} {abs(pct):.1f}% faster "
        f"({factor:.2f}x; baseline={baseline.elapsed_secs:.2f}s, "
        f"sakura={sakura.elapsed_secs:.2f}s) | {tok_part} | "
        f"gpu-util {baseline.gpu_util_mean_pct:.0f}%->{sakura.gpu_util_mean_pct:.0f}%"
    )


__all__ = ["load_reports", "render_markdown_table", "speedup_summary"]
