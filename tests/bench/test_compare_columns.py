"""compare.py surfaces the P0 instrumentation columns + speedup deltas."""
from __future__ import annotations

from sakura.bench.compare import render_markdown_table, speedup_summary
from sakura.bench.harness import RunReport


def _rep(label, **kw):
    return RunReport(
        workload="gpt2-124m", framework="pytorch-ddp",
        sakura_services=None if label == "baseline" else [label], **kw)


def test_render_markdown_table_has_p0_columns():
    reports = [
        _rep("baseline", elapsed_secs=20.0, samples_per_sec=80,
             tokens_per_sec=41000, gpu_util_mean_pct=72.0,
             peak_gpu_mem_mb=8200, gpu_mem_used_peak_mb=9000),
        _rep("mixed_precision", elapsed_secs=11.0, samples_per_sec=150,
             tokens_per_sec=77000, gpu_util_mean_pct=85.0,
             peak_gpu_mem_mb=5200, gpu_mem_used_peak_mb=6000,
             reached_target=True,
             final_metrics={"val_loss": 6.51, "perplexity": 671.2}),
    ]
    tbl = render_markdown_table(reports)
    header = tbl.splitlines()[0]
    for col in ("tokens_per_sec", "gpu_util_mean_pct", "gpu_mem_used_peak_mb",
                "reached_target"):
        assert col in header, f"missing column {col!r} in: {header}"
    # values rendered
    assert "77000" in tbl and "85.0" in tbl and "6000" in tbl
    # reached_target: None on baseline -> empty cell, True on amp -> "True"
    assert "True" in tbl
    # workload metrics still appended after the fixed columns
    assert "val_loss" in header and "perplexity" in header
    # back-compat: first column unchanged (existing export test asserts this)
    assert "| workload " in header


def test_speedup_summary_surfaces_tokens_and_util():
    base = RunReport(workload="gpt2-124m", framework="pytorch-ddp",
                     elapsed_secs=20.0, tokens_per_sec=41000, gpu_util_mean_pct=72.0)
    amp = RunReport(workload="gpt2-124m", framework="pytorch-ddp",
                    elapsed_secs=11.0, tokens_per_sec=77000, gpu_util_mean_pct=85.0,
                    sakura_services=["mixed_precision"])
    s = speedup_summary(base, amp)
    assert "tokens/sec 41000->77000" in s
    assert "1.88x" in s   # 77000 / 41000
    assert "gpu-util 72%->85%" in s
    assert "1.82x" in s   # wall-clock 20 / 11 (unchanged behaviour)


def test_speedup_summary_zero_tokens_back_compat():
    # Old reports have tokens_per_sec == 0.0: must not print "infx".
    base = RunReport(workload="x", framework="pytorch-ddp", elapsed_secs=10.0)
    sak = RunReport(workload="x", framework="pytorch-ddp", elapsed_secs=5.0)
    s = speedup_summary(base, sak)
    assert "2.00x" in s and "tokens/sec n/a" in s and "infx" not in s
    # gpu-util segment still rendered for the legacy/zero path (both sides 0.0)
    assert "gpu-util 0%->0%" in s
