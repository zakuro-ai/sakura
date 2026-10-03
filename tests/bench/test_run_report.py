"""RunReport JSON round-trip + helper tests."""
from __future__ import annotations

import json

from sakura.bench.harness import RunReport, Workload, detect_git_sha, detect_hardware


def test_run_report_json_roundtrip():
    r = RunReport(
        workload="cifar10-resnet50",
        framework="pytorch-ddp",
        sakura_services=["MixedPrecision", "AsyncEval"],
        elapsed_secs=42.5,
        samples_per_sec=2716,
        peak_gpu_mem_mb=1830,
        final_metrics={"val_acc": 0.914},
        per_stage_secs={"compile": 5.2, "epoch_avg": 6.7},
        git_sha="abc123",
        hardware={"gpu_name": "RTX 4090"},
    )
    s = r.to_json()
    r2 = RunReport.from_json(s)
    assert r2 == r


def test_detect_hardware_returns_dict_with_torch_version():
    h = detect_hardware()
    assert isinstance(h, dict)
    assert "torch" in h
    assert "cuda_available" in h


def test_detect_git_sha_returns_string():
    sha = detect_git_sha()
    assert isinstance(sha, str)
    # Either valid hex or empty (if not in a git repo); both are fine.


def test_run_report_new_metric_fields_roundtrip():
    r = RunReport(
        workload="gpt2-124m",
        framework="pytorch-ddp",
        sakura_services=["MixedPrecision"],
        elapsed_secs=12.5,
        samples_per_sec=512.0,
        peak_gpu_mem_mb=7600.0,
        gpu_util_mean_pct=91.3,
        gpu_util_max_pct=99.0,
        gpu_util_samples=128,
        gpu_mem_used_peak_mb=8100.0,
        tokens_per_sec=262144.0,
        total_tokens=262144,
        reached_target=True,
        epochs_to_target=3,
    )
    r2 = RunReport.from_json(r.to_json())
    assert r2 == r


def test_run_report_back_compat_old_json():
    """Reports written before P0 instrumentation still load; new fields default."""
    old = json.dumps({
        "workload": "cifar10-resnet50",
        "framework": "pytorch-ddp",
        "sakura_services": ["MixedPrecision", "AsyncEval"],
        "elapsed_secs": 42.5,
        "samples_per_sec": 2716,
        "peak_gpu_mem_mb": 1830,
        "final_metrics": {"val_acc": 0.914},
        "per_stage_secs": {"compile": 5.2, "epoch_avg": 6.7},
        "git_sha": "abc123",
        "hardware": {"gpu_name": "RTX 4090"},
    })
    r = RunReport.from_json(old)
    assert r.gpu_util_mean_pct == 0.0
    assert r.gpu_util_max_pct == 0.0
    assert r.gpu_util_samples == 0
    assert r.gpu_mem_used_peak_mb == 0.0
    assert r.tokens_per_sec == 0.0
    assert r.total_tokens == 0
    assert r.reached_target is None
    assert r.epochs_to_target is None
    assert r.workload == "cifar10-resnet50"


def test_workload_accepts_tokens_and_block_types():
    class _Block:  # stand-in for a transformer block class
        pass

    w = Workload(
        name="gpt2-124m", tier="perf",
        make_model=lambda: None,
        make_train_loader=lambda: [],
        make_val_loader=lambda: [],
        eval_fn=lambda m, p: {},
        epochs=2,
        tokens_per_sample=512,
        block_types=(_Block,),
    )
    assert w.tokens_per_sample == 512
    assert w.block_types == (_Block,)
    # defaults apply when the new fields are omitted
    w2 = Workload(
        name="x", tier="ci",
        make_model=lambda: None,
        make_train_loader=lambda: [],
        make_val_loader=lambda: [],
        eval_fn=lambda m, p: {},
    )
    assert w2.tokens_per_sample is None
    assert w2.block_types == ()
