"""RunReport JSON round-trip + helper tests."""
from __future__ import annotations

from sakura.bench.harness import RunReport, detect_git_sha, detect_hardware


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
