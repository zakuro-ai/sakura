"""sakura-bench CLI smoke tests."""
from __future__ import annotations

import json
import subprocess
import sys

import pytest

torch = pytest.importorskip("torch")


def test_cli_run_mnist_baseline_writes_report(tmp_path):
    """`sakura-bench run --workload mnist-mlp --runner baseline` writes a RunReport JSON."""
    out_dir = tmp_path / "reports"
    out_dir.mkdir()
    rc = subprocess.run(
        [sys.executable, "-m", "sakura.bench", "run",
          "--workload", "mnist-mlp",
          "--runner", "baseline",
          "--framework", "pytorch-ddp",
          "--output", str(out_dir)],
        capture_output=True, text=True,
    )
    assert rc.returncode == 0, f"stderr:\n{rc.stderr}"
    files = list(out_dir.glob("*.json"))
    assert len(files) == 1
    report = json.loads(files[0].read_text())
    assert report["workload"] == "mnist-mlp"
    assert report["framework"] == "pytorch-ddp"
    assert report["elapsed_secs"] > 0


def test_cli_compare_two_reports(tmp_path):
    """sakura-bench compare prints a speedup summary."""
    from sakura.bench.harness import RunReport

    a = RunReport(workload="x", framework="pytorch-ddp", elapsed_secs=10.0,
                   samples_per_sec=100, final_metrics={"val_acc": 0.5})
    b = RunReport(workload="x", framework="pytorch-ddp", elapsed_secs=5.0,
                   samples_per_sec=200, final_metrics={"val_acc": 0.5},
                   sakura_services=["MixedPrecision"])
    pa = tmp_path / "a.json"
    pb = tmp_path / "b.json"
    pa.write_text(a.to_json())
    pb.write_text(b.to_json())

    rc = subprocess.run(
        [sys.executable, "-m", "sakura.bench", "compare", str(pa), str(pb)],
        capture_output=True, text=True,
    )
    assert rc.returncode == 0, f"stderr:\n{rc.stderr}"
    assert "sakura" in rc.stdout.lower() or "x:" in rc.stdout
    assert "2.00x" in rc.stdout or "50.0%" in rc.stdout  # 10s → 5s = 2x speedup


def test_cli_export_markdown_table(tmp_path):
    """sakura-bench export renders a markdown table."""
    from sakura.bench.harness import RunReport

    r = RunReport(workload="cifar10-resnet50", framework="pytorch-ddp",
                   elapsed_secs=42.5, samples_per_sec=2716,
                   peak_gpu_mem_mb=1830, final_metrics={"val_acc": 0.914})
    p = tmp_path / "r.json"
    p.write_text(r.to_json())

    rc = subprocess.run(
        [sys.executable, "-m", "sakura.bench", "export", str(p)],
        capture_output=True, text=True,
    )
    assert rc.returncode == 0, f"stderr:\n{rc.stderr}"
    assert "| workload " in rc.stdout
    assert "cifar10-resnet50" in rc.stdout
