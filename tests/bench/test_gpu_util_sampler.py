"""_GpuUtilSampler + nvidia-smi line parser tests.

None of these need a real GPU: the no-CUDA path is forced via monkeypatch,
and the reader-thread test drives _read_smi against a `cat` subprocess that
replays a captured nvidia-smi stream.
"""
from __future__ import annotations

import subprocess

import pytest

from sakura.bench.harness import _GpuUtilSampler, _parse_smi_line


def test_parse_smi_line_valid():
    assert _parse_smi_line("37, 1234") == (37.0, 1234.0)
    assert _parse_smi_line("0, 5") == (0.0, 5.0)
    assert _parse_smi_line("100, 10989") == (100.0, 10989.0)


def test_parse_smi_line_rejects_garbage():
    assert _parse_smi_line("") is None
    assert _parse_smi_line("\n") is None
    assert _parse_smi_line("garbage") is None
    assert _parse_smi_line("[N/A], [N/A]") is None  # driver may emit N/A


def test_gpu_util_sampler_no_cuda_yields_zeros(monkeypatch):
    # Force the no-CUDA path regardless of the host (the dev host may have a real GPU).
    monkeypatch.setattr("sakura.bench.harness._cuda_available", lambda: False)
    with _GpuUtilSampler() as s:
        pass  # never starts a thread, never raises
    assert s.mean_pct == 0.0
    assert s.max_pct == 0.0
    assert s.n_samples == 0
    assert s.peak_mem_mb == 0.0


def test_gpu_util_sampler_reader_parses_and_aggregates():
    """Drive _read_smi against a fake nvidia-smi stdout (a `cat` subprocess);
    proves the reader parses, drops N/A lines, and aggregates correctly."""
    s = _GpuUtilSampler()
    s._proc = subprocess.Popen(
        ["cat"], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True,
    )
    s._proc.stdin.write("37, 1234\n50, 1300\n[N/A], [N/A]\n88, 1400\n")
    s._proc.stdin.close()
    s._read_smi()  # drains until EOF
    s._proc.wait(timeout=2.0)  # reap the cat subprocess (no zombie)
    assert s.n_samples == 3            # 3 valid rows, the N/A row dropped
    assert s.max_pct == 88.0
    assert s.mean_pct == pytest.approx((37 + 50 + 88) / 3)
    assert s.peak_mem_mb == 1400.0


def test_gpu_util_sampler_smi_path_cleanup_ordering(monkeypatch):
    """Force the smi fallback through the context manager and prove the
    cleanup ordering: __exit__ terminates the subprocess FIRST so the reader
    thread (blocked in `for line in proc.stdout:`) unblocks on EOF and joins
    quickly. This is the test that catches the join-before-terminate bug:
    if __exit__ joined first it would burn the full 2s timeout and the thread
    would still be alive on return."""
    import time as _time

    monkeypatch.setattr("sakura.bench.harness._cuda_available", lambda: True)
    # Make pynvml unavailable so __enter__ falls through to the smi backend.
    monkeypatch.setattr(
        _GpuUtilSampler, "_start_pynvml", lambda self: False
    )
    # Replace the real nvidia-smi launch with a fake producer that streams
    # parseable lines forever (so the reader thread is genuinely blocked on
    # stdout, exactly like the real -lms subprocess).
    def fake_start_smi(self):
        self._proc = subprocess.Popen(
            ["bash", "-c", "while true; do echo '50, 100'; sleep 0.05; done"],
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True,
        )
        return True

    monkeypatch.setattr(_GpuUtilSampler, "_start_smi", fake_start_smi)

    s = _GpuUtilSampler(interval_s=0.05)
    with s:
        assert s._backend == "smi"
        _time.sleep(0.2)  # let it sample a few lines
        t_exit = _time.perf_counter()
    elapsed = _time.perf_counter() - t_exit

    assert elapsed < 1.0, f"__exit__ too slow ({elapsed:.2f}s) — join-before-terminate?"
    assert s._thread is not None and not s._thread.is_alive()
    assert s.n_samples > 0
    assert s.mean_pct == pytest.approx(50.0)
    assert s.peak_mem_mb == pytest.approx(100.0)


def _tiny_workload(H, torch):
    def make_model():
        return torch.nn.Sequential(
            torch.nn.Linear(8, 16), torch.nn.ReLU(), torch.nn.Linear(16, 4),
        )

    def make_loader():
        torch.manual_seed(0)
        ds = torch.utils.data.TensorDataset(
            torch.randn(16, 8), torch.randint(0, 4, (16,)),
        )
        return torch.utils.data.DataLoader(ds, batch_size=8)

    def eval_fn(model, loader):
        return {"val_acc": 0.0}

    return H.Workload(
        name="tiny-sampler-wire", tier="smoke", make_model=make_model,
        make_train_loader=make_loader, make_val_loader=make_loader,
        eval_fn=eval_fn, epochs=1,
    )


class _FakeSampler:
    """Stand-in for _GpuUtilSampler with deterministic sentinel results, so
    the test proves the runner enters the sampler and threads its results
    into RunReport (independent of any real GPU)."""
    mean_pct = 42.0
    max_pct = 99.0
    n_samples = 7
    peak_mem_mb = 123.0

    def __init__(self, device=0, interval_s=0.1):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return None


def test_baseline_runner_threads_sampler_into_report(monkeypatch):
    torch = pytest.importorskip("torch")
    import sakura.bench.harness as H
    monkeypatch.setattr(H, "_GpuUtilSampler", _FakeSampler)
    report = H.BaselineRunner(framework="pytorch-ddp").run(_tiny_workload(H, torch))
    assert report.gpu_util_mean_pct == 42.0
    assert report.gpu_util_max_pct == 99.0
    assert report.gpu_util_samples == 7
    assert report.gpu_mem_used_peak_mb == 123.0


def test_sakura_runner_threads_sampler_into_report(monkeypatch):
    torch = pytest.importorskip("torch")
    import sakura.bench.harness as H
    monkeypatch.setattr(H, "_GpuUtilSampler", _FakeSampler)
    # No services installed: exercises the plain adapter loop (no async bridge).
    report = H.SakuraRunner(framework="pytorch-ddp", services=[]).run(_tiny_workload(H, torch))
    assert report.gpu_util_mean_pct == 42.0
    assert report.gpu_util_max_pct == 99.0
    assert report.gpu_util_samples == 7
    assert report.gpu_mem_used_peak_mb == 123.0
