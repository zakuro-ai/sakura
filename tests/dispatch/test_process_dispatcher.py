"""ProcessDispatcher — persistent spawned worker with shared-memory tensor transfer."""
from __future__ import annotations

import time

import pytest
import torch

from sakura.dispatch import ProcessDispatcher, worker_state


def _double(x):
    return x * 2


def _boom():
    raise ValueError("nope")


def _init(factor):
    return {"factor": factor}


def _use_state(x):
    return x * worker_state()["init"]["factor"]


def _sum_tensors(sd):
    return {k: float(v.sum()) for k, v in sd.items()}


def _bump_in_place(t):
    t += 1  # mutates the worker's mapping only if it were shared writably
    return float(t.sum())


def _busy(seconds):
    end = time.perf_counter() + seconds
    n = 0
    while time.perf_counter() < end:
        n += 1
    return n


@pytest.fixture(scope="module")
def disp():
    d = ProcessDispatcher(initializer=_init, initargs=(3,))
    d.wait_ready(120)
    yield d
    d.shutdown()


def test_round_trip(disp):
    assert disp.submit(_double, 21).result(30).value == 42


def test_closures_and_lambdas(disp):
    k = 5
    assert disp.submit(lambda x: x + k, 1).result(30).value == 6


def test_initializer_state(disp):
    assert disp.submit(_use_state, 7).result(30).value == 21


def test_exception_type_is_preserved(disp):
    with pytest.raises(ValueError, match="nope"):
        disp.submit(_boom).result(30)


def test_tensor_state_dict_round_trip(disp):
    sd = {"a": torch.ones(4, 4), "b": torch.arange(10, dtype=torch.float32)}
    assert disp.submit(_sum_tensors, sd).result(30).value == {"a": 16.0, "b": 45.0}


def test_elapsed_us_is_reported(disp):
    assert disp.submit(_double, 1).result(30).elapsed_us >= 0


def test_runs_in_parallel_without_gil_contention(disp):
    # A GIL-bound busy loop in the worker must not slow a busy loop in the parent.
    solo = _busy(0.5)
    fut = disp.submit(_busy, 1.0)
    time.sleep(0.2)
    shared = _busy(0.5)
    fut.result(30)
    assert shared > 0.6 * solo, (solo, shared)


def test_done_and_stats(disp):
    fut = disp.submit(_double, 2)
    fut.result(30)
    assert fut.done()
    assert disp.stats()["kind"] == "process"


def test_initializer_failure_surfaces():
    d = ProcessDispatcher(initializer=_boom)
    try:
        with pytest.raises(RuntimeError, match="initializer failed"):
            d.wait_ready(120)
    finally:
        d.shutdown()


def test_dead_worker_fails_pending_futures():
    d = ProcessDispatcher()
    d.wait_ready(120)
    fut = d.submit(_busy, 30)
    time.sleep(0.3)
    d._proc.kill()
    with pytest.raises(RuntimeError):
        fut.result(30)
    d.shutdown()


def test_shutdown_is_idempotent_and_blocks_submit():
    d = ProcessDispatcher()
    d.wait_ready(120)
    d.shutdown()
    d.shutdown()
    with pytest.raises(RuntimeError, match="shutdown"):
        d.submit(_double, 1)


def _echo(x):
    return x


def _count_open_fds():
    import os

    return len(os.listdir("/proc/self/fd")) if os.path.isdir("/proc/self/fd") else -1


def test_nested_structure_and_dtypes_round_trip(disp):
    payload = {
        "w": torch.randn(3, 4),
        "h": torch.randn(5).half(),
        "i": torch.arange(7),
        "b": torch.tensor([True, False]),
        "bf": torch.randn(2, 2).bfloat16(),
        "empty": torch.empty(0, 3),
        "scalar": torch.tensor(3.5),
        "nested": [{"x": torch.ones(2)}, (torch.zeros(1), 5, "s")],
        "meta": {"epoch": 3, "name": "m"},
    }
    out = disp.submit(_echo, payload).result(60).value
    for k in ("w", "h", "i", "b", "bf", "empty", "scalar"):
        assert out[k].dtype == payload[k].dtype and out[k].shape == payload[k].shape
        assert torch.equal(out[k], payload[k])
    assert torch.equal(out["nested"][0]["x"], torch.ones(2))
    assert out["nested"][1][1:] == (5, "s") or tuple(out["nested"][1][1:]) == (5, "s")
    assert out["meta"] == {"epoch": 3, "name": "m"}


def test_many_tensors_do_not_exhaust_file_descriptors(disp):
    # 1500 tensors x 5 messages would need ~7500 fds with per-tensor sharing.
    sd = {f"p{i}": torch.randn(8) for i in range(1500)}
    for _ in range(5):
        out = disp.submit(_sum_tensors, sd).result(60).value
        assert len(out) == 1500


def test_wait_ready_raises_if_worker_dies_during_startup():
    import os

    def _die():
        os._exit(1)

    d = ProcessDispatcher(initializer=_die)
    try:
        with pytest.raises(RuntimeError):
            d.wait_ready(120)
    finally:
        d.shutdown()


def _save_to(sd, path):
    torch.save(sd, path)
    return path


def test_received_tensors_can_be_torch_saved(disp, tmp_path):
    sd = {"w": torch.randn(4, 4), "h": torch.randn(3).half(), "i": torch.arange(5)}
    path = disp.submit(_save_to, sd, str(tmp_path / "sd.pt")).result(60).value
    loaded = torch.load(path)
    assert all(torch.equal(loaded[k], sd[k]) for k in sd)


def test_construction_does_not_block_on_a_large_initializer_payload():
    big = torch.zeros(40_000_000 // 4)  # ~40 MB, far above the spawn pipe buffer
    t0 = time.perf_counter()
    d = ProcessDispatcher(initializer=_echo, initargs=(big,))
    built = time.perf_counter() - t0
    try:
        d.wait_ready(120)
        assert built < 1.5, built  # constructing must not wait for the child's imports
    finally:
        d.shutdown()
