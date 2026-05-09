"""ThreadDispatcher — Python thread pool for in-process parallelism."""
from __future__ import annotations

import threading
import time

import pytest

from sakura.dispatch import ThreadDispatcher


def test_submit_returns_value():
    disp = ThreadDispatcher()
    fut = disp.submit(lambda x: x * 2, 21)
    assert fut.result().value == 42
    disp.shutdown()


def test_submit_propagates_exception():
    disp = ThreadDispatcher()

    def boom():
        raise RuntimeError("nope")

    fut = disp.submit(boom)
    with pytest.raises(RuntimeError, match="nope"):
        fut.result()
    disp.shutdown()


def test_done_reflects_completion():
    disp = ThreadDispatcher()
    started = threading.Event()
    finish = threading.Event()

    def slow():
        started.set()
        finish.wait(timeout=2.0)
        return 99

    fut = disp.submit(slow)
    started.wait(timeout=2.0)
    assert fut.done() is False
    finish.set()
    assert fut.result().value == 99
    assert fut.done() is True
    disp.shutdown()


def test_runs_concurrently_with_caller():
    """A long-running submit doesn't block the caller — that's the entire
    point of dispatching to a thread.
    """
    disp = ThreadDispatcher()

    def slow():
        time.sleep(0.2)
        return "done"

    t0 = time.perf_counter()
    fut = disp.submit(slow)
    submit_ms = (time.perf_counter() - t0) * 1000
    assert submit_ms < 50, f"submit blocked for {submit_ms:.0f}ms — should be near-instant"
    assert fut.result(timeout=2.0).value == "done"
    disp.shutdown()


def test_shutdown_rejects_subsequent_submits():
    disp = ThreadDispatcher()
    disp.shutdown()
    with pytest.raises(RuntimeError, match="shutdown"):
        disp.submit(lambda: None)


def test_stats_reports_kind():
    disp = ThreadDispatcher()
    assert disp.stats() == {"kind": "thread"}
    disp.shutdown()


def test_max_workers_bounds_in_flight():
    """With max_workers=1, the second submission queues behind the first."""
    disp = ThreadDispatcher(max_workers=1)
    started = threading.Event()
    release = threading.Event()

    def first():
        started.set()
        release.wait(timeout=2.0)
        return 1

    def second():
        return 2

    f1 = disp.submit(first)
    started.wait(timeout=2.0)
    # second is queued; not running yet
    f2 = disp.submit(second)
    assert f2.done() is False
    release.set()
    assert f1.result(timeout=2.0).value == 1
    assert f2.result(timeout=2.0).value == 2
    disp.shutdown()
