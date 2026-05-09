"""InThreadDispatcher: runs callables synchronously in the calling thread."""
from __future__ import annotations

import pytest

from sakura.dispatch.in_thread import InThreadDispatcher


def _double(x):
    return x * 2


def _add(a, b, *, scale=1):
    return (a + b) * scale


def _raises():
    raise ValueError("intentional")


class TestInThreadDispatcher:
    def test_submit_runs_synchronously_and_returns_done_future(self):
        d = InThreadDispatcher()
        fut = d.submit(_double, 21)
        assert fut.done() is True
        result = fut.result()
        assert result.value == 42
        assert result.elapsed_us >= 0

    def test_submit_propagates_kwargs(self):
        d = InThreadDispatcher()
        fut = d.submit(_add, 3, 4, scale=10)
        assert fut.result().value == 70

    def test_submit_propagates_exceptions_via_result(self):
        d = InThreadDispatcher()
        fut = d.submit(_raises)
        with pytest.raises(ValueError, match="intentional"):
            fut.result()

    def test_cancel_after_done_returns_false(self):
        d = InThreadDispatcher()
        fut = d.submit(_double, 5)
        assert fut.cancel() is False

    def test_shutdown_is_safe_noop(self):
        d = InThreadDispatcher()
        d.shutdown()
