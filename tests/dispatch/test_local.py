"""LocalDispatcher: spawns localhost sakura-worker on first use."""
from __future__ import annotations

import pytest

sakura_wire = pytest.importorskip("sakura_wire")

from sakura.dispatch.local import LocalDispatcher


def _add(a, b):
    return int(a.sum()) + int(b.sum())


def test_local_dispatcher_auto_spawns_and_round_trips():
    import numpy as np

    d = LocalDispatcher()
    try:
        fut = d.submit(_add,
                       np.array([1, 2, 3], dtype=np.int64),
                       np.array([10, 20, 30], dtype=np.int64))
        result = fut.result(timeout=10.0)
        assert result.value == 6 + 60
    finally:
        d.shutdown()


def test_local_dispatcher_shutdown_idempotent():
    d = LocalDispatcher()
    d.shutdown()
    d.shutdown()
