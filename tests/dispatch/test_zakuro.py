"""ZakuroDispatcher: wraps zakuro.Compute. Skipped when zakuro-ai is not installed."""
from __future__ import annotations

import pytest

zakuro = pytest.importorskip("zakuro")

from sakura.dispatch.zakuro import ZakuroDispatcher


def _square(x):
    return int(x.sum()) ** 2


def test_zakuro_dispatcher_passes_through_to_zk_compute():
    """End-to-end: real ZakuroDispatcher uses zk.Compute() (standalone fallback)."""
    import numpy as np

    zk_compute = zakuro.Compute()  # standalone in-process fallback
    d = ZakuroDispatcher(zk_compute)
    fut = d.submit(_square, np.array([1, 2, 3], dtype=np.int64))
    result = fut.result(timeout=5.0)
    assert result.value == (1 + 2 + 3) ** 2
