"""End-to-end Plan 2 acceptance test.

Top-level integration: instantiate SakuraRuntime, install a service that
dispatches a cloudpickled callable via LocalDispatcher, dispatch an
OnEpochEnd event, verify the service got the result back.
"""
from __future__ import annotations

import numpy as np
import pytest

sakura_wire = pytest.importorskip("sakura_wire")

from sakura import (
    BaseService,
    Compute,
    OnEpochEnd,
    SakuraRuntime,
)
from sakura.dispatch.local import LocalDispatcher


def _compute_loss(x, y):
    """A trivial 'eval' that runs on the worker."""
    return {"loss": float(np.mean((x - y) ** 2))}


class _DispatchOnEpochEnd(BaseService):
    """Toy service: every OnEpochEnd, dispatch _compute_loss to the worker."""
    name = "dispatch_on_epoch_end"
    priority = 80

    def __init__(self, dispatcher):
        super().__init__()
        self._d = dispatcher
        self.results: list[dict] = []

    def on_epoch_end(self, event):
        x = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        y = np.array([1.5, 2.5, 3.5], dtype=np.float32)
        fut = self._d.submit(_compute_loss, x, y)
        self.results.append(fut.result(timeout=10.0).value)


def test_e2e_dispatch_via_runtime_and_local_dispatcher():
    """Full Plan 2 acceptance loop."""
    dispatcher = LocalDispatcher()
    try:
        rt = SakuraRuntime(compute=Compute.local())
        service = _DispatchOnEpochEnd(dispatcher)
        rt.install(service)
        with rt:
            rt.dispatch(OnEpochEnd(epoch=0, model="m", optimizer="o", metrics={},
                                    rank=0, world_size=1))
            rt.dispatch(OnEpochEnd(epoch=1, model="m", optimizer="o", metrics={},
                                    rank=0, world_size=1))
        assert len(service.results) == 2
        for r in service.results:
            assert "loss" in r
            assert abs(r["loss"] - 0.25) < 1e-6
    finally:
        dispatcher.shutdown()
