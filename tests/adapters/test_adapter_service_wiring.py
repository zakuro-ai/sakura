"""Service+adapter wiring: MixedPrecision wraps forward in real autocast."""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from sakura.adapters.ddp import DDPAdapter
from sakura.events import OnTrainBegin, OnTrainEnd
from sakura.runtime import SakuraRuntime
from sakura.services.mixed_precision import MixedPrecision


class _M(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.l = torch.nn.Linear(4, 2)

    def forward(self, x):
        return self.l(x)


def test_mixed_precision_actually_wraps_forward_at_train_begin():
    rt = SakuraRuntime()
    mp = MixedPrecision(dtype="bf16")
    rt.install(mp)
    adapter = DDPAdapter(rt, rank=0, world_size=1)

    model = _M()
    original_forward_fn = model.forward.__func__  # underlying function
    adapter.on_train_begin(model, optimizer=None, train_loader=None)

    # forward must now be the autocast wrapper (instance attribute, not a bound method).
    assert model.forward is not model.__class__.forward

    x = torch.randn(2, 4)
    y = model(x)
    assert y.shape == (2, 2)

    adapter.on_train_end(model)
    # After restore: the instance-attr wrapper is gone; class forward is back.
    assert model.forward.__func__ is original_forward_fn


def test_ddp_adapter_with_telemetry_and_async_eval():
    """Full Plan 4 acceptance: DDPAdapter + Telemetry + AsyncEval (in-thread dispatcher)."""
    from sakura.dispatch.in_thread import InThreadDispatcher
    from sakura.services.async_eval import AsyncEval
    from sakura.services.telemetry import Telemetry

    sink: list[dict] = []
    rt = SakuraRuntime()
    rt.install(Telemetry(output=sink.append))

    def _eval(epoch, payload):
        return {"val_loss": 1.0 / (epoch + 1)}

    rt.install(AsyncEval(eval_fn=_eval, eval_payload={},
                          dispatcher=InThreadDispatcher()))

    adapter = DDPAdapter(rt, rank=0, world_size=1)
    adapter.on_train_begin(model=_M(), optimizer=None, train_loader=None)
    for epoch in range(3):
        adapter.on_epoch_begin(epoch)
        adapter.on_epoch_end(epoch, model=_M(), optimizer=None,
                             metrics={"train_loss": 1.5 / (epoch + 1)})
    adapter.on_train_end(model=_M())

    event_types = [r["event"] for r in sink]
    assert event_types.count("OnTrainBegin") == 1
    assert event_types.count("OnEpochBegin") == 3
    assert event_types.count("OnEpochEnd") == 3
    assert event_types.count("OnTrainEnd") == 1

    eval_svc = next(s for s in rt.services if s.name == "async_eval")
    assert len(eval_svc.history) == 3
