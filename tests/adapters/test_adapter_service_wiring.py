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
