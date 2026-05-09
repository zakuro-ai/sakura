"""ZeRO1: sharded optimizer wrap (Plan 3: single-rank correctness only)."""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from sakura.events import OnOptimizerStep, OnTrainBegin
from sakura.runtime import SakuraRuntime
from sakura.services.zero1 import ZeRO1


class _Lin(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = torch.nn.Linear(4, 2)


class TestZeRO1SingleRank:
    def test_priority_and_name(self):
        s = ZeRO1()
        assert s.priority == 30
        assert s.name == "zero1"

    def test_wraps_optimizer_at_train_begin_and_steps(self):
        model = _Lin()
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        s = ZeRO1()
        rt = SakuraRuntime()
        rt.install(s)
        rt.dispatch(OnTrainBegin(model=model, optimizer=opt, train_loader=None,
                                  val_loader=None, rank=0, world_size=1))
        # Make a fake gradient so step does something.
        for p in model.parameters():
            p.grad = torch.zeros_like(p)
        # Step via the event; world_size=1 means no actual sharding happens —
        # just a passthrough — but the wrap must not break.
        rt.dispatch(OnOptimizerStep(optimizer=opt, rank=0, world_size=1))
        # Optimizer should have stepped without error.

    def test_world_size_1_is_passthrough(self):
        """world_size=1 → no sharding. ZeRO1 wraps but step is identical to opt.step()."""
        model = _Lin()
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        s = ZeRO1()
        rt = SakuraRuntime()
        rt.install(s)
        rt.dispatch(OnTrainBegin(model=model, optimizer=opt, train_loader=None,
                                  val_loader=None, rank=0, world_size=1))
        # Capture pre-step state.
        for p in model.parameters():
            p.grad = torch.ones_like(p)
        before = [p.detach().clone() for p in model.parameters()]
        rt.dispatch(OnOptimizerStep(optimizer=opt, rank=0, world_size=1))
        after = [p.detach().clone() for p in model.parameters()]
        # Params should have moved (Adam step took effect):
        assert any(not torch.allclose(b, a) for b, a in zip(before, after))
