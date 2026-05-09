"""ActivationCheckpoint: wrap matching submodules with torch.utils.checkpoint."""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from sakura.events import OnTrainBegin
from sakura.runtime import SakuraRuntime
from sakura.services.activation_checkpoint import ActivationCheckpoint


class _Block(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = torch.nn.Linear(4, 4)

    def forward(self, x):
        return self.lin(x)


class _Net(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.b1 = _Block()
        self.b2 = _Block()
        self.b3 = _Block()


class TestActivationCheckpoint:
    def test_priority_is_15(self):
        s = ActivationCheckpoint(target_types=(_Block,))
        assert s.priority == 15
        assert s.name == "activation_checkpoint"

    def test_wraps_all_target_modules(self):
        s = ActivationCheckpoint(target_types=(_Block,), selective=True)
        net = _Net()
        rt = SakuraRuntime()
        rt.install(s)
        rt.dispatch(OnTrainBegin(model=net, optimizer=None, train_loader=None,
                                  val_loader=None, rank=0, world_size=1))
        # All 3 blocks should have their forward wrapped.
        assert s.wrapped_count == 3

    def test_selective_int_wraps_every_n(self):
        s = ActivationCheckpoint(target_types=(_Block,), selective=2)
        net = _Net()
        rt = SakuraRuntime()
        rt.install(s)
        rt.dispatch(OnTrainBegin(model=net, optimizer=None, train_loader=None,
                                  val_loader=None, rank=0, world_size=1))
        # Wraps every 2nd block: b1 (idx 0), b3 (idx 2) → 2 wrapped.
        assert s.wrapped_count == 2
