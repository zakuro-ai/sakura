"""Compile: torch.compile lazy wrapping + cache."""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from sakura.events import OnTrainBegin, OnTrainStepBegin
from sakura.runtime import SakuraRuntime
from sakura.services.compile import Compile


class _Lin(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = torch.nn.Linear(4, 2)

    def forward(self, x):
        return self.lin(x)


class TestCompile:
    def test_priority_and_name(self):
        s = Compile()
        assert s.priority == 20
        assert s.name == "compile"

    def test_compile_wraps_model_at_train_begin(self):
        model = _Lin()
        s = Compile(mode="default")  # default mode is fastest to compile
        rt = SakuraRuntime()
        rt.install(s)
        rt.dispatch(OnTrainBegin(model=model, optimizer=None, train_loader=None,
                                  val_loader=None, rank=0, world_size=1))
        # The service stores the original-vs-compiled mapping.
        assert s.compiled_called is True
        # First step records compile_secs.
        rt.dispatch(OnTrainStepBegin(model=model, batch=None, step=0,
                                      rank=0, world_size=1))
        assert s.first_step_secs is not None or s.first_step_secs == 0  # initialized
