"""KernelOpt: cuDNN/TF32/channels_last backend knobs, set then restored."""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from sakura.events import OnTrainBegin, OnTrainEnd
from sakura.runtime import SakuraRuntime
from sakura.services.kernel_opt import KernelOpt


class _RNNModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.rnn = torch.nn.GRU(4, 4, batch_first=True)

    def forward(self, x):
        return self.rnn(x)[0]


def _begin(model):
    return OnTrainBegin(model=model, optimizer="o", train_loader=None,
                        val_loader=None, rank=0, world_size=1)


def _end(model):
    return OnTrainEnd(model=model, history=[], rank=0, world_size=1)


def test_priority_and_name():
    s = KernelOpt()
    assert s.name == "kernel_opt"
    assert s.priority == 5


def test_sets_and_restores_cudnn_benchmark():
    prior = torch.backends.cudnn.benchmark
    torch.backends.cudnn.benchmark = False
    try:
        s = KernelOpt(cudnn_benchmark=True, tf32=False, flatten_rnn=False)
        rt = SakuraRuntime()
        rt.install(s)
        model = _RNNModel()
        rt.dispatch(_begin(model))
        assert torch.backends.cudnn.benchmark is True  # turned on for training
        rt.dispatch(_end(model))
        assert torch.backends.cudnn.benchmark is False  # restored to the value at begin
    finally:
        torch.backends.cudnn.benchmark = prior


def test_no_benchmark_leaves_it_off():
    prior = torch.backends.cudnn.benchmark
    torch.backends.cudnn.benchmark = False
    try:
        s = KernelOpt(cudnn_benchmark=False, tf32=False, flatten_rnn=False)
        rt = SakuraRuntime()
        rt.install(s)
        model = _RNNModel()
        rt.dispatch(_begin(model))
        assert torch.backends.cudnn.benchmark is False
        rt.dispatch(_end(model))
    finally:
        torch.backends.cudnn.benchmark = prior


def test_flatten_rnn_runs_without_error():
    s = KernelOpt(cudnn_benchmark=False, tf32=False, flatten_rnn=True)
    rt = SakuraRuntime()
    rt.install(s)
    model = _RNNModel()
    rt.dispatch(_begin(model))  # calls flatten_parameters() on the GRU
    rt.dispatch(_end(model))
