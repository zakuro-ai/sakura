"""MixedPrecision: autocast wrapping + GradScaler for fp16."""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from sakura.events import OnOptimizerStep, OnTrainBegin, OnTrainStepBegin
from sakura.runtime import SakuraRuntime
from sakura.services.mixed_precision import MixedPrecision


class _DummyParam:
    def __init__(self, dtype=torch.float32):
        self.dtype = dtype
        self.grad = torch.zeros(1)


class _DummyModel:
    def parameters(self):
        return iter([_DummyParam()])

    def forward(self, *args, **kwargs):
        return None


class _DummyOptimizer:
    def __init__(self):
        self.stepped = 0
        self.zero_graded = 0

    def step(self, closure=None):
        self.stepped += 1

    def zero_grad(self, set_to_none=True):
        self.zero_graded += 1


class TestMixedPrecision:
    def test_priority_is_10(self):
        s = MixedPrecision(dtype="bf16")
        assert s.priority == 10
        assert s.name == "mixed_precision"

    def test_bf16_does_not_use_grad_scaler(self):
        opt = _DummyOptimizer()
        s = MixedPrecision(dtype="bf16")
        rt = SakuraRuntime()
        rt.install(s)
        rt.dispatch(OnTrainBegin(model=_DummyModel(), optimizer=opt, train_loader=None,
                                  val_loader=None, rank=0, world_size=1))
        rt.dispatch(OnTrainStepBegin(model=_DummyModel(), batch=("x",), step=0,
                                      rank=0, world_size=1))
        rt.dispatch(OnOptimizerStep(optimizer=opt, rank=0, world_size=1))
        # MixedPrecision must NOT call opt.step() — the framework's training loop
        # is responsible for stepping. Calling it here would double-step.
        assert opt.stepped == 0
        assert s._scaler is None  # bf16 = no scaler

    def test_fp16_creates_grad_scaler_if_cuda_available(self):
        s = MixedPrecision(dtype="fp16")
        if torch.cuda.is_available():
            opt = _DummyOptimizer()
            rt = SakuraRuntime()
            rt.install(s)
            rt.dispatch(OnTrainBegin(model=_DummyModel(), optimizer=opt, train_loader=None,
                                      val_loader=None, rank=0, world_size=1))
            assert s._scaler is not None
        else:
            # On CPU, MixedPrecision(dtype="fp16") still installs but the scaler is a no-op
            # (autocast on CPU with fp16 is supported via torch.autocast(device_type='cpu')).
            pass

    def test_invalid_dtype_raises_at_install(self):
        rt = SakuraRuntime()
        with pytest.raises(ValueError, match="dtype"):
            rt.install(MixedPrecision(dtype="bogus"))
