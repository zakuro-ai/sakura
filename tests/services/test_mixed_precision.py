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

    def test_bf16_wrap_loss_is_passthrough_optimizer_step_returns_false(self):
        """bf16 has no scaler; loss is unwrapped and the loop owns opt.step()."""
        s = MixedPrecision(dtype="bf16")
        # No on_train_begin needed — no scaler regardless of CUDA.
        loss = torch.tensor(2.0)
        assert s.wrap_loss(loss) is loss  # passthrough (same object)
        opt = _DummyOptimizer()
        assert s.optimizer_step(opt) is False
        assert opt.stepped == 0  # service did not step

    def test_fp16_wrap_loss_and_optimizer_step_with_fake_scaler(self):
        """fp16 path: wrap_loss scales, optimizer_step calls scaler.step+update.

        Validates the protocol on CPU by injecting a fake scaler — the real
        GradScaler is CUDA-gated, so we exercise the contract here. Real
        CUDA validation runs when a GPU is available.
        """
        class _FakeScaler:
            def __init__(self):
                self.scaled_calls = 0
                self.step_calls = 0
                self.update_calls = 0

            def scale(self, loss):
                self.scaled_calls += 1
                return loss * 1024.0  # canonical "fp16 init scale"

            def step(self, opt):
                self.step_calls += 1
                # Real scaler calls opt.step() iff grads finite — fake always does.
                opt.step()

            def update(self):
                self.update_calls += 1

            def unscale_(self, opt):
                pass

        s = MixedPrecision(dtype="fp16")
        s._scaler = _FakeScaler()  # bypass on_train_begin (CUDA-gated)

        # wrap_loss
        loss = torch.tensor(2.0)
        scaled = s.wrap_loss(loss)
        assert float(scaled) == pytest.approx(2048.0)
        assert s._scaler.scaled_calls == 1

        # optimizer_step
        opt = _DummyOptimizer()
        assert s.optimizer_step(opt) is True
        assert s._scaler.step_calls == 1
        assert s._scaler.update_calls == 1
        assert opt.stepped == 1  # scaler.step internally called opt.step()

    def test_fp16_loop_integration_no_double_step_via_runtime(self):
        """End-to-end: runtime.optimizer_step returns True for fp16 — loop must not double-step.

        Mirrors the real bench-harness loop semantics:
            adapter.on_optimizer_step(opt)            # unscale + grad-clip
            if not rt.optimizer_step(opt): opt.step() # scaler.step or fallback

        With a fake scaler installed, scaler.step calls opt.step exactly
        once — and the loop's fallback is skipped. Total: 1 step per batch.
        """
        class _FakeScaler:
            def scale(self, loss): return loss
            def step(self, opt): opt.step()
            def update(self): pass
            def unscale_(self, opt): pass

        opt = _DummyOptimizer()
        s = MixedPrecision(dtype="fp16")
        s._scaler = _FakeScaler()
        rt = SakuraRuntime()
        rt.install(s)

        # Pre-step (unscale + clip — no-op here since fake unscale_ is no-op)
        rt.dispatch(OnOptimizerStep(optimizer=opt, rank=0, world_size=1))
        # Step — runtime delegates to MixedPrecision; returns True.
        handled = rt.optimizer_step(opt)
        if not handled:
            opt.step()  # would double-step under the old contract

        assert handled is True
        assert opt.stepped == 1  # exactly one step per batch — no double
