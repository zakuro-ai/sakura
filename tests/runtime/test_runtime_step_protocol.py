"""SakuraRuntime.scale_loss + optimizer_step protocol.

These methods give services a chance to wrap the training loss before
backward (e.g., GradScaler.scale) and to drive opt.step() themselves
(e.g., GradScaler.step+update for fp16). Services that don't implement
these methods are silently skipped.
"""
from __future__ import annotations

from sakura.runtime import SakuraRuntime
from sakura.service import BaseService


class _NoopService(BaseService):
    name = "noop"
    priority = 50

    def on_event(self, event):
        pass


class _LossDoubler(BaseService):
    name = "doubler"
    priority = 10

    def on_event(self, event):
        pass

    def wrap_loss(self, loss):
        return loss * 2


class _LossAdder(BaseService):
    name = "adder"
    priority = 20

    def on_event(self, event):
        pass

    def wrap_loss(self, loss):
        return loss + 1


class _Stepper(BaseService):
    name = "stepper"
    priority = 10

    def __init__(self):
        super().__init__()
        self.calls = 0

    def on_event(self, event):
        pass

    def optimizer_step(self, optimizer) -> bool:
        self.calls += 1
        return True


class _NonStepper(BaseService):
    name = "non_stepper"
    priority = 20

    def __init__(self):
        super().__init__()
        self.calls = 0

    def on_event(self, event):
        pass

    def optimizer_step(self, optimizer) -> bool:
        self.calls += 1
        return False


class TestScaleLoss:
    def test_no_services_returns_loss_unchanged(self):
        rt = SakuraRuntime()
        assert rt.scale_loss(5.0) == 5.0

    def test_service_without_wrap_loss_is_passthrough(self):
        rt = SakuraRuntime()
        rt.install(_NoopService())
        assert rt.scale_loss(5.0) == 5.0

    def test_single_service_wraps_loss(self):
        rt = SakuraRuntime()
        rt.install(_LossDoubler())
        assert rt.scale_loss(5.0) == 10.0

    def test_multiple_services_compose_in_priority_order(self):
        # Doubler (prio 10) runs first → 5*2=10; Adder (prio 20) runs second → 10+1=11
        rt = SakuraRuntime()
        rt.install(_LossDoubler())
        rt.install(_LossAdder())
        assert rt.scale_loss(5.0) == 11.0

    def test_failing_wrap_loss_does_not_break_chain(self):
        class _Bad(BaseService):
            name = "bad"
            priority = 15

            def on_event(self, event):
                pass

            def wrap_loss(self, loss):
                raise RuntimeError("boom")

        rt = SakuraRuntime()
        rt.install(_LossDoubler())  # prio 10
        rt.install(_Bad())          # prio 15 — raises, runtime catches, loss unchanged through it
        rt.install(_LossAdder())    # prio 20
        # 5 → doubler → 10 → bad (skipped) → 10 → adder → 11
        assert rt.scale_loss(5.0) == 11.0


class TestOptimizerStep:
    def test_no_services_returns_false(self):
        rt = SakuraRuntime()
        assert rt.optimizer_step(object()) is False

    def test_service_without_optimizer_step_returns_false(self):
        rt = SakuraRuntime()
        rt.install(_NoopService())
        assert rt.optimizer_step(object()) is False

    def test_first_claim_wins(self):
        # Stepper (prio 10) claims first; NonStepper (prio 20) is never called.
        rt = SakuraRuntime()
        stepper = _Stepper()
        non_stepper = _NonStepper()
        rt.install(stepper)
        rt.install(non_stepper)
        assert rt.optimizer_step(object()) is True
        assert stepper.calls == 1
        assert non_stepper.calls == 0  # never reached

    def test_no_claimers_returns_false(self):
        rt = SakuraRuntime()
        a = _NonStepper()
        b = _NonStepper.__class__("Other", (BaseService,), {
            "name": "non_stepper2", "priority": 30,
            "on_event": lambda self, e: None,
            "optimizer_step": lambda self, o: False,
        })()
        rt.install(a)
        # b is a hand-rolled subclass; skip if it can't be installed cleanly.
        try:
            rt.install(b)
        except Exception:
            pass
        assert rt.optimizer_step(object()) is False

    def test_failing_optimizer_step_falls_through(self):
        class _Bad(BaseService):
            name = "bad_step"
            priority = 5

            def on_event(self, event):
                pass

            def optimizer_step(self, optimizer) -> bool:
                raise RuntimeError("boom")

        rt = SakuraRuntime()
        rt.install(_Bad())
        rt.install(_Stepper())
        # Bad raises, runtime logs+skips, then Stepper claims and returns True.
        assert rt.optimizer_step(object()) is True
