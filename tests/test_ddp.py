"""Smoke tests for the DDP async-eval callback — rank semantics + dispatch."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from sakura.ddp import DDPAsyncEvalCallback


class _TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(4, 2)

    def forward(self, x):
        return self.linear(x)


def _model_factory():
    return _TinyModel()


def _eval_fn(model, payload):
    x, y = payload
    with torch.no_grad():
        logits = model(x)
        correct = int((logits.argmax(-1) == y).sum().item())
    return {"val_acc": correct / max(y.numel(), 1)}


class TestDDPRankSemantics:
    def test_non_driver_rank_is_noop(self):
        """Rank != 0 should not allocate a pool, should not dispatch."""
        cb = DDPAsyncEvalCallback(
            model_factory=_model_factory,
            eval_fn=_eval_fn,
            eval_payload=(torch.randn(4, 4), torch.zeros(4, dtype=torch.long)),
            rank=1,
            world_size=2,
            verbose=False,
        )
        assert cb.is_driver is False
        assert cb._pool is None
        # on_epoch_end + on_train_end must be safe no-ops.
        cb.on_epoch_end(0, _TinyModel())
        cb.on_train_end()
        assert cb.history == []

    def test_driver_rank_dispatches_and_records(self):
        cb = DDPAsyncEvalCallback(
            model_factory=_model_factory,
            eval_fn=_eval_fn,
            eval_payload=(torch.randn(8, 4), torch.zeros(8, dtype=torch.long)),
            rank=0,
            world_size=2,
            verbose=False,
        )
        assert cb.is_driver is True

        model = _TinyModel()
        for epoch in range(3):
            cb.on_epoch_end(epoch, model)
        cb.on_train_end()

        assert len(cb.history) == 3
        for row in cb.history:
            assert "val_acc" in row
            assert "epoch" in row
            assert "elapsed_secs" in row

    def test_ddp_wrapped_model_state_dict_is_unwrapped(self):
        """A module wrapped in DDP-like .module attribute should still produce
        a clean state_dict that the worker-side can load."""
        inner = _TinyModel()

        class _FakeDDP:
            def __init__(self, m):
                self.module = m

            def parameters(self):
                return self.module.parameters()

            def state_dict(self):
                # This is what real DDP does: prefix keys with 'module.'.
                return {f"module.{k}": v for k, v in self.module.state_dict().items()}

        wrapped = _FakeDDP(inner)
        cb = DDPAsyncEvalCallback(
            model_factory=_model_factory,
            eval_fn=_eval_fn,
            eval_payload=(torch.randn(4, 4), torch.zeros(4, dtype=torch.long)),
            rank=0,
            world_size=2,
            verbose=False,
        )
        cb.on_epoch_end(0, wrapped)
        cb.on_train_end()
        # No exception = the worker loaded the (prefix-stripped) state_dict cleanly.
        assert len(cb.history) == 1
        assert "val_acc" in cb.history[0]
