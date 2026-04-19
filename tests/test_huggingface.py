"""Smoke tests for the HuggingFace callback — no real model / data."""

from __future__ import annotations

import pytest

transformers = pytest.importorskip("transformers")
torch = pytest.importorskip("torch")

from sakura.huggingface import SakuraHFCallback


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


class _FakeTrainerState:
    def __init__(self):
        self.epoch = 0.0
        self.log_history: list = []


class TestSakuraHFCallback:
    def test_dispatch_and_record_metrics(self):
        """Callback should dispatch eval, reap the future, and log metrics."""
        x = torch.randn(8, 4)
        y = torch.randint(0, 2, (8,))
        cb = SakuraHFCallback(
            model_factory=_model_factory,
            eval_fn=_eval_fn,
            eval_payload=(x, y),
            verbose=False,
        )
        state = _FakeTrainerState()
        model = _TinyModel()

        # Simulate two epochs: HF would call these in order.
        state.epoch = 1.0
        cb.on_epoch_end(args=None, state=state, control=None, model=model)
        state.epoch = 2.0
        cb.on_epoch_end(args=None, state=state, control=None, model=model)
        cb.on_train_end(args=None, state=state, control=None, model=model)

        assert len(cb.history) == 2
        for row in cb.history:
            assert "val_acc" in row
            assert "epoch" in row
            assert "elapsed_secs" in row
        # Metrics should also have been pushed into TrainerState.log_history.
        assert len(state.log_history) == 2
