"""Async trainer now dispatches test phases via Zakuro instead of MPI."""

from __future__ import annotations

import pickle
from types import SimpleNamespace
from unittest.mock import MagicMock

from sakura.ml.async_trainer import AsyncTrainer


class _Metrics:
    """Writable ``.test`` attribute — matches the old trainer-metrics contract."""

    def __init__(self) -> None:
        self.test: dict = {}


class _FakeTrainer:
    """Minimal object with the interface AsyncTrainer.run() expects."""

    def __init__(self, epochs: int) -> None:
        self._epochs = list(range(epochs))
        self._epoch = 0
        self._metrics = _Metrics()
        self.trained_epochs = 0
        self._state = [42]

    def serialized_state_dict(self) -> bytes:
        # Same shape as the real Trainer: pickled mapping of {name: pickled_tensor}.
        return pickle.dumps({"weight": pickle.dumps(self._state)})

    def train(self, train_loader=None):  # noqa: ARG002
        self._state[0] += 1
        self.trained_epochs += 1


def _model_factory():
    model = MagicMock()
    model.load_state_dict = MagicMock()
    return model


def _test_fn(model) -> dict:  # noqa: ARG001
    """Runs on the worker; returns arbitrary metrics."""
    return {"accuracy": 0.99, "executed": True}


class TestAsyncTrainerZakuroDispatch:
    def test_dispatches_and_attaches_metrics(self):
        """AsyncTrainer should ship state to zakuro, wait, and populate metrics."""
        trainer = _FakeTrainer(epochs=3)
        at = AsyncTrainer(
            trainer=trainer,
            model_factory=_model_factory,
            test_fn=_test_fn,
        )  # val_compute=None → zakuro standalone fallback (in-process)

        at.run(train_loader=None, test_loader=None)

        assert trainer.trained_epochs == 3
        assert trainer._metrics.test.get("executed") is True
        assert trainer._metrics.test.get("accuracy") == 0.99
