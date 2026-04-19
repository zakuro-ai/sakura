"""Sakura's Lightning integration — validation offloaded to Zakuro.

Sakura accelerates training by running validation *in parallel* with the
next training epoch. Historically this required MPI + Redis plus a second
Lightning process with ``SAKURA_ROLE=1``. This module replaces that plumbing
with a single process that dispatches validation to a Zakuro worker via
``@zk.fn`` — no MPI, no Redis, no bifurcated roles.

Usage::

    import lightning as L
    import zakuro as zk
    from sakura.lightning import SakuraTrainer

    def model_factory():
        return MyLightningModule()

    def val_loader_factory():
        return DataLoader(val_dataset, batch_size=256)

    trainer = SakuraTrainer(
        max_epochs=10,
        accelerator="auto",
        val_compute=zk.Compute(uri="quic://worker-b:4433"),  # or None → standalone
        model_factory=model_factory,
        val_loader_factory=val_loader_factory,
    )
    trainer.run(model, train_loader)

When ``val_compute`` is ``None`` (or omitted) Zakuro's standalone fallback
runs the validation in-process so the example still works on a laptop
without a second worker.
"""

from __future__ import annotations

import os
import cloudpickle
import time
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any, Callable, Optional

import lightning as L
import torch
from lightning.pytorch.callbacks import Callback
from torch.utils.data import DataLoader

import zakuro as zk


# ---------------------------------------------------------------------------
# Remote validation primitive
# ---------------------------------------------------------------------------


@zk.fn
def _remote_validate(
    state_bytes: bytes,
    model_factory_bytes: bytes,
    val_loader_factory_bytes: bytes,
) -> dict:
    """Rebuild model + loader on the worker, run validation, return metrics.

    The caller serialises everything with cloudpickle (via ``@zk.fn``), so the
    factories and state dict travel through whatever transport Zakuro picks
    (HTTP, QUIC, or in-process for standalone). The validation loop is
    intentionally plain PyTorch — no Lightning process on the worker side —
    to keep the remote footprint small.
    """
    import cloudpickle as _pickle

    import torch as _torch
    import torch.nn.functional as F

    state_dict = _pickle.loads(state_bytes)
    model_factory = _pickle.loads(model_factory_bytes)
    val_loader_factory = _pickle.loads(val_loader_factory_bytes)

    model = model_factory()
    model.load_state_dict(state_dict)
    model.eval()
    loader = val_loader_factory()

    total_loss = 0.0
    total_count = 0
    with _torch.no_grad():
        for batch in loader:
            x, y = batch
            logits = model(x)
            loss = F.cross_entropy(logits, y, reduction="sum")
            total_loss += float(loss.item())
            total_count += int(y.numel())

    avg_loss = total_loss / max(total_count, 1)
    return {
        "val_loss": avg_loss,
        "worker_name": os.environ.get("ZAKURO_WORKER_NAME", "<standalone>"),
    }


# ---------------------------------------------------------------------------
# Callback: fires per training epoch, submits remote validation
# ---------------------------------------------------------------------------


class SakuraLightningCallback(Callback):
    """Async validation callback.

    On ``on_train_epoch_end`` the current model state is shipped to the
    configured Zakuro compute. The future is reaped at the start of the next
    epoch (so validation overlaps with the next training pass), and the final
    pending future is drained in ``on_train_end``.
    """

    def __init__(
        self,
        compute: zk.Compute,
        model_factory: Callable[[], L.LightningModule],
        val_loader_factory: Callable[[], DataLoader],
        model_path: Optional[str] = None,
        verbose: bool = True,
    ) -> None:
        super().__init__()
        self._compute = compute
        self._model_factory_bytes = cloudpickle.dumps(model_factory)
        self._val_loader_factory_bytes = cloudpickle.dumps(val_loader_factory)
        self._model_path = model_path
        self._verbose = verbose

        self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="sakura-val")
        self._pending: Optional[Future] = None
        self._best_val_loss: Optional[float] = None
        self._history: list[dict] = []

    # .................................................................. fit

    def on_train_epoch_end(
        self,
        trainer: L.Trainer,
        pl_module: L.LightningModule,
    ) -> None:
        # Reap previous epoch's validation before kicking off a new one so the
        # user sees losses in order and so we don't stack pending futures.
        self._drain(trainer)

        state_bytes = cloudpickle.dumps(
            {k: v.detach().cpu() for k, v in pl_module.state_dict().items()}
        )
        epoch = trainer.current_epoch
        self._pending = self._pool.submit(
            self._validate_remote, state_bytes, epoch, trainer, pl_module
        )

    def on_train_end(
        self, trainer: L.Trainer, pl_module: L.LightningModule
    ) -> None:
        self._drain(trainer)
        self._pool.shutdown(wait=True)

    # .............................................................. history

    @property
    def history(self) -> list[dict]:
        """List of ``{"epoch", "val_loss", "worker_name", "elapsed_secs"}`` rows."""
        return list(self._history)

    @property
    def best_val_loss(self) -> Optional[float]:
        return self._best_val_loss

    # ............................................................. internals

    def _validate_remote(
        self,
        state_bytes: bytes,
        epoch: int,
        trainer: L.Trainer,
        pl_module: L.LightningModule,
    ) -> dict:
        started = time.perf_counter()
        result = _remote_validate.to(self._compute)(
            state_bytes,
            self._model_factory_bytes,
            self._val_loader_factory_bytes,
        )
        result["epoch"] = epoch
        result["elapsed_secs"] = time.perf_counter() - started
        return result

    def _drain(self, trainer: L.Trainer) -> None:
        if self._pending is None:
            return
        result = self._pending.result()
        self._pending = None

        val_loss = result["val_loss"]
        self._history.append(result)
        if self._verbose:
            print(
                f"[Sakura] epoch={result['epoch']} val_loss={val_loss:.4f} "
                f"on={result['worker_name']} took={result['elapsed_secs']:.2f}s"
            )
        if self._best_val_loss is None or val_loss < self._best_val_loss:
            self._best_val_loss = val_loss
            if self._model_path:
                trainer.save_checkpoint(self._model_path)


# ---------------------------------------------------------------------------
# Trainer façade
# ---------------------------------------------------------------------------


class SakuraTrainer:
    """Lightning trainer that offloads validation to a remote Zakuro worker.

    A drop-in alternative to ``L.Trainer`` for the async-validation use case.
    Instead of spawning a validator process via MPI + ``SAKURA_ROLE``, Sakura
    dispatches a single ``@zk.fn`` call per epoch; the future overlaps with
    the next training pass on the trainer host.
    """

    def __init__(
        self,
        *,
        val_compute: Optional[zk.Compute] = None,
        model_factory: Optional[Callable[[], L.LightningModule]] = None,
        val_loader_factory: Optional[Callable[[], DataLoader]] = None,
        model_path: Optional[str] = None,
        verbose: bool = True,
        **trainer_kwargs: Any,
    ) -> None:
        self._val_compute = val_compute
        self._model_factory = model_factory
        self._val_loader_factory = val_loader_factory
        self._model_path = model_path
        self._verbose = verbose
        self._trainer_kwargs = trainer_kwargs
        self._callback: Optional[SakuraLightningCallback] = None

    def run(
        self,
        model: L.LightningModule,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        *,
        model_factory: Optional[Callable[[], L.LightningModule]] = None,
        val_loader_factory: Optional[Callable[[], DataLoader]] = None,
        model_path: Optional[str] = None,
        **run_kwargs: Any,
    ) -> L.LightningModule:
        """Train ``model`` with async validation on a remote Zakuro worker."""
        model_factory = model_factory or self._model_factory
        val_loader_factory = val_loader_factory or self._val_loader_factory
        model_path = model_path or self._model_path

        # Fallbacks for a friendlier default experience ---------------------
        # If no factory is given, reuse the in-process model class; the remote
        # worker must have the class importable for this to work. Same for
        # the loader.
        if model_factory is None:
            model_cls = type(model)
            model_factory = lambda: model_cls()  # noqa: E731
        if val_loader_factory is None:
            if val_loader is None:
                raise ValueError(
                    "SakuraTrainer.run needs either val_loader or val_loader_factory"
                )
            loader = val_loader
            val_loader_factory = lambda: loader  # noqa: E731

        compute = self._val_compute if self._val_compute is not None else zk.Compute()

        callback = SakuraLightningCallback(
            compute=compute,
            model_factory=model_factory,
            val_loader_factory=val_loader_factory,
            model_path=model_path,
            verbose=self._verbose,
        )
        self._callback = callback

        user_callbacks = list(self._trainer_kwargs.pop("callbacks", []))
        trainer = L.Trainer(
            callbacks=user_callbacks + [callback],
            **self._trainer_kwargs,
            **run_kwargs,
        )
        trainer.fit(model, train_loader)
        return model

    @property
    def history(self) -> list[dict]:
        return self._callback.history if self._callback else []

    @property
    def best_val_loss(self) -> Optional[float]:
        return self._callback.best_val_loss if self._callback else None


__all__ = [
    "SakuraLightningCallback",
    "SakuraTrainer",
]
