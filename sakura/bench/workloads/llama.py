"""Llama-3-1B fine-tune workload (perf tier — stub).

This workload requires a real GPU with sufficient memory (>=24 GB) to run.
The stub returns a Workload that raises NotImplementedError when its
make_model is called without sufficient hardware. The benchmark CLI's
'perf' tier path is responsible for skipping this workload when the
hardware doesn't qualify.

Future implementation will:
  - Load Llama-3-1B (HuggingFace AutoModelForCausalLM)
  - Fine-tune on a small instruction set (e.g., Alpaca subset)
  - Report tokens/second + final perplexity
"""
from __future__ import annotations

import torch

from sakura.bench.harness import Workload


def _check_gpu():
    if not torch.cuda.is_available():
        raise NotImplementedError(
            "Llama-3-1B workload requires CUDA GPU. Skip in CPU-only environments."
        )
    if torch.cuda.device_count() < 1:
        raise NotImplementedError("Llama-3-1B workload requires at least 1 GPU.")


def _make_model():
    _check_gpu()
    raise NotImplementedError(
        "llama.make_workload is a stub for the perf tier. "
        "Real Llama-3-1B fine-tune comes in v1.x with the GPU runner."
    )


def _make_loader():
    raise NotImplementedError("llama loader is a stub.")


def _eval_fn(model, loader):
    raise NotImplementedError("llama eval_fn is a stub.")


def make_workload(*, batch_size: int = 1, epochs: int = 1) -> Workload:
    """Build the Llama-3-1B fine-tune workload (perf tier stub)."""
    return Workload(
        name="llama3-1b-finetune",
        tier="perf",
        make_model=_make_model,
        make_train_loader=_make_loader,
        make_val_loader=_make_loader,
        eval_fn=_eval_fn,
        epochs=epochs,
    )


__all__ = ["make_workload"]
