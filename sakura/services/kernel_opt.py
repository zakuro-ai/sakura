"""KernelOpt — runtime / cuDNN backend knobs that no other service touches.

Sets the PyTorch backend flags that pick faster kernels for a fixed-shape
training loop, then restores them at train end so the process is left as it was.

Knobs:
- cudnn_benchmark: autotune cuDNN conv/RNN algorithms. The single reliably-
  positive knob for ASR — the spectrogram input shape is FIXED (e.g.
  (B,1,64,256)), which is exactly the precondition the autotuner needs. The
  first few steps pay a one-time search cost, then steady-state is faster.
- tf32: allow TF32 matmul/cuDNN. Only the Ampere+ (sm >= 80) datapath has TF32;
  it is a measured no-op on Turing, so this is hardware-gated and harmless to
  leave on by default.
- channels_last: convert the model to NHWC memory format (helps some Conv2d
  kernels). Marginal for the small ASR conv stack, so OFF by default.
- flatten_rnn: call ``flatten_parameters()`` on cuDNN RNN modules so their
  weights are a single contiguous buffer (silences the cuDNN non-contiguous
  warning; no-op when already contiguous).

Notes: autocast/mixed_precision only changes *dtype* — it does not guarantee
tensor-core kernels are selected. This service is complementary and is meant to
be installed alongside mixed_precision (it runs first; priority 5).
"""
from __future__ import annotations

from typing import Any

from sakura.events import OnTrainBegin, OnTrainEnd
from sakura.service import BaseService


class KernelOpt(BaseService):
    name = "kernel_opt"
    priority = 5  # before mixed_precision (10) / compile (20)

    def __init__(
        self,
        *,
        cudnn_benchmark: bool = True,
        tf32: bool = True,
        channels_last: bool = False,
        flatten_rnn: bool = True,
    ):
        super().__init__()
        self._cudnn_benchmark = bool(cudnn_benchmark)
        self._tf32 = bool(tf32)
        self._channels_last = bool(channels_last)
        self._flatten_rnn = bool(flatten_rnn)
        self._saved: dict[str, Any] = {}

    def on_train_begin(self, event: OnTrainBegin) -> None:
        import torch

        backends = torch.backends

        self._saved["cudnn_benchmark"] = backends.cudnn.benchmark
        backends.cudnn.benchmark = self._cudnn_benchmark

        if self._tf32:
            # TF32 only matters on Ampere+; setting it on older GPUs is a no-op.
            self._saved["cudnn_tf32"] = backends.cudnn.allow_tf32
            self._saved["matmul_tf32"] = backends.cuda.matmul.allow_tf32
            backends.cudnn.allow_tf32 = True
            backends.cuda.matmul.allow_tf32 = True

        if self._channels_last:
            try:
                event.model.to(memory_format=torch.channels_last)
            except Exception:  # noqa: BLE001 — never break training over a layout hint
                pass

        if self._flatten_rnn:
            for module in event.model.modules():
                if isinstance(module, torch.nn.RNNBase):
                    try:
                        module.flatten_parameters()
                    except Exception:  # noqa: BLE001
                        pass

    def on_train_end(self, event: OnTrainEnd) -> None:
        import torch

        backends = torch.backends
        if "cudnn_benchmark" in self._saved:
            backends.cudnn.benchmark = self._saved["cudnn_benchmark"]
        if "cudnn_tf32" in self._saved:
            backends.cudnn.allow_tf32 = self._saved["cudnn_tf32"]
        if "matmul_tf32" in self._saved:
            backends.cuda.matmul.allow_tf32 = self._saved["matmul_tf32"]
        self._saved = {}


__all__ = ["KernelOpt"]
