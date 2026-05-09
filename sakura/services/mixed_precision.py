"""MixedPrecision — autocast wrapping + optional GradScaler for fp16.

Knobs:
- dtype: "fp16" | "bf16" | "fp8" | "auto"
- loss_scale: float | "dynamic" | None (only for fp16)
- grad_clip: float | None (applied after unscale)

bf16 / fp8 / auto don't use a GradScaler. fp16 does. CPU autocast for fp16 is
supported via torch.autocast(device_type='cpu', dtype=torch.float16) but the
GradScaler is a no-op on CPU.
"""
from __future__ import annotations

from typing import Any, Literal, Optional, Union

from sakura.events import OnOptimizerStep, OnTrainBegin, OnTrainEnd, OnTrainStepBegin
from sakura.service import BaseService


_VALID_DTYPES = ("fp16", "bf16", "fp8", "auto")


class MixedPrecision(BaseService):
    name = "mixed_precision"
    priority = 10

    def __init__(
        self,
        *,
        dtype: Literal["fp16", "bf16", "fp8", "auto"] = "auto",
        loss_scale: Union[float, Literal["dynamic"], None] = "dynamic",
        grad_clip: Optional[float] = None,
        cache_enabled: bool = True,
    ):
        super().__init__()
        if dtype not in _VALID_DTYPES:
            raise ValueError(f"dtype must be one of {_VALID_DTYPES}, got {dtype!r}")
        self._dtype = dtype
        self._loss_scale = loss_scale
        self._grad_clip = grad_clip
        self._cache_enabled = cache_enabled
        self._scaler = None
        self._autocast_ctx = None
        self._original_forward = None

    def on_install(self, runtime: Any) -> None:
        # No-op at install — we wait for OnTrainBegin to inspect the model device.
        pass

    def on_train_begin(self, event: OnTrainBegin) -> None:
        import torch

        # Determine device + actual dtype.
        device_type = self._device_type_from(event.model)
        actual_dtype = self._resolve_dtype(device_type)

        # GradScaler only for fp16.
        if actual_dtype == torch.float16 and torch.cuda.is_available():
            init_scale = (
                2.0 ** 16
                if self._loss_scale == "dynamic"
                else float(self._loss_scale or 2.0 ** 16)
            )
            # Prefer torch.amp.GradScaler (torch 2.x new API); fall back to
            # torch.cuda.amp.GradScaler for older torch versions.
            if hasattr(torch.amp, "GradScaler"):
                self._scaler = torch.amp.GradScaler(
                    device="cuda",
                    enabled=True,
                    init_scale=init_scale,
                )
            else:
                self._scaler = torch.cuda.amp.GradScaler(  # type: ignore[attr-defined]
                    enabled=True,
                    init_scale=init_scale,
                )

        # Wrap forward with autocast for the duration of training.
        device_type = self._device_type_from(event.model)
        actual_dtype = self._resolve_dtype(device_type)
        original_forward = event.model.forward
        autocast_dtype = actual_dtype

        def _wrapped_forward(*args, **kwargs):
            with torch.autocast(device_type=device_type, dtype=autocast_dtype,
                                enabled=True, cache_enabled=self._cache_enabled):
                return original_forward(*args, **kwargs)

        # Stash the fact that we wrapped, so we can undo at on_train_end.
        self._original_forward = True  # sentinel: wrapping is active
        event.model.forward = _wrapped_forward

    def on_train_end(self, event: OnTrainEnd) -> None:
        if self._original_forward is not None:
            # Remove the instance-attr wrapper so the class descriptor takes over.
            try:
                del event.model.forward
            except AttributeError:
                pass
            self._original_forward = None

    def on_train_step_begin(self, event: OnTrainStepBegin) -> None:
        # Enter autocast context. We rely on caller to wrap the forward.
        # Plan 3 stores the context manager but does NOT auto-wrap forward —
        # framework adapters (Plan 4) integrate this with each framework's
        # model forward step. For Plan 3 the context is a no-op stash.
        pass

    def on_optimizer_step(self, event: OnOptimizerStep) -> None:
        opt = event.optimizer
        # Path 1: with GradScaler (fp16/CUDA)
        if self._scaler is not None:
            self._scaler.unscale_(opt)
            if self._grad_clip is not None:
                import torch
                params = []
                for group in opt.param_groups:
                    params.extend(group["params"])
                torch.nn.utils.clip_grad_norm_(params, self._grad_clip)
            self._scaler.step(opt)
            self._scaler.update()
            return
        # Path 2: no scaler (bf16/fp8/auto)
        if self._grad_clip is not None:
            import torch
            params = []
            for group in opt.param_groups:
                params.extend(group["params"])
            torch.nn.utils.clip_grad_norm_(params, self._grad_clip)
        if hasattr(opt, "step"):
            opt.step()

    # ............................................................. helpers

    def _device_type_from(self, model: Any) -> str:
        try:
            p = next(iter(model.parameters()))
            return p.device.type if hasattr(p, "device") else "cpu"
        except Exception:
            return "cpu"

    def _resolve_dtype(self, device_type: str):
        import torch
        if self._dtype == "auto":
            if torch.cuda.is_available():
                cap = torch.cuda.get_device_capability()
                # Ampere+ = bf16; older = fp16.
                return torch.bfloat16 if cap[0] >= 8 else torch.float16
            return torch.bfloat16
        return {
            "fp16": torch.float16,
            "bf16": torch.bfloat16,
            "fp8": (
                torch.float8_e4m3fn
                if hasattr(torch, "float8_e4m3fn")
                else torch.bfloat16
            ),
        }[self._dtype]


__all__ = ["MixedPrecision"]
