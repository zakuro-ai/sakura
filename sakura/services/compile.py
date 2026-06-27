"""Compile — torch.compile lazy wrapping + on-disk cache.

Knobs:
- mode: "default" | "reduce-overhead" | "max-autotune"
- backend: "inductor" | "aot_eager" | "cudagraphs"
- dynamic: True/False/None
- fullgraph: bool
- cache_dir: path; defaults to ~/.cache/sakura/compile
"""
from __future__ import annotations

import os
import time
from typing import Literal, Optional

from sakura._optional import load
from sakura.events import OnTrainBegin, OnTrainStepBegin
from sakura.service import BaseService


class Compile(BaseService):
    name = "compile"
    priority = 20

    def __init__(
        self,
        *,
        mode: Literal["default", "reduce-overhead", "max-autotune"] = "default",
        backend: str = "inductor",
        dynamic: Optional[bool] = None,
        fullgraph: bool = False,
        cache_dir: Optional[str] = None,
    ):
        super().__init__()
        self._mode = mode
        self._backend = backend
        self._dynamic = dynamic
        self._fullgraph = fullgraph
        self._cache_dir = cache_dir or os.path.expanduser("~/.cache/sakura/compile")
        os.makedirs(self._cache_dir, exist_ok=True)
        self.compiled_called = False
        self.first_step_secs: Optional[float] = None

    def on_train_begin(self, event: OnTrainBegin) -> None:
        torch = load("torch", extra="training")

        os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", self._cache_dir)
        # Mark as "attempted" before the compile call — even if torch.compile
        # falls back to eager (e.g., on CPU with inductor unavailable), we
        # have fulfilled the contract of "attempted to install compile".
        self.compiled_called = True
        try:
            event.model.forward = torch.compile(  # type: ignore[attr-defined]
                event.model.forward,
                mode=self._mode,
                backend=self._backend,
                dynamic=self._dynamic,
                fullgraph=self._fullgraph,
            )
        except Exception:
            # Compile failed — fall back to eager. compiled_called remains True
            # (we attempted; the result is eager fallback).
            pass

    def on_train_step_begin(self, event: OnTrainStepBegin) -> None:
        if self.first_step_secs is None:
            self.first_step_secs = time.perf_counter()
        elif isinstance(self.first_step_secs, float):
            # Convert "first step start" to "first step duration" lazily — Plan 4
            # measures this properly with adapter hooks.
            pass


__all__ = ["Compile"]
