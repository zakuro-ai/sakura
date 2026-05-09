"""HandlerRegistry: in-process dispatch table for worker-side handlers."""
from __future__ import annotations

from typing import Callable, Tuple

TensorDict = dict
HandlerFn = Callable[[list[TensorDict], bytes], Tuple[list[TensorDict], bytes]]


class HandlerRegistry:
    """Maps `handler_id (u32)` → callable that processes one RPC."""

    def __init__(self) -> None:
        self._handlers: dict[int, HandlerFn] = {}

    def register(self, handler_id: int, fn: HandlerFn) -> None:
        if not isinstance(handler_id, int):
            raise TypeError(f"handler_id must be int, got {type(handler_id).__name__}")
        if not callable(fn):
            raise TypeError(f"fn must be callable, got {type(fn).__name__}")
        self._handlers[handler_id] = fn

    def dispatch(
        self, handler_id: int, tensors: list[TensorDict], aux: bytes
    ) -> Tuple[list[TensorDict], bytes]:
        if handler_id not in self._handlers:
            raise KeyError(f"handler {handler_id:#x} not registered")
        result = self._handlers[handler_id](tensors, aux)
        if (
            not isinstance(result, tuple)
            or len(result) != 2
            or not isinstance(result[0], list)
            or not isinstance(result[1], (bytes, bytearray))
        ):
            raise ValueError(
                "handler must return a 2-tuple (list[TensorDict], bytes); got "
                f"{type(result).__name__}"
            )
        return (result[0], bytes(result[1]))


__all__ = ["HandlerRegistry"]
