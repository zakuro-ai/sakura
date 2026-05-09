"""Adapter ABC: holds runtime reference + emit(event) helper.

Concrete adapters (LightningAdapter, HFAdapter, DDPAdapter) subclass both
Adapter and the framework's callback type (where one exists).
"""
from __future__ import annotations

from sakura.events import Event
from sakura.runtime import SakuraRuntime


class Adapter:
    """Base for framework adapters. Holds the runtime + emit() helper."""

    def __init__(self, runtime: SakuraRuntime) -> None:
        self._runtime = runtime

    @property
    def runtime(self) -> SakuraRuntime:
        return self._runtime

    def emit(self, event: Event) -> None:
        """Dispatch an event to the runtime (and thus to all installed services)."""
        self._runtime.dispatch(event)


__all__ = ["Adapter"]
