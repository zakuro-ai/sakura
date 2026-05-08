"""Service ABC + BaseService helper.

Services are installable units of behavior subscribing to events on a
SakuraRuntime. The Protocol form allows duck-typed compliance; BaseService
is a concrete helper that routes events to on_<event_name> methods.
"""
from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from sakura.events import Event


@runtime_checkable
class Service(Protocol):
    """Duck-typed service contract."""
    name: str
    priority: int
    requires: tuple[str, ...]

    def on_install(self, runtime: Any) -> None: ...
    def on_event(self, event: Event) -> None: ...


class BaseService:
    """Convenience base class with default routing.

    Subclasses set `name` and `priority` (class attributes) and define
    `on_<event_name>(event)` methods for the events they care about.
    `on_event(event)` looks up the right method by `event.name()` and
    falls back to no-op for events not handled.
    """
    name: str = ""
    priority: int = -1
    requires: tuple[str, ...] = ()

    def __init__(self):
        if not self.name:
            raise TypeError(f"{type(self).__name__}: 'name' must be set as a class attribute")
        if self.priority < 0:
            raise TypeError(f"{type(self).__name__}: 'priority' must be a non-negative int")

    def on_install(self, runtime: Any) -> None:
        """Default: no-op. Override for setup that needs the runtime reference."""

    def on_event(self, event: Event) -> None:
        """Route to `on_<event_name>` if present, else no-op."""
        method_name = event.name()
        method = getattr(self, method_name, None)
        if method is not None and callable(method):
            method(event)


__all__ = ["Service", "BaseService"]
