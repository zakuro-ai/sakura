"""SakuraRuntime — central orchestrator for events + services."""
from __future__ import annotations

import logging
from typing import Callable, Optional

from sakura.events import Event, OnError
from sakura.service import Service

_log = logging.getLogger(__name__)


class SakuraRuntime:
    def __init__(
        self,
        *,
        compute: Optional[object] = None,
        logger: Optional[Callable[[dict], None]] = None,
    ) -> None:
        self._compute = compute
        self._logger = logger
        self._services: list[Service] = []
        self._sorted: list[Service] = []
        self._by_name: dict[str, Service] = {}
        self._started = False
        self._history: list[dict] = []

    @property
    def compute(self) -> Optional[object]:
        return self._compute

    @property
    def services(self) -> tuple[Service, ...]:
        return tuple(self._sorted)

    def start(self) -> None:
        if self._started:
            return
        self._started = True
        for s in self._sorted:
            hook = getattr(s, "on_runtime_start", None)
            if callable(hook):
                try:
                    hook(self)
                except BaseException:
                    _log.exception("service '%s' on_runtime_start failed", s.name)

    def shutdown(self, *, timeout: float = 30.0) -> None:
        if not self._started:
            return
        for s in reversed(self._sorted):
            hook = getattr(s, "on_runtime_shutdown", None)
            if callable(hook):
                try:
                    hook(self)
                except BaseException:
                    _log.exception("service '%s' on_runtime_shutdown failed", s.name)
        self._started = False

    def __enter__(self) -> "SakuraRuntime":
        self.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.shutdown()

    def install(self, service: Service) -> None:
        if not service.name:
            raise ValueError("service.name must be a non-empty string")
        if service.name in self._by_name:
            raise ValueError(f"service '{service.name}' already installed")
        for req in service.requires:
            if req not in self._by_name:
                raise ValueError(
                    f"service '{service.name}' requires '{req}' but it is missing"
                )
        self._services.append(service)
        self._by_name[service.name] = service
        self._rebuild_sorted()
        service.on_install(self)
        if self._started:
            hook = getattr(service, "on_runtime_start", None)
            if callable(hook):
                try:
                    hook(self)
                except BaseException:
                    _log.exception("service '%s' on_runtime_start failed", service.name)

    def uninstall(self, name: str) -> None:
        if name not in self._by_name:
            raise KeyError(f"service '{name}' not installed")
        dependents = [s.name for s in self._services if name in s.requires]
        if dependents:
            raise ValueError(f"service '{name}' depended on by: {sorted(dependents)}")
        del self._by_name[name]
        self._services = [s for s in self._services if s.name != name]
        self._rebuild_sorted()

    def find(self, name: str) -> Optional[Service]:
        return self._by_name.get(name)

    def dispatch(self, event: Event) -> None:
        services = list(self._sorted)
        errors: list[tuple[Service, BaseException]] = []
        for s in services:
            try:
                s.on_event(event)
            except BaseException as exc:  # noqa: BLE001
                errors.append((s, exc))
                _log.exception("service '%s' raised during %s", s.name, type(event).__name__)
        record = {
            "event": type(event).__name__,
            "rank": event.rank,
            "world_size": event.world_size,
            "n_services": len(services),
            "n_errors": len(errors),
        }
        self._history.append(record)
        if self._logger is not None:
            try:
                self._logger(record)
            except Exception:
                pass
        if errors and not isinstance(event, OnError):
            for svc, exc in errors:
                err_evt = OnError(
                    rank=event.rank,
                    world_size=event.world_size,
                    exc=exc,
                    context={"service": svc.name, "event": type(event).__name__},
                )
                self.dispatch(err_evt)

    def history(self) -> list[dict]:
        """Return a copy of the rolled-up event log."""
        return list(self._history)

    def _rebuild_sorted(self) -> None:
        indexed = list(enumerate(self._services))
        indexed.sort(key=lambda pair: (pair[1].priority, pair[0]))
        self._sorted = [s for _, s in indexed]


__all__ = ["SakuraRuntime"]
