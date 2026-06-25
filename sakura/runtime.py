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
        record_history: bool = True,
    ) -> None:
        """`record_history=False` skips per-event bookkeeping (memory + ~0.4µs/event).

        Default True for backwards compatibility / debug visibility; benches
        and tight inner loops should pass False.
        """
        self._compute = compute
        self._logger = logger
        self._record_history = record_history
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
        # Fast path: empty service list — common when the runtime is used as
        # a passive coordinator (e.g., bench harness with no services for
        # overhead measurement). Skip allocations entirely.
        services = self._sorted
        if not services:
            if self._record_history:
                self._history.append({
                    "event": type(event).__name__,
                    "rank": event.rank,
                    "world_size": event.world_size,
                    "n_services": 0,
                    "n_errors": 0,
                })
            return

        errors: Optional[list[tuple[Service, BaseException]]] = None
        for s in services:
            try:
                s.on_event(event)
            except BaseException as exc:  # noqa: BLE001
                if errors is None:
                    errors = []
                errors.append((s, exc))
                _log.exception("service '%s' raised during %s", s.name, type(event).__name__)
        if self._record_history:
            self._history.append({
                "event": type(event).__name__,
                "rank": event.rank,
                "world_size": event.world_size,
                "n_services": len(services),
                "n_errors": 0 if errors is None else len(errors),
            })
        if self._logger is not None:
            record = {
                "event": type(event).__name__,
                "rank": event.rank,
                "world_size": event.world_size,
                "n_services": len(services),
                "n_errors": 0 if errors is None else len(errors),
            }
            try:
                self._logger(record)
            except Exception:
                pass
        if errors and not isinstance(event, OnError):
            for svc, svc_exc in errors:
                err_evt = OnError(
                    rank=event.rank,
                    world_size=event.world_size,
                    exc=svc_exc,
                    context={"service": svc.name, "event": type(event).__name__},
                )
                self.dispatch(err_evt)

    def history(self) -> list[dict]:
        """Return a copy of the rolled-up event log."""
        return list(self._history)

    def scale_loss(self, loss):
        """Thread a loss tensor through every service that implements wrap_loss.

        Services iterate in priority order. Each can return a wrapped loss
        (e.g., MixedPrecision returns scaler.scale(loss) for fp16). Services
        that don't implement wrap_loss leave the loss unchanged.

        This is the runtime-level coordinator for "before backward" loss
        manipulation — services should not directly observe loss; the loop
        passes it through here so any combination of services compose.
        """
        for s in self._sorted:
            wrap = getattr(s, "wrap_loss", None)
            if callable(wrap):
                try:
                    loss = wrap(loss)
                except BaseException:
                    _log.exception("service '%s' wrap_loss failed", s.name)
        return loss

    def optimizer_step(self, optimizer) -> bool:
        """Give services a chance to step the optimizer themselves.

        Returns True if any service handled the step (loop should NOT call
        opt.step() afterward). Returns False if no service claimed it (loop
        should call opt.step() as usual).

        First-claim-wins: once a service returns True, subsequent services
        are not called for the step. This avoids double-stepping when
        multiple services are installed but only one (typically
        MixedPrecision in fp16 mode) actually wants to drive the step.
        """
        for s in self._sorted:
            stepper = getattr(s, "optimizer_step", None)
            if callable(stepper):
                try:
                    if stepper(optimizer):
                        return True
                except BaseException:
                    _log.exception("service '%s' optimizer_step failed", s.name)
        return False

    def _rebuild_sorted(self) -> None:
        indexed = list(enumerate(self._services))
        indexed.sort(key=lambda pair: (pair[1].priority, pair[0]))
        self._sorted = [s for _, s in indexed]


__all__ = ["SakuraRuntime"]
