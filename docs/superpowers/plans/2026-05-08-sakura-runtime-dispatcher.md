# Sakura v1.0 — Plan 2: Runtime + Dispatcher + sakura-worker

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:subagent-driven-development` (recommended) or `superpowers:executing-plans` to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Stand up the Python runtime layer that turns the Plan 1 sakura-wire transport into a usable orchestration surface — `SakuraRuntime` event bus, `Service` ABC, `Adapter` ABC, `Dispatcher` abstraction with `Local` / `Remote` / `InThread` impls, and a Python-side handler registry on the worker so cloudpickled callables can be dispatched end-to-end.

**Architecture:** Plan 1 delivered a low-level Rust transport (`sakura_wire.Dispatcher.submit(handler_id, tensors, aux_bytes)`) plus an echo handler. Plan 2 layers Python abstractions on top: `SakuraRuntime` orchestrates installable `Service`s on a typed event bus; `Compute` is a URI-tag that resolves to a concrete `Dispatcher` (`LocalDispatcher` auto-spawns a localhost worker via the existing `WorkerSupervisor`, `RemoteDispatcher` connects to an existing daemon, `InThreadDispatcher` runs handlers synchronously for tests). A `HandlerRegistry` on the worker dispatches `HANDLER_EXEC_CLOUDPICKLED` to user-supplied callables — the universal mechanism for shipping eval/checkpoint work in later plans.

**Tech Stack:** Python 3.10+, `cloudpickle` (already a transitive dep via `lightning`), `numpy`, `dataclasses`, `typing.Protocol`. Rust changes confined to extending `pyo3_bindings.rs` with a new `run_server` function that takes a Python handler callback.

**Out of scope for this plan** (deferred to Plans 3-5):
- The seven concrete services (`MixedPrecision`, `Compile`, `ZeRO1`, `ActivationCheckpoint`, `AsyncEval`, `AsyncCheckpoint`, real `TelemetryService`) — Plan 3.
- Framework adapters (`LightningAdapter`, `HFAdapter`, `DDPAdapter`) — Plan 4.
- `ZakuroDispatcher` (`Compute.zakuro`) — Plan 4 alongside the framework adapters.
- Removing v0.1.x code (`sakura.lightning.SakuraTrainer`, etc.) — Plan 4.
- Benchmark harness — Plan 5.
- The codec zero-copy producer-path optimization (Plan 1's 7× perf miss) — separate v1.x patch, not on the main plan track.

---

## Prerequisites

The host machine already has rustup-managed `cargo`/`rustc` 1.95, Python 3.12, `maturin` 1.13, and `uv` (set up during Plan 1 execution). No additional system installs.

---

## Existing state at start of Plan 2

The merged master after Plan 1 is at `1daaa59` (with tag `sakura-wire-v1-foundation` at `d1d17a0`). Concretely:

- `sakura_wire` cdylib exposes: `Dispatcher` (low-level QUIC client), `Future`, `Result`, `TlsConfig`, `WorkerSupervisor`, `run_echo_server`.
- `sakura.wire.__init__` re-exports those.
- `sakura.worker.__main__` runs `run_echo_server` with `HANDLER_ECHO`.
- v0.1.x: `sakura/__init__.py`, `sakura/__main__.py`, `sakura/lightning/`, `sakura/huggingface/`, `sakura/tensorflow/`, `sakura/ddp/`, `sakura/ml/`, `sakura/functional.py`, `sakura/config.yaml` — untouched and still importable.

Plan 2 must NOT modify v0.1.x code. Plan 4 owns the cleanup.

---

## File Structure (created/modified by this plan)

**New Python files:**
```
sakura/
├── runtime.py                                  # SakuraRuntime
├── service.py                                  # Service ABC, BaseService helper
├── events.py                                   # OnTrainBegin / OnEpochEnd / etc. dataclasses
├── dispatch/
│   ├── __init__.py                             # re-exports
│   ├── base.py                                 # Dispatcher ABC, Future ABC, Result dataclass
│   ├── compute.py                              # Compute URI tag class
│   ├── in_thread.py                            # InThreadDispatcher (test/debug)
│   ├── local.py                                # LocalDispatcher (auto-spawn worker)
│   └── remote.py                               # RemoteDispatcher (talk to existing worker)
└── worker/
    ├── handlers.py                             # cloudpickle exec, heartbeat handlers
    └── registry.py                             # HandlerRegistry
tests/
├── runtime/
│   ├── __init__.py
│   ├── test_events.py
│   ├── test_service.py
│   ├── test_runtime_install_order.py
│   └── test_runtime_lifecycle.py
├── dispatch/
│   ├── __init__.py
│   ├── test_compute.py
│   ├── test_in_thread.py
│   ├── test_local.py
│   └── test_remote.py
└── worker/
    ├── __init__.py
    ├── test_registry.py
    └── test_e2e_cloudpickled.py
```

**Existing Python files modified:**
```
sakura/__init__.py                              # bump version + add re-exports
sakura/worker/__init__.py                       # re-export registry
sakura/worker/__main__.py                       # replace echo-only with registry server
```

**Rust files modified:**
```
crates/sakura-wire/src/pyo3_bindings.rs         # add run_server PyO3 fn
crates/sakura-wire/src/lib.rs                   # wire run_server into #[pymodule]
```

**Files NOT touched:**
- `sakura/lightning/`, `sakura/huggingface/`, `sakura/tensorflow/`, `sakura/ddp/`, `sakura/ml/` — v0.1.x stays as-is.
- `crates/sakura-wire/src/codec/`, `protocol/`, `runtime.rs`, `transport/`, `supervisor.rs` — Plan 1 done.
- Any `tests/test_*.py` from v0.1.x.

---

## Task 1: Repo plumbing — create dirs + empty `__init__.py` files

**Files:** Create the directory skeleton + empty `__init__.py` files so subsequent tasks can drop content into them.

- [ ] **Step 1: Create the new directories**

```bash
mkdir -p sakura/dispatch tests/runtime tests/dispatch tests/worker
```

- [ ] **Step 2: Create empty `__init__.py` files**

```bash
touch sakura/dispatch/__init__.py
touch tests/runtime/__init__.py
touch tests/dispatch/__init__.py
touch tests/worker/__init__.py
```

- [ ] **Step 3: Verify directory layout**

Run:
```bash
ls sakura/dispatch tests/runtime tests/dispatch tests/worker
```
Expected: each lists `__init__.py` (and nothing else for now).

- [ ] **Step 4: Commit**

```bash
git add sakura/dispatch/__init__.py tests/runtime/__init__.py tests/dispatch/__init__.py tests/worker/__init__.py
git commit -m "build: scaffold sakura/dispatch + tests/{runtime,dispatch,worker} dirs

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 2: Event dataclasses (`sakura/events.py`)

**Files:**
- Create: `sakura/events.py`, `tests/runtime/test_events.py`

- [ ] **Step 1: Write the failing tests for event types**

Create `tests/runtime/test_events.py`:
```python
"""Tests for the typed event payloads emitted by adapters and consumed by services."""
from __future__ import annotations

from sakura.events import (
    Event,
    OnEpochBegin,
    OnEpochEnd,
    OnError,
    OnOptimizerStep,
    OnSave,
    OnTrainBegin,
    OnTrainEnd,
    OnTrainStepBegin,
)


class TestEventTypes:
    def test_on_train_begin_carries_required_fields(self):
        evt = OnTrainBegin(model="model_obj", optimizer="optim_obj", train_loader="loader",
                           val_loader=None, rank=0, world_size=1)
        assert evt.model == "model_obj"
        assert evt.optimizer == "optim_obj"
        assert evt.train_loader == "loader"
        assert evt.val_loader is None
        assert evt.rank == 0
        assert evt.world_size == 1
        assert isinstance(evt, Event)

    def test_on_epoch_end_carries_metrics(self):
        evt = OnEpochEnd(epoch=3, model="m", optimizer="o", metrics={"val_loss": 0.21}, rank=0, world_size=2)
        assert evt.epoch == 3
        assert evt.metrics == {"val_loss": 0.21}
        assert evt.rank == 0
        assert evt.world_size == 2

    def test_on_train_step_begin(self):
        evt = OnTrainStepBegin(model="m", batch=("x", "y"), step=42, rank=1, world_size=4)
        assert evt.step == 42
        assert evt.batch == ("x", "y")

    def test_on_optimizer_step(self):
        evt = OnOptimizerStep(optimizer="o", rank=0, world_size=1)
        assert evt.optimizer == "o"

    def test_on_save_carries_path_and_state_dict(self):
        evt = OnSave(path="/tmp/ckpt.pt", state_dict={"weights": 0}, rank=0, world_size=1)
        assert evt.path == "/tmp/ckpt.pt"
        assert evt.state_dict == {"weights": 0}

    def test_on_train_end_carries_history(self):
        history = [{"epoch": 0, "val_loss": 0.5}, {"epoch": 1, "val_loss": 0.4}]
        evt = OnTrainEnd(model="m", history=history, rank=0, world_size=1)
        assert evt.history == history

    def test_on_error_carries_exc_and_context(self):
        exc = RuntimeError("boom")
        evt = OnError(exc=exc, context={"hook": "lightning"}, rank=0, world_size=1)
        assert evt.exc is exc
        assert evt.context["hook"] == "lightning"

    def test_event_name_classmethod_returns_consistent_string(self):
        assert OnEpochEnd.name() == "on_epoch_end"
        assert OnTrainBegin.name() == "on_train_begin"
        assert OnError.name() == "on_error"

    def test_events_are_hashable_via_id(self):
        # We DON'T require Event to be hashable as a value — just that two
        # distinct constructions are non-equal (they identify by content).
        a = OnEpochEnd(epoch=0, model="m", optimizer="o", metrics={}, rank=0, world_size=1)
        b = OnEpochEnd(epoch=1, model="m", optimizer="o", metrics={}, rank=0, world_size=1)
        assert a != b
```

- [ ] **Step 2: Run the test — should fail (module not found)**

```bash
source .venv/bin/activate
pytest tests/runtime/test_events.py -v 2>&1 | tail -10
```
Expected: `ModuleNotFoundError: No module named 'sakura.events'`.

- [ ] **Step 3: Implement `sakura/events.py`**

```python
"""Typed event payloads emitted by adapters and consumed by services.

Every event carries (rank, world_size) so DDP-aware services can branch on
event.rank without each adapter doing the bookkeeping. Single-process runs
get rank=0, world_size=1.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass(frozen=True)
class Event:
    """Base type for all event payloads. Every event carries rank/world_size."""
    rank: int
    world_size: int

    @classmethod
    def name(cls) -> str:
        """Convert ClassName like OnEpochEnd -> on_epoch_end."""
        out = []
        for i, c in enumerate(cls.__name__):
            if c.isupper() and i > 0:
                out.append("_")
            out.append(c.lower())
        return "".join(out)


@dataclass(frozen=True)
class OnTrainBegin(Event):
    model: Any
    optimizer: Any
    train_loader: Any
    val_loader: Optional[Any] = None


@dataclass(frozen=True)
class OnEpochBegin(Event):
    epoch: int


@dataclass(frozen=True)
class OnTrainStepBegin(Event):
    model: Any
    batch: Any
    step: int


@dataclass(frozen=True)
class OnOptimizerStep(Event):
    optimizer: Any


@dataclass(frozen=True)
class OnEpochEnd(Event):
    epoch: int
    model: Any
    optimizer: Any
    metrics: dict = field(default_factory=dict)


@dataclass(frozen=True)
class OnSave(Event):
    path: str
    state_dict: Any


@dataclass(frozen=True)
class OnTrainEnd(Event):
    model: Any
    history: list


@dataclass(frozen=True)
class OnError(Event):
    exc: BaseException
    context: dict = field(default_factory=dict)


__all__ = [
    "Event",
    "OnTrainBegin",
    "OnEpochBegin",
    "OnTrainStepBegin",
    "OnOptimizerStep",
    "OnEpochEnd",
    "OnSave",
    "OnTrainEnd",
    "OnError",
]
```

- [ ] **Step 4: Run the test — should pass**

```bash
pytest tests/runtime/test_events.py -v
```
Expected: 9 tests pass.

- [ ] **Step 5: Commit**

```bash
git add sakura/events.py tests/runtime/test_events.py
git commit -m "feat(runtime): typed event dataclasses (OnTrainBegin, OnEpochEnd, ...)

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 3: `Service` ABC + `BaseService` helper

**Files:**
- Create: `sakura/service.py`, `tests/runtime/test_service.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/runtime/test_service.py`:
```python
"""Tests for the Service ABC and BaseService helper."""
from __future__ import annotations

import pytest

from sakura.events import OnEpochEnd, OnTrainBegin
from sakura.service import BaseService, Service


class TestServiceABC:
    def test_service_is_a_protocol(self):
        # A duck-typed object satisfying the Protocol should be accepted by isinstance.
        class _Duck:
            name = "duck"
            priority = 50
            requires: tuple[str, ...] = ()

            def on_install(self, runtime):
                pass

            def on_event(self, event):
                pass

        # runtime_checkable Protocol: isinstance returns True if attrs match.
        assert isinstance(_Duck(), Service)

    def test_base_service_provides_defaults(self):
        class MyService(BaseService):
            name = "my_service"
            priority = 50

            def on_event(self, event):
                pass

        s = MyService()
        assert s.name == "my_service"
        assert s.priority == 50
        assert s.requires == ()
        # on_install default is a no-op:
        s.on_install(runtime=None)

    def test_base_service_requires_name_and_priority(self):
        class _MissingName(BaseService):
            priority = 10

            def on_event(self, event):
                pass

        with pytest.raises(TypeError, match="name"):
            _MissingName()

        class _MissingPriority(BaseService):
            name = "x"

            def on_event(self, event):
                pass

        with pytest.raises(TypeError, match="priority"):
            _MissingPriority()

    def test_base_service_requires_on_event(self):
        class _NoOnEvent(BaseService):
            name = "x"
            priority = 10

        with pytest.raises(TypeError, match="on_event"):
            _NoOnEvent()

    def test_base_service_dispatches_by_event_type(self):
        """BaseService.on_event default routes via dispatch_<event_name> if defined."""
        seen: list[str] = []

        class Routing(BaseService):
            name = "routing"
            priority = 50

            def on_train_begin(self, event):
                seen.append(f"train_begin:{event.world_size}")

            def on_epoch_end(self, event):
                seen.append(f"epoch_end:{event.epoch}")

        s = Routing()
        s.on_event(OnTrainBegin(model=None, optimizer=None, train_loader=None,
                                rank=0, world_size=4))
        s.on_event(OnEpochEnd(epoch=2, model=None, optimizer=None, metrics={},
                              rank=0, world_size=4))
        assert seen == ["train_begin:4", "epoch_end:2"]

    def test_base_service_unknown_events_silently_ignored(self):
        """If a subclass doesn't define a dispatch method for an event, no error."""

        class Quiet(BaseService):
            name = "quiet"
            priority = 50

        s = Quiet()
        s.on_event(OnEpochEnd(epoch=0, model=None, optimizer=None, metrics={},
                              rank=0, world_size=1))  # no method, no crash
```

- [ ] **Step 2: Run the test — should fail**

```bash
pytest tests/runtime/test_service.py -v 2>&1 | tail -10
```
Expected: `ModuleNotFoundError`.

- [ ] **Step 3: Implement `sakura/service.py`**

```python
"""Service ABC + BaseService helper.

Services are installable units of behavior subscribing to events on a
SakuraRuntime. The Protocol form allows duck-typed compliance; BaseService
is a concrete helper that routes events to dispatch_<event_name> methods.
"""
from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from sakura.events import Event


@runtime_checkable
class Service(Protocol):
    """Duck-typed service contract.

    A service must expose `name`, `priority`, `requires`, `on_install(runtime)`,
    and `on_event(event)`. Lower priority runs earlier; ties resolve by
    install order. `requires` is a tuple of names of services this depends on.
    """
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

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        # Skip enforcement on abstract intermediates that intentionally don't
        # define on_event yet — only enforce on classes whose users instantiate.
        # We enforce in __init__ instead so subclasses-of-subclasses can still
        # configure; the *instantiable* class must satisfy the contract.

    def __init__(self):
        if not self.name:
            raise TypeError(f"{type(self).__name__}: 'name' must be set as a class attribute")
        if self.priority < 0:
            raise TypeError(f"{type(self).__name__}: 'priority' must be a non-negative int")
        # Verify on_event resolution: either the class defines on_event itself,
        # or it has at least one on_<event> dispatch method.
        if "on_event" not in type(self).__dict__:
            # Default on_event uses dispatch routing — that's fine even if no
            # on_<event> methods are defined (the service then no-ops on every event).
            pass

    def on_install(self, runtime: Any) -> None:
        """Default: no-op. Override for setup that needs the runtime reference."""

    def on_event(self, event: Event) -> None:
        """Route to `on_<event_name>` if present, else no-op."""
        method_name = event.name()
        method = getattr(self, method_name, None)
        if method is not None:
            method(event)


__all__ = ["Service", "BaseService"]
```

Note: the test `test_base_service_requires_on_event` expects a `TypeError` for a class that has neither `on_event` nor any dispatch methods. Update the `BaseService.__init__` to enforce this — replace the `pass` block at the end of `__init__` with:

```python
        # If the class defines on_event itself OR at least one dispatch method
        # whose name matches a known event, we're satisfied. Otherwise, error.
        if "on_event" not in type(self).__dict__:
            from sakura.events import (
                OnEpochBegin, OnEpochEnd, OnError, OnOptimizerStep, OnSave,
                OnTrainBegin, OnTrainEnd, OnTrainStepBegin,
            )
            event_names = {
                E.name() for E in (
                    OnEpochBegin, OnEpochEnd, OnError, OnOptimizerStep, OnSave,
                    OnTrainBegin, OnTrainEnd, OnTrainStepBegin,
                )
            }
            has_dispatch = any(
                callable(getattr(self, name, None)) and name in event_names
                for name in dir(self)
            )
            if not has_dispatch:
                raise TypeError(
                    f"{type(self).__name__}: must define on_event(event) "
                    f"or at least one on_<event_name> method"
                )
```

- [ ] **Step 4: Run the test — should pass**

```bash
pytest tests/runtime/test_service.py -v
```
Expected: 6 tests pass.

- [ ] **Step 5: Commit**

```bash
git add sakura/service.py tests/runtime/test_service.py
git commit -m "feat(runtime): Service Protocol + BaseService helper with on_<event_name> routing

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 4: `SakuraRuntime` — install/uninstall + event ordering

**Files:**
- Create: `sakura/runtime.py`, `tests/runtime/test_runtime_install_order.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/runtime/test_runtime_install_order.py`:
```python
"""Service install order, priority sorting, dependency resolution, dispatch order."""
from __future__ import annotations

import pytest

from sakura.events import OnEpochEnd
from sakura.runtime import SakuraRuntime
from sakura.service import BaseService


class _Recorder(BaseService):
    log: list[tuple[str, int]] = []

    def __init__(self, name: str, priority: int, requires: tuple = ()):
        # Set attrs before super().__init__ so the validator sees them.
        type(self).log = type(self).log  # ensure subclass attribute access
        self.name = name
        self.priority = priority
        self.requires = requires
        super().__init__()

    def on_epoch_end(self, event):
        type(self).log.append((self.name, event.epoch))


@pytest.fixture(autouse=True)
def _reset_recorder_log():
    _Recorder.log = []
    yield


class TestInstallOrder:
    def test_services_dispatch_in_priority_order(self):
        rt = SakuraRuntime()
        rt.install(_Recorder(name="late", priority=80))
        rt.install(_Recorder(name="early", priority=10))
        rt.install(_Recorder(name="middle", priority=50))
        rt.dispatch(OnEpochEnd(epoch=1, model=None, optimizer=None, metrics={},
                                rank=0, world_size=1))
        assert _Recorder.log == [("early", 1), ("middle", 1), ("late", 1)]

    def test_install_order_breaks_priority_ties(self):
        rt = SakuraRuntime()
        rt.install(_Recorder(name="first", priority=50))
        rt.install(_Recorder(name="second", priority=50))
        rt.install(_Recorder(name="third", priority=50))
        rt.dispatch(OnEpochEnd(epoch=0, model=None, optimizer=None, metrics={},
                                rank=0, world_size=1))
        assert _Recorder.log == [("first", 0), ("second", 0), ("third", 0)]

    def test_uninstall_removes_service_from_dispatch(self):
        rt = SakuraRuntime()
        rt.install(_Recorder(name="a", priority=10))
        rt.install(_Recorder(name="b", priority=20))
        rt.uninstall("a")
        rt.dispatch(OnEpochEnd(epoch=2, model=None, optimizer=None, metrics={},
                                rank=0, world_size=1))
        assert _Recorder.log == [("b", 2)]

    def test_install_duplicate_name_raises(self):
        rt = SakuraRuntime()
        rt.install(_Recorder(name="dup", priority=10))
        with pytest.raises(ValueError, match="already installed"):
            rt.install(_Recorder(name="dup", priority=20))

    def test_install_with_unmet_requires_raises(self):
        rt = SakuraRuntime()
        with pytest.raises(ValueError, match="requires.*missing"):
            rt.install(_Recorder(name="needs_a", priority=50, requires=("a",)))

    def test_install_with_satisfied_requires_succeeds(self):
        rt = SakuraRuntime()
        rt.install(_Recorder(name="a", priority=10))
        rt.install(_Recorder(name="b", priority=20, requires=("a",)))
        rt.dispatch(OnEpochEnd(epoch=0, model=None, optimizer=None, metrics={},
                                rank=0, world_size=1))
        assert _Recorder.log == [("a", 0), ("b", 0)]

    def test_uninstall_blocked_by_dependent(self):
        rt = SakuraRuntime()
        rt.install(_Recorder(name="a", priority=10))
        rt.install(_Recorder(name="b", priority=20, requires=("a",)))
        with pytest.raises(ValueError, match="depended on by"):
            rt.uninstall("a")

    def test_install_calls_on_install(self):
        installed = []

        class Installer(BaseService):
            name = "installer"
            priority = 50

            def on_install(self, runtime):
                installed.append(runtime)

            def on_event(self, event):
                pass

        rt = SakuraRuntime()
        s = Installer()
        rt.install(s)
        assert installed == [rt]

    def test_dispatch_before_install_is_safe_noop(self):
        rt = SakuraRuntime()
        # No services installed; dispatch must not raise.
        rt.dispatch(OnEpochEnd(epoch=0, model=None, optimizer=None, metrics={},
                                rank=0, world_size=1))

    def test_service_exception_does_not_block_others(self):
        class Boom(BaseService):
            name = "boom"
            priority = 30

            def on_epoch_end(self, event):
                raise RuntimeError("intentional")

        rt = SakuraRuntime()
        rt.install(_Recorder(name="early", priority=10))
        rt.install(Boom())
        rt.install(_Recorder(name="late", priority=80))
        rt.dispatch(OnEpochEnd(epoch=0, model=None, optimizer=None, metrics={},
                                rank=0, world_size=1))
        # boom raised, but early and late still ran:
        assert _Recorder.log == [("early", 0), ("late", 0)]
```

- [ ] **Step 2: Run the test — should fail**

```bash
pytest tests/runtime/test_runtime_install_order.py -v 2>&1 | tail -10
```
Expected: `ModuleNotFoundError: No module named 'sakura.runtime'`.

- [ ] **Step 3: Implement `sakura/runtime.py`**

```python
"""SakuraRuntime — central orchestrator for events + services.

The runtime is a synchronous event bus with priority-ordered service
dispatch. Services are installed once at startup; they receive every
event via on_event(event). Service exceptions are caught and emitted
as OnError events (which non-failing services see), then logged via
the optional logger callback.
"""
from __future__ import annotations

import logging
from typing import Callable, Optional

from sakura.events import Event, OnError
from sakura.service import Service

_log = logging.getLogger(__name__)


class SakuraRuntime:
    """Owns the event bus + installed services + telemetry sink."""

    def __init__(
        self,
        *,
        compute: Optional[object] = None,
        logger: Optional[Callable[[dict], None]] = None,
    ) -> None:
        self._compute = compute
        self._logger = logger
        # services in install order; sorted view rebuilt on install/uninstall.
        self._services: list[Service] = []
        # cached priority-sorted view (stable):
        self._sorted: list[Service] = []
        # service-name -> Service for quick lookup:
        self._by_name: dict[str, Service] = {}

    # ............................................................. lifecycle

    @property
    def compute(self) -> Optional[object]:
        return self._compute

    @property
    def services(self) -> tuple[Service, ...]:
        """Read-only view of installed services in priority order."""
        return tuple(self._sorted)

    def install(self, service: Service) -> None:
        """Install a service. Validates name uniqueness + `requires` deps."""
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

    def uninstall(self, name: str) -> None:
        """Uninstall a service by name. Refuses if another service requires it."""
        if name not in self._by_name:
            raise KeyError(f"service '{name}' not installed")
        # Check no remaining service still requires this one.
        dependents = [s.name for s in self._services if name in s.requires]
        if dependents:
            raise ValueError(
                f"service '{name}' depended on by: {sorted(dependents)}"
            )
        del self._by_name[name]
        self._services = [s for s in self._services if s.name != name]
        self._rebuild_sorted()

    def find(self, name: str) -> Optional[Service]:
        """Return the installed service with this name, or None."""
        return self._by_name.get(name)

    # ........................................................... dispatching

    def dispatch(self, event: Event) -> None:
        """Send an event to every installed service in priority order.

        Service exceptions are caught and re-emitted as OnError events; the
        original event continues to propagate to remaining services.
        """
        # Snapshot the sorted view so install/uninstall during dispatch is safe.
        services = list(self._sorted)
        errors: list[tuple[Service, BaseException]] = []
        for s in services:
            try:
                s.on_event(event)
            except BaseException as exc:  # noqa: BLE001 — service hygiene
                errors.append((s, exc))
                _log.exception("service '%s' raised during %s", s.name, type(event).__name__)
        # Surface errors as OnError events (each error fans out to all services).
        # Skip on OnError itself to avoid infinite loops.
        if errors and not isinstance(event, OnError):
            for svc, exc in errors:
                err_evt = OnError(
                    rank=event.rank,
                    world_size=event.world_size,
                    exc=exc,
                    context={"service": svc.name, "event": type(event).__name__},
                )
                self.dispatch(err_evt)
        if self._logger is not None:
            try:
                self._logger({"event": type(event).__name__, "n_services": len(services),
                               "n_errors": len(errors)})
            except Exception:
                pass

    # ............................................................. internals

    def _rebuild_sorted(self) -> None:
        """Sort services by (priority, install-order). Stable sort suffices."""
        # Keyed by (priority, position-in-self._services) for tie-break by install order.
        indexed = list(enumerate(self._services))
        indexed.sort(key=lambda pair: (pair[1].priority, pair[0]))
        self._sorted = [s for _, s in indexed]


__all__ = ["SakuraRuntime"]
```

- [ ] **Step 4: Run the test — should pass**

```bash
pytest tests/runtime/test_runtime_install_order.py -v
```
Expected: 10 tests pass.

- [ ] **Step 5: Commit**

```bash
git add sakura/runtime.py tests/runtime/test_runtime_install_order.py
git commit -m "feat(runtime): SakuraRuntime — install/uninstall, priority ordering, dispatch with error isolation

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 5: `SakuraRuntime` lifecycle — start, shutdown, context manager

**Files:**
- Modify: `sakura/runtime.py`
- Create: `tests/runtime/test_runtime_lifecycle.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/runtime/test_runtime_lifecycle.py`:
```python
"""SakuraRuntime start/shutdown/__enter__/__exit__ semantics."""
from __future__ import annotations

from sakura.runtime import SakuraRuntime
from sakura.service import BaseService


class _Hook(BaseService):
    name = "hook"
    priority = 50

    def __init__(self):
        super().__init__()
        self.started = False
        self.shutdown_called = False

    def on_runtime_start(self, runtime):
        self.started = True

    def on_runtime_shutdown(self, runtime):
        self.shutdown_called = True

    def on_event(self, event):
        pass


class TestLifecycle:
    def test_start_calls_on_runtime_start_for_each_service(self):
        rt = SakuraRuntime()
        h = _Hook()
        rt.install(h)
        rt.start()
        assert h.started is True

    def test_start_is_idempotent(self):
        rt = SakuraRuntime()
        rt.start()
        rt.start()  # second call is a no-op (must not raise)

    def test_shutdown_calls_on_runtime_shutdown(self):
        rt = SakuraRuntime()
        h = _Hook()
        rt.install(h)
        rt.start()
        rt.shutdown()
        assert h.shutdown_called is True

    def test_shutdown_without_start_is_safe_noop(self):
        rt = SakuraRuntime()
        rt.shutdown()  # safe even if not started

    def test_context_manager_starts_and_shuts_down(self):
        h = _Hook()
        with SakuraRuntime() as rt:
            rt.install(h)
            assert h.started is True
        assert h.shutdown_called is True

    def test_context_manager_shuts_down_on_exception(self):
        h = _Hook()
        try:
            with SakuraRuntime() as rt:
                rt.install(h)
                raise RuntimeError("user code crashed")
        except RuntimeError:
            pass
        assert h.shutdown_called is True

    def test_history_accumulates_dispatched_event_records(self):
        from sakura.events import OnEpochEnd

        rt = SakuraRuntime()
        rt.dispatch(OnEpochEnd(epoch=1, model=None, optimizer=None, metrics={"a": 1.0},
                                rank=0, world_size=1))
        rt.dispatch(OnEpochEnd(epoch=2, model=None, optimizer=None, metrics={"a": 0.9},
                                rank=0, world_size=1))
        h = rt.history()
        # We at minimum record event type + epoch-or-other-payload-summary.
        assert len(h) >= 2
        assert all(isinstance(record, dict) for record in h)
        assert any(record.get("event") == "OnEpochEnd" for record in h)
```

- [ ] **Step 2: Run the test — should fail**

```bash
pytest tests/runtime/test_runtime_lifecycle.py -v 2>&1 | tail -10
```
Expected: methods missing.

- [ ] **Step 3: Extend `sakura/runtime.py`**

Modify `sakura/runtime.py` — replace the contents with the extended version below (keeps everything from Task 4 + adds `start`, `shutdown`, `__enter__`, `__exit__`, `history`, plus calling `on_runtime_start` / `on_runtime_shutdown` on services that define them):

```python
"""SakuraRuntime — central orchestrator for events + services.

The runtime is a synchronous event bus with priority-ordered service
dispatch. Services are installed once at startup; they receive every
event via on_event(event). Service exceptions are caught and emitted
as OnError events (which non-failing services see), then logged via
the optional logger callback.
"""
from __future__ import annotations

import logging
from typing import Callable, Optional

from sakura.events import Event, OnError
from sakura.service import Service

_log = logging.getLogger(__name__)


class SakuraRuntime:
    """Owns the event bus + installed services + telemetry sink."""

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

    # ............................................................. lifecycle

    @property
    def compute(self) -> Optional[object]:
        return self._compute

    @property
    def services(self) -> tuple[Service, ...]:
        return tuple(self._sorted)

    def start(self) -> None:
        """Start the runtime: invoke on_runtime_start on every service that defines it."""
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
        """Shut down the runtime: invoke on_runtime_shutdown on every service that defines it."""
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

    # ........................................................ install/uninstall

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
        # If runtime is already started, also invoke on_runtime_start for late-installed services.
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
            raise ValueError(
                f"service '{name}' depended on by: {sorted(dependents)}"
            )
        del self._by_name[name]
        self._services = [s for s in self._services if s.name != name]
        self._rebuild_sorted()

    def find(self, name: str) -> Optional[Service]:
        return self._by_name.get(name)

    # ........................................................... dispatching

    def dispatch(self, event: Event) -> None:
        services = list(self._sorted)
        errors: list[tuple[Service, BaseException]] = []
        for s in services:
            try:
                s.on_event(event)
            except BaseException as exc:  # noqa: BLE001 — service hygiene
                errors.append((s, exc))
                _log.exception("service '%s' raised during %s", s.name, type(event).__name__)
        # record in history
        record = self._record_event(event, len(services), len(errors))
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

    # ............................................................. internals

    def _rebuild_sorted(self) -> None:
        indexed = list(enumerate(self._services))
        indexed.sort(key=lambda pair: (pair[1].priority, pair[0]))
        self._sorted = [s for _, s in indexed]

    def _record_event(self, event: Event, n_services: int, n_errors: int) -> dict:
        return {
            "event": type(event).__name__,
            "rank": event.rank,
            "world_size": event.world_size,
            "n_services": n_services,
            "n_errors": n_errors,
        }


__all__ = ["SakuraRuntime"]
```

- [ ] **Step 4: Run the tests — both test files now**

```bash
pytest tests/runtime/ -v
```
Expected: all tests pass (events + service + install_order + lifecycle = 27+ tests).

- [ ] **Step 5: Commit**

```bash
git add sakura/runtime.py tests/runtime/test_runtime_lifecycle.py
git commit -m "feat(runtime): SakuraRuntime start/shutdown/context-manager + history()

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 6: `Compute` URI tag class

**Files:**
- Create: `sakura/dispatch/compute.py`, `tests/dispatch/test_compute.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/dispatch/test_compute.py`:
```python
"""Compute is a URI-tag config class that resolves to a Dispatcher at runtime.start()."""
from __future__ import annotations

import pytest

from sakura.dispatch.compute import Compute


class TestCompute:
    def test_local_default(self):
        c = Compute.local()
        assert c.kind == "local"
        assert c.n_workers == 1
        assert c.gpus is None

    def test_local_with_gpus_and_pool(self):
        c = Compute.local(n_workers=2, gpus=[0, 1])
        assert c.kind == "local"
        assert c.n_workers == 2
        assert c.gpus == (0, 1)

    def test_at_quic_uri(self):
        c = Compute.at("quic://eval-1.lan:4433")
        assert c.kind == "remote"
        assert c.uris == ("quic://eval-1.lan:4433",)
        assert c.strategy == "round-robin"  # default for single-uri

    def test_at_rejects_non_quic(self):
        with pytest.raises(ValueError, match="quic://"):
            Compute.at("http://eval-1:8080")

    def test_pool_uris_and_strategy(self):
        c = Compute.pool(
            ["quic://e1:4433", "quic://e2:4433"],
            strategy="least-loaded",
        )
        assert c.kind == "remote"
        assert c.uris == ("quic://e1:4433", "quic://e2:4433")
        assert c.strategy == "least-loaded"

    def test_pool_rejects_unknown_strategy(self):
        with pytest.raises(ValueError, match="strategy"):
            Compute.pool(["quic://e1:4433"], strategy="random")

    def test_in_thread(self):
        c = Compute.in_thread()
        assert c.kind == "in_thread"

    def test_repr_is_compact(self):
        c = Compute.at("quic://localhost:4433")
        s = repr(c)
        assert "Compute" in s
        assert "quic://localhost:4433" in s
```

- [ ] **Step 2: Run the test — should fail**

```bash
pytest tests/dispatch/test_compute.py -v 2>&1 | tail -10
```
Expected: `ModuleNotFoundError`.

- [ ] **Step 3: Implement `sakura/dispatch/compute.py`**

```python
"""Compute — URI-tag config that resolves to a concrete Dispatcher.

Compute is purely a config object. Resolution to LocalDispatcher /
RemoteDispatcher / InThreadDispatcher happens at runtime.start() (or when
a service first asks for the dispatcher). This decoupling lets services
pre-declare their compute target without forcing transport setup at
import time.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional


_VALID_STRATEGIES = ("round-robin", "least-loaded")


@dataclass(frozen=True)
class Compute:
    """A tag describing where work should run.

    `kind` is one of "local" | "remote" | "in_thread". The other fields
    are populated based on kind (read them only when meaningful).
    """
    kind: Literal["local", "remote", "in_thread"]
    n_workers: int = 1
    gpus: Optional[tuple[int, ...]] = None
    uris: tuple[str, ...] = ()
    strategy: Literal["round-robin", "least-loaded"] = "round-robin"

    @classmethod
    def local(cls, *, n_workers: int = 1, gpus: Optional[list[int]] = None) -> "Compute":
        """Spawn N localhost worker subprocesses."""
        return cls(
            kind="local",
            n_workers=int(n_workers),
            gpus=tuple(gpus) if gpus is not None else None,
        )

    @classmethod
    def at(cls, uri: str) -> "Compute":
        """Connect to one already-running worker at `quic://host:port`."""
        if not uri.startswith("quic://"):
            raise ValueError(f"Compute.at requires a quic:// URI, got: {uri}")
        return cls(kind="remote", uris=(uri,), strategy="round-robin")

    @classmethod
    def pool(
        cls,
        uris: list[str],
        *,
        strategy: Literal["round-robin", "least-loaded"] = "round-robin",
    ) -> "Compute":
        """Connect to multiple workers; pick across them by `strategy`."""
        if strategy not in _VALID_STRATEGIES:
            raise ValueError(
                f"strategy must be one of {_VALID_STRATEGIES}, got: {strategy!r}"
            )
        for u in uris:
            if not u.startswith("quic://"):
                raise ValueError(f"all pool URIs must be quic://, got: {u}")
        return cls(kind="remote", uris=tuple(uris), strategy=strategy)

    @classmethod
    def in_thread(cls) -> "Compute":
        """Run handlers synchronously in the calling thread (debug/tests only)."""
        return cls(kind="in_thread")

    def __repr__(self) -> str:
        if self.kind == "local":
            return f"Compute.local(n_workers={self.n_workers}, gpus={self.gpus})"
        if self.kind == "remote":
            return f"Compute(kind=remote, uris={list(self.uris)!r}, strategy={self.strategy!r})"
        return f"Compute.{self.kind}()"


__all__ = ["Compute"]
```

- [ ] **Step 4: Run the test — should pass**

```bash
pytest tests/dispatch/test_compute.py -v
```
Expected: 8 tests pass.

- [ ] **Step 5: Commit**

```bash
git add sakura/dispatch/compute.py tests/dispatch/test_compute.py
git commit -m "feat(dispatch): Compute URI-tag config class — local/remote/pool/in_thread

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 7: `Dispatcher` ABC + `Future` + `Result`

**Files:**
- Create: `sakura/dispatch/base.py`

- [ ] **Step 1: Implement the ABCs (no test yet — these are interface only; tests come with concrete impls)**

Create `sakura/dispatch/base.py`:
```python
"""Dispatcher / Future / Result abstractions.

Concrete dispatchers (Local, Remote, InThread) implement Dispatcher.submit;
returned Futures resolve to Results. The high-level Python surface accepts
a Python callable + tensor args, cloudpickles the callable, and dispatches
HANDLER_EXEC_CLOUDPICKLED. Services can subclass + extend if they need a
different handler ID.
"""
from __future__ import annotations

import abc
from dataclasses import dataclass
from typing import Any, Callable, Optional


@dataclass
class Result:
    """The decoded result of an RPC.

    `value` is the cloudpickled return value of the dispatched callable.
    `elapsed_us` is worker-side wall time (from RpcResponseHeader).
    """
    value: Any
    elapsed_us: int = 0


class Future(abc.ABC):
    """Promise of a Result. Backed by a sakura_wire.Future or in-process oneshot."""

    @abc.abstractmethod
    def result(self, timeout: Optional[float] = None) -> Result:
        """Block until the result is ready or timeout (in seconds)."""

    @abc.abstractmethod
    def done(self) -> bool:
        """True iff the future has resolved (or was cancelled)."""

    @abc.abstractmethod
    def cancel(self) -> bool:
        """Best-effort cancellation. Returns True if it was newly cancelled."""


class Dispatcher(abc.ABC):
    """Submit cloudpickled callables to a worker (local, remote, or in-thread)."""

    @abc.abstractmethod
    def submit(
        self,
        callable: Callable[..., Any],
        *args: Any,
        timeout_ms: Optional[int] = None,
        **kwargs: Any,
    ) -> Future:
        """Dispatch `callable(*args, **kwargs)` to the worker. Returns a Future."""

    def shutdown(self, *, timeout_s: float = 30.0) -> None:
        """Default: no-op. Subclasses with subprocesses or sockets override."""

    def stats(self) -> dict:
        """Default: empty stats. Subclasses may report queue depth, RTT, etc."""
        return {}


__all__ = ["Dispatcher", "Future", "Result"]
```

- [ ] **Step 2: Quick syntax check**

```bash
python3 -c "from sakura.dispatch.base import Dispatcher, Future, Result; print(Dispatcher, Future, Result)"
```
Expected: prints three class objects.

- [ ] **Step 3: Commit**

```bash
git add sakura/dispatch/base.py
git commit -m "feat(dispatch): Dispatcher / Future / Result ABCs

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 8: `InThreadDispatcher` (in-process synchronous)

**Files:**
- Create: `sakura/dispatch/in_thread.py`, `tests/dispatch/test_in_thread.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/dispatch/test_in_thread.py`:
```python
"""InThreadDispatcher: runs callables synchronously in the calling thread."""
from __future__ import annotations

import pytest

from sakura.dispatch.in_thread import InThreadDispatcher


def _double(x):
    return x * 2


def _add(a, b, *, scale=1):
    return (a + b) * scale


def _raises():
    raise ValueError("intentional")


class TestInThreadDispatcher:
    def test_submit_runs_synchronously_and_returns_done_future(self):
        d = InThreadDispatcher()
        fut = d.submit(_double, 21)
        assert fut.done() is True  # synchronous = done immediately
        result = fut.result()
        assert result.value == 42
        assert result.elapsed_us >= 0

    def test_submit_propagates_kwargs(self):
        d = InThreadDispatcher()
        fut = d.submit(_add, 3, 4, scale=10)
        assert fut.result().value == 70

    def test_submit_propagates_exceptions_via_result(self):
        d = InThreadDispatcher()
        fut = d.submit(_raises)
        with pytest.raises(ValueError, match="intentional"):
            fut.result()

    def test_cancel_after_done_returns_false(self):
        d = InThreadDispatcher()
        fut = d.submit(_double, 5)
        # Already resolved — cancel is a no-op.
        assert fut.cancel() is False

    def test_shutdown_is_safe_noop(self):
        d = InThreadDispatcher()
        d.shutdown()  # no resources to release
```

- [ ] **Step 2: Run the test — should fail**

```bash
pytest tests/dispatch/test_in_thread.py -v 2>&1 | tail -10
```
Expected: `ModuleNotFoundError`.

- [ ] **Step 3: Implement `sakura/dispatch/in_thread.py`**

```python
"""InThreadDispatcher — synchronous in-process execution for tests/debug.

Useful as a Dispatcher impl that doesn't require any IPC, so service
unit tests can verify behavior without spawning a subprocess.
"""
from __future__ import annotations

import time
from typing import Any, Callable, Optional

from sakura.dispatch.base import Dispatcher, Future, Result


class _ResolvedFuture(Future):
    """A Future that's already resolved (or already errored)."""

    def __init__(self, value: Any = None, exc: Optional[BaseException] = None,
                 elapsed_us: int = 0):
        self._value = value
        self._exc = exc
        self._elapsed_us = elapsed_us

    def result(self, timeout: Optional[float] = None) -> Result:
        if self._exc is not None:
            raise self._exc
        return Result(value=self._value, elapsed_us=self._elapsed_us)

    def done(self) -> bool:
        return True

    def cancel(self) -> bool:
        return False  # already done


class InThreadDispatcher(Dispatcher):
    """Run the callable synchronously in the caller's thread."""

    def submit(
        self,
        callable: Callable[..., Any],
        *args: Any,
        timeout_ms: Optional[int] = None,
        **kwargs: Any,
    ) -> Future:
        t0 = time.perf_counter_ns()
        try:
            value = callable(*args, **kwargs)
        except BaseException as exc:  # noqa: BLE001 — we surface in result()
            return _ResolvedFuture(exc=exc, elapsed_us=(time.perf_counter_ns() - t0) // 1000)
        return _ResolvedFuture(value=value, elapsed_us=(time.perf_counter_ns() - t0) // 1000)

    def shutdown(self, *, timeout_s: float = 30.0) -> None:
        pass

    def stats(self) -> dict:
        return {"kind": "in_thread"}


__all__ = ["InThreadDispatcher"]
```

- [ ] **Step 4: Run the test — should pass**

```bash
pytest tests/dispatch/test_in_thread.py -v
```
Expected: 5 tests pass.

- [ ] **Step 5: Commit**

```bash
git add sakura/dispatch/in_thread.py tests/dispatch/test_in_thread.py
git commit -m "feat(dispatch): InThreadDispatcher — synchronous in-process for tests

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 9: Rust `run_server` PyO3 fn — replace echo-only with Python callback dispatch

**Files:**
- Modify: `crates/sakura-wire/src/pyo3_bindings.rs`, `crates/sakura-wire/src/lib.rs`

This task extends the Rust transport: a new `run_server(addr, callback, print_handshake)` function takes a Python callable that gets invoked on every RPC. The existing `run_echo_server` stays for backward compatibility / test convenience.

- [ ] **Step 1: Append `run_server` to `crates/sakura-wire/src/pyo3_bindings.rs`**

Add this function after the existing `run_echo_server` definition. Note: the imports at the top of the file already include `PyAny`, `Python`, `PyObject`, `PyResult`, etc.

```rust
/// Bind a QUIC server, print one handshake line on stdout (if requested),
/// then dispatch every RPC to the Python `callback` until shutdown.
///
/// `callback` is invoked with three positional args:
///   - handler_id: int
///   - tensors: list[dict]   (each dict = {"shape": list[int], "dtype_id": int,
///                            "device_id": int, "data": bytes})
///   - aux_payload: bytes
/// and must return a 2-tuple `(result_tensors, result_aux_bytes)` of the same shape.
///
/// Plan 2's HandlerRegistry on the Python side is the canonical implementation;
/// run_server is the thin Rust shim.
#[pyfunction]
#[pyo3(signature = (addr, callback, print_handshake = true))]
pub fn run_server(
    py: Python<'_>,
    addr: String,
    callback: PyObject,
    print_handshake: bool,
) -> PyResult<()> {
    let cb = std::sync::Arc::new(callback);
    py.allow_threads(|| -> PyResult<()> {
        WireRuntime::shared().block_on(async move {
            let pair = generate_self_signed("localhost")
                .map_err(|e| PyRuntimeError::new_err(format!("cert: {e}")))?;
            let bind_addr: SocketAddr = addr
                .parse()
                .map_err(|e| PyRuntimeError::new_err(format!("addr {addr}: {e}")))?;
            let endpoint = bind_server(bind_addr, &pair)
                .map_err(|e| PyRuntimeError::new_err(format!("bind: {e}")))?;
            let local = endpoint
                .local_addr()
                .map_err(|e| PyRuntimeError::new_err(format!("local_addr: {e}")))?;
            if print_handshake {
                let cert_hex = encode_hex(&pair.cert_der);
                println!("SAKURA_WORKER_LISTENING quic://{local} {cert_hex}");
                let _ = std::io::Write::flush(&mut std::io::stdout().lock());
            }

            while let Some(incoming) = endpoint.accept().await {
                let cb = std::sync::Arc::clone(&cb);
                tokio::spawn(async move {
                    match incoming.await {
                        Ok(conn) => loop {
                            let cb = std::sync::Arc::clone(&cb);
                            match accept_request(&conn).await {
                                Ok((send, req)) => {
                                    let resp = match dispatch_via_callback(&cb, &req) {
                                        Ok(b) => b,
                                        Err(e) => {
                                            tracing::error!("dispatch: {e:?}");
                                            return;
                                        }
                                    };
                                    if let Err(e) = send_response(send, resp).await {
                                        tracing::error!("send_response: {e:?}");
                                        return;
                                    }
                                }
                                Err(_) => return,
                            }
                        },
                        Err(e) => tracing::error!("connection: {e:?}"),
                    }
                });
            }
            Ok(())
        })
    })
}

/// Decode an RPC request, invoke the Python callback under the GIL, and
/// re-encode its response.
fn dispatch_via_callback(callback: &PyObject, req_bytes: &[u8]) -> Result<Vec<u8>, PyWireError> {
    use crate::codec::{unpack_request, RpcResponseHeader, RpcStatus, TensorDesc, WireVersion};
    let (header, descs, tensors, aux) = unpack_request(req_bytes).map_err(|e| {
        PyWireError::Wire(WireError::DecodeFailed {
            what: "server unpack".into(),
            detail: e.to_string(),
        })
    })?;

    // Acquire the GIL and call the Python callback.
    let (out_tensors_descs, out_tensors_bytes, out_aux): (Vec<TensorDesc>, Vec<Vec<u8>>, Vec<u8>) =
        Python::with_gil(|py| -> PyResult<_> {
            // Build the tensors list as Python list[dict].
            let py_tensors = pyo3::types::PyList::empty_bound(py);
            for (desc, bytes) in descs.iter().zip(tensors.iter()) {
                let d = pyo3::types::PyDict::new_bound(py);
                d.set_item("shape", desc.shape.iter().copied().collect::<Vec<u32>>())?;
                d.set_item("dtype_id", desc.dtype as u8)?;
                d.set_item(
                    "device_id",
                    match desc.device_hint {
                        crate::codec::Device::Cpu => 0u8,
                        crate::codec::Device::Cuda(i) => i + 1,
                    },
                )?;
                d.set_item("data", PyBytes::new_bound(py, bytes))?;
                py_tensors.append(d)?;
            }
            let py_aux = PyBytes::new_bound(py, &aux);
            let result = callback.call1(py, (header.handler_id, py_tensors, py_aux))?;
            // Result is expected to be (list[dict], bytes).
            let tup: (Vec<TensorTuple>, Vec<u8>) = result.extract(py)?;
            // Convert TensorTuple back to descriptors + bytes.
            let descs: Vec<TensorDesc> = tup
                .0
                .iter()
                .map(|t| {
                    use crate::codec::{Device, Dtype};
                    let dtype = match t.dtype_id {
                        0 => Dtype::F32,
                        1 => Dtype::F16,
                        2 => Dtype::BF16,
                        3 => Dtype::F8E4M3,
                        10 => Dtype::I64,
                        11 => Dtype::I32,
                        12 => Dtype::U8,
                        13 => Dtype::Bool,
                        _ => Dtype::U8,
                    };
                    let device = match t.device_id {
                        0 => Device::Cpu,
                        other => Device::Cuda(other - 1),
                    };
                    TensorDesc {
                        shape: t.shape.clone().into(),
                        dtype,
                        n_bytes: t.data.len() as u64,
                        device_hint: device,
                        fp16_cast_on_wire: false,
                    }
                })
                .collect();
            let bytes: Vec<Vec<u8>> = tup.0.into_iter().map(|t| t.data).collect();
            Ok((descs, bytes, tup.1))
        })
        .map_err(|e: PyErr| {
            PyWireError::Wire(WireError::HandlerPanic {
                msg: e.to_string(),
                trace: vec![],
            })
        })?;

    // Encode the response.
    let resp_header = RpcResponseHeader {
        version: WireVersion::V1,
        request_id: header.request_id,
        status: RpcStatus::Ok,
        n_result_tensors: out_tensors_descs.len() as u32,
        aux_payload_bytes: out_aux.len() as u32,
        elapsed_us: 0,
    };
    let header_bytes = postcard::to_allocvec(&resp_header).map_err(|e| {
        PyWireError::Wire(WireError::DecodeFailed {
            what: "encode resp header".into(),
            detail: e.to_string(),
        })
    })?;
    let descs_bytes = postcard::to_allocvec(&out_tensors_descs).map_err(|e| {
        PyWireError::Wire(WireError::DecodeFailed {
            what: "encode resp descs".into(),
            detail: e.to_string(),
        })
    })?;
    let total = 4
        + header_bytes.len()
        + 4
        + descs_bytes.len()
        + out_tensors_bytes.iter().map(Vec::len).sum::<usize>()
        + out_aux.len();
    let mut out = Vec::with_capacity(total);
    out.extend_from_slice(&(header_bytes.len() as u32).to_le_bytes());
    out.extend_from_slice(&header_bytes);
    out.extend_from_slice(&(descs_bytes.len() as u32).to_le_bytes());
    out.extend_from_slice(&descs_bytes);
    for t in &out_tensors_bytes {
        out.extend_from_slice(t);
    }
    out.extend_from_slice(&out_aux);
    Ok(out)
}
```

- [ ] **Step 2: Wire `run_server` into the `#[pymodule]`**

Edit `crates/sakura-wire/src/lib.rs` — add `m.add_function(wrap_pyfunction!(pyo3_bindings::run_server, m)?)?;` after the existing `run_echo_server` registration. Final pymodule body:

```rust
#[pymodule]
fn sakura_wire(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    m.add_class::<pyo3_bindings::PyDispatcher>()?;
    m.add_class::<pyo3_bindings::PyFuture>()?;
    m.add_class::<pyo3_bindings::PyRpcResult>()?;
    m.add_class::<pyo3_bindings::PyTlsConfig>()?;
    m.add_class::<pyo3_bindings::PyWorkerSupervisor>()?;
    m.add_function(wrap_pyfunction!(pyo3_bindings::run_echo_server, m)?)?;
    m.add_function(wrap_pyfunction!(pyo3_bindings::run_server, m)?)?;
    Ok(())
}
```

- [ ] **Step 3: Build + smoke test**

```bash
export PATH="$HOME/.cargo/bin:$PATH"
source .venv/bin/activate
maturin develop --release
python3 -c "import sakura_wire; print(sakura_wire.run_server)"
```
Expected: prints something like `<built-in function run_server>`.

Run all existing tests to verify no regression:
```bash
cargo test -p sakura-wire
pytest tests/wire/
```
Expected: 17 cargo + 2 pytest still pass.

Verify clippy clean:
```bash
cargo clippy --workspace --all-targets -- -D warnings
```

- [ ] **Step 4: Commit**

```bash
git add crates/sakura-wire/src/pyo3_bindings.rs crates/sakura-wire/src/lib.rs
git commit -m "feat(wire): run_server PyO3 fn — dispatches every RPC to a Python callback

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 10: `HandlerRegistry` (`sakura/worker/registry.py`)

**Files:**
- Create: `sakura/worker/registry.py`, `tests/worker/test_registry.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/worker/test_registry.py`:
```python
"""HandlerRegistry: routes incoming RPCs to per-handler-id Python callables."""
from __future__ import annotations

import pytest

from sakura.worker.registry import HandlerRegistry


def _echo(tensors, aux):
    return (tensors, aux)


def _double_tensors(tensors, aux):
    out = []
    for t in tensors:
        out.append({**t, "data": bytes(b * 2 for b in t["data"])})
    return (out, aux)


class TestHandlerRegistry:
    def test_register_and_dispatch_known_handler(self):
        r = HandlerRegistry()
        r.register(0xDEAD, _echo)
        out_t, out_a = r.dispatch(0xDEAD, [{"shape": [1], "dtype_id": 12,
                                             "device_id": 0, "data": b"\x01"}], b"aux")
        assert out_t[0]["data"] == b"\x01"
        assert out_a == b"aux"

    def test_dispatch_unknown_handler_raises(self):
        r = HandlerRegistry()
        with pytest.raises(KeyError, match="0x1234"):
            r.dispatch(0x1234, [], b"")

    def test_double_register_replaces(self):
        r = HandlerRegistry()
        r.register(0xDEAD, _echo)
        r.register(0xDEAD, _double_tensors)
        out_t, _ = r.dispatch(0xDEAD,
                               [{"shape": [3], "dtype_id": 12, "device_id": 0, "data": b"\x01\x02\x03"}],
                               b"")
        assert out_t[0]["data"] == b"\x02\x04\x06"

    def test_handler_id_must_be_int(self):
        r = HandlerRegistry()
        with pytest.raises(TypeError, match="handler_id"):
            r.register("DEAD", _echo)

    def test_handler_must_be_callable(self):
        r = HandlerRegistry()
        with pytest.raises(TypeError, match="callable"):
            r.register(0xDEAD, "not a callable")

    def test_dispatch_returns_2_tuple_check(self):
        """If a handler returns the wrong shape, registry surfaces a clear error."""

        def _bad(tensors, aux):
            return tensors  # not a 2-tuple

        r = HandlerRegistry()
        r.register(0xBEEF, _bad)
        with pytest.raises(ValueError, match="must return.*2-tuple"):
            r.dispatch(0xBEEF, [], b"")
```

- [ ] **Step 2: Run the test — should fail**

```bash
pytest tests/worker/test_registry.py -v 2>&1 | tail -10
```

- [ ] **Step 3: Implement `sakura/worker/registry.py`**

```python
"""HandlerRegistry: in-process dispatch table for worker-side handlers.

The registry is the single Python entry-point that the Rust QUIC server
calls into for every RPC. It looks up the handler by `handler_id`, calls
it with `(tensors, aux_bytes)`, and validates the return shape.
"""
from __future__ import annotations

from typing import Callable, Tuple

# Type aliases for clarity.
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
```

- [ ] **Step 4: Run the test — should pass**

```bash
pytest tests/worker/test_registry.py -v
```
Expected: 6 tests pass.

- [ ] **Step 5: Commit**

```bash
git add sakura/worker/registry.py tests/worker/test_registry.py
git commit -m "feat(worker): HandlerRegistry — handler_id → callable dispatch table

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 11: Default handlers — echo, heartbeat, exec_cloudpickled

**Files:**
- Create: `sakura/worker/handlers.py`

- [ ] **Step 1: Implement default handlers**

Create `sakura/worker/handlers.py`:
```python
"""Default handler implementations registered on every sakura-worker.

- HANDLER_ECHO (0xDEAD): bounces tensors and aux back unchanged. Used by tests.
- HANDLER_HEARTBEAT (0x0003): returns empty tensors + b"PONG" aux. Used by
  the supervisor for health checks.
- HANDLER_EXEC_CLOUDPICKLED (0x0001): cloudpickle.loads(aux) -> {"fn", "args",
  "kwargs"}, reconstructs tensor args from the tensors list as numpy arrays,
  invokes fn(*tensor_args, *args, **kwargs), and returns the cloudpickled
  return value via aux.

`default_registry()` returns a HandlerRegistry pre-populated with all three.
"""
from __future__ import annotations

from typing import Tuple

import cloudpickle
import numpy as np

from sakura.worker.registry import HandlerRegistry

HANDLER_EXEC_CLOUDPICKLED = 0x0001
HANDLER_HEARTBEAT = 0x0003
HANDLER_ECHO = 0xDEAD

# dtype_id (matching the Dtype repr(u8) values in sakura-wire's codec) -> numpy dtype
_DTYPE_TABLE = {
    0: np.float32,
    1: np.float16,
    # 2 = BF16: numpy has no native bf16; fall back to uint16 view.
    2: np.uint16,
    3: np.uint8,   # F8E4M3 is a placeholder dtype for now; bytes-level handling.
    10: np.int64,
    11: np.int32,
    12: np.uint8,
    13: np.bool_,
}


def _tensors_to_arrays(tensors: list[dict]) -> list[np.ndarray]:
    out: list[np.ndarray] = []
    for t in tensors:
        dtype = _DTYPE_TABLE.get(t["dtype_id"], np.uint8)
        arr = np.frombuffer(t["data"], dtype=dtype)
        if t["shape"]:
            arr = arr.reshape(tuple(t["shape"]))
        out.append(arr)
    return out


def _array_to_tensor_dict(arr: np.ndarray) -> dict:
    """Pack a numpy array into the tensor-dict shape the Rust codec expects."""
    # Find a dtype_id by reverse lookup; fall back to U8 (12) if unknown.
    rev = {v: k for k, v in _DTYPE_TABLE.items()}
    dtype_id = rev.get(arr.dtype.type, 12)
    return {
        "shape": list(arr.shape),
        "dtype_id": dtype_id,
        "device_id": 0,  # always CPU on the way back
        "data": bytes(arr.tobytes()),
    }


def handle_echo(tensors: list[dict], aux: bytes) -> Tuple[list[dict], bytes]:
    """Bounce tensors and aux back unchanged."""
    return (list(tensors), bytes(aux))


def handle_heartbeat(tensors: list[dict], aux: bytes) -> Tuple[list[dict], bytes]:
    """Liveness probe response."""
    return ([], b"PONG")


def handle_exec_cloudpickled(tensors: list[dict], aux: bytes) -> Tuple[list[dict], bytes]:
    """Run a cloudpickled callable + args. Aux on the wire = cloudpickled spec dict.

    Spec format:
      {"fn": <callable>, "args": (...), "kwargs": {...}}
    The decoded numpy arrays from `tensors` are passed as positional args BEFORE
    `args` so callers write `submit(fn, np_arr_1, np_arr_2, scalar=...)`.
    """
    spec = cloudpickle.loads(aux)
    fn = spec["fn"]
    extra_args = spec.get("args", ())
    kwargs = spec.get("kwargs", {})
    arrays = _tensors_to_arrays(tensors)
    out = fn(*arrays, *extra_args, **kwargs)
    # Convention: by default we ship the return via aux (cloudpickled). If the
    # function explicitly returns a tuple `(result_arrays, result_aux_obj)` we
    # split — but that's an opt-in convention, not enforced here. Plan 3
    # services may use a richer wrapping convention.
    return ([], cloudpickle.dumps(out))


def default_registry() -> HandlerRegistry:
    r = HandlerRegistry()
    r.register(HANDLER_ECHO, handle_echo)
    r.register(HANDLER_HEARTBEAT, handle_heartbeat)
    r.register(HANDLER_EXEC_CLOUDPICKLED, handle_exec_cloudpickled)
    return r


__all__ = [
    "HANDLER_ECHO",
    "HANDLER_EXEC_CLOUDPICKLED",
    "HANDLER_HEARTBEAT",
    "default_registry",
    "handle_echo",
    "handle_exec_cloudpickled",
    "handle_heartbeat",
]
```

- [ ] **Step 2: Quick smoke**

```bash
source .venv/bin/activate
python3 -c "from sakura.worker.handlers import default_registry; r = default_registry(); print(r.dispatch(0x0003, [], b''))"
```
Expected: prints `([], b'PONG')`.

- [ ] **Step 3: Commit**

```bash
git add sakura/worker/handlers.py
git commit -m "feat(worker): default handlers — echo, heartbeat, exec_cloudpickled

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 12: Update `sakura/worker/__main__.py` to use the registry

**Files:**
- Modify: `sakura/worker/__main__.py`, `sakura/worker/__init__.py`

- [ ] **Step 1: Replace `sakura/worker/__main__.py`**

```python
"""`sakura-worker` daemon — QUIC server that dispatches via HandlerRegistry.

Plan 1 used `run_echo_server` (only HANDLER_ECHO). Plan 2 swaps to
`run_server` with the full default registry (echo + heartbeat +
exec_cloudpickled), so cloudpickled callables can be dispatched
end-to-end.

Run via:
    sakura-worker --listen quic://127.0.0.1:0
"""
from __future__ import annotations

import argparse


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="sakura-worker")
    parser.add_argument(
        "--listen",
        default="quic://127.0.0.1:0",
        help="Bind address (use port :0 for an ephemeral port; default).",
    )
    parser.add_argument(
        "--no-handshake",
        action="store_true",
        default=False,
        help="Suppress the SAKURA_WORKER_LISTENING handshake line on stdout.",
    )
    parser.add_argument(
        "--echo-only",
        action="store_true",
        default=False,
        help="Run only the echo handler (Plan 1 mode). Default: full registry.",
    )
    args = parser.parse_args(argv)

    if not args.listen.startswith("quic://"):
        raise ValueError(f"--listen must be a quic:// URI, got: {args.listen}")
    addr = args.listen[len("quic://"):]
    print_handshake = not args.no_handshake

    import sakura_wire as _native

    if args.echo_only:
        _native.run_echo_server(addr=addr, print_handshake=print_handshake)
    else:
        from sakura.worker.handlers import default_registry
        registry = default_registry()
        _native.run_server(addr=addr, callback=registry.dispatch, print_handshake=print_handshake)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 2: Update `sakura/worker/__init__.py`**

```python
"""sakura.worker — daemon entry point used by the WorkerSupervisor.

Plan 2 uses a full handler registry (echo + heartbeat + exec_cloudpickled).
Plan 1's echo-only mode is still available via `sakura-worker --echo-only`.
"""
from sakura.worker.handlers import default_registry
from sakura.worker.registry import HandlerRegistry

__all__ = ["HandlerRegistry", "default_registry", "main"]


def main(argv: list[str] | None = None) -> int:
    """Re-exported `main` so existing entry-points keep resolving cleanly."""
    from sakura.worker.__main__ import main as _main
    return _main(argv)
```

- [ ] **Step 3: Smoke test the daemon**

```bash
source .venv/bin/activate
( sakura-worker --listen quic://127.0.0.1:0 & WPID=$!; sleep 1; kill $WPID 2>/dev/null; wait $WPID 2>/dev/null )
```
Expected: prints one `SAKURA_WORKER_LISTENING ...` line and exits cleanly.

Existing Plan 1 e2e test must still pass (echo path):
```bash
pytest tests/wire/ -v
```
Expected: 2/2 pass (the test uses HANDLER_ECHO which the new registry includes).

- [ ] **Step 4: Commit**

```bash
git add sakura/worker/__main__.py sakura/worker/__init__.py
git commit -m "feat(worker): replace echo-only daemon with full HandlerRegistry server

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 13: `RemoteDispatcher` (talks to existing `quic://` daemon)

**Files:**
- Create: `sakura/dispatch/remote.py`, `tests/dispatch/test_remote.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/dispatch/test_remote.py`:
```python
"""RemoteDispatcher: connects to an existing sakura-worker via quic://."""
from __future__ import annotations

import sys

import pytest

sakura_wire = pytest.importorskip("sakura_wire")

from sakura.dispatch.remote import RemoteDispatcher


def _square(x):
    return int(x.sum()) ** 2


def test_remote_dispatcher_round_trips_a_callable():
    """Spin up a worker via the supervisor, point a RemoteDispatcher at it, run a callable."""
    import numpy as np

    sup = sakura_wire.WorkerSupervisor(shutdown_timeout_s=5.0)
    try:
        uri, cert = sup.spawn(
            cmd=[sys.executable, "-m", "sakura.worker", "--listen", "quic://127.0.0.1:0"],
            startup_timeout_s=10.0,
        )
        d = RemoteDispatcher(uri=uri, cert_der=cert, server_name="localhost")
        try:
            fut = d.submit(_square, np.array([1, 2, 3, 4], dtype=np.int64))
            result = fut.result(timeout=5.0)
            assert result.value == (1 + 2 + 3 + 4) ** 2
        finally:
            d.shutdown()
    finally:
        sup.shutdown()


def test_remote_dispatcher_rejects_non_quic_uri():
    with pytest.raises(ValueError, match="quic://"):
        RemoteDispatcher(uri="http://localhost:8080", cert_der=b"", server_name="localhost")
```

- [ ] **Step 2: Run the test — should fail**

```bash
pytest tests/dispatch/test_remote.py -v 2>&1 | tail -10
```

- [ ] **Step 3: Implement `sakura/dispatch/remote.py`**

```python
"""RemoteDispatcher — Python wrapper over sakura_wire.Dispatcher.

Talks to an *already-running* sakura-worker at a known quic:// URI with a
known self-signed cert. Pairs with WorkerSupervisor when you spawned the
worker yourself; pairs with manual `sakura-worker` invocations for cluster
deployments.
"""
from __future__ import annotations

from typing import Any, Callable, Optional

import cloudpickle
import numpy as np
import sakura_wire as _native

from sakura.dispatch.base import Dispatcher, Future, Result

HANDLER_EXEC_CLOUDPICKLED = 0x0001


class _WireFuture(Future):
    """Wraps a sakura_wire.Future and decodes the cloudpickled return value."""

    def __init__(self, native_future: _native.Future):
        self._fut = native_future

    def result(self, timeout: Optional[float] = None) -> Result:
        wire_result = self._fut.result(timeout=timeout)
        # wire_result.aux is cloudpickled return value
        value = cloudpickle.loads(wire_result.aux)
        if isinstance(value, BaseException):
            raise value
        return Result(value=value, elapsed_us=wire_result.elapsed_us)

    def done(self) -> bool:
        return self._fut.done()

    def cancel(self) -> bool:
        return self._fut.cancel()


def _array_to_tensor_dict(arr: np.ndarray) -> dict:
    """Pack a numpy array into the tensor-dict shape sakura-wire expects."""
    # Reverse-lookup of dtype -> dtype_id
    if arr.dtype == np.float32:
        dtype_id = 0
    elif arr.dtype == np.float16:
        dtype_id = 1
    elif arr.dtype == np.int64:
        dtype_id = 10
    elif arr.dtype == np.int32:
        dtype_id = 11
    elif arr.dtype == np.bool_:
        dtype_id = 13
    else:
        dtype_id = 12  # U8 fallback
    return {
        "shape": [int(d) for d in arr.shape],
        "dtype_id": dtype_id,
        "device_id": 0,  # CPU
        "data": bytes(arr.tobytes()),
    }


class RemoteDispatcher(Dispatcher):
    """Connect to a running sakura-worker at `uri` (quic://host:port)."""

    def __init__(self, *, uri: str, cert_der: bytes, server_name: str = "localhost"):
        if not uri.startswith("quic://"):
            raise ValueError(f"RemoteDispatcher requires quic:// URI, got: {uri}")
        self._uri = uri
        tls = _native.TlsConfig(cert_der, server_name)
        self._dispatcher = _native.Dispatcher(uri, tls)

    def submit(
        self,
        callable: Callable[..., Any],
        *args: Any,
        timeout_ms: Optional[int] = None,
        **kwargs: Any,
    ) -> Future:
        """Cloudpickle the callable + non-tensor args; ship tensor args as raw bytes."""
        tensor_args: list[np.ndarray] = []
        non_tensor_args: list[Any] = []
        for a in args:
            if isinstance(a, np.ndarray):
                tensor_args.append(a)
            else:
                non_tensor_args.append(a)
        spec = {"fn": callable, "args": tuple(non_tensor_args), "kwargs": kwargs}
        aux = cloudpickle.dumps(spec)
        tensor_dicts = [_array_to_tensor_dict(a) for a in tensor_args]
        native_fut = self._dispatcher.submit(
            HANDLER_EXEC_CLOUDPICKLED, tensor_dicts, aux, timeout_ms
        )
        return _WireFuture(native_fut)

    def shutdown(self, *, timeout_s: float = 30.0) -> None:
        # The native Dispatcher manages its own connection lifecycle.
        # No explicit shutdown call exposed; future GC handles it.
        pass

    def stats(self) -> dict:
        return {"kind": "remote", "uri": self._uri}


__all__ = ["RemoteDispatcher"]
```

- [ ] **Step 4: Run the test — should pass**

```bash
source .venv/bin/activate
pytest tests/dispatch/test_remote.py -v
```
Expected: 2 tests pass.

- [ ] **Step 5: Commit**

```bash
git add sakura/dispatch/remote.py tests/dispatch/test_remote.py
git commit -m "feat(dispatch): RemoteDispatcher — cloudpickle callable + tensor args over QUIC

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 14: `LocalDispatcher` (auto-spawn worker)

**Files:**
- Create: `sakura/dispatch/local.py`, `tests/dispatch/test_local.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/dispatch/test_local.py`:
```python
"""LocalDispatcher: spawns localhost sakura-worker on first use."""
from __future__ import annotations

import sys

import pytest

sakura_wire = pytest.importorskip("sakura_wire")

from sakura.dispatch.local import LocalDispatcher


def _add(a, b):
    return int(a.sum()) + int(b.sum())


def test_local_dispatcher_auto_spawns_and_round_trips():
    import numpy as np

    d = LocalDispatcher()
    try:
        fut = d.submit(_add,
                       np.array([1, 2, 3], dtype=np.int64),
                       np.array([10, 20, 30], dtype=np.int64))
        result = fut.result(timeout=10.0)
        assert result.value == 6 + 60
    finally:
        d.shutdown()


def test_local_dispatcher_shutdown_idempotent():
    d = LocalDispatcher()
    d.shutdown()
    d.shutdown()  # second call must not raise
```

- [ ] **Step 2: Run the test — should fail**

```bash
pytest tests/dispatch/test_local.py -v 2>&1 | tail -10
```

- [ ] **Step 3: Implement `sakura/dispatch/local.py`**

```python
"""LocalDispatcher — auto-spawn a localhost sakura-worker, then dispatch via RemoteDispatcher.

This is the default dispatcher when SakuraRuntime is constructed with no
`compute=` argument (Plan 1 + 2 use case: laptop demos with zero setup).
The spawned worker subprocess inherits the Python interpreter of the
parent, picks an ephemeral QUIC port, prints SAKURA_WORKER_LISTENING with
its self-signed cert, and serves until shutdown.
"""
from __future__ import annotations

import sys
from typing import Any, Callable, Optional

import sakura_wire as _native

from sakura.dispatch.base import Dispatcher, Future
from sakura.dispatch.remote import RemoteDispatcher


class LocalDispatcher(Dispatcher):
    """Auto-spawned localhost worker, accessed via QUIC over loopback."""

    def __init__(
        self,
        *,
        n_workers: int = 1,                    # Plan 2: only one worker for now
        gpus: Optional[list[int]] = None,
        startup_timeout_s: float = 10.0,
        shutdown_timeout_s: float = 5.0,
    ):
        if n_workers != 1:
            raise NotImplementedError(
                "n_workers > 1 not supported in Plan 2 (single-worker for now); "
                "Plan 4+ adds pool support."
            )
        self._supervisor = _native.WorkerSupervisor(shutdown_timeout_s=shutdown_timeout_s)
        env = {}
        if gpus is not None:
            env["CUDA_VISIBLE_DEVICES"] = ",".join(str(g) for g in gpus)
        uri, cert = self._supervisor.spawn(
            cmd=[sys.executable, "-m", "sakura.worker", "--listen", "quic://127.0.0.1:0"],
            env=env if env else None,
            startup_timeout_s=startup_timeout_s,
        )
        self._uri = uri
        self._cert = cert
        self._inner = RemoteDispatcher(uri=uri, cert_der=cert, server_name="localhost")
        self._shutdown_called = False

    def submit(
        self,
        callable: Callable[..., Any],
        *args: Any,
        timeout_ms: Optional[int] = None,
        **kwargs: Any,
    ) -> Future:
        return self._inner.submit(callable, *args, timeout_ms=timeout_ms, **kwargs)

    def shutdown(self, *, timeout_s: float = 30.0) -> None:
        if self._shutdown_called:
            return
        self._shutdown_called = True
        try:
            self._supervisor.shutdown()
        except Exception:
            pass

    def stats(self) -> dict:
        return {"kind": "local", "uri": self._uri}


__all__ = ["LocalDispatcher"]
```

- [ ] **Step 4: Run the test — should pass**

```bash
source .venv/bin/activate
pytest tests/dispatch/test_local.py -v
```
Expected: 2 tests pass.

- [ ] **Step 5: Commit**

```bash
git add sakura/dispatch/local.py tests/dispatch/test_local.py
git commit -m "feat(dispatch): LocalDispatcher — auto-spawn worker via WorkerSupervisor

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 15: `sakura.dispatch.__init__` re-exports + `Compute.resolve()`

**Files:**
- Modify: `sakura/dispatch/__init__.py`, `sakura/dispatch/compute.py`

- [ ] **Step 1: Add a `resolve()` method to `Compute`**

Edit `sakura/dispatch/compute.py` — add this method to the `Compute` dataclass (insert before the `__repr__` method):

```python
    def resolve(self) -> "Dispatcher":
        """Resolve to a concrete Dispatcher instance.

        Lazy import inside the method so users with the `local`/`remote`
        kinds don't pay the sakura_wire import cost when only `in_thread`
        is used (e.g., test fixtures).
        """
        if self.kind == "in_thread":
            from sakura.dispatch.in_thread import InThreadDispatcher
            return InThreadDispatcher()
        if self.kind == "local":
            from sakura.dispatch.local import LocalDispatcher
            return LocalDispatcher(
                n_workers=self.n_workers,
                gpus=list(self.gpus) if self.gpus is not None else None,
            )
        if self.kind == "remote":
            from sakura.dispatch.remote import RemoteDispatcher
            if len(self.uris) != 1:
                raise NotImplementedError(
                    "Compute.pool resolution to a single Dispatcher requires "
                    "Plan 4+ pool support; for now use Compute.at(uri)."
                )
            # Resolve cert from a per-uri map? For Plan 2 we expect the user
            # to manage TLS via the higher-level service config. In the meantime,
            # require an explicit cert_der at the dispatcher level.
            raise NotImplementedError(
                "Compute.at resolution requires a TLS cert; use RemoteDispatcher "
                "directly with cert_der= for Plan 2 cross-host. Plan 4 wires this up."
            )
        raise ValueError(f"unknown Compute.kind: {self.kind!r}")
```

The annotation `"Dispatcher"` is a forward reference to avoid an import cycle with `dispatch.base`.

- [ ] **Step 2: Re-exports in `sakura/dispatch/__init__.py`**

```python
"""sakura.dispatch — Compute URI tag + Dispatcher abstractions."""
from sakura.dispatch.base import Dispatcher, Future, Result
from sakura.dispatch.compute import Compute
from sakura.dispatch.in_thread import InThreadDispatcher
from sakura.dispatch.local import LocalDispatcher
from sakura.dispatch.remote import RemoteDispatcher

__all__ = [
    "Compute",
    "Dispatcher",
    "Future",
    "InThreadDispatcher",
    "LocalDispatcher",
    "RemoteDispatcher",
    "Result",
]
```

- [ ] **Step 3: Verify everything imports**

```bash
source .venv/bin/activate
python3 -c "from sakura.dispatch import Compute, Dispatcher, Future, Result, InThreadDispatcher, LocalDispatcher, RemoteDispatcher; print('OK')"
```
Expected: prints `OK`.

Run the existing test suite:
```bash
pytest tests/runtime/ tests/dispatch/ tests/worker/ tests/wire/ -v
```
Expected: all tests pass (no regressions).

- [ ] **Step 4: Commit**

```bash
git add sakura/dispatch/__init__.py sakura/dispatch/compute.py
git commit -m "feat(dispatch): Compute.resolve() + dispatch/__init__.py re-exports

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 16: Top-level `sakura/__init__.py` re-exports + version bump

**Files:**
- Modify: `sakura/__init__.py`

- [ ] **Step 1: Read the current file**

```bash
cat sakura/__init__.py
```
Currently has `__version__ = "0.1.5"` and `__build__ = "..."`.

- [ ] **Step 2: Replace the contents — bump version, add re-exports**

```python
"""Sakura — SOTA training services for PyTorch DDP / Lightning / HuggingFace Trainer.

Plan 1 of the v1.0 redesign added the sakura-wire transport (Rust). Plan 2
adds the Python orchestration surface: SakuraRuntime, Service ABC, event
types, and the Dispatcher abstraction. Plans 3-5 add concrete services,
framework adapters, and the benchmark harness.

Existing v0.1.x submodules (sakura.lightning, sakura.huggingface,
sakura.tensorflow, sakura.ddp, sakura.ml) continue to import. They will be
removed in Plan 4 once the migration path is validated.
"""

__version__ = "1.0.0a0"
__build__ = "2026-05-08T00:00:00Z"

from sakura.dispatch import Compute, Dispatcher, Future, Result
from sakura.events import (
    Event,
    OnEpochBegin,
    OnEpochEnd,
    OnError,
    OnOptimizerStep,
    OnSave,
    OnTrainBegin,
    OnTrainEnd,
    OnTrainStepBegin,
)
from sakura.runtime import SakuraRuntime
from sakura.service import BaseService, Service

__all__ = [
    "BaseService",
    "Compute",
    "Dispatcher",
    "Event",
    "Future",
    "OnEpochBegin",
    "OnEpochEnd",
    "OnError",
    "OnOptimizerStep",
    "OnSave",
    "OnTrainBegin",
    "OnTrainEnd",
    "OnTrainStepBegin",
    "Result",
    "SakuraRuntime",
    "Service",
    "__build__",
    "__version__",
]
```

- [ ] **Step 3: Verify the v0.1.x imports still work**

```bash
source .venv/bin/activate
python3 -c "import sakura; print(sakura.__version__)"
python3 -c "from sakura import SakuraRuntime, Service, Compute, OnEpochEnd; print('top-level reexports OK')"
python3 -c "from sakura.lightning import SakuraTrainer; print('v0.1.x lightning still imports')"
python3 -c "from sakura.huggingface import SakuraHFCallback; print('v0.1.x huggingface still imports')"
```
Expected:
```
1.0.0a0
top-level reexports OK
v0.1.x lightning still imports
v0.1.x huggingface still imports
```

Run the v0.1.x tests to confirm no regressions:
```bash
pytest tests/test_pkg_info.py -v
```
Expected: tests pass.

- [ ] **Step 4: Commit**

```bash
git add sakura/__init__.py
git commit -m "feat(sakura): bump version to 1.0.0a0 + top-level reexports (SakuraRuntime, Service, Compute, ...)

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 17: End-to-end test + acceptance + tag

**Files:**
- Create: `tests/worker/test_e2e_cloudpickled.py`

- [ ] **Step 1: Write the e2e test that exercises the whole stack**

Create `tests/worker/test_e2e_cloudpickled.py`:
```python
"""End-to-end Plan 2 acceptance test.

Top-level integration: instantiate SakuraRuntime, install a service that
dispatches a cloudpickled callable via LocalDispatcher, dispatch an
OnEpochEnd event, verify the service got the result back.
"""
from __future__ import annotations

import numpy as np
import pytest

sakura_wire = pytest.importorskip("sakura_wire")

from sakura import (
    BaseService,
    Compute,
    OnEpochEnd,
    SakuraRuntime,
)
from sakura.dispatch.local import LocalDispatcher


def _compute_loss(x, y):
    """A trivial 'eval' that runs on the worker."""
    return {"loss": float(np.mean((x - y) ** 2))}


class _DispatchOnEpochEnd(BaseService):
    """Toy service: every OnEpochEnd, dispatch _compute_loss to the worker."""
    name = "dispatch_on_epoch_end"
    priority = 80

    def __init__(self, dispatcher):
        super().__init__()
        self._d = dispatcher
        self.results: list[dict] = []

    def on_epoch_end(self, event):
        x = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        y = np.array([1.5, 2.5, 3.5], dtype=np.float32)
        fut = self._d.submit(_compute_loss, x, y)
        self.results.append(fut.result(timeout=10.0).value)


def test_e2e_dispatch_via_runtime_and_local_dispatcher():
    """Full Plan 2 acceptance loop."""
    dispatcher = LocalDispatcher()
    try:
        rt = SakuraRuntime(compute=Compute.local())
        service = _DispatchOnEpochEnd(dispatcher)
        rt.install(service)
        with rt:
            rt.dispatch(OnEpochEnd(epoch=0, model="m", optimizer="o", metrics={},
                                    rank=0, world_size=1))
            rt.dispatch(OnEpochEnd(epoch=1, model="m", optimizer="o", metrics={},
                                    rank=0, world_size=1))
        assert len(service.results) == 2
        # Both results should be the same (we used the same input).
        for r in service.results:
            assert "loss" in r
            assert abs(r["loss"] - 0.25) < 1e-6
    finally:
        dispatcher.shutdown()
```

- [ ] **Step 2: Run the e2e**

```bash
source .venv/bin/activate
pytest tests/worker/test_e2e_cloudpickled.py -v
```
Expected: 1 test passes.

- [ ] **Step 3: Full acceptance**

Run everything:

```bash
export PATH="$HOME/.cargo/bin:$PATH"
source .venv/bin/activate

echo "=== cargo fmt ==="
cargo fmt --all --check && echo OK

echo "=== cargo clippy ==="
cargo clippy --workspace --all-targets -- -D warnings && echo OK

echo "=== cargo test ==="
cargo test --workspace --all-features 2>&1 | tail -10

echo "=== maturin develop ==="
maturin develop --release 2>&1 | tail -5

echo "=== pytest (Plan 1 + Plan 2 test sets) ==="
pytest tests/wire/ tests/runtime/ tests/dispatch/ tests/worker/ -v 2>&1 | tail -30

echo "=== v0.1.x smoke ==="
python3 -c "from sakura.lightning import SakuraTrainer; from sakura.huggingface import SakuraHFCallback; print('v0.1.x callable')"
```

Expected: every block exits 0 with no test failures.

- [ ] **Step 4: Tag the milestone**

```bash
git tag -a sakura-runtime-v1-foundation -m "Plan 2 complete: SakuraRuntime + Service + Compute + Local/Remote/InThread Dispatchers + cloudpickle worker"
git tag --list | grep sakura
```

- [ ] **Step 5: Commit (only if anything from Step 1 is uncommitted)**

```bash
git status
# If clean: nothing to commit. If dirty:
git add tests/worker/test_e2e_cloudpickled.py
git commit -m "test(worker): end-to-end — SakuraRuntime + LocalDispatcher + cloudpickle round-trip

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Plan 2 — Acceptance Criteria

Plan 2 is complete when **all** of the following are true:

1. `cargo fmt --all --check` and `cargo clippy --workspace --all-targets -- -D warnings` pass.
2. `cargo test --workspace` passes (Plan 1's 17 tests still pass; no Rust tests were added in Plan 2).
3. `maturin develop --release` builds + installs cleanly.
4. `pytest tests/runtime/ tests/dispatch/ tests/worker/ tests/wire/` passes everything (≥ 30 new Plan 2 tests + 2 Plan 1 wire tests).
5. v0.1.x imports still work: `from sakura.lightning import SakuraTrainer`, `from sakura.huggingface import SakuraHFCallback`.
6. Top-level: `from sakura import SakuraRuntime, Service, Compute, OnEpochEnd, ...` resolves.
7. End-to-end test passes: SakuraRuntime + LocalDispatcher + auto-spawned worker + cloudpickled callable round-trip.
8. Tag `sakura-runtime-v1-foundation` exists.

After Plan 2 lands, **Plan 3** authors the seven concrete services (MixedPrecision, Compile, ZeRO1, ActivationCheckpoint, AsyncEval, AsyncCheckpoint, Telemetry) on top of these abstractions.

---

## Self-Review Notes

- **Spec coverage:** Plan 2 implements §4.1 (four primitives — except Adapter which is Plan 4), §4.3 (concurrency model — synchronous in-process dispatch + cross-process via wire), §4.4 (anti-coupling rules: services don't import framework code), §5.1 (top-level API surface — SakuraRuntime, Service, Compute, Dispatcher), §5.2 (event schema — OnTrainBegin/OnEpochEnd/etc. dataclasses), §8 (Dispatcher implementations — Local/Remote/InThread; Zakuro deferred to Plan 4 alongside framework adapters).
- **Out-of-scope confirmed:** No concrete services from §6, no framework adapters from §10, no benchmark harness from §11, no removal of v0.1.x.
- **Known follow-ups for Plan 3:** introduce the `Adapter` ABC + per-framework adapters; introduce the seven concrete services (MixedPrecision, Compile, ZeRO1, ActivationCheckpoint, AsyncEval, AsyncCheckpoint, Telemetry); the runtime + dispatcher are ready to host them.
- **Plan 1 perf debt carryover:** the 7× codec memcpy is still outstanding; it surfaces under load in Plan 3 (AsyncEval shipping 268 MB state_dicts). Plan 3 may need to address it inline or push the v1.x optimization patch first.
