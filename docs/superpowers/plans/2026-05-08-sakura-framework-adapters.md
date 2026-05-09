# Sakura v1.0 — Plan 4: Framework Adapters + v0.1.x Removal

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:subagent-driven-development` (recommended) or `superpowers:executing-plans` to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Wire the Plan 3 service catalog into real PyTorch-frontend training loops via three adapters (`LightningAdapter`, `HFAdapter`, `DDPAdapter`); add `ZakuroDispatcher` for users with existing Zakuro infrastructure; finalize the v1.0 clean break by removing all v0.1.x submodules.

**Architecture:** Each adapter is a thin per-framework bridge (~150-250 LOC) that translates framework hooks to runtime events — they never run techniques themselves, only emit events that services consume. `LightningAdapter` is a `lightning.Callback`, `HFAdapter` is a `transformers.TrainerCallback`, `DDPAdapter` is a free-standing class with explicit hook methods for raw loops. The adapter pattern is uniform: instantiate the runtime, install services, install the adapter, run the framework's normal `fit`/`train`/loop. `MixedPrecision`'s `on_train_step_begin` hook (Plan 3 stub) is wired into actual `torch.autocast` contexts via the adapter integration.

**Tech Stack:** Python 3.10+, `torch>=2.1`, `lightning>=2.0` (optional dep), `transformers>=4.38` (optional dep), `cloudpickle`, `numpy`. No new Rust changes — Plan 4 is purely Python.

**Out of scope for this plan** (deferred to Plan 5):
- Multi-rank ZeRO1 sharding (Plan 3 carryover) — Plan 5 with real DDP runs.
- Benchmark harness (`Workload` / `BaselineRunner` / `SakuraRunner`, `sakura-bench` CLI) — Plan 5.
- maturin packaging fix (Plan 2 carryover) — Plan 5.
- Codec memcpy zero-copy fix (Plan 1 carryover) — separate v1.x patch.
- Migration guide doc (`docs/migration-from-0.1.md`) — Plan 5.

---

## Existing state at start of Plan 4

Master after Plan 3 merge is at `819156f`. The v1.0 architecture is in place: `SakuraRuntime`, 7 services, 4 dispatchers (Local/Remote/InThread/`Compute.in_thread()`/etc.). v0.1.x submodules (`sakura/lightning`, `sakura/huggingface`, `sakura/tensorflow`, `sakura/ddp`, `sakura/ml`, `sakura/functional.py`, `sakura/config.yaml`) coexist alongside the v1.0 code.

Plan 4 reverses that coexistence: NEW adapters live at `sakura/adapters/{lightning,huggingface,ddp}.py`, and the v0.1.x submodule trees + their tests are deleted at the end of the plan.

---

## File Structure (created/modified/deleted by this plan)

**New Python files:**
```
sakura/adapters/
├── __init__.py                            # Adapter ABC + re-exports
├── base.py                                # Adapter ABC
├── lightning.py                           # LightningAdapter (lightning.Callback)
├── huggingface.py                         # HFAdapter (transformers.TrainerCallback)
└── ddp.py                                 # DDPAdapter (explicit hooks)
sakura/dispatch/zakuro.py                  # ZakuroDispatcher (wraps zk.Compute)
tests/adapters/
├── __init__.py
├── test_adapter_base.py
├── test_lightning_adapter.py
├── test_hf_adapter.py
├── test_ddp_adapter.py
└── test_adapter_service_wiring.py         # MixedPrecision autocast actually wraps forward
tests/dispatch/test_zakuro.py              # smoke: skip if zakuro-ai missing
```

**Existing files modified:**
```
sakura/__init__.py                         # add Adapter reexports; rm v0.1.x reexports if any
sakura/services/mixed_precision.py         # actual autocast context (Plan 3 stub → real)
sakura/dispatch/__init__.py                # add ZakuroDispatcher reexport
```

**Files / dirs DELETED at the end of the plan:**
```
sakura/lightning/                          # v0.1.x SakuraTrainer
sakura/huggingface/                        # v0.1.x SakuraHFCallback
sakura/tensorflow/                         # TF integration (per spec, dropped entirely)
sakura/ddp/                                # v0.1.x DDPAsyncEvalCallback
sakura/ml/                                 # sakura_trainer + async_trainer dead code
sakura/functional.py                       # unused metric defaults
sakura/__main__.py                         # v0.1.x CLI; replaced by package-level `sakura` script
tests/test_huggingface.py                  # v0.1.x test
tests/test_huggingface_e2e.py              # v0.1.x test
tests/test_lightning.py                    # v0.1.x test (replaced by tests/adapters/test_lightning_adapter.py)
tests/test_ddp.py                          # v0.1.x test (replaced by tests/adapters/test_ddp_adapter.py)
tests/test_tensorflow.py                   # v0.1.x test
tests/test_async_trainer.py                # v0.1.x test
tests/test_sakura_trainer.py               # v0.1.x test
tests/test_epoch_range.py                  # v0.1.x test
tests/test_functional.py                   # v0.1.x test
tests/test_worker_integration.py           # v0.1.x worker integration test (Plan 1+2 cover this differently)
tests/test_cli.py                          # v0.1.x CLI test
mnist_demo/                                # v0.1.x demo using SakuraTrainer
bert_demo/                                 # v0.1.x demo using SakuraHFCallback
main.py                                    # v0.1.x benchmark script using SakuraTrainer
```

---

## Task 1: `Adapter` ABC

**Files:** Create `sakura/adapters/__init__.py`, `sakura/adapters/base.py`, `tests/adapters/__init__.py`, `tests/adapters/test_adapter_base.py`.

The `Adapter` ABC is an interface that holds a runtime reference and provides an `emit(event)` helper. Concrete adapters subclass framework callbacks AND `Adapter`.

- [ ] **Step 1: empty placeholder files**

```bash
mkdir -p sakura/adapters tests/adapters
echo "" > sakura/adapters/__init__.py
echo "" > tests/adapters/__init__.py
```

- [ ] **Step 2: write the failing tests** (`tests/adapters/test_adapter_base.py`):

```python
"""Adapter ABC: holds the runtime + provides an emit() helper."""
from __future__ import annotations

import pytest

from sakura.adapters.base import Adapter
from sakura.events import OnEpochEnd
from sakura.runtime import SakuraRuntime


class TestAdapterBase:
    def test_emit_dispatches_event_to_runtime(self):
        rt = SakuraRuntime()
        seen = []
        from sakura.service import BaseService

        class Recorder(BaseService):
            name = "rec"
            priority = 10

            def on_epoch_end(self, event):
                seen.append(event.epoch)

        rt.install(Recorder())
        adapter = Adapter(rt)
        adapter.emit(OnEpochEnd(epoch=7, model="m", optimizer="o", metrics={},
                                 rank=0, world_size=1))
        assert seen == [7]

    def test_runtime_property(self):
        rt = SakuraRuntime()
        adapter = Adapter(rt)
        assert adapter.runtime is rt
```

- [ ] **Step 3: implement `sakura/adapters/base.py`**

```python
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
```

- [ ] **Step 4: run + commit**

```bash
source .venv/bin/activate
pytest tests/adapters/test_adapter_base.py -v
git add sakura/adapters/__init__.py sakura/adapters/base.py tests/adapters/__init__.py tests/adapters/test_adapter_base.py
git commit -m "feat(adapters): Adapter ABC — runtime ref + emit() helper

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 2: `LightningAdapter`

**Files:** Create `sakura/adapters/lightning.py`, `tests/adapters/test_lightning_adapter.py`.

Subclasses both `Adapter` and `lightning.pytorch.Callback`. Hook mapping per spec §10.1.

- [ ] **Step 1: failing tests**

```python
"""LightningAdapter: maps lightning.Callback hooks to runtime events."""
from __future__ import annotations

import pytest

lightning = pytest.importorskip("lightning")
torch = pytest.importorskip("torch")

from sakura.adapters.lightning import LightningAdapter
from sakura.events import OnEpochEnd, OnTrainBegin, OnTrainEnd
from sakura.runtime import SakuraRuntime
from sakura.service import BaseService


class _EventCollector(BaseService):
    name = "collector"
    priority = 10

    def __init__(self):
        super().__init__()
        self.events: list[str] = []

    def on_event(self, event):
        self.events.append(type(event).__name__)


def test_lightning_adapter_subclasses_callback():
    rt = SakuraRuntime()
    adapter = LightningAdapter(rt)
    assert isinstance(adapter, lightning.pytorch.Callback)


def test_lightning_adapter_translates_lifecycle_to_events():
    """Direct invocation of the adapter's callback methods emits the right events."""
    rt = SakuraRuntime()
    collector = _EventCollector()
    rt.install(collector)
    adapter = LightningAdapter(rt)

    # Build a minimal fake "trainer" + "module" so the adapter can extract optim/loaders.
    class _FakeOptim:
        pass
    class _FakeLoader:
        pass
    class _FakeTrainer:
        current_epoch = 0
        optimizers = [_FakeOptim()]
        train_dataloader = _FakeLoader()
        val_dataloaders = None
        callback_metrics = {"val_loss": 0.5}
    class _FakeModule(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.l = torch.nn.Linear(2, 1)

    trainer = _FakeTrainer()
    module = _FakeModule()

    # Lightning calls these in sequence:
    adapter.on_train_start(trainer, module)
    adapter.on_train_epoch_start(trainer, module)
    adapter.on_train_epoch_end(trainer, module)
    adapter.on_train_end(trainer, module)

    # We should have collected: OnTrainBegin, OnEpochBegin, OnEpochEnd, OnTrainEnd.
    expected = ["OnTrainBegin", "OnEpochBegin", "OnEpochEnd", "OnTrainEnd"]
    assert collector.events == expected


def test_lightning_adapter_carries_metrics_on_epoch_end():
    rt = SakuraRuntime()
    collected: list[dict] = []

    class _Capturer(BaseService):
        name = "capturer"
        priority = 10

        def on_epoch_end(self, event):
            collected.append(dict(event.metrics))

    rt.install(_Capturer())
    adapter = LightningAdapter(rt)

    class _FakeTrainer:
        current_epoch = 5
        optimizers = []
        callback_metrics = {"val_loss": 0.123, "val_acc": 0.95}
    class _FakeModule(torch.nn.Module):
        pass

    adapter.on_train_epoch_end(_FakeTrainer(), _FakeModule())
    assert len(collected) == 1
    assert collected[0]["val_loss"] == 0.123
    assert collected[0]["val_acc"] == 0.95
```

- [ ] **Step 2: implement `sakura/adapters/lightning.py`**

```python
"""LightningAdapter — lightning.Callback that emits Sakura runtime events.

Hook mapping (per spec §10.1):
  setup-after / on_train_start  → on_train_begin
  on_train_epoch_start          → on_epoch_begin
  on_train_batch_start          → on_train_step_begin
  on_before_optimizer_step      → on_optimizer_step
  on_train_epoch_end            → on_epoch_end (carries trainer.callback_metrics)
  on_train_end                  → on_train_end
  on_exception                  → on_error
"""
from __future__ import annotations

from typing import Any

try:
    from lightning.pytorch import Callback
except ImportError:  # pragma: no cover
    class Callback:  # type: ignore[no-redef]
        pass

from sakura.adapters.base import Adapter
from sakura.events import (
    OnEpochBegin,
    OnEpochEnd,
    OnError,
    OnOptimizerStep,
    OnTrainBegin,
    OnTrainEnd,
    OnTrainStepBegin,
)
from sakura.runtime import SakuraRuntime


class LightningAdapter(Callback, Adapter):
    """Lightning callback that translates framework hooks into Sakura events."""

    def __init__(self, runtime: SakuraRuntime, *, rank: int = 0, world_size: int = 1):
        # NOTE: we call Adapter.__init__ explicitly because Callback's __init__
        # may not accept positional args.
        Callback.__init__(self)
        Adapter.__init__(self, runtime)
        self._rank = rank
        self._world_size = world_size
        self._collected: list[dict] = []

    # ........................................................... lifecycle

    def on_train_start(self, trainer, pl_module):
        opt = trainer.optimizers[0] if trainer.optimizers else None
        self.emit(OnTrainBegin(
            model=pl_module,
            optimizer=opt,
            train_loader=getattr(trainer, "train_dataloader", None),
            val_loader=getattr(trainer, "val_dataloaders", None),
            rank=self._rank,
            world_size=self._world_size,
        ))

    def on_train_epoch_start(self, trainer, pl_module):
        self.emit(OnEpochBegin(
            epoch=int(trainer.current_epoch),
            rank=self._rank,
            world_size=self._world_size,
        ))

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        self.emit(OnTrainStepBegin(
            model=pl_module, batch=batch, step=int(batch_idx),
            rank=self._rank, world_size=self._world_size,
        ))

    def on_before_optimizer_step(self, trainer, pl_module, optimizer):
        self.emit(OnOptimizerStep(
            optimizer=optimizer, rank=self._rank, world_size=self._world_size,
        ))

    def on_train_epoch_end(self, trainer, pl_module):
        opt = trainer.optimizers[0] if getattr(trainer, "optimizers", None) else None
        metrics = dict(getattr(trainer, "callback_metrics", {}) or {})
        # Convert any tensor metrics to float for telemetry serializability.
        clean = {}
        for k, v in metrics.items():
            try:
                clean[k] = float(v)
            except Exception:
                clean[k] = v
        self.emit(OnEpochEnd(
            epoch=int(trainer.current_epoch),
            model=pl_module, optimizer=opt, metrics=clean,
            rank=self._rank, world_size=self._world_size,
        ))

    def on_train_end(self, trainer, pl_module):
        self.emit(OnTrainEnd(
            model=pl_module, history=list(self._collected),
            rank=self._rank, world_size=self._world_size,
        ))

    def on_exception(self, trainer, pl_module, exception):
        self.emit(OnError(
            exc=exception, context={"hook": "lightning"},
            rank=self._rank, world_size=self._world_size,
        ))


__all__ = ["LightningAdapter"]
```

- [ ] **Step 3: run + commit**

```bash
pytest tests/adapters/test_lightning_adapter.py -v
git add sakura/adapters/lightning.py tests/adapters/test_lightning_adapter.py
git commit -m "feat(adapters): LightningAdapter — lightning.Callback that emits runtime events

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 3: `HFAdapter`

**Files:** Create `sakura/adapters/huggingface.py`, `tests/adapters/test_hf_adapter.py`.

Subclasses `transformers.TrainerCallback`. Hook mapping per spec §10.2. Tests use synthetic state objects since spinning up a real HF Trainer is too heavy.

- [ ] **Step 1: failing tests**

```python
"""HFAdapter: maps transformers.TrainerCallback hooks to runtime events."""
from __future__ import annotations

import pytest

transformers = pytest.importorskip("transformers")

from sakura.adapters.huggingface import HFAdapter
from sakura.runtime import SakuraRuntime
from sakura.service import BaseService


class _EventCollector(BaseService):
    name = "collector"
    priority = 10

    def __init__(self):
        super().__init__()
        self.events: list[str] = []
        self.metrics: list[dict] = []

    def on_train_begin(self, event):
        self.events.append("OnTrainBegin")

    def on_train_step_begin(self, event):
        self.events.append("OnTrainStepBegin")

    def on_optimizer_step(self, event):
        self.events.append("OnOptimizerStep")

    def on_epoch_end(self, event):
        self.events.append("OnEpochEnd")
        self.metrics.append(dict(event.metrics))

    def on_train_end(self, event):
        self.events.append("OnTrainEnd")


def test_hf_adapter_subclasses_trainer_callback():
    rt = SakuraRuntime()
    adapter = HFAdapter(rt)
    assert isinstance(adapter, transformers.TrainerCallback)


def test_hf_adapter_emits_lifecycle_events():
    rt = SakuraRuntime()
    collector = _EventCollector()
    rt.install(collector)
    adapter = HFAdapter(rt)

    class _State:
        epoch = 1.0
        log_history = [{"val_loss": 0.21, "step": 100}]
        global_step = 100

    args = object()
    state = _State()
    control = object()
    fake_model = object()
    fake_optimizer = object()
    fake_loader = object()

    # Lightning calls these in sequence:
    adapter.on_train_begin(args, state, control,
                            model=fake_model, optimizer=fake_optimizer,
                            train_dataloader=fake_loader)
    adapter.on_step_begin(args, state, control,
                           model=fake_model, inputs={"x": 1})
    adapter.on_pre_optimizer_step(args, state, control,
                                    optimizer=fake_optimizer)
    adapter.on_epoch_end(args, state, control,
                          model=fake_model, optimizer=fake_optimizer)
    adapter.on_train_end(args, state, control, model=fake_model)

    assert collector.events == [
        "OnTrainBegin", "OnTrainStepBegin", "OnOptimizerStep",
        "OnEpochEnd", "OnTrainEnd",
    ]
    # Last log_history entry is exposed as metrics on epoch_end
    assert collector.metrics[0].get("val_loss") == 0.21
```

- [ ] **Step 2: implement `sakura/adapters/huggingface.py`**

```python
"""HFAdapter — transformers.TrainerCallback that emits Sakura runtime events."""
from __future__ import annotations

try:
    from transformers import TrainerCallback
except ImportError:  # pragma: no cover
    class TrainerCallback:  # type: ignore[no-redef]
        pass

from sakura.adapters.base import Adapter
from sakura.events import (
    OnEpochEnd,
    OnError,
    OnOptimizerStep,
    OnTrainBegin,
    OnTrainEnd,
    OnTrainStepBegin,
)
from sakura.runtime import SakuraRuntime


class HFAdapter(TrainerCallback, Adapter):
    """transformers.TrainerCallback that translates HF Trainer hooks into Sakura events."""

    min_transformers_version: str = "4.38"

    def __init__(self, runtime: SakuraRuntime, *, rank: int = 0, world_size: int = 1):
        TrainerCallback.__init__(self)
        Adapter.__init__(self, runtime)
        self._rank = rank
        self._world_size = world_size

    # ........................................................... lifecycle

    def on_train_begin(self, args, state, control, **kw):
        self.emit(OnTrainBegin(
            model=kw.get("model"),
            optimizer=kw.get("optimizer"),
            train_loader=kw.get("train_dataloader"),
            val_loader=kw.get("eval_dataloader"),
            rank=self._rank, world_size=self._world_size,
        ))

    def on_step_begin(self, args, state, control, **kw):
        self.emit(OnTrainStepBegin(
            model=kw.get("model"),
            batch=kw.get("inputs"),
            step=int(getattr(state, "global_step", 0)),
            rank=self._rank, world_size=self._world_size,
        ))

    def on_pre_optimizer_step(self, args, state, control, **kw):
        self.emit(OnOptimizerStep(
            optimizer=kw.get("optimizer"),
            rank=self._rank, world_size=self._world_size,
        ))

    def on_epoch_end(self, args, state, control, **kw):
        # Pull most-recent metrics from state.log_history if present.
        metrics: dict = {}
        log = getattr(state, "log_history", None)
        if log:
            metrics = dict(log[-1])
        self.emit(OnEpochEnd(
            epoch=int(state.epoch) if state.epoch is not None else 0,
            model=kw.get("model"),
            optimizer=kw.get("optimizer"),
            metrics=metrics,
            rank=self._rank, world_size=self._world_size,
        ))

    def on_train_end(self, args, state, control, **kw):
        self.emit(OnTrainEnd(
            model=kw.get("model"),
            history=list(getattr(state, "log_history", []) or []),
            rank=self._rank, world_size=self._world_size,
        ))


__all__ = ["HFAdapter"]
```

- [ ] **Step 3: run + commit**

```bash
pytest tests/adapters/test_hf_adapter.py -v
git add sakura/adapters/huggingface.py tests/adapters/test_hf_adapter.py
git commit -m "feat(adapters): HFAdapter — transformers.TrainerCallback that emits runtime events

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 4: `DDPAdapter`

**Files:** Create `sakura/adapters/ddp.py`, `tests/adapters/test_ddp_adapter.py`.

No callback model — explicit hook methods called by user code. The simplest of the three adapters; mainly a uniform interface over raw PyTorch DDP loops.

- [ ] **Step 1: failing tests**

```python
"""DDPAdapter: explicit hooks for raw PyTorch DDP loops."""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from sakura.adapters.ddp import DDPAdapter
from sakura.runtime import SakuraRuntime
from sakura.service import BaseService


class _EventCollector(BaseService):
    name = "collector"
    priority = 10

    def __init__(self):
        super().__init__()
        self.events: list[tuple[str, int]] = []  # (event_type, rank)

    def on_event(self, event):
        self.events.append((type(event).__name__, event.rank))


def test_ddp_adapter_emits_events_with_correct_rank_and_world_size():
    rt = SakuraRuntime()
    collector = _EventCollector()
    rt.install(collector)
    adapter = DDPAdapter(rt, rank=0, world_size=4)

    fake_model = object()
    fake_opt = object()
    fake_loader = object()

    adapter.on_train_begin(fake_model, fake_opt, fake_loader)
    adapter.on_epoch_begin(0)
    adapter.on_train_step_begin(fake_model, batch=("x", "y"), step=0)
    adapter.on_optimizer_step(fake_opt)
    adapter.on_epoch_end(0, fake_model, fake_opt, metrics={"val_loss": 0.3})
    adapter.on_train_end(fake_model)

    expected = [
        ("OnTrainBegin", 0),
        ("OnEpochBegin", 0),
        ("OnTrainStepBegin", 0),
        ("OnOptimizerStep", 0),
        ("OnEpochEnd", 0),
        ("OnTrainEnd", 0),
    ]
    assert collector.events == expected


def test_ddp_adapter_passes_world_size_through():
    rt = SakuraRuntime()
    seen_world: list[int] = []

    class _W(BaseService):
        name = "w"
        priority = 10

        def on_epoch_end(self, event):
            seen_world.append(event.world_size)

    rt.install(_W())
    adapter = DDPAdapter(rt, rank=2, world_size=8)
    adapter.on_epoch_end(epoch=0, model=None, optimizer=None, metrics={})
    assert seen_world == [8]
```

- [ ] **Step 2: implement `sakura/adapters/ddp.py`**

```python
"""DDPAdapter — explicit-hook adapter for raw PyTorch DDP loops.

Unlike LightningAdapter / HFAdapter, DDPAdapter has no callback subclass.
Users invoke its methods directly from their training loop. Embeds rank
and world_size in every emitted event.
"""
from __future__ import annotations

from typing import Any, Optional

from sakura.adapters.base import Adapter
from sakura.events import (
    OnEpochBegin,
    OnEpochEnd,
    OnOptimizerStep,
    OnTrainBegin,
    OnTrainEnd,
    OnTrainStepBegin,
)
from sakura.runtime import SakuraRuntime


class DDPAdapter(Adapter):
    """Explicit-hook adapter for raw PyTorch DDP training loops."""

    def __init__(self, runtime: SakuraRuntime, *, rank: int, world_size: int):
        super().__init__(runtime)
        self._rank = int(rank)
        self._world_size = int(world_size)

    @property
    def rank(self) -> int:
        return self._rank

    @property
    def world_size(self) -> int:
        return self._world_size

    def on_train_begin(self, model: Any, optimizer: Any, train_loader: Any,
                        val_loader: Optional[Any] = None) -> None:
        self.emit(OnTrainBegin(
            model=model, optimizer=optimizer, train_loader=train_loader,
            val_loader=val_loader, rank=self._rank, world_size=self._world_size,
        ))

    def on_epoch_begin(self, epoch: int) -> None:
        self.emit(OnEpochBegin(epoch=int(epoch), rank=self._rank, world_size=self._world_size))

    def on_train_step_begin(self, model: Any, batch: Any, step: int) -> None:
        self.emit(OnTrainStepBegin(
            model=model, batch=batch, step=int(step),
            rank=self._rank, world_size=self._world_size,
        ))

    def on_optimizer_step(self, optimizer: Any) -> None:
        self.emit(OnOptimizerStep(
            optimizer=optimizer, rank=self._rank, world_size=self._world_size,
        ))

    def on_epoch_end(self, epoch: int, model: Any, optimizer: Any,
                      metrics: dict) -> None:
        self.emit(OnEpochEnd(
            epoch=int(epoch), model=model, optimizer=optimizer, metrics=dict(metrics),
            rank=self._rank, world_size=self._world_size,
        ))

    def on_train_end(self, model: Any) -> None:
        self.emit(OnTrainEnd(
            model=model, history=[], rank=self._rank, world_size=self._world_size,
        ))


__all__ = ["DDPAdapter"]
```

- [ ] **Step 3: run + commit**

```bash
pytest tests/adapters/test_ddp_adapter.py -v
git add sakura/adapters/ddp.py tests/adapters/test_ddp_adapter.py
git commit -m "feat(adapters): DDPAdapter — explicit-hook adapter for raw PyTorch DDP loops

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 5: `ZakuroDispatcher` (optional dep)

**Files:** Create `sakura/dispatch/zakuro.py`, `tests/dispatch/test_zakuro.py`.

Wraps `zakuro.Compute` so users with existing Zakuro infrastructure can keep using it. Skip the test if `zakuro-ai` isn't installed.

- [ ] **Step 1: failing test (importorskip-guarded)**

```python
"""ZakuroDispatcher: wraps zakuro.Compute. Skipped when zakuro-ai is not installed."""
from __future__ import annotations

import pytest

zakuro = pytest.importorskip("zakuro")

from sakura.dispatch.zakuro import ZakuroDispatcher


def _square(x):
    return int(x.sum()) ** 2


def test_zakuro_dispatcher_passes_through_to_zk_compute():
    """End-to-end: real ZakuroDispatcher uses zk.Compute() (standalone fallback)."""
    import numpy as np

    zk_compute = zakuro.Compute()  # standalone in-process fallback
    d = ZakuroDispatcher(zk_compute)
    fut = d.submit(_square, np.array([1, 2, 3], dtype=np.int64))
    result = fut.result(timeout=5.0)
    assert result.value == (1 + 2 + 3) ** 2
```

- [ ] **Step 2: implement `sakura/dispatch/zakuro.py`**

```python
"""ZakuroDispatcher — wraps zakuro.Compute for users with existing Zakuro infra.

Loses the sakura-wire codec wins (Zakuro's wire format is its own); kept for
backward compatibility and for users who want to leverage Zakuro's worker
allocation features.
"""
from __future__ import annotations

import time
from typing import Any, Callable, Optional

from sakura.dispatch.base import Dispatcher, Future, Result


class _ZkFuture(Future):
    def __init__(self, value, exc, elapsed_us):
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
        return False


class ZakuroDispatcher(Dispatcher):
    """Wraps zakuro.Compute and dispatches via @zk.fn."""

    def __init__(self, zk_compute: Any):
        self._zk_compute = zk_compute

    def submit(
        self,
        callable: Callable[..., Any],
        *args: Any,
        timeout_ms: Optional[int] = None,
        **kwargs: Any,
    ) -> Future:
        import zakuro as zk

        # Wrap user callable with @zk.fn so Zakuro can ship it.
        @zk.fn
        def _wrapped(*a, **kw):
            return callable(*a, **kw)

        t0 = time.perf_counter_ns()
        try:
            value = _wrapped.to(self._zk_compute)(*args, **kwargs)
        except BaseException as exc:  # noqa: BLE001
            return _ZkFuture(value=None, exc=exc,
                              elapsed_us=(time.perf_counter_ns() - t0) // 1000)
        return _ZkFuture(value=value, exc=None,
                          elapsed_us=(time.perf_counter_ns() - t0) // 1000)

    def stats(self) -> dict:
        return {"kind": "zakuro"}


__all__ = ["ZakuroDispatcher"]
```

- [ ] **Step 3: update `sakura/dispatch/__init__.py`** to add `ZakuroDispatcher` to the re-exports:

```python
from sakura.dispatch.zakuro import ZakuroDispatcher
# (add to __all__)
```

- [ ] **Step 4: run + commit**

```bash
pytest tests/dispatch/test_zakuro.py -v
git add sakura/dispatch/zakuro.py sakura/dispatch/__init__.py tests/dispatch/test_zakuro.py
git commit -m "feat(dispatch): ZakuroDispatcher — wraps zakuro.Compute for existing Zakuro users

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 6: Wire `MixedPrecision` autocast into adapter forward

**Files:** Modify `sakura/services/mixed_precision.py`.

Plan 3's `MixedPrecision.on_train_step_begin` was a stub (no actual autocast wrapping). Plan 4's adapters call it at the right point in the training step, but we need to make the wrapping actually happen.

The cleanest approach: `MixedPrecision` exposes an `autocast()` context manager that adapters and user code can use, AND a `wrap_forward(model)` method that swaps `model.forward` with an autocast-wrapped version.

- [ ] **Step 1: extend `sakura/services/mixed_precision.py`** — wrap the model's forward at `on_train_begin`:

Modify the `on_train_begin` method so it ALSO swaps `event.model.forward` with an autocast-wrapped version. After the existing GradScaler creation:

```python
        # Wrap forward with autocast for the duration of training.
        device_type = self._device_type_from(event.model)
        actual_dtype = self._resolve_dtype(device_type)
        original_forward = event.model.forward
        autocast_dtype = actual_dtype

        def _wrapped_forward(*args, **kwargs):
            with torch.autocast(device_type=device_type, dtype=autocast_dtype,
                                enabled=True, cache_enabled=self._cache_enabled):
                return original_forward(*args, **kwargs)

        # Stash the original so we can restore at on_train_end (clean-up).
        self._original_forward = original_forward
        event.model.forward = _wrapped_forward
```

Add a class attribute `_original_forward = None` and an `on_train_end` that restores:

```python
    def on_train_end(self, event):
        if self._original_forward is not None:
            event.model.forward = self._original_forward
            self._original_forward = None
```

- [ ] **Step 2: add a regression test** at `tests/adapters/test_adapter_service_wiring.py`:

```python
"""Service+adapter wiring: MixedPrecision wraps forward in real autocast."""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from sakura.adapters.ddp import DDPAdapter
from sakura.events import OnTrainBegin, OnTrainEnd
from sakura.runtime import SakuraRuntime
from sakura.services.mixed_precision import MixedPrecision


class _M(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.l = torch.nn.Linear(4, 2)


def test_mixed_precision_actually_wraps_forward_at_train_begin():
    rt = SakuraRuntime()
    mp = MixedPrecision(dtype="bf16")  # bf16 works on CPU
    rt.install(mp)
    adapter = DDPAdapter(rt, rank=0, world_size=1)

    model = _M()
    original_forward = model.forward
    adapter.on_train_begin(model, optimizer=None, train_loader=None)

    # forward should now be a wrapper.
    assert model.forward is not original_forward

    # Calling it should still produce the same result tensor (autocast on CPU is fine).
    x = torch.randn(2, 4)
    y = model(x)
    assert y.shape == (2, 2)

    # Cleanup restores forward.
    adapter.on_train_end(model)
    assert model.forward is original_forward
```

- [ ] **Step 3: run + commit**

```bash
pytest tests/adapters/test_adapter_service_wiring.py -v
git add sakura/services/mixed_precision.py tests/adapters/test_adapter_service_wiring.py
git commit -m "feat(services): MixedPrecision actually wraps forward with torch.autocast at train_begin

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 7: Adapter integration tests (cross-service)

**Files:** Append to `tests/adapters/test_adapter_service_wiring.py`.

Test that an adapter + multiple services + dispatch work end-to-end on a synthetic training loop.

- [ ] **Step 1: add the integration test**

Append to the existing file:

```python
def test_ddp_adapter_with_telemetry_and_async_eval():
    """Full Plan 4 acceptance: DDPAdapter + Telemetry + AsyncEval (in-thread dispatcher)."""
    from sakura.dispatch.in_thread import InThreadDispatcher
    from sakura.services.async_eval import AsyncEval
    from sakura.services.telemetry import Telemetry

    sink: list[dict] = []
    rt = SakuraRuntime()
    rt.install(Telemetry(output=sink.append))

    def _eval(epoch, payload):
        return {"val_loss": 1.0 / (epoch + 1)}

    rt.install(AsyncEval(eval_fn=_eval, eval_payload={},
                          dispatcher=InThreadDispatcher()))

    adapter = DDPAdapter(rt, rank=0, world_size=1)
    adapter.on_train_begin(model=_M(), optimizer=None, train_loader=None)
    for epoch in range(3):
        adapter.on_epoch_begin(epoch)
        adapter.on_epoch_end(epoch, model=_M(), optimizer=None,
                             metrics={"train_loss": 1.5 / (epoch + 1)})
    adapter.on_train_end(model=_M())

    # Telemetry observed all 8 events: 1 train_begin + 3*(begin+end) + 1 train_end = 8.
    event_types = [r["event"] for r in sink]
    assert event_types.count("OnTrainBegin") == 1
    assert event_types.count("OnEpochBegin") == 3
    assert event_types.count("OnEpochEnd") == 3
    assert event_types.count("OnTrainEnd") == 1

    # AsyncEval has 3 results.
    eval_svc = next(s for s in rt.services if s.name == "async_eval")
    assert len(eval_svc.history) == 3
```

- [ ] **Step 2: run + commit**

```bash
pytest tests/adapters/test_adapter_service_wiring.py -v
git add tests/adapters/test_adapter_service_wiring.py
git commit -m "test(adapters): full Plan 4 acceptance — adapter + Telemetry + AsyncEval e2e

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 8: Remove v0.1.x submodules + tests

**Files:** Delete v0.1.x trees and tests.

This is the v1.0 clean break per the spec's API migration decision. Before deleting, capture the LOC for the commit message.

- [ ] **Step 1: capture LOC removed**

```bash
wc -l sakura/lightning/*.py sakura/huggingface/*.py sakura/tensorflow/*.py sakura/ddp/*.py sakura/ml/*.py sakura/ml/epoch/*.py sakura/functional.py sakura/__main__.py main.py 2>/dev/null
wc -l tests/test_lightning.py tests/test_huggingface.py tests/test_huggingface_e2e.py tests/test_ddp.py tests/test_tensorflow.py tests/test_async_trainer.py tests/test_sakura_trainer.py tests/test_epoch_range.py tests/test_functional.py tests/test_worker_integration.py tests/test_cli.py 2>/dev/null
```

- [ ] **Step 2: delete v0.1.x sources + tests + demos**

```bash
git rm -r sakura/lightning sakura/huggingface sakura/tensorflow sakura/ddp sakura/ml
git rm sakura/functional.py sakura/__main__.py
git rm main.py
git rm -r mnist_demo bert_demo
git rm tests/test_lightning.py tests/test_huggingface.py tests/test_huggingface_e2e.py
git rm tests/test_ddp.py tests/test_tensorflow.py tests/test_async_trainer.py
git rm tests/test_sakura_trainer.py tests/test_epoch_range.py tests/test_functional.py
git rm tests/test_worker_integration.py tests/test_cli.py
```

- [ ] **Step 3: update `sakura/__init__.py`** to remove the v0.1.x docstring reference:

Change the docstring in `sakura/__init__.py` from "Existing v0.1.x submodules ... continue to import. They will be removed in Plan 4 once the migration path is validated." to "v0.1.x has been removed at v1.0; the new architecture is the only entry point."

Replace the docstring block with:

```python
"""Sakura — SOTA training services for PyTorch DDP / Lightning / HuggingFace Trainer.

Plan 1 added the sakura-wire transport (Rust). Plan 2 the Python orchestration
surface (SakuraRuntime, Service ABC, Dispatcher). Plan 3 the seven v1 services.
Plan 4 the framework adapters (Lightning/HF/DDP) and removed v0.1.x.
Plan 5 (future) the benchmark harness + multi-rank ZeRO1 + maturin packaging.

Users on v0.1.x should pin `sakura-ml<1.0` if they're not migrating to the
new SakuraRuntime + Adapter + Service surface.
"""
```

- [ ] **Step 4: also remove `sakura-benchmark` console script** (it referenced `main.py` which is now deleted):

In `pyproject.toml`'s `[project.scripts]` block, remove the line `sakura-benchmark = "main:main"`.

Also remove the old `sakura = "sakura.__main__:main"` line — that's gone too. Keep only `sakura-worker`.

- [ ] **Step 5: also clean up `[tool.maturin].include`** — the `include = [...]` block referenced `sakura/config.yaml` (now gone with the v0.1.x removal) and `main.py` (now gone). Remove the `include = [...]` block from `pyproject.toml` entirely.

- [ ] **Step 6: verify imports + test suite**

```bash
source .venv/bin/activate
pytest tests/ 2>&1 | tail -10
python3 -c "import sakura; print('still works:', sakura.__version__)"
python3 -c "from sakura import SakuraRuntime, Telemetry, AsyncEval; print('top-level OK')"
# These should now FAIL (v0.1.x removed):
python3 -c "import sakura.lightning" 2>&1 | tail -3
python3 -c "import sakura.huggingface" 2>&1 | tail -3
```

Expected: `import sakura.lightning` raises `ModuleNotFoundError`; `import sakura.huggingface` same; full pytest still passes (Plans 1-3 + Plan 4 tests).

- [ ] **Step 7: commit the removal**

```bash
git add -A
git commit -m "refactor: remove v0.1.x submodules — v1.0 clean break

Removed:
  • sakura/lightning, huggingface, tensorflow, ddp, ml/ trees
  • sakura/functional.py, sakura/__main__.py
  • main.py (v0.1.x benchmark script), mnist_demo/, bert_demo/
  • tests/test_{lightning,huggingface,...,cli}.py (11 files)
  • pyproject.toml's [project.scripts].sakura{,-benchmark} entries
  • [tool.maturin].include block (references files now gone)

v1.0 surface: SakuraRuntime + Adapter + Service. Users on v0.1.x should
pin sakura-ml<1.0 if not migrating. Migration guide is a Plan 5 deliverable.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 9: `sakura.adapters` re-exports + top-level + `Compute.zakuro` + acceptance + tag

**Files:** Modify `sakura/adapters/__init__.py`, `sakura/__init__.py`, `sakura/dispatch/compute.py`.

- [ ] **Step 1: populate `sakura/adapters/__init__.py`**

```python
"""sakura.adapters — per-framework bridges that translate hooks to runtime events."""
from sakura.adapters.base import Adapter
from sakura.adapters.ddp import DDPAdapter
from sakura.adapters.huggingface import HFAdapter
from sakura.adapters.lightning import LightningAdapter

__all__ = ["Adapter", "DDPAdapter", "HFAdapter", "LightningAdapter"]
```

- [ ] **Step 2: top-level `sakura/__init__.py`** — add adapter re-exports + ZakuroDispatcher:

Append to existing imports:
```python
from sakura.adapters import Adapter, DDPAdapter, HFAdapter, LightningAdapter
from sakura.dispatch.zakuro import ZakuroDispatcher
```

Append to `__all__`:
```python
    "Adapter",
    "DDPAdapter",
    "HFAdapter",
    "LightningAdapter",
    "ZakuroDispatcher",
```

- [ ] **Step 3: `Compute.zakuro` factory**

In `sakura/dispatch/compute.py`, add a classmethod and update `resolve()` to handle the zakuro kind. After `Compute.in_thread`:

```python
    @classmethod
    def zakuro(cls, zk_compute: Any) -> "Compute":
        """Wrap an existing zakuro.Compute object for users with Zakuro infra."""
        # Stored on the dataclass via a fresh field — see resolve().
        return cls(kind="zakuro", uris=(), strategy="round-robin", _zakuro_compute=zk_compute)  # type: ignore[call-arg]
```

This requires adding `_zakuro_compute: Any = None` to the dataclass fields and updating `resolve()` to handle `kind="zakuro"`. Pragmatic alternative — store the zk_compute in a module-level dict keyed by id, and look it up in resolve(). For Plan 4 we keep the simpler form: ZakuroDispatcher is constructed directly by the user (`ZakuroDispatcher(zk_compute)`); `Compute.zakuro` is a future-Plan-5 ergonomic helper. **Skip Compute.zakuro for Plan 4.** Document this in the commit message.

- [ ] **Step 4: full acceptance**

```bash
export PATH="$HOME/.cargo/bin:/home/foo/.local/bin:$PATH"
source .venv/bin/activate

cargo fmt --all --check && echo fmt_OK
cargo clippy --workspace --all-targets -- -D warnings && echo clippy_OK
cargo test --workspace 2>&1 | grep "test result"
maturin develop --release 2>&1 | tail -3

pytest tests/ 2>&1 | tail -5

python3 -c "from sakura import (SakuraRuntime, LightningAdapter, HFAdapter, DDPAdapter, ZakuroDispatcher, Telemetry, AsyncEval); print('full top-level surface importable')"
# v0.1.x must NOT be importable:
python3 -c "import sakura.lightning" 2>&1 | grep -q "ModuleNotFoundError" && echo "v0.1.x correctly removed"
```

- [ ] **Step 5: tag**

```bash
git tag -a sakura-adapters-v1-foundation -m "Plan 4 complete: framework adapters (Lightning/HF/DDP) + ZakuroDispatcher + v0.1.x clean break"
git tag --list 'sakura-*'
```

- [ ] **Step 6: commit re-exports**

```bash
git add sakura/adapters/__init__.py sakura/__init__.py
git commit -m "feat(adapters): top-level re-exports — Adapter, LightningAdapter, HFAdapter, DDPAdapter, ZakuroDispatcher

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Plan 4 — Acceptance Criteria

Plan 4 is complete when **all** of the following are true:

1. `cargo fmt --all --check` and `cargo clippy --workspace --all-targets -- -D warnings` pass.
2. `cargo test --workspace`: 17 tests still pass (no Rust regressions).
3. `maturin develop --release` builds cleanly.
4. `pytest tests/`: ≥ 100 tests pass (Plans 1-3 had 91; Plan 4 adds ~10-15 adapter tests; v0.1.x test deletions remove some).
5. `from sakura import SakuraRuntime, LightningAdapter, HFAdapter, DDPAdapter, ZakuroDispatcher, Telemetry, ...` resolves.
6. `import sakura.lightning` raises `ModuleNotFoundError`.
7. Tag `sakura-adapters-v1-foundation` exists.

After Plan 4 lands, **Plan 5** (benchmark harness + maturin packaging fix + multi-rank ZeRO1 + migration guide) closes out the redesign.

---

## Self-Review Notes

- **Spec coverage:** Plan 4 implements §10 (framework adapters), §8.3 (`ZakuroDispatcher`), and §15 (migration table — v0.1.x removal).
- **Out-of-scope confirmed:** No benchmark harness, no maturin packaging fix, no multi-rank ZeRO1, no migration guide, no codec zero-copy fix.
- **Plan 4's `Compute.zakuro` deferred:** the convenience factory needs a way to thread `zk_compute` through the dataclass cleanly; Plan 5 handles it. For Plan 4, users construct `ZakuroDispatcher(zk_compute)` directly.
- **Plan 5 carryovers:** maturin packaging gap (cdylib + Python pkg shipping); codec memcpy 7× over budget; multi-rank ZeRO1 sharding; benchmark harness; migration guide.
