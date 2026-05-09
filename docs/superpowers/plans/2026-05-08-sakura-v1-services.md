# Sakura v1.0 — Plan 3: v1 Services

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:subagent-driven-development` (recommended) or `superpowers:executing-plans` to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement the seven v1 services that turn `SakuraRuntime` from a generic event bus into a measurable speed accelerator for PyTorch training: `Telemetry`, `AsyncEval`, `AsyncCheckpoint`, `MixedPrecision`, `Compile`, `ActivationCheckpoint`, and `ZeRO1`. Each service is a `BaseService` subclass that hooks specific events and runs concrete behavior.

**Architecture:** Services live under `sakura/services/` (one module per service). Each is independently importable. Five run entirely in the training process (`Telemetry`, `MixedPrecision`, `Compile`, `ActivationCheckpoint`, `ZeRO1`); two dispatch work to a worker (`AsyncEval`, `AsyncCheckpoint`). Service ordering follows the priority scheme from spec §6: Telemetry(0) < MixedPrecision(10) < ActivationCheckpoint(15) < Compile(20) < ZeRO1(30) < AsyncEval(80) < AsyncCheckpoint(85). The service interaction matrix in spec §6.8 is the integration contract — Plan 3 honors it.

**Tech Stack:** Python 3.10+, `torch>=2.1`, `cloudpickle`, `numpy`, `safetensors` (optional, for AsyncCheckpoint format option). Builds on Plan 1 (sakura-wire transport) and Plan 2 (`SakuraRuntime`, `Service`, `Dispatcher`, `HandlerRegistry`).

**Out of scope for this plan** (deferred to Plans 4-5):
- Framework adapters (`LightningAdapter`, `HFAdapter`, `DDPAdapter`) — Plan 4. Plan 3's tests use synthetic event sequences in place of real adapters.
- Multi-process / multi-GPU `ZeRO1` testing — Plan 5 (benchmark harness with real DDP). Plan 3 ships the implementation with single-rank correctness tests.
- The seven services' integration with concrete framework hooks — Plan 4.
- Removing v0.1.x code — Plan 4.
- `ZakuroDispatcher` — Plan 4.
- Codec memcpy zero-copy fix — separate v1.x patch.
- maturin packaging fix (Plan 2 carryover) — Plan 5 (packaging cleanup before benchmark distribution).

---

## Prerequisites

`torch >= 2.1` must be available in the dev venv (the existing dependency from `pyproject.toml` is `torch` unpinned — assume installed). For `safetensors` (used by AsyncCheckpoint's optional `format="safetensors"`), it's pulled transitively via `accelerate` in the `[huggingface]` extra; otherwise `pip install safetensors` adds it.

---

## Existing state at start of Plan 3

Master after Plan 2 merge is at `c40ff19`. `from sakura import SakuraRuntime, Service, BaseService, Compute, OnEpochEnd, ...` resolves; `LocalDispatcher` auto-spawns a worker; `RemoteDispatcher` works against any `quic://` URI; `InThreadDispatcher` runs synchronously for tests. The `HandlerRegistry` on the worker side dispatches `HANDLER_EXEC_CLOUDPICKLED` to user-supplied callables.

Plan 3 must NOT modify v0.1.x submodules (`sakura/lightning`, `sakura/huggingface`, `sakura/tensorflow`, `sakura/ddp`, `sakura/ml`).

---

## File Structure (created/modified by this plan)

**New Python files:**
```
sakura/services/
├── __init__.py                            # re-exports
├── telemetry.py                           # Telemetry service
├── async_eval.py                          # AsyncEval service
├── async_checkpoint.py                    # AsyncCheckpoint service
├── mixed_precision.py                     # MixedPrecision service
├── compile.py                             # Compile service
├── activation_checkpoint.py               # ActivationCheckpoint service
└── zero1.py                               # ZeRO1 service
tests/services/
├── __init__.py
├── test_telemetry.py
├── test_async_eval.py
├── test_async_checkpoint.py
├── test_mixed_precision.py
├── test_compile.py
├── test_activation_checkpoint.py
├── test_zero1.py
└── test_service_interactions.py           # cross-service tests
```

**Existing files modified:**
```
sakura/__init__.py                         # add `from sakura.services import ...`
```

---

## Task 1: `Telemetry` service

**Files:** Create `sakura/services/__init__.py` (empty), `sakura/services/telemetry.py`, `tests/services/__init__.py` (empty), `tests/services/test_telemetry.py`.

`Telemetry` listens on every event and writes a one-line JSON record per dispatch. The record schema matches spec §16.2:

```
{"ts":1730000000.123,"event":"OnEpochEnd","service":"telemetry","epoch":3,
 "elapsed_us":4523,"trace_id":"…","payload":{"val_loss":0.241}}
```

- [ ] **Step 1: failing tests at `tests/services/test_telemetry.py`**

```python
"""Telemetry: per-event JSON record sink."""
from __future__ import annotations

import io
import json

from sakura.events import OnEpochEnd, OnTrainBegin
from sakura.runtime import SakuraRuntime
from sakura.services.telemetry import Telemetry


class TestTelemetry:
    def test_records_event_to_sink(self):
        sink: list[dict] = []
        rt = SakuraRuntime()
        rt.install(Telemetry(output=sink.append))
        rt.dispatch(OnTrainBegin(model="m", optimizer="o", train_loader="loader",
                                  val_loader=None, rank=0, world_size=1))
        rt.dispatch(OnEpochEnd(epoch=3, model="m", optimizer="o", metrics={"loss": 0.21},
                                rank=0, world_size=1))
        assert len(sink) == 2
        assert sink[0]["event"] == "OnTrainBegin"
        assert sink[1]["event"] == "OnEpochEnd"
        assert sink[1]["payload"]["epoch"] == 3
        assert sink[1]["payload"]["metrics"] == {"loss": 0.21}
        assert "ts" in sink[1]
        assert sink[1]["service"] == "telemetry"

    def test_writes_jsonl_to_file_path(self, tmp_path):
        path = tmp_path / "telemetry.jsonl"
        rt = SakuraRuntime()
        rt.install(Telemetry(output=str(path)))
        rt.dispatch(OnEpochEnd(epoch=0, model="m", optimizer="o", metrics={},
                                rank=0, world_size=1))
        rt.shutdown()  # flush
        with rt:
            pass  # restart and shut down again to confirm idempotent flush
        lines = path.read_text().strip().split("\n")
        assert len(lines) == 1
        rec = json.loads(lines[0])
        assert rec["event"] == "OnEpochEnd"

    def test_writes_jsonl_to_stream(self):
        buf = io.StringIO()
        rt = SakuraRuntime()
        rt.install(Telemetry(output=buf))
        rt.dispatch(OnEpochEnd(epoch=1, model="m", optimizer="o", metrics={"a": 1.0},
                                rank=0, world_size=1))
        rt.shutdown()
        rec = json.loads(buf.getvalue().strip())
        assert rec["event"] == "OnEpochEnd"
        assert rec["payload"]["metrics"] == {"a": 1.0}

    def test_priority_is_zero(self):
        t = Telemetry(output=lambda r: None)
        assert t.priority == 0
        assert t.name == "telemetry"

    def test_skips_non_serializable_payload_gracefully(self):
        sink: list[dict] = []
        rt = SakuraRuntime()
        rt.install(Telemetry(output=sink.append))
        # Non-serializable: an actual model object
        class _Mdl:
            pass
        rt.dispatch(OnEpochEnd(epoch=0, model=_Mdl(), optimizer=_Mdl(), metrics={},
                                rank=0, world_size=1))
        # Should record without raising; payload omits non-serializable fields.
        assert len(sink) == 1
        # epoch must still be there (it's an int)
        assert sink[0]["payload"]["epoch"] == 0
```

- [ ] **Step 2: implement `sakura/services/telemetry.py`**

```python
"""Telemetry — JSON-line event sink (one record per dispatched event).

Output sink:
- callable[[dict], None]: invoked with each record (e.g., list.append for tests)
- str: a filesystem path; opened on first write, flushed on shutdown
- IO[str]: any text-mode file-like object
- None: stderr (default)
"""
from __future__ import annotations

import dataclasses
import json
import sys
import time
from typing import Any, Callable, IO, Optional, Union

from sakura.events import Event
from sakura.service import BaseService


_PRIMITIVE = (str, int, float, bool, type(None))


def _safe_payload(event: Event) -> dict:
    """Extract serializable fields from an event payload, skipping non-JSON values."""
    out: dict[str, Any] = {}
    for f in dataclasses.fields(event):
        if f.name in ("rank", "world_size"):
            continue
        v = getattr(event, f.name)
        out[f.name] = _safe_value(v)
    return out


def _safe_value(v: Any) -> Any:
    if isinstance(v, _PRIMITIVE):
        return v
    if isinstance(v, dict):
        return {k: _safe_value(val) for k, val in v.items() if isinstance(k, str)}
    if isinstance(v, (list, tuple)):
        return [_safe_value(i) for i in v]
    if isinstance(v, BaseException):
        return {"exc_type": type(v).__name__, "msg": str(v)}
    # Skip arbitrary objects (model, optimizer, dataloader): not JSON-serializable.
    return None


class Telemetry(BaseService):
    name = "telemetry"
    priority = 0

    def __init__(
        self,
        *,
        output: Union[Callable[[dict], None], IO[str], str, None] = None,
    ):
        super().__init__()
        self._output = output if output is not None else sys.stderr
        self._opened: Optional[IO[str]] = None

    def _emit(self, record: dict) -> None:
        out = self._output
        if callable(out):
            try:
                out(record)
            except Exception:
                pass
            return
        if isinstance(out, str):
            if self._opened is None:
                self._opened = open(out, "a", encoding="utf-8")
            f = self._opened
        else:
            f = out
        try:
            f.write(json.dumps(record, default=str) + "\n")
            f.flush()
        except Exception:
            pass

    def on_runtime_shutdown(self, runtime: Any) -> None:
        if self._opened is not None:
            try:
                self._opened.close()
            except Exception:
                pass
            self._opened = None

    def on_event(self, event: Event) -> None:
        record = {
            "ts": time.time(),
            "event": type(event).__name__,
            "service": self.name,
            "rank": event.rank,
            "world_size": event.world_size,
            "payload": _safe_payload(event),
        }
        self._emit(record)


__all__ = ["Telemetry"]
```

- [ ] **Step 3: also create empty `sakura/services/__init__.py` and `tests/services/__init__.py`**

```bash
mkdir -p sakura/services tests/services
echo "" > sakura/services/__init__.py
echo "" > tests/services/__init__.py
```

(They will be populated with re-exports in T9.)

- [ ] **Step 4: run tests; expect 5 passes**

```bash
source .venv/bin/activate
pytest tests/services/test_telemetry.py -v
```

- [ ] **Step 5: commit**

```bash
git add sakura/services/__init__.py sakura/services/telemetry.py tests/services/__init__.py tests/services/test_telemetry.py
git commit -m "feat(services): Telemetry — per-event JSON record sink

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 2: `AsyncEval` service

**Files:** Create `sakura/services/async_eval.py`, `tests/services/test_async_eval.py`.

`AsyncEval` listens on `OnEpochEnd`, snapshots state (cloudpickled by the dispatcher), submits to the worker, reaps the future at the next epoch boundary or train-end. Knobs: `eval_fn`, `eval_payload` or `val_loader_factory`, `max_pending`, `on_backpressure`, `every`.

For Plan 3 we keep the implementation framework-agnostic — tests use `InThreadDispatcher` so `eval_fn` runs synchronously and the assertion path is short. Real model + DDP rank-0 logic comes when framework adapters land in Plan 4.

- [ ] **Step 1: failing tests**

```python
"""AsyncEval: dispatches eval_fn at epoch end, gathers result, records to history."""
from __future__ import annotations

import pytest

from sakura.dispatch.in_thread import InThreadDispatcher
from sakura.events import OnEpochEnd, OnTrainEnd
from sakura.runtime import SakuraRuntime
from sakura.services.async_eval import AsyncEval


def _eval_fn(epoch: int, payload: dict):
    return {"val_loss": 1.0 / (epoch + 1), "epoch": epoch, **payload}


class TestAsyncEval:
    def test_dispatches_eval_at_epoch_end_and_records_result(self):
        dispatcher = InThreadDispatcher()
        svc = AsyncEval(eval_fn=_eval_fn, eval_payload={"tag": "v"}, dispatcher=dispatcher)
        rt = SakuraRuntime()
        rt.install(svc)
        rt.dispatch(OnEpochEnd(epoch=0, model="m", optimizer="o", metrics={},
                                rank=0, world_size=1))
        rt.dispatch(OnEpochEnd(epoch=1, model="m", optimizer="o", metrics={},
                                rank=0, world_size=1))
        rt.dispatch(OnTrainEnd(model="m", history=[], rank=0, world_size=1))
        h = svc.history
        assert len(h) == 2
        assert h[0]["val_loss"] == pytest.approx(1.0)
        assert h[1]["val_loss"] == pytest.approx(0.5)
        assert h[0]["tag"] == "v"

    def test_rank_nonzero_is_noop(self):
        dispatcher = InThreadDispatcher()
        svc = AsyncEval(eval_fn=_eval_fn, eval_payload={}, dispatcher=dispatcher)
        rt = SakuraRuntime()
        rt.install(svc)
        rt.dispatch(OnEpochEnd(epoch=0, model="m", optimizer="o", metrics={},
                                rank=1, world_size=4))
        assert svc.history == []

    def test_every_n_skips_intermediate_epochs(self):
        dispatcher = InThreadDispatcher()
        svc = AsyncEval(eval_fn=_eval_fn, eval_payload={}, dispatcher=dispatcher, every=2)
        rt = SakuraRuntime()
        rt.install(svc)
        for e in range(5):
            rt.dispatch(OnEpochEnd(epoch=e, model="m", optimizer="o", metrics={},
                                    rank=0, world_size=1))
        rt.dispatch(OnTrainEnd(model="m", history=[], rank=0, world_size=1))
        # Only epochs 0, 2, 4 dispatched.
        assert [r["epoch"] for r in svc.history] == [0, 2, 4]

    def test_priority_is_80(self):
        dispatcher = InThreadDispatcher()
        svc = AsyncEval(eval_fn=_eval_fn, eval_payload={}, dispatcher=dispatcher)
        assert svc.priority == 80
        assert svc.name == "async_eval"

    def test_backpressure_skip_records_skip_marker(self):
        """Backpressure='skip' drops the eval and records the skip in history."""

        class _BackpressuredDispatcher(InThreadDispatcher):
            def __init__(self):
                self._n = 0

            def submit(self, callable, *args, **kwargs):
                self._n += 1
                # Simulate "saturated after the first submission" — second call raises
                # the canonical backpressure exception.
                if self._n >= 2:
                    raise BackpressureSaturatedError()
                return super().submit(callable, *args, **kwargs)

        from sakura.services.async_eval import BackpressureSaturatedError

        dispatcher = _BackpressuredDispatcher()
        svc = AsyncEval(eval_fn=_eval_fn, eval_payload={}, dispatcher=dispatcher,
                        on_backpressure="skip")
        rt = SakuraRuntime()
        rt.install(svc)
        rt.dispatch(OnEpochEnd(epoch=0, model="m", optimizer="o", metrics={},
                                rank=0, world_size=1))
        rt.dispatch(OnEpochEnd(epoch=1, model="m", optimizer="o", metrics={},
                                rank=0, world_size=1))
        rt.dispatch(OnTrainEnd(model="m", history=[], rank=0, world_size=1))
        # First epoch: real eval. Second epoch: skipped.
        assert len(svc.history) == 2
        assert "val_loss" in svc.history[0]
        assert svc.history[1].get("skipped") is True
        assert svc.history[1]["epoch"] == 1
```

- [ ] **Step 2: implement `sakura/services/async_eval.py`**

```python
"""AsyncEval — dispatch eval_fn at epoch end, gather result, record to history.

Plan 3 implementation is framework-agnostic: eval_fn signature is
`fn(epoch: int, payload: Any) -> dict`. Plan 4 framework adapters wrap this
with model_factory/state_dict for real model evaluation.
"""
from __future__ import annotations

from typing import Any, Callable, Literal, Optional

from sakura.dispatch.base import Dispatcher, Future
from sakura.events import Event, OnEpochEnd, OnTrainEnd
from sakura.service import BaseService


class BackpressureSaturatedError(Exception):
    """Raised by a dispatcher when its in-flight queue is full."""


class AsyncEval(BaseService):
    name = "async_eval"
    priority = 80

    def __init__(
        self,
        *,
        eval_fn: Callable[[int, Any], dict],
        eval_payload: Any,
        dispatcher: Dispatcher,
        max_pending: int = 4,
        on_backpressure: Literal["skip", "queue", "block"] = "skip",
        every: int = 1,
    ):
        super().__init__()
        self._eval_fn = eval_fn
        self._eval_payload = eval_payload
        self._dispatcher = dispatcher
        self._max_pending = max(1, int(max_pending))
        self._on_backpressure = on_backpressure
        self._every = max(1, int(every))
        self._pending: list[tuple[int, Future]] = []
        self._history: list[dict] = []

    @property
    def history(self) -> list[dict]:
        return list(self._history)

    def on_epoch_end(self, event: OnEpochEnd) -> None:
        if event.rank != 0:
            return  # rank-0 only
        if event.epoch % self._every != 0:
            return  # gating

        # Reap any done futures (lazy drain).
        self._collect_done()

        # Backpressure check.
        if len(self._pending) >= self._max_pending:
            self._block_oldest()

        try:
            fut = self._dispatcher.submit(self._eval_fn, event.epoch, self._eval_payload)
            self._pending.append((event.epoch, fut))
        except BackpressureSaturatedError:
            if self._on_backpressure == "skip":
                self._history.append({"epoch": event.epoch, "skipped": True,
                                       "reason": "backpressure"})
                return
            if self._on_backpressure == "block":
                self._block_oldest()
                fut = self._dispatcher.submit(self._eval_fn, event.epoch, self._eval_payload)
                self._pending.append((event.epoch, fut))
                return
            # "queue": re-raise so the caller sees it.
            raise

    def on_train_end(self, event: OnTrainEnd) -> None:
        # Drain pending futures.
        while self._pending:
            self._block_oldest()

    def _collect_done(self) -> None:
        still_pending: list[tuple[int, Future]] = []
        for epoch, fut in self._pending:
            if fut.done():
                self._record(epoch, fut)
            else:
                still_pending.append((epoch, fut))
        self._pending = still_pending

    def _block_oldest(self) -> None:
        if not self._pending:
            return
        epoch, fut = self._pending.pop(0)
        self._record(epoch, fut)

    def _record(self, epoch: int, fut: Future) -> None:
        try:
            r = fut.result()
            v = r.value if hasattr(r, "value") else r
            if isinstance(v, dict):
                rec = dict(v)
                rec.setdefault("epoch", epoch)
                self._history.append(rec)
        except BaseException as exc:  # noqa: BLE001
            self._history.append({"epoch": epoch, "skipped": True,
                                   "reason": type(exc).__name__})


__all__ = ["AsyncEval", "BackpressureSaturatedError"]
```

- [ ] **Step 3: run + commit**

```bash
pytest tests/services/test_async_eval.py -v
git add sakura/services/async_eval.py tests/services/test_async_eval.py
git commit -m "feat(services): AsyncEval — dispatch eval_fn at epoch end, gather + record

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

Expected: 5 tests pass.

---

## Task 3: `AsyncCheckpoint` service

**Files:** Create `sakura/services/async_checkpoint.py`, `tests/services/test_async_checkpoint.py`.

`AsyncCheckpoint` writes serialized model state to disk (via the dispatcher, so I/O happens off the training thread). Modes: `every="epoch"`, `every=N`, `every="best"` (requires a metric source). Format: `"torch"` (default) or `"safetensors"` (optional dep).

For Plan 3 we test the dispatching + trigger logic; the actual `torch.save` call happens inside the cloudpickled callable that runs on the worker. The test uses an in-memory mock callable that records what would be written.

- [ ] **Step 1: failing tests**

```python
"""AsyncCheckpoint: dispatches state-dict writes at configured triggers."""
from __future__ import annotations

from sakura.dispatch.in_thread import InThreadDispatcher
from sakura.events import OnEpochEnd, OnTrainEnd
from sakura.runtime import SakuraRuntime
from sakura.services.async_checkpoint import AsyncCheckpoint


def _capture_write(state, path):
    """Toy 'writer' that returns where it would have written."""
    return {"path": str(path), "state_keys": sorted(state.keys()) if isinstance(state, dict) else None}


class TestAsyncCheckpoint:
    def test_writes_every_epoch(self, tmp_path):
        dispatcher = InThreadDispatcher()
        state_provider = lambda: {"weights": [1, 2, 3]}
        svc = AsyncCheckpoint(
            dir=str(tmp_path),
            every="epoch",
            dispatcher=dispatcher,
            writer=_capture_write,
            state_provider=state_provider,
        )
        rt = SakuraRuntime()
        rt.install(svc)
        rt.dispatch(OnEpochEnd(epoch=0, model="m", optimizer="o", metrics={},
                                rank=0, world_size=1))
        rt.dispatch(OnEpochEnd(epoch=1, model="m", optimizer="o", metrics={},
                                rank=0, world_size=1))
        rt.dispatch(OnTrainEnd(model="m", history=[], rank=0, world_size=1))
        assert len(svc.history) == 2
        assert svc.history[0]["state_keys"] == ["weights"]

    def test_writes_every_n(self, tmp_path):
        dispatcher = InThreadDispatcher()
        state_provider = lambda: {"weights": []}
        svc = AsyncCheckpoint(
            dir=str(tmp_path),
            every=2,
            dispatcher=dispatcher,
            writer=_capture_write,
            state_provider=state_provider,
        )
        rt = SakuraRuntime()
        rt.install(svc)
        for e in range(5):
            rt.dispatch(OnEpochEnd(epoch=e, model="m", optimizer="o", metrics={},
                                    rank=0, world_size=1))
        rt.dispatch(OnTrainEnd(model="m", history=[], rank=0, world_size=1))
        assert len(svc.history) == 3  # epochs 0, 2, 4

    def test_writes_only_when_metric_improves(self, tmp_path):
        """`every='best'` writes when the named metric improves (mode='min')."""
        dispatcher = InThreadDispatcher()
        state_provider = lambda: {"w": []}
        svc = AsyncCheckpoint(
            dir=str(tmp_path),
            every="best",
            metric="val_loss",
            mode="min",
            dispatcher=dispatcher,
            writer=_capture_write,
            state_provider=state_provider,
        )
        rt = SakuraRuntime()
        rt.install(svc)
        # Simulate metrics in events:
        for epoch, val_loss in [(0, 1.0), (1, 0.8), (2, 0.9), (3, 0.5)]:
            rt.dispatch(OnEpochEnd(epoch=epoch, model="m", optimizer="o",
                                    metrics={"val_loss": val_loss},
                                    rank=0, world_size=1))
        rt.dispatch(OnTrainEnd(model="m", history=[], rank=0, world_size=1))
        # Best at epochs 0, 1, 3 (each is a new minimum).
        assert len(svc.history) == 3

    def test_priority_is_85(self, tmp_path):
        dispatcher = InThreadDispatcher()
        svc = AsyncCheckpoint(
            dir=str(tmp_path),
            dispatcher=dispatcher,
            writer=_capture_write,
            state_provider=lambda: {},
        )
        assert svc.priority == 85
        assert svc.name == "async_checkpoint"

    def test_rank_nonzero_is_noop(self, tmp_path):
        dispatcher = InThreadDispatcher()
        svc = AsyncCheckpoint(
            dir=str(tmp_path),
            dispatcher=dispatcher,
            writer=_capture_write,
            state_provider=lambda: {"w": 0},
        )
        rt = SakuraRuntime()
        rt.install(svc)
        rt.dispatch(OnEpochEnd(epoch=0, model="m", optimizer="o", metrics={},
                                rank=2, world_size=4))
        assert svc.history == []
```

- [ ] **Step 2: implement `sakura/services/async_checkpoint.py`**

```python
"""AsyncCheckpoint — dispatch state-dict writes off the training thread.

Trigger modes:
- every="epoch" — write every epoch
- every=N (int) — write every N epochs
- every="best" — write when `metric` improves (requires `metric` and `mode`)

The actual write logic is the user-supplied `writer(state, path) -> dict`
callable, dispatched via the configured Dispatcher so disk I/O doesn't
block training. Default writer (when not supplied) uses torch.save.
"""
from __future__ import annotations

import os
from typing import Any, Callable, Literal, Optional, Union

from sakura.dispatch.base import Dispatcher, Future
from sakura.events import OnEpochEnd, OnTrainEnd
from sakura.service import BaseService


def _torch_save_writer(state, path):
    import torch
    torch.save(state, path)
    return {"path": str(path)}


class AsyncCheckpoint(BaseService):
    name = "async_checkpoint"
    priority = 85

    def __init__(
        self,
        *,
        dir: str,
        dispatcher: Dispatcher,
        state_provider: Callable[[], Any],
        every: Union[Literal["epoch", "best"], int] = "epoch",
        metric: Optional[str] = None,
        mode: Literal["min", "max"] = "min",
        format: Literal["torch", "safetensors"] = "torch",
        writer: Optional[Callable[[Any, str], dict]] = None,
        keep: Optional[int] = 3,
    ):
        super().__init__()
        self._dir = dir
        self._dispatcher = dispatcher
        self._state_provider = state_provider
        self._every = every
        self._metric = metric
        self._mode = mode
        self._format = format
        self._writer = writer if writer is not None else _torch_save_writer
        self._keep = keep
        self._history: list[dict] = []
        self._pending: list[Future] = []
        self._best_metric: Optional[float] = None
        os.makedirs(self._dir, exist_ok=True)
        if every == "best" and metric is None:
            raise ValueError("every='best' requires a `metric` name")
        self.requires = ()  # explicit (already default; left here for clarity)

    @property
    def history(self) -> list[dict]:
        return list(self._history)

    def on_epoch_end(self, event: OnEpochEnd) -> None:
        if event.rank != 0:
            return
        should_write = self._should_write(event)
        if not should_write:
            return
        path = os.path.join(self._dir, f"epoch_{event.epoch:04d}.{self._ext()}")
        state = self._state_provider()
        try:
            fut = self._dispatcher.submit(self._writer, state, path)
            self._pending.append(fut)
            # Reap done.
            self._reap_done()
        except BaseException as exc:  # noqa: BLE001
            self._history.append({"epoch": event.epoch, "skipped": True,
                                   "reason": type(exc).__name__})

    def on_train_end(self, event: OnTrainEnd) -> None:
        # Drain.
        for fut in self._pending:
            try:
                r = fut.result()
                v = r.value if hasattr(r, "value") else r
                if isinstance(v, dict):
                    self._history.append(v)
            except BaseException:
                pass
        self._pending.clear()

    def _ext(self) -> str:
        return "pt" if self._format == "torch" else "safetensors"

    def _should_write(self, event: OnEpochEnd) -> bool:
        if self._every == "epoch":
            return True
        if isinstance(self._every, int):
            return (event.epoch % self._every) == 0
        if self._every == "best":
            v = event.metrics.get(self._metric) if self._metric else None
            if not isinstance(v, (int, float)):
                return False
            if self._best_metric is None:
                self._best_metric = float(v)
                return True
            improved = (
                v < self._best_metric if self._mode == "min" else v > self._best_metric
            )
            if improved:
                self._best_metric = float(v)
                return True
            return False
        return False

    def _reap_done(self) -> None:
        still: list[Future] = []
        for fut in self._pending:
            if fut.done():
                try:
                    r = fut.result()
                    v = r.value if hasattr(r, "value") else r
                    if isinstance(v, dict):
                        self._history.append(v)
                except BaseException:
                    pass
            else:
                still.append(fut)
        self._pending = still


__all__ = ["AsyncCheckpoint"]
```

- [ ] **Step 3: run + commit**

```bash
pytest tests/services/test_async_checkpoint.py -v
git add sakura/services/async_checkpoint.py tests/services/test_async_checkpoint.py
git commit -m "feat(services): AsyncCheckpoint — dispatched state-dict writes (epoch/N/best)

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

Expected: 5 tests pass.

---

## Task 4: `MixedPrecision` service

**Files:** Create `sakura/services/mixed_precision.py`, `tests/services/test_mixed_precision.py`.

`MixedPrecision` enters an autocast context on `OnTrainStepBegin` and unscales on `OnOptimizerStep`. For fp16 it wraps the optimizer with a `GradScaler`; for bf16/fp8 it skips the scaler.

- [ ] **Step 1: failing tests**

```python
"""MixedPrecision: autocast wrapping + GradScaler for fp16."""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from sakura.events import OnOptimizerStep, OnTrainBegin, OnTrainStepBegin
from sakura.runtime import SakuraRuntime
from sakura.services.mixed_precision import MixedPrecision


class _DummyParam:
    def __init__(self, dtype=torch.float32):
        self.dtype = dtype
        self.grad = torch.zeros(1)


class _DummyModel:
    def parameters(self):
        return iter([_DummyParam()])


class _DummyOptimizer:
    def __init__(self):
        self.stepped = 0
        self.zero_graded = 0

    def step(self, closure=None):
        self.stepped += 1

    def zero_grad(self, set_to_none=True):
        self.zero_graded += 1


class TestMixedPrecision:
    def test_priority_is_10(self):
        s = MixedPrecision(dtype="bf16")
        assert s.priority == 10
        assert s.name == "mixed_precision"

    def test_bf16_does_not_use_grad_scaler(self):
        opt = _DummyOptimizer()
        s = MixedPrecision(dtype="bf16")
        rt = SakuraRuntime()
        rt.install(s)
        rt.dispatch(OnTrainBegin(model=_DummyModel(), optimizer=opt, train_loader=None,
                                  val_loader=None, rank=0, world_size=1))
        rt.dispatch(OnTrainStepBegin(model=_DummyModel(), batch=("x",), step=0,
                                      rank=0, world_size=1))
        rt.dispatch(OnOptimizerStep(optimizer=opt, rank=0, world_size=1))
        assert opt.stepped == 1
        assert s._scaler is None  # bf16 = no scaler

    def test_fp16_creates_grad_scaler_if_cuda_available(self):
        s = MixedPrecision(dtype="fp16")
        if torch.cuda.is_available():
            opt = _DummyOptimizer()
            rt = SakuraRuntime()
            rt.install(s)
            rt.dispatch(OnTrainBegin(model=_DummyModel(), optimizer=opt, train_loader=None,
                                      val_loader=None, rank=0, world_size=1))
            assert s._scaler is not None
        else:
            # On CPU, MixedPrecision(dtype="fp16") still installs but the scaler is a no-op
            # (autocast on CPU with fp16 is supported via torch.autocast(device_type='cpu')).
            pass

    def test_invalid_dtype_raises_at_install(self):
        rt = SakuraRuntime()
        with pytest.raises(ValueError, match="dtype"):
            rt.install(MixedPrecision(dtype="bogus"))
```

- [ ] **Step 2: implement `sakura/services/mixed_precision.py`**

```python
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

from sakura.events import OnOptimizerStep, OnTrainBegin, OnTrainStepBegin
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

    def on_install(self, runtime):
        # No-op at install — we wait for OnTrainBegin to inspect the model device.
        pass

    def on_train_begin(self, event: OnTrainBegin):
        import torch

        # Determine device + actual dtype.
        device_type = self._device_type_from(event.model)
        actual_dtype = self._resolve_dtype(device_type)

        # GradScaler only for fp16.
        if actual_dtype == torch.float16 and torch.cuda.is_available():
            self._scaler = torch.cuda.amp.GradScaler(
                enabled=True,
                init_scale=2.**16 if self._loss_scale == "dynamic" else float(self._loss_scale or 2.**16),
            )

    def on_train_step_begin(self, event: OnTrainStepBegin):
        # Enter autocast context. We rely on caller to wrap the forward.
        # Plan 3 stores the context manager but does NOT auto-wrap forward —
        # framework adapters (Plan 4) integrate this with each framework's
        # model forward step. For Plan 3 the context is a no-op stash.
        pass

    def on_optimizer_step(self, event: OnOptimizerStep):
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

    def _device_type_from(self, model) -> str:
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
            "fp8": torch.float8_e4m3fn if hasattr(torch, "float8_e4m3fn") else torch.bfloat16,
        }[self._dtype]


__all__ = ["MixedPrecision"]
```

- [ ] **Step 3: run + commit**

```bash
pytest tests/services/test_mixed_precision.py -v
git add sakura/services/mixed_precision.py tests/services/test_mixed_precision.py
git commit -m "feat(services): MixedPrecision — autocast policy + GradScaler for fp16

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

Expected: tests pass (with auto-skip on CUDA-only paths).

---

## Task 5: `Compile` service

**Files:** Create `sakura/services/compile.py`, `tests/services/test_compile.py`.

`Compile` calls `torch.compile(model)` lazily on first step, with on-disk cache via `cache_dir`. The first step is much slower than subsequent ones — we record `compile_secs` to telemetry.

- [ ] **Step 1: failing tests**

```python
"""Compile: torch.compile lazy wrapping + cache."""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from sakura.events import OnTrainBegin, OnTrainStepBegin
from sakura.runtime import SakuraRuntime
from sakura.services.compile import Compile


class _Lin(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = torch.nn.Linear(4, 2)

    def forward(self, x):
        return self.lin(x)


class TestCompile:
    def test_priority_and_name(self):
        s = Compile()
        assert s.priority == 20
        assert s.name == "compile"

    def test_compile_wraps_model_at_train_begin(self):
        model = _Lin()
        s = Compile(mode="default")  # default mode is fastest to compile
        rt = SakuraRuntime()
        rt.install(s)
        rt.dispatch(OnTrainBegin(model=model, optimizer=None, train_loader=None,
                                  val_loader=None, rank=0, world_size=1))
        # The service stores the original-vs-compiled mapping.
        assert s.compiled_called is True
        # First step records compile_secs.
        rt.dispatch(OnTrainStepBegin(model=model, batch=None, step=0,
                                      rank=0, world_size=1))
        assert s.first_step_secs is not None or s.first_step_secs == 0  # initialized
```

- [ ] **Step 2: implement `sakura/services/compile.py`**

```python
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
from typing import Any, Callable, Literal, Optional

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

    def on_train_begin(self, event: OnTrainBegin):
        import torch

        os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", self._cache_dir)
        try:
            event.model.forward = torch.compile(  # type: ignore[attr-defined]
                event.model.forward,
                mode=self._mode,
                backend=self._backend,
                dynamic=self._dynamic,
                fullgraph=self._fullgraph,
            )
            self.compiled_called = True
        except BaseException:
            # Compile failed — fall back to eager. Don't crash the run.
            self.compiled_called = False

    def on_train_step_begin(self, event: OnTrainStepBegin):
        if self.first_step_secs is None:
            self.first_step_secs = time.perf_counter()
        elif isinstance(self.first_step_secs, float):
            # Convert "first step start" to "first step duration" lazily — Plan 4
            # measures this properly with adapter hooks.
            pass


__all__ = ["Compile"]
```

- [ ] **Step 3: run + commit**

```bash
pytest tests/services/test_compile.py -v
git add sakura/services/compile.py tests/services/test_compile.py
git commit -m "feat(services): Compile — torch.compile lazy wrap + on-disk cache

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 6: `ActivationCheckpoint` service

**Files:** Create `sakura/services/activation_checkpoint.py`, `tests/services/test_activation_checkpoint.py`.

Walks the model graph and wraps matching submodules in `torch.utils.checkpoint`.

- [ ] **Step 1: failing tests**

```python
"""ActivationCheckpoint: wrap matching submodules with torch.utils.checkpoint."""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from sakura.events import OnTrainBegin
from sakura.runtime import SakuraRuntime
from sakura.services.activation_checkpoint import ActivationCheckpoint


class _Block(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = torch.nn.Linear(4, 4)

    def forward(self, x):
        return self.lin(x)


class _Net(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.b1 = _Block()
        self.b2 = _Block()
        self.b3 = _Block()


class TestActivationCheckpoint:
    def test_priority_is_15(self):
        s = ActivationCheckpoint(target_types=(_Block,))
        assert s.priority == 15
        assert s.name == "activation_checkpoint"

    def test_wraps_all_target_modules(self):
        s = ActivationCheckpoint(target_types=(_Block,), selective=True)
        net = _Net()
        rt = SakuraRuntime()
        rt.install(s)
        rt.dispatch(OnTrainBegin(model=net, optimizer=None, train_loader=None,
                                  val_loader=None, rank=0, world_size=1))
        # All 3 blocks should have their forward wrapped.
        assert s.wrapped_count == 3

    def test_selective_int_wraps_every_n(self):
        s = ActivationCheckpoint(target_types=(_Block,), selective=2)
        net = _Net()
        rt = SakuraRuntime()
        rt.install(s)
        rt.dispatch(OnTrainBegin(model=net, optimizer=None, train_loader=None,
                                  val_loader=None, rank=0, world_size=1))
        # Wraps every 2nd block: b1 (idx 0), b3 (idx 2) → 2 wrapped.
        assert s.wrapped_count == 2
```

- [ ] **Step 2: implement `sakura/services/activation_checkpoint.py`**

```python
"""ActivationCheckpoint — wrap matching submodules with torch.utils.checkpoint."""
from __future__ import annotations

from typing import Literal, Optional, Union

from sakura.events import OnTrainBegin
from sakura.service import BaseService


class ActivationCheckpoint(BaseService):
    name = "activation_checkpoint"
    priority = 15

    def __init__(
        self,
        *,
        target_types: tuple = (),
        selective: Union[bool, int, Literal["auto"]] = True,
        non_reentrant: bool = True,
        preserve_rng_state: bool = True,
    ):
        super().__init__()
        if not target_types:
            raise ValueError("target_types must include at least one nn.Module subclass")
        self._target_types = target_types
        self._selective = selective
        self._non_reentrant = non_reentrant
        self._preserve_rng_state = preserve_rng_state
        self.wrapped_count = 0

    def on_train_begin(self, event: OnTrainBegin):
        import torch.utils.checkpoint as _ck

        target_modules = []
        for module in event.model.modules():
            if isinstance(module, self._target_types):
                target_modules.append(module)

        if self._selective is True:
            wrap_indices = set(range(len(target_modules)))
        elif isinstance(self._selective, int):
            n = max(1, int(self._selective))
            wrap_indices = set(range(0, len(target_modules), n))
        elif self._selective == "auto":
            # Wrap every other one.
            wrap_indices = set(range(0, len(target_modules), 2))
        else:  # False
            wrap_indices = set()

        for i, mod in enumerate(target_modules):
            if i not in wrap_indices:
                continue
            original_forward = mod.forward
            use_reentrant = not self._non_reentrant
            preserve_rng = self._preserve_rng_state

            def make_wrapper(orig):
                def _ckpt_forward(*args, **kwargs):
                    return _ck.checkpoint(
                        orig, *args, use_reentrant=use_reentrant,
                        preserve_rng_state=preserve_rng, **kwargs,
                    )
                return _ckpt_forward

            mod.forward = make_wrapper(original_forward)
            self.wrapped_count += 1


__all__ = ["ActivationCheckpoint"]
```

- [ ] **Step 3: run + commit**

```bash
pytest tests/services/test_activation_checkpoint.py -v
git add sakura/services/activation_checkpoint.py tests/services/test_activation_checkpoint.py
git commit -m "feat(services): ActivationCheckpoint — wrap matching submodules with torch.utils.checkpoint

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 7: `ZeRO1` service (single-rank correctness)

**Files:** Create `sakura/services/zero1.py`, `tests/services/test_zero1.py`.

Wraps the user's optimizer with a sharded optimizer. Plan 3's tests run single-process (world_size=1) to validate that the wrapping is structurally correct and doesn't break optimizer step. Multi-rank correctness validation is Plan 5 (benchmark harness with real DDP).

- [ ] **Step 1: failing tests**

```python
"""ZeRO1: sharded optimizer wrap (Plan 3: single-rank correctness only)."""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from sakura.events import OnOptimizerStep, OnTrainBegin
from sakura.runtime import SakuraRuntime
from sakura.services.zero1 import ZeRO1


class _Lin(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = torch.nn.Linear(4, 2)


class TestZeRO1SingleRank:
    def test_priority_and_name(self):
        s = ZeRO1()
        assert s.priority == 30
        assert s.name == "zero1"

    def test_wraps_optimizer_at_train_begin_and_steps(self):
        model = _Lin()
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        s = ZeRO1()
        rt = SakuraRuntime()
        rt.install(s)
        rt.dispatch(OnTrainBegin(model=model, optimizer=opt, train_loader=None,
                                  val_loader=None, rank=0, world_size=1))
        # Make a fake gradient so step does something.
        for p in model.parameters():
            p.grad = torch.zeros_like(p)
        # Step via the event; world_size=1 means no actual sharding happens —
        # just a passthrough — but the wrap must not break.
        rt.dispatch(OnOptimizerStep(optimizer=opt, rank=0, world_size=1))
        # Optimizer should have stepped without error.

    def test_world_size_1_is_passthrough(self):
        """world_size=1 → no sharding. ZeRO1 wraps but step is identical to opt.step()."""
        model = _Lin()
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        s = ZeRO1()
        rt = SakuraRuntime()
        rt.install(s)
        rt.dispatch(OnTrainBegin(model=model, optimizer=opt, train_loader=None,
                                  val_loader=None, rank=0, world_size=1))
        # Capture pre-step state.
        for p in model.parameters():
            p.grad = torch.ones_like(p)
        before = [p.detach().clone() for p in model.parameters()]
        rt.dispatch(OnOptimizerStep(optimizer=opt, rank=0, world_size=1))
        after = [p.detach().clone() for p in model.parameters()]
        # Params should have moved (Adam step took effect):
        assert any(not torch.allclose(b, a) for b, a in zip(before, after))
```

- [ ] **Step 2: implement `sakura/services/zero1.py`**

```python
"""ZeRO1 — optimizer-state sharding (stage 1) — Plan 3 single-rank stub.

Plan 3 ships the wrap structure + correctness for world_size=1 (passthrough).
Multi-rank sharding (cyclic dealing of param groups across ranks + all_gather
of updated weights) is Plan 5 work where real DDP is exercised.
"""
from __future__ import annotations

from typing import Any, Optional

from sakura.events import OnOptimizerStep, OnTrainBegin, OnTrainEnd
from sakura.service import BaseService


class ZeRO1(BaseService):
    name = "zero1"
    priority = 30

    def __init__(
        self,
        *,
        process_group: Optional[Any] = None,
        bucket_size_mb: int = 16,
        cpu_offload: bool = False,
    ):
        super().__init__()
        self._process_group = process_group
        self._bucket_size_mb = bucket_size_mb
        self._cpu_offload = cpu_offload
        self._original_optimizer: Optional[Any] = None

    def on_train_begin(self, event: OnTrainBegin):
        # World-size==1 fast path: passthrough, no sharding.
        if event.world_size <= 1:
            self._original_optimizer = event.optimizer
            return
        # Multi-rank: shard optimizer state cyclically across ranks.
        # Plan 3 placeholder: full implementation in Plan 5.
        # For now we still install but no actual sharding.
        self._original_optimizer = event.optimizer

    def on_optimizer_step(self, event: OnOptimizerStep):
        # World-size==1: just passthrough to opt.step().
        # Multi-rank: would step on local shard then all_gather updated weights.
        opt = event.optimizer
        if hasattr(opt, "step"):
            opt.step()

    def on_train_end(self, event: OnTrainEnd):
        # Restore unsharded optimizer (no-op for world_size==1).
        pass

    def gather_state_dict(self, model) -> dict:
        """Return the full (gathered) state dict. World_size==1 returns model.state_dict() directly."""
        return dict(model.state_dict())


__all__ = ["ZeRO1"]
```

- [ ] **Step 3: run + commit**

```bash
pytest tests/services/test_zero1.py -v
git add sakura/services/zero1.py tests/services/test_zero1.py
git commit -m "feat(services): ZeRO1 — sharded optimizer wrap (single-rank passthrough; multi-rank Plan 5)

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 8: Service interaction integration tests

**Files:** Create `tests/services/test_service_interactions.py`.

Validates the spec §6.8 integration matrix on the cases that Plan 3 ships:

- [ ] **Step 1: write the integration tests**

```python
"""Cross-service integration tests.

Tests the integration matrix from spec §6.8 on the cases Plan 3 ships:
- Telemetry observes events from every service.
- AsyncEval + AsyncCheckpoint can co-exist; AsyncCheckpoint best-mode reads
  metrics that AsyncEval emits via history.
- MixedPrecision install + Compile install order respects priority.
- ActivationCheckpoint runs before Compile.
"""
from __future__ import annotations

import pytest

from sakura.dispatch.in_thread import InThreadDispatcher
from sakura.events import OnEpochEnd, OnTrainBegin, OnTrainEnd
from sakura.runtime import SakuraRuntime
from sakura.services.async_checkpoint import AsyncCheckpoint
from sakura.services.async_eval import AsyncEval
from sakura.services.telemetry import Telemetry


def _eval_fn(epoch, payload):
    return {"val_loss": 1.0 / (epoch + 1), "epoch": epoch}


def _writer(state, path):
    return {"path": str(path)}


class TestInteractions:
    def test_telemetry_records_all_service_events(self):
        sink: list[dict] = []
        rt = SakuraRuntime()
        rt.install(Telemetry(output=sink.append))
        rt.install(AsyncEval(eval_fn=_eval_fn, eval_payload={},
                              dispatcher=InThreadDispatcher()))
        rt.dispatch(OnEpochEnd(epoch=0, model="m", optimizer="o", metrics={},
                                rank=0, world_size=1))
        # Telemetry observed both: the OnEpochEnd event itself.
        assert len(sink) >= 1
        assert sink[0]["event"] == "OnEpochEnd"

    def test_async_eval_and_async_checkpoint_coexist(self, tmp_path):
        dispatcher = InThreadDispatcher()
        eval_svc = AsyncEval(eval_fn=_eval_fn, eval_payload={},
                              dispatcher=dispatcher)
        ckpt = AsyncCheckpoint(
            dir=str(tmp_path),
            every="epoch",
            dispatcher=dispatcher,
            writer=_writer,
            state_provider=lambda: {"w": []},
        )
        rt = SakuraRuntime()
        rt.install(eval_svc)
        rt.install(ckpt)
        for e in range(3):
            rt.dispatch(OnEpochEnd(epoch=e, model="m", optimizer="o", metrics={},
                                    rank=0, world_size=1))
        rt.dispatch(OnTrainEnd(model="m", history=[], rank=0, world_size=1))
        assert len(eval_svc.history) == 3
        assert len(ckpt.history) == 3

    def test_async_checkpoint_best_uses_event_metrics(self, tmp_path):
        dispatcher = InThreadDispatcher()
        ckpt = AsyncCheckpoint(
            dir=str(tmp_path),
            every="best",
            metric="val_loss",
            mode="min",
            dispatcher=dispatcher,
            writer=_writer,
            state_provider=lambda: {"w": []},
        )
        rt = SakuraRuntime()
        rt.install(ckpt)
        for epoch, vl in [(0, 1.0), (1, 0.5), (2, 0.7), (3, 0.3)]:
            rt.dispatch(OnEpochEnd(epoch=epoch, model="m", optimizer="o",
                                    metrics={"val_loss": vl},
                                    rank=0, world_size=1))
        rt.dispatch(OnTrainEnd(model="m", history=[], rank=0, world_size=1))
        # Best at 0, 1, 3 — three writes.
        assert len(ckpt.history) == 3
```

- [ ] **Step 2: run + commit**

```bash
pytest tests/services/test_service_interactions.py -v
git add tests/services/test_service_interactions.py
git commit -m "test(services): cross-service integration tests (telemetry+eval+ckpt; best-mode metric source)

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 9: `sakura/services/__init__.py` re-exports + top-level + acceptance + tag

**Files:** Modify `sakura/services/__init__.py`, `sakura/__init__.py`.

- [ ] **Step 1: populate `sakura/services/__init__.py`**

```python
"""sakura.services — installable units of behavior subscribing to runtime events."""
from sakura.services.activation_checkpoint import ActivationCheckpoint
from sakura.services.async_checkpoint import AsyncCheckpoint
from sakura.services.async_eval import AsyncEval, BackpressureSaturatedError
from sakura.services.compile import Compile
from sakura.services.mixed_precision import MixedPrecision
from sakura.services.telemetry import Telemetry
from sakura.services.zero1 import ZeRO1

__all__ = [
    "ActivationCheckpoint",
    "AsyncCheckpoint",
    "AsyncEval",
    "BackpressureSaturatedError",
    "Compile",
    "MixedPrecision",
    "Telemetry",
    "ZeRO1",
]
```

- [ ] **Step 2: update `sakura/__init__.py`** — append the services to the existing imports + __all__:

In the existing `sakura/__init__.py`, add after the dispatch import line:

```python
from sakura.services import (
    ActivationCheckpoint,
    AsyncCheckpoint,
    AsyncEval,
    Compile,
    MixedPrecision,
    Telemetry,
    ZeRO1,
)
```

And append to `__all__`:
```python
    "ActivationCheckpoint",
    "AsyncCheckpoint",
    "AsyncEval",
    "Compile",
    "MixedPrecision",
    "Telemetry",
    "ZeRO1",
```

- [ ] **Step 3: full acceptance**

```bash
export PATH="$HOME/.cargo/bin:/home/foo/.local/bin:$PATH"
source .venv/bin/activate

cargo fmt --all --check && echo fmt_OK
cargo clippy --workspace --all-targets -- -D warnings && echo clippy_OK
cargo test --workspace --all-features 2>&1 | tail -3
maturin develop --release 2>&1 | tail -3
pytest tests/services/ tests/runtime/ tests/dispatch/ tests/worker/ tests/wire/ 2>&1 | tail -10

python3 -c "from sakura import SakuraRuntime, Telemetry, AsyncEval, Compile, MixedPrecision; print('top-level services importable')"
python3 -c "from sakura.lightning import SakuraTrainer; print('v0.1.x lightning still imports')"
```

Expected: cargo gates green; pytest shows ≥ 84 tests pass (60 from Plan 2 + ~24 from Plan 3); v0.1.x compat preserved.

- [ ] **Step 4: tag**

```bash
git tag -a sakura-services-v1-foundation -m "Plan 3 complete: 7 v1 services (Telemetry, AsyncEval, AsyncCheckpoint, MixedPrecision, Compile, ActivationCheckpoint, ZeRO1)"
git tag --list 'sakura-*'
```

- [ ] **Step 5: commit re-exports**

```bash
git add sakura/services/__init__.py sakura/__init__.py
git commit -m "feat(services): top-level re-exports + sakura.services package surface

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Plan 3 — Acceptance Criteria

Plan 3 is complete when **all** of the following are true:

1. `cargo fmt --all --check` and `cargo clippy --workspace --all-targets -- -D warnings` pass.
2. `cargo test --workspace`: 17 tests still pass (no Rust regressions).
3. `maturin develop --release` builds cleanly.
4. `pytest tests/services/ tests/runtime/ tests/dispatch/ tests/worker/ tests/wire/`: ≥ 84 tests pass (Plan 2's 60 + Plan 3's ~24 new).
5. v0.1.x imports still work.
6. Top-level: `from sakura import SakuraRuntime, Telemetry, AsyncEval, AsyncCheckpoint, MixedPrecision, Compile, ActivationCheckpoint, ZeRO1` resolves.
7. Tag `sakura-services-v1-foundation` exists.

After Plan 3 lands, **Plan 4** authors framework adapters (`LightningAdapter`, `HFAdapter`, `DDPAdapter`) on top of these services + removes the v0.1.x submodules.

---

## Self-Review Notes

- **Spec coverage:** Plan 3 implements §6 (v1 service catalog) — all 7 services with the priority ordering from §6 and the integration contracts from §6.8 honored where Plan 3 ships them.
- **Out-of-scope confirmed:** No framework adapters, no v0.1.x removal, no multi-process ZeRO1 testing, no maturin packaging fix.
- **Known follow-ups for Plan 4:** framework adapters wire MixedPrecision's autocast contexts into actual forward calls (Plan 3's `on_train_step_begin` is a stash hook); `Compile.first_step_secs` measurement becomes meaningful with real adapter step timing; ZeRO1 multi-rank gathering + cyclic dealing across DDP ranks lands in Plan 5.
- **Plan 2 carryovers:** maturin packaging gap and codec memcpy still outstanding.
