"""Reactive training: one strict-xfail test per open gap (epic #249, Phase 0 #243).

Each test asserts the behaviour the roadmap wants, so today it fails and is
marked ``xfail(strict=True)``. When a later phase closes a gap the test starts
passing, strict mode turns that into a failure, and the phase PR removes the
marker. The ``reason`` names the phase issue that closes the gap.

Gaps that need hardware or a multi-rank lane (snapshot consistency, sharded
eval, same-step decisions across DDP ranks) are settled by investigation in
#243 rather than encoded here.
"""
from __future__ import annotations

import dataclasses
import inspect
import os
import signal
import threading
import time
from typing import Any, Optional

import pytest

from sakura import events
from sakura.dispatch.base import Dispatcher, Future, Result
from sakura.dispatch.in_thread import InThreadDispatcher
from sakura.events import OnEpochEnd, OnTrainEnd
from sakura.runtime import SakuraRuntime
from sakura.service import BaseService
from sakura.services.async_checkpoint import AsyncCheckpoint
from sakura.services.async_eval import AsyncEval

PHASE1 = "#244 Phase 1 (foundations)"
PHASE2 = "#245 Phase 2 (control channel)"
PHASE3 = "#246 Phase 3 (streaming eval)"
PHASE5 = "#248 Phase 5 (robustness)"


# --------------------------------------------------------------------- fakes


class _HeldFuture(Future):
    """A future that resolves only when the test releases it."""

    def __init__(self, value: Any) -> None:
        self._value = value
        self._event = threading.Event()

    def release(self) -> None:
        self._event.set()

    def result(self, timeout: Optional[float] = None) -> Result:
        if not self._event.wait(timeout):
            raise TimeoutError("held future not released")
        return Result(value=self._value)

    def done(self) -> bool:
        return self._event.is_set()

    def cancel(self) -> bool:
        return False


class _HoldingDispatcher(Dispatcher):
    """Runs the callable eagerly but holds every future until released, so the
    test controls exactly how many evals are in flight."""

    def __init__(self) -> None:
        self.submitted: list[tuple[Any, ...]] = []
        self.futures: list[_HeldFuture] = []

    def submit(self, callable, *args, timeout_ms=None, **kwargs):  # noqa: A002
        self.submitted.append(args)
        fut = _HeldFuture(callable(*args, **kwargs))
        self.futures.append(fut)
        return fut

    def release_all(self) -> None:
        for f in self.futures:
            f.release()


def _eval_fn(epoch: int, payload: Any) -> dict[str, Any]:
    return {"val_loss": 1.0 / (epoch + 1)}


def _epoch_end(epoch: int, model: Any = "m", **extra: Any) -> OnEpochEnd:
    return OnEpochEnd(epoch=epoch, model=model, optimizer="o", metrics={},
                      rank=0, world_size=1, **extra)


def _train_end() -> OnTrainEnd:
    return OnTrainEnd(model="m", history=[], rank=0, world_size=1)


def _skipped_epochs(svc: AsyncEval) -> list[int]:
    return [r["epoch"] for r in svc.history if r.get("skipped")]


def _run_saturated(on_backpressure: str) -> tuple[AsyncEval, _HoldingDispatcher]:
    """Steady async mode, one eval slot, three epochs while the first is still
    running: the queue is full for epochs 1 and 2."""
    disp = _HoldingDispatcher()
    svc = AsyncEval(eval_fn=_eval_fn, eval_payload=None, dispatcher=disp,
                    adaptive=False, max_pending=1, on_backpressure=on_backpressure)
    rt = SakuraRuntime()
    rt.install(svc)
    for e in range(3):
        rt.dispatch(_epoch_end(e))
    disp.release_all()
    rt.dispatch(_train_end())
    return svc, disp


# ------------------------------------------------------------------- Phase 1


@pytest.mark.xfail(strict=True, reason=f"{PHASE1}: no OnTrainStepEnd, no global_step on OnOptimizerStep")
def test_step_level_events_carry_global_step():
    step_end = getattr(events, "OnTrainStepEnd", None)
    assert step_end is not None, "sakura.events has no OnTrainStepEnd"
    assert "global_step" in {f.name for f in dataclasses.fields(step_end)}
    assert "global_step" in {f.name for f in dataclasses.fields(events.OnOptimizerStep)}


@pytest.mark.xfail(strict=True, reason=f"{PHASE1}: eval results are tagged by epoch only")
def test_eval_results_are_tagged_by_global_step():
    assert "global_step" in {f.name for f in dataclasses.fields(OnEpochEnd)}, \
        "OnEpochEnd carries no global_step to tag results with"
    svc = AsyncEval(eval_fn=_eval_fn, eval_payload=None,
                    dispatcher=InThreadDispatcher(), adaptive=False)
    svc.on_epoch_end(_epoch_end(0, global_step=100))
    svc.on_train_end(_train_end())
    assert svc.history[0].get("global_step") == 100


@pytest.mark.xfail(strict=True, reason=f"{PHASE1}: AsyncEval keeps results in a private _history; no event")
def test_eval_results_reach_other_services():
    seen: list[Any] = []

    class Recorder(BaseService):
        name = "recorder"
        priority = 10

        def on_eval_result(self, event):
            seen.append(event)

    svc = AsyncEval(eval_fn=_eval_fn, eval_payload=None,
                    dispatcher=InThreadDispatcher(), adaptive=False)
    rt = SakuraRuntime()
    rt.install(svc)
    rt.install(Recorder())
    for e in range(2):
        rt.dispatch(_epoch_end(e))
    rt.dispatch(_train_end())
    assert len(seen) == 2, "no OnEvalResult was dispatched to other services"


@pytest.mark.xfail(strict=True, reason=f"{PHASE1}: AsyncEval submits a payload fixed at construction")
def test_async_eval_sees_the_weights_of_the_epoch_it_scores():
    disp = _HoldingDispatcher()
    svc = AsyncEval(eval_fn=_eval_fn, eval_payload={"static": True},
                    dispatcher=disp, adaptive=False, max_pending=8)
    svc.on_epoch_end(_epoch_end(0, model={"w": 0}))
    svc.on_epoch_end(_epoch_end(1, model={"w": 1}))
    disp.release_all()
    svc.on_train_end(_train_end())
    # Drop the epoch argument; what is left is what eval_fn gets to score.
    scored = [args[1:] for args in disp.submitted]
    assert scored[0] != scored[1], "both epochs were scored against the same fixed payload"


@pytest.mark.xfail(strict=True, reason=f"{PHASE1}: AsyncCheckpoint(every='best') reads OnEpochEnd.metrics only")
def test_best_checkpoint_follows_async_eval(tmp_path):
    def writer(state, path):
        return {"path": str(path)}

    ckpt = AsyncCheckpoint(dir=str(tmp_path), dispatcher=InThreadDispatcher(),
                           state_provider=lambda: {"w": 1}, writer=writer,
                           every="best", metric="val_loss", source="async_eval")
    ev = AsyncEval(eval_fn=_eval_fn, eval_payload=None,
                   dispatcher=InThreadDispatcher(), adaptive=False)
    rt = SakuraRuntime()
    rt.install(ev)
    rt.install(ckpt)
    for e in range(3):  # val_loss improves every epoch; framework metrics are empty
        rt.dispatch(_epoch_end(e))
    rt.dispatch(_train_end())
    assert [r for r in ckpt.history if not r.get("skipped")], "no best checkpoint written"


# ------------------------------------------------------------------- Phase 2


@pytest.mark.xfail(strict=True, reason=f"{PHASE2}: the runtime has no command path back to the trainer")
def test_runtime_accepts_commands():
    assert callable(getattr(SakuraRuntime, "issue", None)), "SakuraRuntime.issue is missing"


@pytest.mark.xfail(strict=True, reason=f"{PHASE2}: nothing records or replays decisions")
def test_runtime_can_replay_a_command_log(tmp_path):
    log = tmp_path / "commands.jsonl"
    log.write_text("")
    assert "replay" in inspect.signature(SakuraRuntime).parameters
    SakuraRuntime(replay=str(log))


# ------------------------------------------------------------------- Phase 3


@pytest.mark.xfail(strict=True, reason=f"{PHASE3}: AsyncEval fires on epoch_end + every N epochs only")
def test_async_eval_has_step_and_time_triggers():
    params = inspect.signature(AsyncEval).parameters
    assert "every_steps" in params and "every_seconds" in params


@pytest.mark.xfail(strict=True, reason=f"{PHASE3}: a full queue drops the newest eval, not the stale one")
def test_backpressure_keeps_the_newest_snapshot():
    svc, _ = _run_saturated("skip")
    assert 2 not in _skipped_epochs(svc), \
        f"newest epoch 2 was dropped (skipped={_skipped_epochs(svc)})"


@pytest.mark.xfail(strict=True, reason=f"{PHASE3}: on_backpressure='queue' acts as 'skip' in async mode")
def test_queue_backpressure_differs_from_skip():
    q, _ = _run_saturated("queue")
    s, _ = _run_saturated("skip")
    assert q.decisions != s.decisions, f"queue and skip made identical decisions: {q.decisions}"


# ------------------------------------------------------------------- Phase 5


def _never_resolves(*_a, **_k) -> dict[str, Any]:
    return {}


@pytest.mark.xfail(strict=True, reason=f"{PHASE5}: on_train_end waits forever on an eval that never resolves")
def test_train_end_does_not_hang_when_the_eval_worker_is_gone():
    disp = _HoldingDispatcher()  # never released: a worker that died mid-eval
    svc = AsyncEval(eval_fn=_never_resolves, eval_payload=None,
                    dispatcher=disp, adaptive=False)
    svc.on_epoch_end(_epoch_end(0))
    t = threading.Thread(target=svc.on_train_end, args=(_train_end(),), daemon=True)
    t.start()
    t.join(timeout=3.0)
    assert not t.is_alive(), "on_train_end is still blocked on the lost eval"


def _double(x):
    return x * 2


@pytest.mark.xfail(strict=True, reason=f"{PHASE5}: ProcessDispatcher has no respawn; later submits never resolve")
def test_process_dispatcher_recovers_after_worker_death():
    from sakura.dispatch import ProcessDispatcher

    d = ProcessDispatcher()
    try:
        d.wait_ready(120)
        assert d.submit(_double, 2).result(30).value == 4
        os.kill(d.stats()["pid"], signal.SIGKILL)
        time.sleep(1.0)  # let the collector notice
        assert d.submit(_double, 3).result(15).value == 6
    finally:
        d.shutdown(timeout_s=5)
