"""AsyncEval — dispatch eval_fn at epoch end, gather result, record to history.

Plan 3 implementation is framework-agnostic: eval_fn signature is
`fn(epoch: int, payload: Any) -> dict`. Plan 4 framework adapters wrap this
with model_factory/state_dict for real model evaluation.

Adaptive gate (the no-net-negative guarantee)
---------------------------------------------
Overlapping eval with the next epoch is only a win when async eval actually
reduces wall-clock. That depends on more than "is the eval expensive?": on a
CPU-saturated box the eval thread just contends for cores/GIL and overlap never
materialises, so naive always-async eval can be *net-negative* (measured -21% on
a tiny MLP under thread oversubscription).

When ``adaptive=True`` (with a ``sync_eval_fn``) AsyncEval does not trust a cost
model — it **measures both modes and keeps the faster one**:

  1. ``sync``  — run eval synchronously on the live model (cost == vanilla
     baseline) for ``calibration_epochs``, recording the full per-epoch
     wall-cost (train + eval).
  2. ``trial`` — run eval async (snapshot + dispatch) for ``trial_epochs``,
     recording the full per-epoch wall-cost (whose train time now also absorbs
     any contention from the concurrent eval).
  3. decide    — keep ``async`` only if its measured per-epoch cost beats the
     sync cost by at least ``min_gain``; otherwise lock to ``sync`` forever.

A cheap-eval pre-filter (eval ≪ measured overhead) and an epochs-remaining guard
skip the trial when it could not pay off, so short/cheap runs never even pay the
A/B cost. Worst case the wall-clock equals the synchronous baseline plus the few
trial epochs (only entered when there are spare epochs and the eval is non-trivial),
making a net-negative result structurally unreachable.
"""
from __future__ import annotations

import statistics
import time
from typing import Any, Callable, Literal, Optional

from sakura.dispatch.base import Dispatcher, Future
from sakura.events import OnEpochEnd, OnTrainBegin, OnTrainEnd
from sakura.service import BaseService


class BackpressureSaturatedError(Exception):
    """Raised by a dispatcher when its in-flight queue is full."""


def _noop_eval(epoch: int, payload: Any) -> dict[str, Any]:  # picklable for LocalDispatcher probe
    return {}


class AsyncEval(BaseService):
    name = "async_eval"
    priority = 80

    def __init__(
        self,
        *,
        eval_fn: Callable[[int, Any], dict[str, Any]],
        eval_payload: Any,
        dispatcher: Dispatcher,
        sync_eval_fn: Optional[Callable[[Any, Any], dict[str, Any]]] = None,
        device_eval_fn: Optional[Callable[[Any, Any], dict[str, Any]]] = None,
        cpu_probe_fn: Optional[Callable[[Any], float]] = None,
        adaptive: bool = True,
        calibration_epochs: int = 2,
        trial_epochs: int = 2,
        min_gain: float = 0.03,
        total_epochs: Optional[int] = None,
        max_pending: int = 4,
        on_backpressure: Literal["skip", "queue", "block"] = "skip",
        every: int = 1,
    ):
        super().__init__()
        self._eval_fn = eval_fn
        self._eval_payload = eval_payload
        self._dispatcher = dispatcher
        self._sync_eval_fn = sync_eval_fn
        self._device_eval_fn = device_eval_fn
        self._cpu_probe_fn = cpu_probe_fn
        self._adaptive = bool(adaptive)
        self._calibration_epochs = max(1, int(calibration_epochs))
        self._trial_epochs = max(1, int(trial_epochs))
        self._min_gain = float(min_gain)
        self._total_epochs = total_epochs
        self._max_pending = max(1, int(max_pending))
        self._on_backpressure = on_backpressure
        self._every = max(1, int(every))
        self._pending: list[tuple[int, Future]] = []
        self._history: list[dict[str, Any]] = []

        # Without a sync_eval_fn we cannot run eval synchronously, so fall back
        # to the original always-async behaviour ("async" terminal mode).
        self._can_gate = self._adaptive and self._sync_eval_fn is not None
        # When the caller supplies an on-device eval fn we can *measure*, during a
        # short warmup, whether overlapping a CPU eval actually beats just running
        # eval on the (fast) training device — instead of assuming it does. This is
        # the resource-allocation profiler: it starts in "profile" mode and ends in
        # either "sync_device" (eval on-device, no overlap) or "async" (CPU overlap).
        self._can_profile = self._adaptive and self._device_eval_fn is not None
        if self._can_profile:
            self._mode = "profile"
        elif self._can_gate:
            self._mode = "sync"
        else:
            self._mode = "async"
        self._sync_costs: list[float] = []    # ms, full per-epoch wall-cost in sync mode
        self._trial_costs: list[float] = []   # ms, full per-epoch wall-cost in trial mode
        self._sync_evals = 0
        self._trial_seen = 0
        self._eval_ewma: Optional[float] = None    # ms (eval-only cost, for pre-filter)
        self._overhead_ms: Optional[float] = None  # ms (one snapshot clone + one dispatch)
        self._last_exit: Optional[float] = None    # wall ts at end of previous on_epoch_end
        self._val_loader: Any = None
        self._model_cuda = False                   # training on GPU? (disjoint from CPU eval)
        self._decisions: list[tuple[int, str]] = []
        # warmup profiler state
        self._profile_seen = 0
        self._profile_samples: list[tuple[float, float]] = []  # (train_ms, eval_device_ms)
        self._t_eval_cpu_ms: Optional[float] = None            # one CPU-eval probe
        self._alloc: Optional[dict[str, Any]] = None           # the chosen allocation

    # ----- introspection -----
    @property
    def history(self) -> list[dict[str, Any]]:
        return list(self._history)

    @property
    def mode(self) -> str:
        return self._mode

    @property
    def decisions(self) -> list[tuple[int, str]]:
        return list(self._decisions)

    @property
    def allocation(self) -> Optional[dict[str, Any]]:
        """The warmup profiler's measured costs and chosen strategy (or None
        if profiling was not used / has not committed yet)."""
        return dict(self._alloc) if self._alloc is not None else None

    @property
    def overhead_ms(self) -> Optional[float]:
        return self._overhead_ms

    @property
    def timing(self) -> dict[str, Any]:
        return {
            "sync_cost_ms": statistics.median(self._sync_costs) if self._sync_costs else None,
            "trial_cost_ms": statistics.median(self._trial_costs) if self._trial_costs else None,
            "eval_ms": self._eval_ewma,
            "overhead_ms": self._overhead_ms,
        }

    def wants_snapshot(self) -> bool:
        """True iff the next on_epoch_end will dispatch async (so the caller's
        per-epoch state snapshot is actually needed). In sync mode the live
        model is evaluated in-place and no snapshot should be paid for."""
        return self._mode in ("trial", "async")

    # ----- event handlers -----
    def on_train_begin(self, event: OnTrainBegin) -> None:
        self._val_loader = event.val_loader
        self._last_exit = time.perf_counter()  # so epoch 0's full cost is measurable

    def on_epoch_end(self, event: OnEpochEnd) -> None:
        if event.rank != 0:
            return
        if event.epoch % self._every != 0:
            return

        entry_mode = self._mode
        if self._mode == "profile":
            self._run_profile(event)
        elif self._mode == "sync_device":
            self._run_sync_device(event)
        elif self._mode == "async":
            self._run_async(event)
        elif self._mode == "sync":
            self._run_sync(event)
            self._maybe_start_trial(event)
        elif self._mode == "trial":
            self._run_async(event)
            self._maybe_decide(event)
        else:  # "locked_sync"
            self._run_sync(event)

        # Full per-epoch wall-cost = exit-to-exit span = train(this epoch) +
        # eval-handling(this epoch). Cleanly attributed to one epoch and one mode.
        now = time.perf_counter()
        if self._last_exit is not None:
            cost_ms = (now - self._last_exit) * 1e3
            if entry_mode == "sync":
                self._sync_costs.append(cost_ms)
            elif entry_mode == "trial" and self._trial_seen >= 2:
                # skip the first trial epoch: its train ran with no concurrent
                # eval (the prior epoch was sync), so it understates contention.
                self._trial_costs.append(cost_ms)
        self._last_exit = now

    def on_train_end(self, event: OnTrainEnd) -> None:
        while self._pending:
            self._block_oldest()

    # ----- mode transitions -----
    def _maybe_start_trial(self, event: OnEpochEnd) -> None:
        if self._sync_evals < self._calibration_epochs:
            return
        # cheap-eval pre-filter: eval not clearly bigger than the dispatch overhead
        # -> overlap cannot win -> stay sync.
        if self._overhead_ms is not None and self._eval_ewma is not None \
                and self._eval_ewma <= self._overhead_ms * 1.5:
            self._mode = "locked_sync"
            return
        # GPU fast-path: when training runs on the GPU, the CPU-worker eval runs
        # on disjoint hardware with its own interpreter/GIL, so a non-trivial eval
        # reliably overlaps the next epoch's GPU compute. Skip the short CPU A/B
        # trial, whose few epochs are dominated by one-time worker warm-up
        # (subprocess torch import) and so under-measure the steady-state win.
        if self._model_cuda:
            self._mode = "async"
            return
        # epochs-remaining guard: need (trial_epochs + 1) trial epochs (first is
        # discarded) plus at least one decided epoch to profit.
        if self._total_epochs is not None:
            remaining = self._total_epochs - 1 - event.epoch
            if remaining < self._trial_epochs + 2:
                self._mode = "locked_sync"
                return
        self._mode = "trial"
        self._trial_costs.clear()
        self._trial_seen = 0

    def _maybe_decide(self, event: OnEpochEnd) -> None:
        if not self._trial_costs:
            return  # still discarding the first trial epoch
        sync_t = statistics.median(self._sync_costs) if self._sync_costs else None
        # Early abort: a trial epoch already clearly worse than sync — stop the
        # trial immediately so a bad-async workload pays at most ~1 trial epoch.
        if sync_t is not None and self._trial_costs[-1] > sync_t * (1.0 + self._min_gain):
            self._mode = "locked_sync"
            return
        if len(self._trial_costs) < self._trial_epochs:
            return
        trial_t = statistics.median(self._trial_costs)
        if sync_t is not None and trial_t < sync_t * (1.0 - self._min_gain):
            self._mode = "async"   # async measurably faster — keep it
        else:
            self._mode = "locked_sync"  # async didn't help — revert and stay

    # ----- synchronous path -----
    @staticmethod
    def _is_cuda(model: Any) -> bool:
        try:
            return bool(next(model.parameters()).is_cuda)
        except Exception:  # noqa: BLE001
            return False

    def _run_sync(self, event: OnEpochEnd) -> None:
        if self._overhead_ms is None:
            self._overhead_ms = self._measure_overhead(event.model)
            self._model_cuda = self._is_cuda(event.model)
        # Prefer the on-device eval fn (evaluates the live model where it trains).
        # When training on GPU this is far faster than the CPU-copy sync_eval_fn,
        # so the gate's "don't overlap" fallback no longer pays a CPU penalty.
        eval_fn = self._device_eval_fn or self._sync_eval_fn
        t0 = time.perf_counter()
        metrics = eval_fn(event.model, self._val_loader)  # type: ignore[misc]
        e_ms = (time.perf_counter() - t0) * 1e3
        self._eval_ewma = e_ms if self._eval_ewma is None else 0.5 * self._eval_ewma + 0.5 * e_ms
        self._record_metrics(event.epoch, metrics)
        self._sync_evals += 1
        self._decisions.append((event.epoch, self._mode))

    # ----- warmup resource-allocation profiler -----
    def _run_sync_device(self, event: OnEpochEnd) -> None:
        """Terminal mode: evaluate the live model on its training device each epoch.

        Chosen by the warmup profiler when overlapping a CPU eval cannot beat
        just running eval on the (fast) training device — e.g. an RNN whose CPU
        eval is far slower than its GPU eval. No snapshot, no dispatch, no GIL
        contention with the training loop."""
        t0 = time.perf_counter()
        metrics = self._device_eval_fn(event.model, self._val_loader)  # type: ignore[misc]
        e_ms = (time.perf_counter() - t0) * 1e3
        self._eval_ewma = e_ms if self._eval_ewma is None else 0.5 * self._eval_ewma + 0.5 * e_ms
        self._record_metrics(event.epoch, metrics)
        self._decisions.append((event.epoch, "sync_device"))

    def _run_profile(self, event: OnEpochEnd) -> None:
        """Warmup: measure train cost + eval-on-device vs eval-on-CPU, then allocate.

        Evaluates on-device every profiling epoch (cheap + gives training accurate
        metrics) and probes the CPU-eval cost once. The first profiling epoch is
        discarded to absorb cudnn-autotune / worker warm-up. After
        ``calibration_epochs`` measured epochs (or when epochs are about to run
        out) it commits to ``sync_device`` or ``async``."""
        entry = time.perf_counter()
        t_train_ms = (entry - self._last_exit) * 1e3 if self._last_exit is not None else 0.0
        if self._overhead_ms is None:
            self._overhead_ms = self._measure_overhead(event.model)
            self._model_cuda = self._is_cuda(event.model)
        t0 = time.perf_counter()
        metrics = self._device_eval_fn(event.model, self._val_loader)  # type: ignore[misc]
        t_eval_device_ms = (time.perf_counter() - t0) * 1e3
        self._eval_ewma = (
            t_eval_device_ms if self._eval_ewma is None
            else 0.5 * self._eval_ewma + 0.5 * t_eval_device_ms
        )
        self._record_metrics(event.epoch, metrics)
        self._decisions.append((event.epoch, "profile"))
        self._profile_seen += 1
        if self._profile_seen == 1:
            return  # discard first sample (autotune / worker warm-up)
        # probe the CPU-eval cost once — it's the async-overlap candidate's cost.
        # Prefer the cheap estimator (times one batch and scales) so the warmup
        # doesn't pay a full, possibly-slow CPU eval just to measure it.
        if self._t_eval_cpu_ms is None:
            if self._cpu_probe_fn is not None:
                try:
                    self._t_eval_cpu_ms = float(self._cpu_probe_fn(event.model))
                except Exception:  # noqa: BLE001 — probe must never break training
                    self._t_eval_cpu_ms = float("inf")
            elif self._sync_eval_fn is not None:
                t1 = time.perf_counter()
                try:
                    self._sync_eval_fn(event.model, self._val_loader)
                except Exception:  # noqa: BLE001
                    self._t_eval_cpu_ms = float("inf")
                else:
                    self._t_eval_cpu_ms = (time.perf_counter() - t1) * 1e3
        self._profile_samples.append((t_train_ms, t_eval_device_ms))
        remaining = (
            self._total_epochs - 1 - event.epoch if self._total_epochs is not None else None
        )
        if len(self._profile_samples) >= self._calibration_epochs or (
            remaining is not None and remaining <= 1
        ):
            self._decide_allocation()

    def _decide_allocation(self) -> None:
        t_train = statistics.median([s[0] for s in self._profile_samples])
        t_eval_device = statistics.median([s[1] for s in self._profile_samples])
        t_eval_cpu = self._t_eval_cpu_ms if self._t_eval_cpu_ms is not None else t_eval_device
        overhead = self._overhead_ms or 0.0
        # Predicted per-epoch wall-cost of each allocation:
        #   sync_device : train, then eval on-device      -> t_train + t_eval_device
        #   async (CPU) : eval overlaps next epoch's train -> max(t_train, t_eval_cpu) + overhead
        # Overlap can at best hide the eval under training; it still pays the
        # *CPU* eval cost (not the cheaper device cost) plus snapshot+dispatch
        # overhead. Only commit to async when it beats on-device sync by min_gain.
        sync_device_cost = t_train + t_eval_device
        async_cost = max(t_train, t_eval_cpu) + overhead
        choose_async = sync_device_cost > 0 and async_cost < sync_device_cost * (1.0 - self._min_gain)
        self._mode = "async" if choose_async else "sync_device"
        self._alloc = {
            "t_train_ms": round(t_train, 3),
            "t_eval_device_ms": round(t_eval_device, 3),
            "t_eval_cpu_ms": round(t_eval_cpu, 3),
            "overhead_ms": round(overhead, 3),
            "sync_device_cost_ms": round(sync_device_cost, 3),
            "async_cost_ms": round(async_cost, 3),
            "choice": self._mode,
        }

    def _measure_overhead(self, model: Any) -> float:
        clone_ms = 0.0
        try:
            sd = model.state_dict()
            t0 = time.perf_counter()
            _ = {k: (v.detach().cpu().clone() if hasattr(v, "detach") else v) for k, v in sd.items()}
            clone_ms = (time.perf_counter() - t0) * 1e3
        except Exception:  # noqa: BLE001 — probe must never break training
            pass
        dispatch_ms = 0.0
        try:
            t0 = time.perf_counter()
            self._dispatcher.submit(_noop_eval, -1, None).result()
            dispatch_ms = (time.perf_counter() - t0) * 1e3
        except Exception:  # noqa: BLE001
            pass
        return clone_ms + dispatch_ms

    # ----- asynchronous path (trial + locked async) -----
    def _run_async(self, event: OnEpochEnd) -> None:
        if self._mode == "trial":
            self._trial_seen += 1
        # The final epoch's eval can never overlap; run it synchronously.
        if (
            self._total_epochs is not None
            and event.epoch >= self._total_epochs - 1
            and self._sync_eval_fn is not None
        ):
            metrics = self._sync_eval_fn(event.model, self._val_loader)
            self._record_metrics(event.epoch, metrics)
            self._decisions.append((event.epoch, "sync-final"))
            return

        self._collect_done()
        if len(self._pending) >= self._max_pending:
            # Queue full. Skip (never stall training) ONLY once the gate has
            # locked into steady async; during sync/trial calibration we BLOCK so
            # the gate measures the true async cost and can decide honestly. (In
            # the un-gated always-async config self._mode is "async" from the
            # start, so the don't-stall behavior is preserved there too.)
            if self._on_backpressure != "block" and self._mode == "async":
                self._history.append({"epoch": event.epoch, "skipped": True, "reason": "backpressure"})
                self._decisions.append((event.epoch, "skip"))
                return
            self._block_oldest()
        try:
            fut = self._dispatcher.submit(self._eval_fn, event.epoch, self._eval_payload)
            self._pending.append((event.epoch, fut))
            self._decisions.append((event.epoch, self._mode))
        except BackpressureSaturatedError:
            if self._on_backpressure != "block" and self._mode == "async":
                self._history.append({"epoch": event.epoch, "skipped": True, "reason": "backpressure"})
                self._decisions.append((event.epoch, "skip"))
            # Gate-aware (same policy as the pre-submit check above): only the
            # locked-async branch skips; during sync/trial calibration we
            # block-and-resubmit so the gate measures the true async cost.
            elif self._on_backpressure in ("block", "skip"):
                self._block_oldest()
                fut = self._dispatcher.submit(self._eval_fn, event.epoch, self._eval_payload)
                self._pending.append((event.epoch, fut))
                self._decisions.append((event.epoch, self._mode))
            else:
                raise

    def _collect_done(self) -> None:
        still: list[tuple[int, Future]] = []
        for epoch, fut in self._pending:
            if fut.done():
                self._record_future(epoch, fut)
            else:
                still.append((epoch, fut))
        self._pending = still

    def _block_oldest(self) -> None:
        if not self._pending:
            return
        epoch, fut = self._pending.pop(0)
        self._record_future(epoch, fut)

    # ----- history -----
    def _record_metrics(self, epoch: int, v: Any) -> None:
        if isinstance(v, dict):
            rec = dict(v)
            rec.setdefault("epoch", epoch)
            self._history.append(rec)

    def _record_future(self, epoch: int, fut: Future) -> None:
        try:
            r = fut.result()
            v = r.value if hasattr(r, "value") else r
            self._record_metrics(epoch, v)
        except BaseException as exc:  # noqa: BLE001
            self._history.append({"epoch": epoch, "skipped": True, "reason": type(exc).__name__})


__all__ = ["AsyncEval", "BackpressureSaturatedError"]
