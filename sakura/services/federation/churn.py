"""Dynamic-membership stress over the real federation stack (#95).

INTELLECT-1 trained with ~30 providers joining and leaving mid-run. This
module replays that on loopback: a real :class:`Leader` and one real
:class:`FederationService` per worker (each in its own thread, driven by the
DDP adapter), while a controller applies a churn schedule at round
boundaries through the SAK-0014 paths:

* ``join``   — a worker outside the initial roster presents the join token,
  a retired roster member is re-authorised with ``Leader.admit``, and a
  roster member that only disconnected reconnects into its kept slot. Every
  (re)start is a fresh ``FederationService`` (``last_committed = -1``), so it
  is bootstrapped through FULL_SYNC like a restarted process;
* ``leave``  — the worker shuts down; the gather drops it;
* ``retire`` — the worker shuts down and the operator revokes it with
  ``Leader.retire`` (a later rejoin needs ``admit`` or the token).

The run is checked for correctness, not speed: a *consensus-hash
discrepancy* is a committed round whose ``consensus_ok`` is false, or a
worker (live at the end, or snapshotted when it left) whose anchor+BN hash
differs from the leader's hash for the round it last committed. Accuracy is
measured on the leader's final anchor by a caller-supplied ``eval_fn``, so a
static-roster control (empty schedule) gives the accuracy impact of churn.
"""

from __future__ import annotations

import random
import threading
import time
from dataclasses import asdict, dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple

import torch

from .codec import anchor_hash16
from .leader import Leader
from .service import FederationService

ACTIONS = ("join", "leave", "retire")


@dataclass(frozen=True)
class ChurnEvent:
    """Apply `action` to `worker_id` once the leader has committed `at_round`."""

    at_round: int
    action: str
    worker_id: str


def churn_schedule(
    roster: Sequence[str],
    pool: Sequence[str],
    rounds: int,
    every: int = 2,
    min_alive: int = 2,
    seed: int = 0,
    first: int = 2,
    tail: int = 3,
) -> List[ChurnEvent]:
    """A seeded churn schedule: one event every `every` rounds from `first`
    to ``rounds - tail`` (the tail leaves rounds to observe the last event).

    `roster` starts online, `pool` starts offline. At each slot a random
    offline worker joins, or a random online worker leaves (alternating
    graceful ``leave`` and operator ``retire``), never dropping below
    `min_alive`. The first slots join the pool so the fleet grows first."""
    rng = random.Random(seed)
    online, offline = list(roster), list(pool)
    events: List[ChurnEvent] = []
    leaves = 0
    for i, r in enumerate(range(first, rounds - tail + 1, every)):
        can_leave = len(online) > min_alive
        join = bool(offline) and (not can_leave or i < len(pool) or rng.random() < 0.5)
        if join:
            wid = offline.pop(rng.randrange(len(offline)))
            online.append(wid)
            events.append(ChurnEvent(r, "join", wid))
        elif can_leave:
            wid = online.pop(rng.randrange(len(online)))
            offline.append(wid)
            events.append(ChurnEvent(r, ("leave", "retire")[leaves % 2], wid))
            leaves += 1
    return events


class _Worker:
    """One FederationService-driven trainer thread (a 'process' of the fleet)."""

    def __init__(
        self,
        wid: str,
        port: int,
        model_fn: Callable[[], torch.nn.Module],
        data: Tuple[torch.Tensor, torch.Tensor],
        batch: int,
        lr: float,
        codec: str,
        join_token: Optional[str],
        seed: int,
    ) -> None:
        self.wid = wid
        self.stop = threading.Event()
        self.result: Dict[str, Any] = {}
        self.started = threading.Event()  # bootstrap (FULL_SYNC) done
        self._args = (port, model_fn, data, batch, lr, codec, join_token, seed)
        self.thread = threading.Thread(target=self._run, name=f"churn-{wid}", daemon=True)
        self.thread.start()

    def _run(self) -> None:
        from sakura.adapters.ddp import DDPAdapter
        from sakura.runtime import SakuraRuntime

        port, model_fn, (X, y), batch, lr, codec, token, seed = self._args
        g = torch.Generator().manual_seed(seed)
        with torch.random.fork_rng():  # model init seed without racing threads
            torch.manual_seed(seed)
            model = model_fn()
        opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
        fed = FederationService(
            "127.0.0.1", port, self.wid, batch, codec=codec,
            shard_size=int(X.shape[0]), join_token=token,
        )
        rt = SakuraRuntime()
        rt.install(fed)
        adapter = DDPAdapter(rt, rank=0, world_size=1)
        try:
            with rt:
                adapter.on_train_begin(model=model, optimizer=opt, train_loader=None)
                self.started.set()
                step = 0
                while not self.stop.is_set():
                    adapter.on_train_step_begin(model=model, batch=None, step=step)
                    bi = torch.randint(0, X.shape[0], (min(batch, X.shape[0]),), generator=g)
                    opt.zero_grad()
                    loss = torch.nn.functional.cross_entropy(model(X[bi]), y[bi])
                    loss.backward()  # type: ignore[no-untyped-call]
                    adapter.on_optimizer_step(optimizer=opt)
                    opt.step()
                    step += 1
                    time.sleep(0.002)  # keep a many-thread fleet schedulable
            self.result = {
                "steps": step,
                "committed_round": fed.committed_round,
                "hash16": anchor_hash16(fed.anchor, fed._bn_anchor).hex(),
            }
        except Exception as e:  # noqa: BLE001 — reported, counted as a failure
            self.result = {"error": repr(e)}
        finally:
            self.started.set()

    def shutdown(self, timeout: float = 30.0) -> Dict[str, Any]:
        self.stop.set()
        self.thread.join(timeout)
        if self.thread.is_alive():
            return {"error": "worker thread did not exit"}
        return self.result


def run_fleet(
    model_fn: Callable[[], torch.nn.Module],
    data_fn: Callable[[str], Tuple[torch.Tensor, torch.Tensor]],
    roster: Sequence[str],
    schedule: Sequence[ChurnEvent],
    rounds: int,
    gap_s: float = 1.0,
    codec: str = "int8",
    batch: int = 64,
    lr: float = 0.05,
    seed: int = 1,
    join_token: str = "churn-join-token",
    eval_fn: Optional[Callable[[torch.Tensor, torch.Tensor], Dict[str, float]]] = None,
    timeout_s: float = 600.0,
) -> Dict[str, Any]:
    """Run `rounds` real federation rounds on loopback while applying
    `schedule`; return the event log, per-round stats summary, discrepancy
    count and (with `eval_fn`) the final anchor's metrics."""
    for ev in schedule:
        if ev.action not in ACTIONS:
            raise ValueError(f"unknown churn action {ev.action!r}")
    roster = list(roster)
    leader = Leader(
        model_fn=model_fn, port=0, expected=roster, codec=codec, gap_s=gap_s,
        seed=seed, max_rounds=rounds, join_token=join_token,
    )
    hashes: Dict[int, str] = {0: anchor_hash16(leader.anchor, leader.bn).hex()}
    lt = threading.Thread(target=leader.serve_forever, kwargs={"warmup_s": 0.5}, daemon=True)
    lt.start()
    t_end = time.monotonic() + timeout_s
    while getattr(leader, "_srv", None) is None:  # bound to port 0 by _serve
        if time.monotonic() > t_end:
            raise TimeoutError("leader did not bind")
        time.sleep(0.02)
    port = leader._srv.getsockname()[1]

    workers: Dict[str, _Worker] = {}
    # (wid, result, left_mid_run) per stint
    finals: List[Tuple[str, Dict[str, Any], bool]] = []
    retired: Set[str] = set()
    restarts: Dict[str, int] = {}

    def start(wid: str) -> None:
        n = restarts.get(wid, 0)
        restarts[wid] = n + 1
        token = None if wid in roster else join_token
        workers[wid] = _Worker(
            wid, port, model_fn, data_fn(wid), batch, lr, codec, token,
            seed=seed * 1000 + 17 * len(restarts) + n,
        )

    log: List[Dict[str, Any]] = []
    try:
        for wid in roster:
            start(wid)
        pending = sorted(schedule, key=lambda e: e.at_round)
        while lt.is_alive() and time.monotonic() < t_end:
            if pending and leader.round >= pending[0].at_round:
                ev = pending.pop(0)
                rec = {**asdict(ev), "applied_after_round": leader.round}
                if ev.action == "join":
                    if ev.worker_id in workers:
                        raise ValueError(f"{ev.worker_id} is already online")
                    if ev.worker_id in retired:
                        if ev.worker_id in roster:  # no token: re-authorise
                            leader.admit(ev.worker_id)
                            rec["auth"] = "admit"
                        else:
                            rec["auth"] = "token"
                        retired.discard(ev.worker_id)
                    elif ev.worker_id in roster:
                        rec["auth"] = "roster"
                    else:
                        rec["auth"] = "token"
                    start(ev.worker_id)
                else:
                    w = workers.pop(ev.worker_id)
                    finals.append((ev.worker_id, w.shutdown(), True))
                    if ev.action == "retire":
                        rec["retired"] = leader.retire(ev.worker_id)
                        retired.add(ev.worker_id)
                log.append(rec)
                continue
            time.sleep(0.05)
        timed_out = lt.is_alive()
    finally:
        leader.stop.set()
        for wid, w in list(workers.items()):
            finals.append((wid, w.shutdown(), False))
        lt.join(30)
        try:
            leader._srv.close()
        except Exception:  # noqa: BLE001
            pass

    stats = list(leader.stats)
    for s in stats:
        hashes[s["round"]] = s["hash16"]
    bad_rounds = [s["round"] for s in stats if not s["consensus_ok"]]
    worker_checks = []
    for wid, res, mid_run in finals:
        cr = res.get("committed_round")
        h = res.get("hash16")
        ok = "error" not in res and cr in hashes and h == hashes[cr]
        rec_c: Dict[str, Any] = {"worker_id": wid, "committed_round": cr, "left_mid_run": mid_run}
        if not ok and mid_run and isinstance(cr, int) and h == hashes.get(cr + 1):
            # stopped between applying round cr+1 and committing it: the
            # anchor is the leader's cr+1 state, which the leader's resync
            # recognises on reconnect (SAK-0006) — consistent, not forked
            ok = rec_c["applied_uncommitted"] = True
        rec_c["ok"] = ok
        if "error" in res:
            rec_c["error"] = res["error"]
        worker_checks.append(rec_c)
    # first round each (re)joiner took part in, after its join was applied
    for rec in log:
        if rec["action"] == "join":
            after = rec["applied_after_round"]
            rec["first_round"] = next(
                (s["round"] for s in stats
                 if s["round"] > after and rec["worker_id"] in s["participants"]),
                None,
            )
    sizes = [len(s["participants"]) for s in stats]
    out: Dict[str, Any] = {
        "rounds": len(stats),
        "rounds_target": rounds,
        "timed_out": timed_out,
        "roster": roster,
        "events": log,
        "n_join": sum(1 for e in log if e["action"] == "join"),
        "n_leave": sum(1 for e in log if e["action"] != "join"),
        "distinct_workers": len({w for s in stats for w in s["participants"]}),
        "participants_min": min(sizes) if sizes else 0,
        "participants_max": max(sizes) if sizes else 0,
        "consensus_bad_rounds": bad_rounds,
        "worker_checks": worker_checks,
        "hash_discrepancies": len(bad_rounds) + sum(1 for c in worker_checks if not c["ok"]),
        "final_hash16": stats[-1]["hash16"] if stats else None,
    }
    if eval_fn is not None:
        out["eval"] = eval_fn(leader.anchor.clone(), leader.bn.clone())
    return out
