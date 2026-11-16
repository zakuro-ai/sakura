"""Dynamic-membership stress (#95): workers join and leave a running fleet.

Drives the real Leader and FederationService over loopback (leader bound to
port 0) through `sakura.services.federation.churn`, applying a churn
schedule that covers every SAK-0014 path — token join, graceful leave,
operator retire, admit-rejoin of a retired roster member, reconnect into a
kept slot, and token rejoin after a retire — and requires zero
consensus-hash discrepancies. Nothing asserts on wall-clock ratios.
"""

from __future__ import annotations

import os

import pytest

torch = pytest.importorskip("torch")

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from sakura.services.federation.churn import (  # noqa: E402
    ChurnEvent,
    churn_schedule,
    run_fleet,
)

from tests.services.test_federation import TinyNet  # noqa: E402

ROSTER = ("a", "b", "c")
POOL = ("d", "e")
ROUNDS = 16


def _data(wid: str):
    g = torch.Generator().manual_seed(sum(map(ord, wid)))
    return torch.randn(128, 32, generator=g), torch.randint(0, 8, (128,), generator=g)


# --------------------------------------------------------------- schedule


class TestChurnSchedule:
    def test_deterministic_and_bounded(self):
        s1 = churn_schedule(ROSTER, POOL, 30, seed=3)
        assert s1 == churn_schedule(ROSTER, POOL, 30, seed=3)
        assert s1 != churn_schedule(ROSTER, POOL, 30, seed=4)
        assert all(2 <= e.at_round <= 27 for e in s1)
        rounds = [e.at_round for e in s1]
        assert rounds == sorted(rounds) and len(set(rounds)) == len(rounds)

    @pytest.mark.parametrize("seed", range(6))
    def test_replays_consistently(self, seed):
        online, offline = set(ROSTER), set(POOL)
        sched = churn_schedule(ROSTER, POOL, 40, min_alive=2, seed=seed)
        assert {e.action for e in sched} == {"join", "leave", "retire"}
        for e in sched:
            if e.action == "join":
                assert e.worker_id in offline
                offline.remove(e.worker_id)
                online.add(e.worker_id)
            else:
                assert e.worker_id in online
                online.remove(e.worker_id)
                offline.add(e.worker_id)
            assert len(online) >= 2
        # the fleet grows first: the pool joins before anyone leaves
        assert [e.action for e in sched[: len(POOL)]] == ["join"] * len(POOL)

    def test_unknown_action_rejected(self):
        with pytest.raises(ValueError, match="unknown churn action"):
            run_fleet(TinyNet, _data, ROSTER, [ChurnEvent(1, "explode", "a")], 1)


# ------------------------------------------------------------------ loopback


def test_churn_keeps_consensus_hash_green():
    sched = [
        ChurnEvent(2, "join", "d"),  # outside the roster: join token
        ChurnEvent(3, "retire", "a"),  # operator revokes a roster member
        ChurnEvent(4, "join", "e"),
        ChurnEvent(5, "leave", "b"),  # graceful leave, slot kept
        ChurnEvent(6, "join", "a"),  # retired roster member: Leader.admit
        ChurnEvent(7, "retire", "d"),
        ChurnEvent(8, "join", "b"),  # reconnect into the kept slot
        ChurnEvent(9, "join", "d"),  # token rejoin after a retire
        ChurnEvent(10, "leave", "c"),
    ]
    r = run_fleet(TinyNet, _data, ROSTER, sched, ROUNDS, gap_s=0.75, batch=16,
                  timeout_s=240)

    assert not r["timed_out"], r
    assert r["rounds"] == ROUNDS, r
    assert r["hash_discrepancies"] == 0, r
    assert r["consensus_bad_rounds"] == []
    assert len(r["events"]) == len(sched)
    assert (r["n_join"], r["n_leave"]) == (5, 4)
    auth = {(e["worker_id"], e["at_round"]): e.get("auth") for e in r["events"]}
    assert auth[("d", 2)] == "token" and auth[("a", 6)] == "admit"
    assert auth[("b", 8)] == "roster" and auth[("d", 9)] == "token"
    assert all(e["retired"] for e in r["events"] if e["action"] == "retire")
    # every (re)joiner was bootstrapped and took part in a later round
    for e in r["events"]:
        if e["action"] == "join":
            assert e["first_round"] is not None, e
    assert r["distinct_workers"] == 5
    # every stint (leavers snapshotted at leave, the rest at the end) sits on
    # a leader-committed anchor
    assert len(r["worker_checks"]) == len(ROSTER) + 5
    assert all(c["ok"] for c in r["worker_checks"]), r["worker_checks"]
    finals = [c for c in r["worker_checks"] if not c["left_mid_run"]]
    assert {c["worker_id"] for c in finals} == {"a", "b", "d", "e"}
    assert all(c["committed_round"] == ROUNDS for c in finals), finals
