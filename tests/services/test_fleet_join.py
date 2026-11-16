"""Runtime fleet membership (SAK-0014).

A worker that is NOT in `Leader(expected=[...])` joins a running leader when
the operator opted in (allow-list / predicate / join token), is bootstrapped
through the ordinary FULL_SYNC path, takes part in the very next round, and
later leaves — while the consensus hash stays green for every other worker.
Unauthorised ids are still rejected at handshake (fail-closed default).

Loopback only, free ports (bind port 0); nothing asserts on wall-clock ratios.
"""

from __future__ import annotations

import json
import os
import socket
import threading
import time

import pytest

torch = pytest.importorskip("torch")

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from sakura.services.federation.codec import anchor_hash16  # noqa: E402
from sakura.services.federation.leader import Leader  # noqa: E402
from sakura.services.federation.transport import (  # noqa: E402
    FT_HELLO,
    IO_TIMEOUT,
    WireDead,
    connect,
    recv_frame,
    send_frame,
)

from tests.services.test_federation import TinyNet, _worker_thread  # noqa: E402

ROSTER = ("a", "b", "c")
TOKEN = "join-s3cret"
MAX_ROUNDS = 14  # rounds 3..14 (>= 10) span the join AND the leave
JOIN_AFTER = 2


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _leader(**kw):
    return Leader(model_fn=TinyNet, port=0, expected=list(ROSTER), **kw)


# --------------------------------------------------------------- policy units


class TestJoinPolicy:
    def test_default_is_fail_closed(self):
        ld = _leader()
        assert ld._roster_entry("a", None).worker_id == "a"
        with pytest.raises(AssertionError, match="joining not authorised"):
            ld._roster_entry("z", None)
        with pytest.raises(AssertionError):
            ld._roster_entry("z", "any-token")  # no token configured
        assert "z" not in ld.workers

    def test_allow_list(self):
        ld = _leader(allow_join={"z"})
        assert ld._roster_entry("z", None).worker_id == "z"
        assert "z" in ld.workers
        with pytest.raises(AssertionError):
            ld._roster_entry("y", None)

    def test_predicate(self):
        def pol(w):
            if w == "boom":
                raise RuntimeError("policy bug")
            return w.startswith("sim-")

        ld = _leader(allow_join=pol)
        ld._roster_entry("sim-7", None)
        for bad in ("y", "boom"):  # a raising policy denies
            with pytest.raises(AssertionError):
                ld._roster_entry(bad, None)

    def test_join_token(self):
        ld = _leader(join_token=TOKEN)
        for bad in (None, "", "wrong", 42, TOKEN + "x"):
            with pytest.raises(AssertionError):
                ld._roster_entry("z", bad)
        assert ld._roster_entry("z", TOKEN).worker_id == "z"

    def test_admit_and_retire(self):
        ld = _leader()
        with pytest.raises(AssertionError):
            ld._roster_entry("z", None)
        ld.admit("z")
        wc = ld._roster_entry("z", None)
        assert ld._roster_entry("z", None) is wc  # idempotent
        assert ld.retire("z") is True
        assert "z" not in ld.workers
        with pytest.raises(AssertionError):  # authorisation revoked
            ld._roster_entry("z", None)
        assert ld.retire("z") is False
        # retiring a roster member also stops serve_forever waiting for it
        assert ld.retire("c") is True
        assert ld.expected == ["a", "b"]

    def test_bad_config(self):
        with pytest.raises(TypeError):
            _leader(allow_join="z")
        with pytest.raises(ValueError):
            _leader(join_token="")


# ------------------------------------------------------------------ loopback


class _SpyLeader(Leader):
    """Records which worker took which resync path (last_committed=-1 is the
    FULL_SYNC bootstrap) and at which leader round."""

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.resyncs = []

    def _resync_if_needed(self, wc, their_hash):
        self.resyncs.append((wc.worker_id, wc.last_committed, self.round))
        return super()._resync_if_needed(wc, their_hash)


def _raw_hello_rejected(port, wid, token):
    """Unauthorised HELLO: the leader must close the connection, not ack."""
    s = connect("127.0.0.1", port)
    try:
        hello = {
            "worker_id": wid,
            "region": "default",
            "role": "ctl",
            "torch": torch.__version__.split("+")[0],
            "last_committed": -1,
            "anchor_hash16": "00" * 16,
        }
        if token is not None:
            hello["join_token"] = token
        send_frame(s, FT_HELLO, 0, json.dumps(hello).encode())
        try:
            recv_frame(s, time.monotonic() + IO_TIMEOUT)
        except (WireDead, OSError):
            return True
        return False
    finally:
        s.close()


def _wait(cond, timeout, what):
    t0 = time.monotonic()
    while not cond():
        assert time.monotonic() - t0 < timeout, f"timed out waiting for {what}"
        time.sleep(0.1)


def test_worker_joins_running_leader_and_leaves_with_consensus_green():
    port = _free_port()
    leader = _SpyLeader(
        model_fn=TinyNet,
        port=port,
        expected=list(ROSTER),
        codec="int8",
        gap_s=1.0,
        seed=42,
        max_rounds=MAX_ROUNDS,
        join_token=TOKEN,
    )
    lt = threading.Thread(
        target=leader.serve_forever, kwargs={"warmup_s": 0.5}, daemon=True
    )
    stop, stop_d = threading.Event(), threading.Event()
    out: dict = {}
    ths = []
    try:
        lt.start()
        for wid in ROSTER:
            t = threading.Thread(
                target=_worker_thread, args=(wid, port, stop, out), daemon=True
            )
            t.start()
            ths.append(t)
        _wait(lambda: leader.round >= JOIN_AFTER, 90, "initial rounds")

        # unauthorised joiners are still rejected at handshake
        assert _raw_hello_rejected(port, "x", None)
        assert _raw_hello_rejected(port, "x", "wrong-token")
        assert "x" not in leader.workers

        # 1. join: d is not in `expected`; it presents the join token
        td = threading.Thread(
            target=_worker_thread,
            args=("d", port, stop_d, out),
            kwargs={"fed_kw": {"join_token": TOKEN}},
            daemon=True,
        )
        td.start()

        def d_rounds():
            return [r["round"] for r in list(leader.stats) if "d" in r["participants"]]

        _wait(lambda: len(d_rounds()) >= 3, 90, "d to take part in 3 rounds")
        boot = [r for r in leader.resyncs if r[0] == "d"]
        assert boot and boot[0][1] == -1, leader.resyncs  # FULL_SYNC bootstrap
        assert d_rounds()[0] == boot[0][2] + 1, (boot, d_rounds())  # next round

        # 2. leave: d shuts down; rounds keep committing without it
        stop_d.set()
        td.join(30)
        assert not td.is_alive()
        left_at = leader.round
        assert left_at <= MAX_ROUNDS - 3, "no rounds left to observe the leave"
        lt.join(120)
        assert not lt.is_alive(), "leader stalled after the leave"
    finally:
        leader.stop.set()
        stop.set()
        stop_d.set()
        for t in ths:
            t.join(30)
        try:
            leader._srv.close()
        except Exception:
            pass

    rounds = {r["round"]: r for r in leader.stats}
    assert sorted(rounds) == list(range(1, MAX_ROUNDS + 1))
    # >= 10 rounds around the join and the leave, hash green on every one,
    # and the original roster never dropped out
    window = [rounds[n] for n in range(JOIN_AFTER + 1, MAX_ROUNDS + 1)]
    assert len(window) >= 10
    assert all(r["consensus_ok"] for r in rounds.values()), leader.stats
    assert all(set(ROSTER) <= set(r["participants"]) for r in rounds.values())
    assert all("d" not in rounds[n]["participants"] for n in range(left_at + 2, MAX_ROUNDS + 1))

    errors = {w: out[w]["error"] for w in out if "error" in out[w]}
    assert not errors, errors
    final = rounds[MAX_ROUNDS]["hash16"]
    for w in ROSTER:
        assert anchor_hash16(out[w]["anchor"], out[w]["bn"]).hex() == final
        assert out[w]["fed"].committed_round == MAX_ROUNDS
    # the leaver sits bit-exact on the anchor of the last round it committed
    d = out["d"]
    assert anchor_hash16(d["anchor"], d["bn"]).hex() == rounds[d["fed"].committed_round]["hash16"]
