"""Leader kill-and-restart e2e (SAK-0015).

The leader runs in a child process and is SIGKILLed at a precise point inside
a round (crash injection, not a timer), then restarted from its atomic
`persist_path` file. Three in-process workers must reconnect, resync, and keep
committing rounds with the consensus hash green on every round — a round that
was applied by some workers before the crash must never be applied twice.

The leader's optimizer state (broadcast error-feedback residual, outer
momentum, delayed-Nesterov buffer and per-shard step counts, #143) rides in
the same persisted file: the restarted leader must load it bit-exact, as of
the last committed round, and continue the per-shard schedule from it.

Loopback only; the crash point is deterministic, so nothing here asserts on
wall-clock ratios (not a `timing` test).
"""

from __future__ import annotations

import hashlib
import json
import os
import signal
import socket
import subprocess
import sys
import threading
import time

import pytest

torch = pytest.importorskip("torch")

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from sakura.services.federation.codec import anchor_hash16  # noqa: E402

from tests.services.test_federation import _worker_thread  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
WORKERS = ("a", "b", "c")
# opt profile -> (Leader kwargs, crash round). "dn" keeps every optimizer
# buffer non-trivial by the crash: with K=2 shards and delay 2, rounds 1 and 3
# hit shard 0 (momentum refreshed at 3), round 2 leaves shard 1's DN buffer
# full, and int8 keeps a broadcast EF residual.
PROFILES = {
    "plain": ({}, 2),
    "dn": ({"outer_opt": "delayed_nesterov", "outer_delay": 2, "shards": 2}, 4),
}
_OPT_TENSORS = ("ef_bcast", "outer_m", "outer_buf")


def _opt_digest(st) -> dict:
    """sha256 of every optimizer buffer present in a persisted dict (or a
    Leader's ``_opt_state()``), plus the per-shard step counts."""
    out = {
        k: hashlib.sha256(st[k].contiguous().numpy().tobytes()).hexdigest()
        for k in _OPT_TENSORS
        if st.get(k) is not None
    }
    out["outer_steps"] = {str(lo): n for lo, n in sorted(st.get("outer_steps", {}).items())}
    return out


# Child leader process. argv: port persist_path crash crash_round max_rounds opt
#   crash = "none"   — run to max_rounds and exit 0
#           "gather" — SIGKILL after ROUND_BEGIN went out, while gathering
#           "bcast"  — SIGKILL after the FIRST worker received the whole
#                      broadcast (it applies the round; the others do not)
#           "persist"— SIGKILL after every ack, before the atomic persist
_LEADER_CHILD = r"""
import json, os, signal, sys
import torch
from sakura.services.federation import leader as L
from tests.services.test_federation import TinyNet
from tests.services.test_leader_restart import PROFILES, _opt_digest

port, persist, crash, crash_round, max_rounds, opt = (
    int(sys.argv[1]), sys.argv[2], sys.argv[3], int(sys.argv[4]), int(sys.argv[5]),
    sys.argv[6],
)

# the optimizer state as of each commit, printed just before it is persisted
_persist = L.Leader._persist
def _persist_spy(self, rnd, path):
    print("LEADER_OPT " + json.dumps({"round": rnd, **_opt_digest(self._opt_state())}),
          flush=True)
    return _persist(self, rnd, path)
L.Leader._persist = _persist_spy

def die():
    sys.stdout.flush()
    os.kill(os.getpid(), signal.SIGKILL)

if crash == "gather":
    _gather = L.Leader._gather_one
    def _gather_one(self, wc, gen, rnd, out, deadline):
        if rnd == crash_round:
            die()
        return _gather(self, wc, gen, rnd, out, deadline)
    L.Leader._gather_one = _gather_one
elif crash == "bcast":
    _send = L._WorkerConn.send
    sent = [0]
    def send(self, ftype, rnd, payload=b""):
        _send(self, ftype, rnd, payload)
        if ftype == L.FT_BCAST_META and rnd == crash_round:
            sent[0] += 1
            if sent[0] == 2:  # meta + BN sideband of the first worker
                die()
    L._WorkerConn.send = send
elif crash == "persist":
    _save = torch.save
    def save(obj, f, *a, **k):
        if isinstance(obj, dict) and obj.get("round") == crash_round:
            die()
        return _save(obj, f, *a, **k)
    torch.save = save

leader = L.Leader(
    model_fn=TinyNet, port=port, expected=["a", "b", "c"], codec="int8",
    gap_s=1.0, seed=42, max_rounds=max_rounds, persist_path=persist,
    **PROFILES[opt][0],
)
print("LEADER_READY " + json.dumps({"round": leader.round}), flush=True)
print("LEADER_RESTORED " + json.dumps(_opt_digest(leader._opt_state())), flush=True)
leader.serve_forever(warmup_s=0.5)
leader.stop.set()
print("LEADER_DONE " + json.dumps({"round": leader.round}), flush=True)
"""


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class _LeaderProc:
    """Child leader; stdout lines parsed into READY / ROUND / DONE records."""

    def __init__(self, port, persist, crash, max_rounds, log_path, opt, crash_round):
        env = dict(os.environ)
        env["PYTHONPATH"] = ROOT + os.pathsep + env.get("PYTHONPATH", "")
        env["CUDA_VISIBLE_DEVICES"] = ""
        self._log = open(log_path, "ab")
        self.proc = subprocess.Popen(
            [
                sys.executable, "-c", _LEADER_CHILD, str(port), str(persist),
                crash, str(crash_round), str(max_rounds), opt,
            ],
            cwd=ROOT,
            env=env,
            stdout=subprocess.PIPE,
            stderr=self._log,
            text=True,
        )
        self.ready = None
        self.restored = None
        self.done = None
        self.rounds = []
        self.opt = {}
        self._reader = threading.Thread(target=self._read, daemon=True)
        self._reader.start()

    def _read(self):
        for line in self.proc.stdout:
            tag, _, body = line.strip().partition(" ")
            if tag == "LEADER_READY":
                self.ready = json.loads(body)
            elif tag == "LEADER_RESTORED":
                self.restored = json.loads(body)
            elif tag == "LEADER_ROUND":
                self.rounds.append(json.loads(body))
            elif tag == "LEADER_OPT":
                rec = json.loads(body)
                self.opt[rec.pop("round")] = rec
            elif tag == "LEADER_DONE":
                self.done = json.loads(body)

    def wait(self, timeout):
        try:
            rc = self.proc.wait(timeout)
        finally:
            self.kill()
        self._reader.join(5)
        return rc

    def kill(self):
        if self.proc.poll() is None:
            self.proc.kill()
            self.proc.wait(10)
        self._log.close()


@pytest.mark.parametrize(
    "crash,opt",
    [
        ("gather", "plain"),
        ("bcast", "plain"),
        ("persist", "plain"),
        # optimizer buffers already mutated in memory for the lost round
        ("bcast", "dn"),
        ("persist", "dn"),
    ],
)
def test_leader_killed_mid_round_restarts_without_double_apply(crash, opt, tmp_path):
    CRASH_ROUND = PROFILES[opt][1]
    port = _free_port()
    persist = tmp_path / "leader_state.pt"
    log_path = tmp_path / "leader.log"
    stop = threading.Event()
    out: dict = {}
    ths = []
    first = second = None
    try:
        # 1. leader #1 commits round CRASH_ROUND-1, then dies inside CRASH_ROUND
        first = _LeaderProc(port, persist, crash, 10, log_path, opt, CRASH_ROUND)
        for wid in WORKERS:
            t = threading.Thread(
                target=_worker_thread, args=(wid, port, stop, out), daemon=True,
                # Workers start before the leader child has bound its port and
                # lose it again at the crash; the default 30 s retry ceiling
                # sat out up to ~30 s per gap on slow CI pods (#161). Same
                # retry path, found within 0.5 s of the leader listening.
                kwargs={"fed_kw": {"reconnect_backoff_max_s": 0.5}},
            )
            t.start()
            ths.append(t)
        rc = first.wait(90)
        assert rc == -signal.SIGKILL, (
            f"leader #1 exited {rc} instead of dying at the crash point; log:\n"
            + log_path.read_text()
        )
        assert first.ready == {"round": 0}
        pre = {r["round"]: r for r in first.rounds}
        assert set(pre) >= set(range(1, CRASH_ROUND))
        if crash == "persist":
            # the round was fully acked (every worker applied it) but never
            # persisted: the restarted leader must roll the fleet back to it.
            assert CRASH_ROUND in pre and len(pre[CRASH_ROUND]["participants"]) == 3

        # 2. the persisted file is exactly the last committed round
        st = torch.load(persist, map_location="cpu")
        assert st["round"] == CRASH_ROUND - 1
        assert (
            anchor_hash16(st["anchor"], st["bn"]).hex()
            == pre[CRASH_ROUND - 1]["hash16"]
        )
        # ...and so is the optimizer state: the snapshot taken at that commit,
        # not a later one (round CRASH_ROUND already mutated it in memory for
        # the bcast / persist crashes).
        saved = _opt_digest(st)
        assert saved == first.opt[CRASH_ROUND - 1], (saved, first.opt)
        assert "ef_bcast" in saved  # int8 is lossy
        assert st["ef_bcast"].abs().sum() > 0
        if opt == "dn":
            assert st["outer_m"].abs().sum() > 0 and st["outer_buf"].abs().sum() > 0
            per = -(-st["anchor"].numel() // 2)  # shard_bounds: shard 1 starts here
            assert saved["outer_steps"] == {"0": 2, str(per): 1}
        else:
            assert saved["outer_steps"] == {"0": CRASH_ROUND - 1}

        # 3. supervised restart: leader #2 resumes from the file and runs two
        #    more rounds (the re-run of CRASH_ROUND and the one after it).
        max_rounds = CRASH_ROUND + 1
        second = _LeaderProc(port, persist, "none", max_rounds, log_path, opt, CRASH_ROUND)
        rc = second.wait(120)
        assert rc == 0, f"leader #2 exited {rc}; log:\n" + log_path.read_text()
        assert second.ready == {"round": CRASH_ROUND - 1}, "did not resume"
        # optimizer buffers restored bit-exact, and the per-shard step count
        # continues: the re-run round is the next update of its shard.
        assert second.restored == saved, (second.restored, saved)
        steps = dict(saved["outer_steps"])
        lo = next(iter(set(second.opt[CRASH_ROUND]["outer_steps"]) - set(steps)), None)
        assert lo is None, f"re-run round started a fresh shard window at {lo}"
        bumped = {
            k: n - steps[k]
            for k, n in second.opt[CRASH_ROUND]["outer_steps"].items()
            if n != steps[k]
        }
        assert list(bumped.values()) == [1], (steps, second.opt[CRASH_ROUND])
        assert second.done == {"round": max_rounds}
        post = second.rounds
        assert [r["round"] for r in post] == list(range(CRASH_ROUND, max_rounds + 1))
        # hash green on every post-restart round: a double-applied delta on
        # any worker would surface as a CONSENSUS HASH MISMATCH here.
        assert all(r["consensus_ok"] for r in post), post
        assert all(sorted(r["participants"]) == list(WORKERS) for r in post), post
    finally:
        stop.set()
        for t in ths:
            t.join(30)
        for p in (first, second):
            if p is not None:
                p.kill()

    # 4. every worker sits on the restarted leader's final anchor, bit-exact
    errors = {w: out[w]["error"] for w in WORKERS if "error" in out.get(w, {})}
    assert not errors, errors
    final = post[-1]["hash16"]
    hashes = {w: anchor_hash16(out[w]["anchor"], out[w]["bn"]).hex() for w in WORKERS}
    assert all(h == final for h in hashes.values()), (final, hashes)
    assert all(out[w]["fed"].committed_round == max_rounds for w in WORKERS)


def _leader(persist, **kw):
    from sakura.services.federation.leader import Leader
    from tests.services.test_federation import TinyNet

    return Leader(
        model_fn=TinyNet, port=0, expected=["a"], codec="int8", seed=42,
        persist_path=str(persist), **kw,
    )


def test_persist_round_trips_optimizer_state_bit_exact(tmp_path):
    """In-process: every optimizer buffer (and the per-shard step counts)
    written by ``_persist`` loads back bit-exact into a fresh leader."""
    persist = tmp_path / "leader_state.pt"
    kw = PROFILES["dn"][0]
    a = _leader(persist, **kw)
    g = torch.Generator().manual_seed(7)
    for k in _OPT_TENSORS:
        getattr(a, k).copy_(torch.randn(a.fs.numel, generator=g))
    a.outer_steps = {0: 3, -(-a.fs.numel // 2): 2}
    a.round = 5
    a._persist(5, str(persist))
    assert not os.path.exists(str(persist) + ".tmp")
    b = _leader(persist, **kw)
    assert b.round == 5
    for k in _OPT_TENSORS:
        assert torch.equal(getattr(b, k), getattr(a, k)), k
    assert b.outer_steps == a.outer_steps
    assert _opt_digest(b._opt_state()) == _opt_digest(a._opt_state())


def test_old_persist_file_without_optimizer_state_still_loads(tmp_path):
    """A file written before #143 holds only {round, anchor, bn}: it must
    resume the anchor and leave the optimizer buffers at zero, as before."""
    persist = tmp_path / "leader_state.pt"
    a = _leader(persist)
    a.anchor.add_(1.0)
    torch.save({"round": 3, "anchor": a.anchor, "bn": a.bn}, str(persist))
    b = _leader(persist, **PROFILES["dn"][0])
    assert b.round == 3
    assert torch.equal(b.anchor, a.anchor)
    for k in _OPT_TENSORS:
        assert torch.count_nonzero(getattr(b, k)) == 0, k
    assert b.outer_steps == {}


def test_persisted_buffer_not_used_by_new_config_is_ignored(tmp_path):
    """Restarting with a different outer optimizer keeps the anchor and
    simply drops buffers the new configuration does not keep."""
    persist = tmp_path / "leader_state.pt"
    a = _leader(persist, **PROFILES["dn"][0])
    a.outer_m.fill_(2.0)
    a._persist(1, str(persist))
    b = _leader(persist)  # plain: no outer_m / outer_buf
    assert b.round == 1 and b.outer_m is None and b.outer_buf is None
