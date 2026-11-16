"""Link re-probe, plan persistence and the SAK-0010 window hook (SAK-0013).

- a worker that joins a running fleet on a slow link changes the fleet plan
  at a round boundary, with the consensus hash green before and after;
- a slow worker that retires frees the plan again;
- a restarted leader resumes its persisted plan instead of int8 / K=1, and a
  pre-plan persist file still loads;
- `LinkPlan.link_estimate()` carries the measured BDP into
  `sakura.wire.FlowControl`'s QUIC windows.

Loopback only, free ports (bind port 0); nothing asserts on wall-clock ratios.
"""

from __future__ import annotations

import os
import threading
import time

import pytest

torch = pytest.importorskip("torch")

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from sakura.services.federation import Leader  # noqa: E402
from sakura.services.federation.codec import anchor_hash16  # noqa: E402
from sakura.services.federation.probe import (  # noqa: E402
    LinkEstimate,
    LinkPlan,
    LinkSample,
    plan_links,
)

from tests.services.test_link_probe import (  # noqa: E402
    BN_NUMEL,
    BUDGET_S,
    FAST,
    GAP_S,
    NUMEL,
    SLOW,
    WideNet,
    _free_port,
    _worker,
)

TOKEN = "reprobe-s3cret"
ROSTER = ("a", "b", "c")


def _fast(wid: str) -> LinkSample:
    return LinkSample(wid, FAST.bytes_per_s, FAST.rtt_s)


def _slow(wid: str) -> LinkSample:
    return LinkSample(wid, SLOW.bytes_per_s, SLOW.rtt_s)


def _wait(cond, timeout, what):
    t0 = time.monotonic()
    while not cond():
        assert time.monotonic() - t0 < timeout, f"timed out waiting for {what}"
        time.sleep(0.1)


# ------------------------------------------------------------------- re-plan


def test_joiner_on_a_slow_link_changes_the_plan_at_a_round_boundary():
    port = _free_port()
    holder: dict = {}

    def probe():  # every roster member is fast, except a slow "d" once it joined
        ld = holder["leader"]
        return {w: (_slow(w) if w == "d" else _fast(w)) for w in list(ld.workers)}

    leader = Leader(model_fn=WideNet, port=port, expected=list(ROSTER),
                    codec="auto", shards=0, gap_s=GAP_S, seed=42,
                    max_rounds=10**6, link_probe=probe, join_token=TOKEN)
    holder["leader"] = leader
    lt = threading.Thread(target=leader.serve_forever, kwargs={"warmup_s": 0.5},
                          daemon=True)
    stop, stop_d = threading.Event(), threading.Event()
    out: dict = {}
    ths = [threading.Thread(target=_worker, args=(w, port, stop, out), daemon=True)
           for w in ROSTER]
    try:
        lt.start()
        for t in ths:
            t.start()
        _wait(lambda: leader.round >= 2, 90, "initial rounds")
        assert (leader.codec, leader.shards) == ("int8", 1)
        assert [p["reason"] for p in leader.plan_history] == ["bootstrap"]

        td = threading.Thread(target=_worker, args=("d", port, stop_d, out),
                              kwargs={"fed_kw": {"join_token": TOKEN}}, daemon=True)
        td.start()

        def d_rounds():
            return [r for r in list(leader.stats) if "d" in r["participants"]]

        _wait(lambda: len(d_rounds()) >= 4, 90, "d to take part in 4 rounds")
        stop_d.set()
        td.join(30)
    finally:
        leader.stop.set()
        stop.set()
        stop_d.set()
        for t in ths:
            t.join(15)
        try:
            leader._srv.close()
        except Exception:  # noqa: BLE001
            pass

    # the join re-planned: the slow joiner is the bottleneck
    assert (leader.codec, leader.shards) == ("topk", 4)
    assert leader.upload_codecs == {"a": "int8", "b": "int8", "c": "int8", "d": "topk"}
    joined = [p for p in leader.plan_history if p["reason"] == "join:d"]
    assert len(joined) == 1, leader.plan_history
    change = joined[0]
    assert change["bottleneck"] == "d" and change["codec"] == "topk"
    # ... and took effect at a round boundary: every round before
    # `from_round` ran the old plan, every round from it the new one
    rounds = {r["round"]: r for r in leader.stats}
    first = change["from_round"]
    assert all((rounds[n]["codec"], rounds[n]["shards"]) == ("int8", 1)
               for n in rounds if n < first)
    assert all((rounds[n]["codec"], rounds[n]["shards"]) == ("topk", 4)
               for n in rounds if n >= first)
    assert min(r["round"] for r in d_rounds()) >= first  # d never ran the old plan
    assert all(r["consensus_ok"] for r in leader.stats), leader.stats
    errors = {w: out[w]["error"] for w in out if "error" in out.get(w, {})}
    assert not errors, errors
    lh = anchor_hash16(leader.anchor, leader.bn).hex()
    hashes = [anchor_hash16(out[w]["anchor"], out[w]["bn"]).hex() for w in ROSTER]
    assert sum(h == lh for h in hashes) >= 2
    assert out["d"]["codec"] == "topk" and out["d"]["bcast_codec"] == "topk"


def test_retiring_the_slow_worker_frees_the_plan():
    links = {"a": _fast("a"), "slow": _slow("slow")}
    leader = Leader(model_fn=WideNet, port=_free_port(), expected=["a", "slow"],
                    codec="auto", shards=0, gap_s=GAP_S,
                    link_probe=lambda: dict(links))
    leader._plan_links()
    leader._bootstrapped = True
    assert (leader.codec, leader.shards) == ("topk", 4)
    assert leader.retire("slow")
    assert (leader.codec, leader.shards) == ("int8", 1)
    assert leader.upload_codecs == {"a": "int8"}
    assert [p["reason"] for p in leader.plan_history] == ["bootstrap", "retire:slow"]


def test_unchanged_decision_is_not_a_plan_change():
    leader = Leader(model_fn=WideNet, port=_free_port(), expected=["a"],
                    codec="auto", shards=0, gap_s=GAP_S,
                    link_probe=lambda: {"a": _fast("a")})
    leader._plan_links()
    leader._bootstrapped = True
    leader._plan_links(reason="join:a")
    assert len(leader.plan_history) == 1


def test_k_change_restarts_delayed_nesterov_windows():
    links = {"a": _fast("a")}
    leader = Leader(model_fn=WideNet, port=_free_port(), expected=["a", "s"],
                    codec="auto", shards=0, gap_s=GAP_S,
                    outer_opt="delayed_nesterov", outer_delay=4,
                    link_probe=lambda: dict(links))
    leader._plan_links()
    leader.outer_steps = {0: 3}
    assert leader.outer_buf is not None
    leader.outer_buf.fill_(1.0)
    links["s"] = _slow("s")
    leader._plan_links(reason="join:s")
    assert leader.shards == 4 and leader.outer_steps == {}
    assert float(leader.outer_buf.abs().sum()) == 0.0
    assert leader.ef_bcast is not None  # topk is lossy: residual kept / allocated


# ------------------------------------------------------------------- persist


def _persisted_leader(path, **kw):
    """A leader that committed state with a slow-link plan: the plan is
    written the way run_round persists it."""
    ld = Leader(model_fn=WideNet, port=_free_port(), expected=["a", "c"],
                codec="auto", shards=0, gap_s=GAP_S, persist_path=path,
                link_probe=lambda: {"a": _fast("a"), "c": _slow("c")}, **kw)
    ld._plan_links()
    assert (ld.codec, ld.shards) == ("topk", 4)
    torch.save({"round": 7, "anchor": ld.anchor, "bn": ld.bn,
                "link_plan": ld.link_plan.to_state()}, path)
    return ld


def test_restarted_leader_resumes_its_plan(tmp_path):
    path = str(tmp_path / "leader.pt")
    first = _persisted_leader(path)
    # restart with every worker in sync: no FULL_SYNC, no injected probe
    again = Leader(model_fn=WideNet, port=_free_port(), expected=["a", "c"],
                   codec="auto", shards=0, gap_s=GAP_S, persist_path=path)
    assert again.round == 7
    assert (again.codec, again.shards) == ("topk", 4)
    assert again.upload_codecs == first.upload_codecs == {"a": "int8", "c": "topk"}
    assert again.ef_bcast is not None
    assert again.link_plan is not None and again.link_plan.same_decision(first.link_plan)
    again._plan_links()  # bootstrap with no fresh samples: the plan holds
    assert (again.codec, again.shards) == ("topk", 4)
    assert again.link_plan.link_estimate() == first.link_plan.link_estimate()


def test_persisted_plan_and_optimizer_state_restore_together(tmp_path):
    """#143 x SAK-0013: one atomic file carries both the plan and the outer
    optimizer state. The plan is applied first (it moves K from 1 to 4, which
    resets the per-shard windows), then the buffers and the K=4 step counts
    saved under that plan are restored bit-exact on top."""
    path = str(tmp_path / "leader.pt")
    kw = {"outer_opt": "delayed_nesterov", "outer_delay": 2}
    first = _persisted_leader(path, **kw)
    g = torch.Generator().manual_seed(3)
    for buf in (first.ef_bcast, first.outer_m, first.outer_buf):
        buf.copy_(torch.randn(first.fs.numel, generator=g))
    first.outer_steps = {0: 2, 1: 1}
    first._persist(7, path)
    again = Leader(model_fn=WideNet, port=_free_port(), expected=["a", "c"],
                   codec="auto", shards=0, gap_s=GAP_S, persist_path=path, **kw)
    assert again.round == 7 and (again.codec, again.shards) == ("topk", 4)
    assert again.link_plan is not None and again.link_plan.same_decision(first.link_plan)
    for k in ("ef_bcast", "outer_m", "outer_buf"):
        assert torch.equal(getattr(again, k), getattr(first, k)), k
    assert again.outer_steps == {0: 2, 1: 1}


def test_restart_with_operator_settings_ignores_the_persisted_plan(tmp_path):
    path = str(tmp_path / "leader.pt")
    _persisted_leader(path)
    again = Leader(model_fn=WideNet, port=_free_port(), expected=["a", "c"],
                   codec="int8", shards=1, gap_s=GAP_S, persist_path=path)
    assert (again.codec, again.shards) == ("int8", 1)
    assert again.upload_codecs == {}


def test_pre_plan_persist_file_still_loads(tmp_path):
    path = str(tmp_path / "leader.pt")
    ld = Leader(model_fn=WideNet, port=_free_port(), expected=["a"])
    torch.save({"round": 3, "anchor": ld.anchor, "bn": ld.bn}, path)
    again = Leader(model_fn=WideNet, port=_free_port(), expected=["a"],
                   codec="auto", shards=0, persist_path=path)
    assert again.round == 3 and again.link_plan is None
    assert (again.codec, again.shards) == ("int8", 1)


def test_plan_state_round_trips():
    plan = plan_links({"fast": FAST, "slow": SLOW}, NUMEL, BN_NUMEL, BUDGET_S)
    assert plan is not None
    back = LinkPlan.from_state(plan.to_state())
    assert back == plan


# ------------------------------------------------------- SAK-0010 window hook

WAN = LinkSample("wan", 12_500_000.0, 0.248)  # 248 ms x 12.5 MB/s = 3.1 MB BDP


def test_link_estimate_is_the_largest_bdp_link():
    plan = plan_links({"fast": FAST, "slow": SLOW, "wan": WAN}, NUMEL, BN_NUMEL, 100.0)
    assert plan is not None
    est = plan.link_estimate()
    assert est == LinkEstimate(0.248, 12_500_000.0)
    assert est.bdp_bytes == pytest.approx(3.1e6)
    assert est.rtt_ms == pytest.approx(248.0)
    assert plan.link_estimate("slow") == LinkEstimate(SLOW.rtt_s, SLOW.bytes_per_s)
    assert plan.link_estimate("nobody") is None
    no_rtt = plan_links({"w": LinkSample("w", 1e6, 0.0)}, NUMEL, BN_NUMEL, 100.0)
    assert no_rtt is not None and no_rtt.link_estimate() is None


@pytest.mark.wire
def test_leader_plan_bdp_reaches_the_quic_window_config():
    pytest.importorskip("sakura_wire")
    leader = Leader(model_fn=WideNet, port=_free_port(), expected=["a", "wan"],
                    codec="auto", shards=0, gap_s=GAP_S,
                    link_probe=lambda: {"a": _fast("a"), "wan": WAN})
    leader._plan_links()
    assert leader.link_plan is not None
    est = leader.link_plan.link_estimate()
    assert est is not None
    w = est.flow_control().windows()
    # SAK-0010 sizing: stream window = clamp(2 x BDP); connection = 8x
    assert w == {
        "stream_receive_window": 6_200_000,
        "receive_window": 49_600_000,
        "send_window": 49_600_000,
    }
    capped = est.flow_control(max_window=4_000_000).windows()
    assert capped["stream_receive_window"] == 4_000_000
