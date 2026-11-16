"""Bootstrap link probe (SAK-0013): codec / shard-count auto-selection.

Unit tests pin the decision table; the loopback tests run a real leader and
three in-process workers, feed the leader a fake slow link through the
injectable `link_probe`, and check that the chosen codec / K are used on the
wire with the consensus hash green (including mixed upload codecs).
"""

from __future__ import annotations

import json
import logging
import os
import socket
import threading
import time

import pytest

torch = pytest.importorskip("torch")

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from sakura.services.federation import FederationService, Leader  # noqa: E402
from sakura.services.federation.codec import anchor_hash16, codec_encode  # noqa: E402
from sakura.services.federation.probe import (  # noqa: E402
    LinkSample,
    codec_payload_bytes,
    plan_links,
    round_budget_s,
    round_bytes,
    sample_from_fullsync,
    tcp_rtt_s,
)

# WideNet: 76 808 params (19 int8 buckets) + 2 048 BN floats — big enough
# that sharding actually shrinks the per-round payload.
NUMEL, BN_NUMEL = 76_808, 2_048
GAP_S = 1.2
BUDGET_S = round_budget_s(GAP_S, 0.0)  # 0.6 s (default duty 0.5)
FAST = LinkSample("fast", 1e9, 0.0005)  # ~8 Gbit/s loopback-class
# Slow link: 60 kB/s, 50 ms RTT -> 0.5 s of serialization in a 0.6 s
# budget, i.e. at most 15 KB per leg (int8 K=1 needs 86 KB).
SLOW = LinkSample("slow", 60_000.0, 0.05)


class WideNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(64, 1024),
            torch.nn.BatchNorm1d(1024),
            torch.nn.ReLU(),
            torch.nn.Linear(1024, 8),
        )

    def forward(self, x):
        return self.net(x)


# ------------------------------------------------------------- decision table


@pytest.mark.parametrize("kind", ["int8", "int4", "int2", "topk", "topk:0.03",
                                  "topk2", "fp32raw"])
@pytest.mark.parametrize("n", [1, 4095, 4096, 4097, 3 * 4096 + 17])
def test_payload_bytes_match_encoder(kind, n):
    x = torch.randn(n)
    payload, _, _ = codec_encode(x, None, kind)
    assert codec_payload_bytes(kind, n) == len(payload)


def test_fast_link_keeps_int8_unsharded():
    plan = plan_links({"fast": FAST}, NUMEL, BN_NUMEL, BUDGET_S)
    assert plan is not None and plan.fits
    assert (plan.codec, plan.shards) == ("int8", 1)
    assert plan.upload_codecs == {"fast": "int8"}


def test_slow_link_escalates_codec_then_shards():
    plan = plan_links({"fast": FAST, "slow": SLOW}, NUMEL, BN_NUMEL, BUDGET_S)
    assert plan is not None and plan.fits
    # bytes per leg (payload + 8 KB BN): int8 K1 86.1 KB, int4 K1 47.2 KB,
    # topk K1 31.6 KB, topk K2 20.5 KB, int8 K16 16.4 KB, topk K4 14.4 KB
    # -> 2*0.05 + 2*14.4 KB / 60 kB/s = 0.58 s <= 0.6 s
    assert (plan.codec, plan.shards) == ("topk", 4)
    assert plan.bottleneck == "slow"
    assert plan.est_round_s <= BUDGET_S
    # the fast worker's own link has room for int8 uploads; the slow one
    # uploads with the broadcast codec — mixed codecs, one broadcast.
    assert plan.upload_codecs == {"fast": "int8", "slow": "topk"}


def test_rungs_are_most_bytes_first_and_first_fit_wins():
    # just enough for int4 K=1 but not int8 K=1
    b_int4 = round_bytes("int4", NUMEL, 1, BN_NUMEL)
    b_int8 = round_bytes("int8", NUMEL, 1, BN_NUMEL)
    bw = 2 * b_int4 / BUDGET_S * 1.01
    assert 2 * b_int8 / bw > BUDGET_S
    plan = plan_links({"w": LinkSample("w", bw, 0.0)}, NUMEL, BN_NUMEL, BUDGET_S)
    assert plan is not None and (plan.codec, plan.shards) == ("int4", 1)


def test_operator_codec_wins_only_k_moves():
    plan = plan_links({"slow": SLOW}, NUMEL, BN_NUMEL, BUDGET_S, codec="int8")
    assert plan is not None and plan.codec == "int8" and plan.shards > 1
    assert plan.upload_codecs == {"slow": "int8"}


def test_operator_shards_wins_only_codec_moves():
    plan = plan_links({"slow": SLOW}, NUMEL, BN_NUMEL, BUDGET_S, shards=1)
    assert plan is not None and plan.shards == 1 and plan.codec == "topk"


def test_both_explicit_is_a_noop_decision():
    plan = plan_links({"slow": SLOW}, NUMEL, BN_NUMEL, BUDGET_S,
                      codec="int8", shards=1)
    assert plan is not None and (plan.codec, plan.shards) == ("int8", 1)
    assert not plan.fits  # still logged as an overrun


def test_nothing_fits_takes_most_aggressive_rung():
    crawl = LinkSample("crawl", 1_000.0, 0.4)  # RTT alone exceeds the budget
    plan = plan_links({"crawl": crawl}, NUMEL, BN_NUMEL, BUDGET_S)
    assert plan is not None and not plan.fits
    assert (plan.codec, plan.shards) == ("topk", 16)


def test_no_samples_no_plan():
    assert plan_links({}, NUMEL, BN_NUMEL, BUDGET_S) is None


def test_budget_uses_link_duty():
    assert round_budget_s(10.0, 0.0) == pytest.approx(5.0)
    assert round_budget_s(10.0, 0.2) == pytest.approx(2.0)


def test_fullsync_sample_subtracts_one_rtt():
    s = sample_from_fullsync("w", 1_000_000, 1.1, 0.1)
    assert s.bytes_per_s == pytest.approx(1e6)
    assert s.bdp_bytes == pytest.approx(1e5)
    assert sample_from_fullsync("w", 1_000, 0.5, None).rtt_s == 0.0


def test_tcp_rtt_on_a_connected_socket():
    srv = socket.socket()
    srv.bind(("127.0.0.1", 0))
    srv.listen(1)
    cli = socket.create_connection(srv.getsockname())
    peer, _ = srv.accept()
    try:
        cli.sendall(b"x")
        peer.recv(1)
        rtt = tcp_rtt_s(cli)
        if hasattr(socket, "TCP_INFO"):  # Linux: kernel srtt, sub-second on loopback
            assert rtt is not None and 0.0 <= rtt < 1.0
        else:
            assert rtt is None
    finally:
        for s in (cli, peer, srv):
            s.close()
    assert tcp_rtt_s(None) is None


# ------------------------------------------------------------------ loopback


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _worker(wid, port, stop_evt, out, fed_kw=None):
    from sakura.adapters.ddp import DDPAdapter
    from sakura.runtime import SakuraRuntime

    torch.manual_seed(hash(wid) % 2**31)
    model = WideNet()
    opt = torch.optim.SGD(model.parameters(), lr=0.05, momentum=0.9)
    loss_fn = torch.nn.CrossEntropyLoss()
    g = torch.Generator().manual_seed(hash(wid) % 1000)
    x = torch.randn(16, 64, generator=g)
    y = torch.randint(0, 8, (16,), generator=g)
    rt = SakuraRuntime()
    # constructor codec deliberately int8: the leader's announcement wins
    fed = FederationService("127.0.0.1", port, wid, 16, codec="int8",
                            **(fed_kw or {}))
    rt.install(fed)
    adapter = DDPAdapter(rt, rank=0, world_size=3)
    try:
        with rt:
            adapter.on_train_begin(model=model, optimizer=opt, train_loader=None)
            step = 0
            while not stop_evt.is_set():
                adapter.on_train_step_begin(model=model, batch=(x, y), step=step)
                opt.zero_grad()
                loss = loss_fn(model(x), y)
                loss.backward()
                adapter.on_optimizer_step(optimizer=opt)
                opt.step()
                step += 1
                time.sleep(0.002)
            out[wid] = {
                "anchor": fed.anchor.clone(),
                "bn": fed._bn_anchor.clone(),
                "codec": fed.codec,
                "bcast_codec": fed.bcast_codec,
                "shards": sorted({tuple(r["shard"]) for r in fed.records
                                  if "shard" in r}),
            }
    except Exception as e:  # noqa: BLE001
        out[wid] = {"error": repr(e)}


def _run(n_rounds, **leader_kw):
    port = _free_port()
    leader = Leader(model_fn=WideNet, port=port, expected=["a", "b", "c"],
                    gap_s=GAP_S, seed=42, max_rounds=n_rounds, **leader_kw)
    threading.Thread(target=leader.serve_forever, kwargs={"warmup_s": 0.5},
                     daemon=True).start()
    stop = threading.Event()
    out: dict = {}
    ths = [threading.Thread(target=_worker, args=(w, port, stop, out), daemon=True)
           for w in ("a", "b", "c")]
    for t in ths:
        t.start()
    t0 = time.monotonic()
    while leader.round < n_rounds and time.monotonic() - t0 < 90:
        time.sleep(0.25)
    # let the last round's acks land on the workers before stopping them
    time.sleep(0.5)
    leader.stop.set()
    stop.set()
    for t in ths:
        t.join(15)
    try:
        leader._srv.close()
    except Exception:  # noqa: BLE001
        pass
    return leader, out


def _assert_consensus(leader, out, n_rounds):
    assert leader.round >= n_rounds
    assert leader.stats and all(r["consensus_ok"] for r in leader.stats)
    assert all("error" not in out.get(w, {"error": "missing"}) for w in "abc"), out
    lh = anchor_hash16(leader.anchor, leader.bn).hex()
    hashes = [anchor_hash16(out[w]["anchor"], out[w]["bn"]).hex() for w in "abc"]
    assert sum(h == lh for h in hashes) >= 2


def test_loopback_fake_slow_link_picks_codec_k_and_mixed_uploads(caplog):
    def probe():
        return {
            "a": LinkSample("a", FAST.bytes_per_s, FAST.rtt_s),
            "b": LinkSample("b", FAST.bytes_per_s, FAST.rtt_s),
            "c": LinkSample("c", SLOW.bytes_per_s, SLOW.rtt_s),
        }

    caplog.set_level(logging.INFO, logger="sakura.federation")
    leader, out = _run(5, codec="auto", shards=0, link_probe=probe)
    assert (leader.codec, leader.shards) == ("topk", 4)
    assert leader.upload_codecs == {"a": "int8", "b": "int8", "c": "topk"}
    logged = [r.getMessage() for r in caplog.records
              if r.getMessage().startswith("link probe: {")]
    assert logged, "probe output must be logged"
    rec = json.loads(logged[0][len("link probe: "):])
    assert rec["codec"] == "topk" and rec["shards"] == 4
    assert rec["bottleneck"] == "c" and rec["links"]["c"]["rtt_ms"] == 50.0
    _assert_consensus(leader, out, 5)
    # the workers really used what the leader chose
    assert {out[w]["codec"] for w in "ab"} == {"int8"}
    assert out["c"]["codec"] == "topk"
    assert {out[w]["bcast_codec"] for w in "abc"} == {"topk"}
    assert len(out["a"]["shards"]) == 4  # rounds 1-4 hit all four shards
    # mixed upload sizes on the wire: c's topk upload is smaller than a's int8
    up = leader.stats[0]["bytes_up"]
    assert up["c"] < up["a"]


def test_loopback_passive_probe_measures_every_worker():
    leader, out = _run(2, codec="auto", shards=0)
    assert leader.link_plan is not None
    assert set(s.worker_id for s in leader.link_plan.samples) == {"a", "b", "c"}
    assert all(s.bytes_per_s > 0 for s in leader.link_plan.samples)
    # loopback is never the bottleneck for an 80k-param model
    assert (leader.codec, leader.shards) == ("int8", 1)
    _assert_consensus(leader, out, 2)


def test_explicit_operator_settings_win_over_probe():
    leader = Leader(model_fn=WideNet, port=_free_port(), expected=["c"],
                    codec="int8", shards=1, gap_s=GAP_S,
                    link_probe=lambda: {"c": SLOW})
    leader._plan_links()
    assert (leader.codec, leader.shards) == ("int8", 1)
    assert leader.link_plan is not None and not leader.link_plan.fits


def test_auto_without_samples_falls_back_to_int8_k1():
    leader = Leader(model_fn=WideNet, port=_free_port(), expected=["c"],
                    codec="auto", shards=0, link_probe=dict)
    leader._plan_links()
    assert (leader.codec, leader.shards) == ("int8", 1)
    assert leader.link_plan is None
    with pytest.raises(ValueError):
        Leader(model_fn=WideNet, port=_free_port(), expected=["c"], shards=-1)
