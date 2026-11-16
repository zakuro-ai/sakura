"""Hierarchical per-region aggregation (SAK-0011): loopback e2e tests.

Topology: one root Leader, one RegionAggregator per region, workers connect
to their region's aggregator. CPU-only, all on 127.0.0.1.
"""

from __future__ import annotations

import logging
import os
import socket
import threading
import time
from typing import Any, Dict, List

import pytest

torch = pytest.importorskip("torch")

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from sakura.services.federation import (  # noqa: E402
    FederationService,
    Leader,
    RegionAggregator,
)
from sakura.services.federation.codec import anchor_hash16  # noqa: E402
from sakura.services.federation.region import round_wire_bytes  # noqa: E402
from sakura.services.federation.transport import shutclose  # noqa: E402

from .test_federation import TinyNet  # noqa: E402

REGIONS = {"eu": ["a", "b"], "us": ["c", "d"]}
WORKERS = [w for ws in REGIONS.values() for w in ws]


def _free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def _worker(wid, region, port, stop_evt, out, codec="int8", die_after=None, drop_link_at=None,
            join_token=None, fallback_hosts=None, failover_after=3,
            reconnect_backoff_max_s=30.0):
    """Train TinyNet through FederationService. die_after: close the links
    and exit (a crashed process). drop_link_at: close the links once and keep
    training (the service's reconnect path)."""
    from sakura.adapters.ddp import DDPAdapter
    from sakura.runtime import SakuraRuntime

    torch.manual_seed(hash(wid) % 2**31)
    model = TinyNet()
    opt = torch.optim.SGD(model.parameters(), lr=0.05, momentum=0.9)
    loss_fn = torch.nn.CrossEntropyLoss()
    g = torch.Generator().manual_seed(hash(wid) % 1000)
    x = torch.randn(16, 32, generator=g)
    y = torch.randint(0, 8, (16,), generator=g)
    rt = SakuraRuntime()
    fed = FederationService("127.0.0.1", port, wid, 16, codec=codec, region=region,
                            join_token=join_token, fallback_hosts=fallback_hosts,
                            failover_after=failover_after,
                            reconnect_backoff_max_s=reconnect_backoff_max_s)
    rt.install(fed)
    adapter = DDPAdapter(rt, rank=0, world_size=1)
    t0 = time.monotonic()
    try:
        with rt:
            adapter.on_train_begin(model=model, optimizer=opt, train_loader=None)
            step = 0
            while not stop_evt.is_set():
                now = time.monotonic() - t0
                if die_after is not None and now > die_after:
                    for s in [fed.ctl, *fed.data.values()]:
                        shutclose(s)
                    out[wid] = {"died": True}
                    return
                if drop_link_at is not None and now > drop_link_at:
                    drop_link_at = None
                    shutclose(fed.ctl)
                adapter.on_train_step_begin(model=model, batch=(x, y), step=step)
                opt.zero_grad()
                loss = loss_fn(model(x), y)
                loss.backward()
                adapter.on_optimizer_step(optimizer=opt)
                opt.step()
                if step % 10 == 0:
                    adapter.on_epoch_end(0, model=model, optimizer=opt,
                                         metrics={"loss": float(loss.item())})
                step += 1
                time.sleep(0.002)
            out[wid] = {
                "anchor": fed.anchor.clone(),
                "bn": fed._bn_anchor.clone(),
                "committed": fed.committed_round,
                "endpoint": fed.leader,
            }
    except Exception as e:  # noqa: BLE001
        out[wid] = {"error": repr(e)}


class Hier:
    """Root + aggregators + workers on loopback."""

    def __init__(self, codec="int8", root_rounds=6, root_gap=1.0, region_gap=0.2,
                 worker_kw=None, root_kw=None, root_expected=None, rosters=None,
                 agg_kw=None):
        """root_expected / rosters: the root's and each aggregator's
        ``expected`` (default: every aggregator / every regional worker);
        agg_kw: extra RegionAggregator kwargs per region."""
        self.codec = codec
        self.rosters = {**REGIONS, **(rosters or {})}
        self.agg_kw = agg_kw or {}
        self.root_port = _free_port()
        self.root = Leader(
            model_fn=TinyNet,
            port=self.root_port,
            expected=list(root_expected) if root_expected is not None
            else [f"agg-{r}" for r in REGIONS],
            **{
                "codec": codec,
                "gap_s": root_gap,
                "seed": 42,
                "max_rounds": root_rounds,
                **(root_kw or {}),
            },
        )
        self.region_gap = region_gap
        self.ports = {r: _free_port() for r in REGIONS}
        self.aggs: Dict[str, RegionAggregator] = {}
        self.agg_threads: Dict[str, threading.Thread] = {}
        self.retired: List[RegionAggregator] = []
        self.stop = threading.Event()
        self.out: Dict[str, Any] = {}
        self.wthreads: List[threading.Thread] = []
        self._gate_root_on_regions()
        threading.Thread(target=self.root.serve_forever, kwargs={"warmup_s": 0.5},
                         daemon=True).start()
        for r in REGIONS:
            self.start_aggregator(r)
        for r, ws in REGIONS.items():
            for w in ws:
                self.start_worker(w, r, **(worker_kw or {}).get(w, {}))

    def _gate_root_on_regions(self, timeout: float = 60.0) -> None:
        """Hold the root's round loop until every region has run >= 1
        regional round. The root starts counting rounds as soon as both
        aggregators have bootstrapped from it, but an aggregator only binds
        its regional port after that bootstrap: under load its workers can
        still be in connection-refused backoff for the whole (short) root
        run, so a region ran no regional rounds at all (#91 flake). Hooks
        the root's one-shot pre-loop step (``_plan_links``, called after all
        aggregators are present and before the warmup), on this instance
        only, so the production loop is unchanged."""
        plan = self.root._plan_links

        def gated_plan() -> None:
            t0 = time.monotonic()
            while time.monotonic() - t0 < timeout and not self.root.stop.is_set():
                if len(self.aggs) == len(REGIONS) and all(a.stats for a in self.aggs.values()):
                    break
                time.sleep(0.05)
            plan()

        self.root._plan_links = gated_plan  # type: ignore[method-assign]

    def start_aggregator(self, region: str) -> RegionAggregator:
        agg = RegionAggregator(
            model_fn=TinyNet,
            port=self.ports[region],
            expected=self.rosters[region],
            region=region,
            root_host="127.0.0.1",
            root_port=self.root_port,
            aggregator_id=f"agg-{region}",
            codec=self.codec,
            gap_s=self.region_gap,
            seed=7,  # deliberately != root seed: bootstrap must FULL_SYNC
            **self.agg_kw.get(region, {}),
        )
        t = threading.Thread(target=agg.serve_forever, kwargs={"warmup_s": 0.3}, daemon=True)
        t.start()
        self.aggs[region], self.agg_threads[region] = agg, t
        return agg

    def restart_aggregator(self, region: str) -> RegionAggregator:
        old = self.aggs[region]
        old.shutdown()
        self.agg_threads[region].join(10)
        time.sleep(1.2)  # let the old accept loop's 1 s slice run out
        self.retired.append(old)
        return self.start_aggregator(region)

    def kill_aggregator(self, region: str) -> RegionAggregator:
        """Crash an aggregator for good: its region keeps only the workers
        that fail over elsewhere."""
        old = self.aggs.pop(region)
        old.shutdown()
        self.agg_threads.pop(region).join(10)
        self.retired.append(old)
        return old

    def start_worker(self, wid: str, region: str, **kw: Any) -> None:
        # fallback_regions: fail over to those regions' aggregators, in order
        fb = kw.pop("fallback_regions", None)
        if fb:
            kw["fallback_hosts"] = [("127.0.0.1", self.ports[r]) for r in fb]
        t = threading.Thread(
            target=_worker,
            args=(wid, region, self.ports[region], self.stop, self.out),
            kwargs={"codec": self.codec, **kw},
            daemon=True,
        )
        t.start()
        self.wthreads.append(t)

    def wait_root(self, n: int, timeout: float = 90.0) -> None:
        t0 = time.monotonic()
        while self.root.round < n and time.monotonic() - t0 < timeout:
            time.sleep(0.1)

    def finish(self) -> None:
        self.wait_root(self.root.max_rounds)
        self.root.stop.set()
        # freeze the regions first (round loops exit), then the workers, so
        # the final anchors are comparable.
        for agg in self.aggs.values():
            agg.stop.set()
        for t in self.agg_threads.values():
            t.join(15)
        self.stop.set()
        for t in self.wthreads:
            t.join(15)
        for agg in self.aggs.values():
            agg.shutdown()
        try:
            self.root._srv.close()
        except Exception:  # noqa: BLE001
            pass


def _root_hash(root: Leader) -> str:
    return anchor_hash16(root.anchor, root.bn).hex()


def _assert_two_level_consensus(h: Hier) -> None:
    root = h.root
    assert root.round >= root.max_rounds
    # level 2: every root round green, every aggregator acks its g_anchor hash
    assert all(r["consensus_ok"] for r in root.stats), root.stats
    for agg in h.aggs.values():
        assert agg.g_round == root.round
        assert anchor_hash16(agg.g_anchor, agg.g_bn).hex() == _root_hash(root)
    # every aggregator that took part in root round r holds the root's hash
    root_hash_by_round = {r["round"]: r["hash16"] for r in root.stats}
    for agg in [*h.retired, *h.aggs.values()]:
        for g in agg.global_stats:
            assert g["global_hash16"] == root_hash_by_round[g["global_round"]]
    # level 1: every regional round green; workers hold the regional anchor
    for region, agg in h.aggs.items():
        assert agg.stats, f"region {region} ran no rounds"
        assert all(r["consensus_ok"] for r in agg.stats), agg.stats
        lh = anchor_hash16(agg.anchor, agg.bn).hex()
        alive = [w for w in REGIONS[region] if "anchor" in h.out.get(w, {})]
        matches = [w for w in alive if anchor_hash16(h.out[w]["anchor"], h.out[w]["bn"]).hex() == lh]
        assert matches, f"no worker of {region} holds the regional anchor"


@pytest.fixture(scope="module")
def hier_int8():
    h = Hier(codec="int8", root_rounds=6)
    h.finish()
    return h


class TestHierarchyE2E:
    def test_two_level_consensus_every_round(self, hier_int8):
        h = hier_int8
        _assert_two_level_consensus(h)
        for rec in h.root.stats:
            assert set(rec["participants"]) <= {"agg-eu", "agg-us"}
            assert rec["regions"] and set(rec["regions"]) <= {"eu", "us"}
        assert any(set(r["participants"]) == {"agg-eu", "agg-us"} for r in h.root.stats)
        # regional records name the root round they build on
        for agg in h.aggs.values():
            assert all("global_round" in r and "global_hash16" in r for r in agg.stats)

    def test_intra_region_cadence_higher(self, hier_int8):
        h = hier_int8
        for agg in h.aggs.values():
            assert len(agg.stats) > len(h.root.stats)
            assert len({r["global_round"] for r in agg.stats}) > 1

    def test_region_weight_is_sample_count(self, hier_int8):
        h = hier_int8
        for agg in h.aggs.values():
            committed = [g for g in agg.global_stats if g["n_samples"] > 0]
            assert committed, "aggregator never reported regional samples"
        # the root weighs each region by its committed regional samples
        # (floored at 1 for an idle window), never by region count
        reported = [float(n) for r in h.root.stats for n in r["n_samples"].values()]
        assert all(n >= 1.0 for n in reported)
        assert sum(n > 16 for n in reported) >= len(reported) // 2

    def test_wan_bytes_per_round_below_star(self, hier_int8):
        """Same 4 workers / 2 regions as a star: every worker's round traffic
        crosses regions. Hierarchy: only aggregator<->root traffic does."""
        port = _free_port()
        star = Leader(model_fn=TinyNet, port=port, expected=WORKERS, codec="int8",
                      gap_s=0.5, seed=42, max_rounds=3)
        threading.Thread(target=star.serve_forever, kwargs={"warmup_s": 0.3},
                         daemon=True).start()
        stop = threading.Event()
        out: Dict[str, Any] = {}
        ths = []
        for region, ws in REGIONS.items():
            for w in ws:
                t = threading.Thread(target=_worker, args=(w, region, port, stop, out),
                                     daemon=True)
                t.start()
                ths.append(t)
        t0 = time.monotonic()
        while star.round < 3 and time.monotonic() - t0 < 60:
            time.sleep(0.1)
        star.stop.set()
        stop.set()
        for t in ths:
            t.join(15)
        star._srv.close()
        assert star.round >= 3
        full = lambda recs, n: [r for r in recs if len(r["participants"]) == n]  # noqa: E731
        star_b = [round_wire_bytes(r) for r in full(star.stats, len(WORKERS))]
        hier_b = [round_wire_bytes(r) for r in full(hier_int8.root.stats, len(REGIONS))]
        assert star_b and hier_b
        star_mean = sum(star_b) / len(star_b)
        hier_mean = sum(hier_b) / len(hier_b)
        assert hier_mean < star_mean
        # one upload + one broadcast per region instead of per worker
        assert hier_mean / star_mean == pytest.approx(len(REGIONS) / len(WORKERS), rel=0.05)


class TestHierarchyLossless:
    def test_fp32raw_two_level_consensus(self):
        h = Hier(codec="fp32raw", root_rounds=4)
        h.finish()
        _assert_two_level_consensus(h)


def _bare_aggregator(codec="int8", root_codec="int8"):
    agg = RegionAggregator(model_fn=TinyNet, port=0, expected=[], region="x",
                           root_host="127.0.0.1", root_port=0, codec=codec)
    agg.root_codec = root_codec
    agg.ef_up = torch.zeros(agg.fs.numel) if root_codec != "fp32raw" else None
    return agg


class TestTwoLevelAccounting:
    """The intended regional anchor (anchor + ef_bcast + pending) tracks the
    global anchor plus regional progress; no residual is counted twice."""

    def test_root_commit_keeps_progress_since_snapshot(self):
        from sakura.services.federation.codec import codec_encode

        agg = _bare_aggregator(root_codec="fp32raw")
        n = agg.fs.numel
        g = torch.Generator().manual_seed(0)
        agg.anchor.add_(torch.randn(n, generator=g) * 1e-2)  # regional progress
        assert agg.ef_bcast is not None
        agg.ef_bcast.copy_(torch.randn(n, generator=g) * 1e-4)  # undelivered residual
        agg._samples = 64.0
        drift, _, samples = agg._snapshot_drift(0, n)
        assert samples == 64.0 and agg._samples == 0.0
        assert torch.allclose(drift, agg._intended(0, n) - agg.g_anchor)
        later = torch.randn(n, generator=g) * 1e-2
        agg.anchor.add_(later)  # a regional round lands while the root round runs
        U = torch.randn(n, generator=g) * 1e-2  # root update (fp32raw: exact)
        payload, _, _ = codec_encode(U, None, "fp32raw")
        g_old = agg.g_anchor.clone()
        agg._commit_root(1, payload, agg.bn.numpy().tobytes(), drift, None, 0, n)
        assert agg.g_round == 1
        assert torch.equal(agg.g_anchor, g_old + U)
        # intended = new global anchor + progress made after the snapshot
        assert torch.allclose(agg._intended(0, n), agg.g_anchor + later, atol=1e-6)

    def test_fold_adds_pending_slice(self):
        agg = _bare_aggregator()
        n = agg.fs.numel
        agg.pending.fill_(1.0)
        out = agg._fold_update(torch.zeros(n // 2), 0, n // 2)
        assert torch.equal(out, torch.ones(n // 2))
        assert agg._fold == (0, n // 2)

    def test_reset_to_global_drops_regional_progress(self):
        agg = _bare_aggregator()
        n = agg.fs.numel
        agg.anchor.add_(0.5)
        assert agg.ef_bcast is not None and agg.ef_up is not None
        agg.ef_bcast.fill_(0.25)
        agg.ef_up.fill_(0.1)
        agg._samples = 32.0
        agg._reset_to_global()
        assert torch.allclose(agg._intended(0, n), agg.g_anchor, atol=1e-6)
        assert float(agg.ef_up.abs().sum()) == 0.0 and agg._samples == 0.0


class TestRejoinBothLevels:
    def test_worker_rejoin_and_fullsync_level1(self, caplog):
        caplog.set_level(logging.INFO, logger="sakura.federation")
        # a: link drop + reconnect (resync path: nothing / resend / FULL_SYNC);
        # d: crash, then a fresh process with the same id (bootstrap FULL_SYNC)
        h = Hier(codec="int8", root_rounds=8,
                 worker_kw={"a": {"drop_link_at": 3.0}, "d": {"die_after": 3.0}})
        t0 = time.monotonic()
        while not h.out.get("d", {}).get("died") and time.monotonic() - t0 < 60:
            time.sleep(0.1)
        assert h.out.get("d", {}).get("died")
        h.wait_root(h.root.round + 1)
        h.start_worker("d", "us")
        h.finish()
        _assert_two_level_consensus(h)
        msgs = [r.getMessage() for r in caplog.records]
        assert sum(m.startswith("FULL_SYNC -> d ") for m in msgs) >= 2
        assert any(m.startswith("hello a/ctl") and "last_committed=-1" not in m for m in msgs)
        us = h.aggs["us"].stats
        died_rounds = [r for r in us if r["participants"] == ["c"]]
        assert died_rounds, "region us never ran a subset round while d was down"
        assert any(set(r["participants"]) == {"c", "d"} and r["round"] > died_rounds[0]["round"]
                   for r in us), "fresh d never rejoined regional rounds"
        assert any(set(r["participants"]) == {"a", "b"} for r in h.aggs["eu"].stats[-3:])

    def test_aggregator_link_drop_and_restart_level2(self, caplog):
        caplog.set_level(logging.INFO, logger="sakura.federation")
        h = Hier(codec="int8", root_rounds=10)
        h.wait_root(2)
        # (1) root link drop: the aggregator reconnects with its committed
        # root round + g_anchor hash; the root resyncs it (nothing/resend/FULL)
        eu = h.aggs["eu"]
        shutclose(eu.ctl)
        h.wait_root(4)
        # (2) aggregator crash + restart: fresh process -> root FULL_SYNC at
        # level 2, then its workers (old regional anchor) FULL_SYNC at level 1
        h.restart_aggregator("us")
        h.finish()
        _assert_two_level_consensus(h)
        msgs = [r.getMessage() for r in caplog.records]
        assert sum(m.startswith("FULL_SYNC -> agg-us") for m in msgs) >= 2
        assert any("agg-eu connected to root at committed=" in m and "committed=-1" not in m
                   for m in msgs)
        for w in ("c", "d"):
            assert sum(m.startswith(f"FULL_SYNC -> {w} ") for m in msgs) >= 2
        new_us = h.aggs["us"]
        assert any(set(r["participants"]) == {"c", "d"} for r in new_us.stats)
        late = [r for r in h.root.stats if r["round"] > h.root.round - 2]
        assert any(set(r["participants"]) == {"agg-eu", "agg-us"} for r in late)


class TestRootLinkProbe:
    def test_aggregators_follow_root_auto_codecs(self):
        """SAK-0013 at the root: the probe sees agg-us on a slow link, so the
        root broadcasts top-k; agg-eu keeps int8 uploads. Aggregators follow
        the codecs announced in ROUND_BEGIN, not their HELLO_ACK."""
        from sakura.services.federation import LinkSample

        # TinyNet is one bucket per leg: topk 1.75 KB, int4 2.56 KB (with BN).
        # 8 kB/s in a 0.5 s budget carries 2 KB per leg -> only topk fits.
        def probe():
            return {"agg-eu": LinkSample("agg-eu", 1e9, 0.0),
                    "agg-us": LinkSample("agg-us", 8_000.0, 0.0)}

        h = Hier(codec="int8", root_rounds=4,
                 root_kw={"codec": "auto", "link_probe": probe})
        h.finish()
        assert (h.root.codec, h.root.shards) == ("topk", 1)
        assert h.root.upload_codecs == {"agg-eu": "int8", "agg-us": "topk"}
        assert {a.root_codec for a in h.aggs.values()} == {"topk"}
        assert h.aggs["eu"].root_up_codec == "int8"
        assert h.aggs["us"].root_up_codec == "topk"
        _assert_two_level_consensus(h)


ROOT_TOKEN = "test-root-join-token"
REGION_TOKEN = "test-region-join-token"


class TestJoinViaToken:
    """SAK-0014 at both levels of the hierarchy: a worker outside its
    aggregator's roster joins the region with the aggregator's ``join_token``,
    and an aggregator outside the root's ``expected`` joins the root with its
    ``root_join_token``. Defaults stay fail-closed."""

    def test_hello_carries_root_token_only_when_configured(self):
        root = Leader(model_fn=TinyNet, port=0, expected=["agg-eu"], join_token=ROOT_TOKEN)
        plain = _bare_aggregator()
        assert "join_token" not in plain._hello()
        with pytest.raises(AssertionError, match="joining not authorised"):
            root._roster_entry(plain.aggregator_id, plain._hello().get("join_token"))
        wrong = RegionAggregator(model_fn=TinyNet, port=0, expected=[], region="x",
                                 root_host="127.0.0.1", root_port=0,
                                 root_join_token="wrong")
        with pytest.raises(AssertionError):
            root._roster_entry(wrong.aggregator_id, wrong._hello().get("join_token"))
        agg = RegionAggregator(model_fn=TinyNet, port=0, expected=[], region="x",
                               root_host="127.0.0.1", root_port=0,
                               root_join_token=ROOT_TOKEN)
        tok = agg._hello().get("join_token")
        assert root._roster_entry(agg.aggregator_id, tok).worker_id == agg.aggregator_id
        # the regional token is the aggregator's own Leader policy, separate
        # from the one it presents upstream
        assert agg._join_token is None

    def test_regional_policy_forwarded_to_leader(self):
        agg = RegionAggregator(model_fn=TinyNet, port=0, expected=["a"], region="eu",
                               root_host="127.0.0.1", root_port=0,
                               allow_join={"z"}, join_token=REGION_TOKEN,
                               root_join_token=ROOT_TOKEN)
        assert agg._roster_entry("z", None).worker_id == "z"
        assert agg._roster_entry("b", REGION_TOKEN).worker_id == "b"
        with pytest.raises(AssertionError):
            agg._roster_entry("y", ROOT_TOKEN)  # the upstream token is not regional
        default = _bare_aggregator()
        with pytest.raises(AssertionError, match="joining not authorised"):
            default._roster_entry("b", REGION_TOKEN)

    def test_bad_config(self):
        kw = dict(model_fn=TinyNet, port=0, expected=[], region="x",
                  root_host="127.0.0.1", root_port=0)
        with pytest.raises(ValueError):
            RegionAggregator(**kw, root_join_token="")
        with pytest.raises(ValueError):
            RegionAggregator(**kw, join_token="")
        with pytest.raises(TypeError):
            RegionAggregator(**kw, allow_join="z")

    def test_joiners_at_both_levels_loopback(self):
        # eu lists only "a": "b" joins the region with the regional token.
        # The root lists only agg-eu: agg-us joins it with root_join_token.
        h = Hier(codec="int8", root_rounds=5,
                 root_expected=["agg-eu"], root_kw={"join_token": ROOT_TOKEN},
                 rosters={"eu": ["a"]},
                 agg_kw={"eu": {"join_token": REGION_TOKEN},
                         "us": {"root_join_token": ROOT_TOKEN}},
                 worker_kw={"b": {"join_token": REGION_TOKEN}})
        h.finish()
        assert h.root.expected == ["agg-eu"] and "agg-us" in h.root.workers
        assert h.aggs["eu"].expected == ["a"] and "b" in h.aggs["eu"].workers
        assert any(set(r["participants"]) == {"agg-eu", "agg-us"} for r in h.root.stats)
        assert any("b" in r["participants"] for r in h.aggs["eu"].stats)
        _assert_two_level_consensus(h)


class TestShardsTwoBothLevels:
    """Streaming shards (``shards=2``) at the root and in every region: each
    round moves one half of the parameters at its own level; both hashes stay
    green every round."""

    def test_shards2_two_level_consensus(self):
        h = Hier(codec="int8", root_rounds=6, root_kw={"shards": 2},
                 agg_kw={r: {"shards": 2} for r in REGIONS})
        h.finish()
        _assert_two_level_consensus(h)
        assert h.root.stats and all(r["shards"] == 2 for r in h.root.stats)
        for agg in h.aggs.values():
            assert all(r["shards"] == 2 for r in agg.stats)
            # both halves committed at the region (alternating shards)
            assert len(agg.stats) >= 2
        # both halves committed at the root, from every aggregator's view
        for agg in h.aggs.values():
            assert len({g["global_round"] % 2 for g in agg.global_stats}) == 2


def _persisting_aggregator(path, codec="int8"):
    return RegionAggregator(model_fn=TinyNet, port=0, expected=[], region="x",
                            root_host="127.0.0.1", root_port=0, codec=codec,
                            persist_path=str(path))


class TestAggregatorPersistence:
    """#144: an aggregator with ``persist_path`` resumes both tiers after a
    restart instead of re-bootstrapping from the root."""

    def test_round_trip_restores_both_tiers(self, tmp_path):
        path = tmp_path / "agg.pt"
        agg = _persisting_aggregator(path)
        n = agg.fs.numel
        g = torch.Generator().manual_seed(1)
        agg.anchor.add_(torch.randn(n, generator=g))
        assert agg.ef_bcast is not None
        agg.ef_bcast.copy_(torch.randn(n, generator=g) * 1e-3)
        agg.g_anchor.add_(torch.randn(n, generator=g))
        agg.g_bn.add_(0.5)
        agg.pending.copy_(torch.randn(n, generator=g) * 1e-2)
        agg.root_codec, agg.root_up_codec = "int8", "int4"
        agg.ef_up = torch.randn(n, generator=g) * 1e-3
        agg.round, agg.g_round, agg._samples = 17, 5, 96.0
        agg.outer_steps = {0: 17}
        with agg.round_mutex:
            agg._persist_tiers()
        assert not os.path.exists(str(path) + ".tmp")
        back = _persisting_aggregator(path)
        assert back.restored_from == (17, 5)
        assert (back.round, back.g_round, back._samples) == (17, 5, 96.0)
        assert (back.root_codec, back.root_up_codec) == ("int8", "int4")
        assert back.outer_steps == {0: 17}
        for k in ("anchor", "bn", "ef_bcast", "g_anchor", "g_bn", "pending", "ef_up"):
            assert torch.equal(getattr(back, k), getattr(agg, k)), k
        # same intended anchor, same global hash: the next root round uploads
        # the same drift the old process would have (no double count, no loss)
        assert torch.equal(back._intended(0, n), agg._intended(0, n))
        assert back._hello()["last_committed"] == 5
        assert back._hello()["anchor_hash16"] == anchor_hash16(agg.g_anchor, agg.g_bn).hex()

    def test_regional_commit_persists_after_pending_fold(self, tmp_path):
        """The file written at a regional commit already has the folded
        pending slice zeroed: a restart must not fold it a second time."""
        path = tmp_path / "agg.pt"
        agg = _persisting_aggregator(path)
        n = agg.fs.numel
        agg.pending.fill_(0.5)

        def fake_round():  # a committed regional round that folded [0, n/2)
            agg._fold = (0, n // 2)
            agg.anchor[: n // 2].add_(0.5)
            agg.round += 1
            return {"n_samples": {"w": 16}}

        agg._run_round_locked = fake_round  # type: ignore[method-assign]
        assert agg.run_round() is not None
        st = torch.load(str(path), map_location="cpu")
        assert st["round"] == 1
        assert float(st["region"]["pending"][: n // 2].abs().sum()) == 0.0
        assert torch.all(st["region"]["pending"][n // 2:] == 0.5)
        assert st["region"]["samples"] == 16.0
        assert torch.equal(st["anchor"], agg.anchor)

    def test_unusable_file_falls_back_to_root_bootstrap(self, tmp_path):
        path = tmp_path / "agg.pt"
        path.write_bytes(b"not a checkpoint")
        agg = _persisting_aggregator(path)
        assert agg.restored_from is None and agg.g_round == -1

    def test_restart_resumes_without_fullsync(self, tmp_path, caplog):
        caplog.set_level(logging.INFO, logger="sakura.federation")
        path = str(tmp_path / "us.pt")
        h = Hier(codec="int8", root_rounds=10, agg_kw={"us": {"persist_path": path}})
        h.wait_root(3)
        old = h.aggs["us"]
        # stop between regional rounds (the loop exits after a full round),
        # then crash the process
        old.stop.set()
        h.agg_threads["us"].join(15)
        old.shutdown()
        time.sleep(1.2)  # let the old accept loop's 1 s slice run out
        h.retired.append(old)
        new = h.start_aggregator("us")
        assert new.restored_from == (old.round, old.g_round)
        assert old.round > 0 and old.g_round >= 0
        h.finish()
        _assert_two_level_consensus(h)
        msgs = [r.getMessage() for r in caplog.records]
        # level 2: the restored aggregator resumes at its committed root round
        assert any(f"agg-us connected to root at committed={old.g_round}" in m for m in msgs)
        # level 1: its workers resume against the restored regional anchor,
        # no second FULL_SYNC (only the bootstrap one)
        for w in ("c", "d"):
            assert sum(m.startswith(f"FULL_SYNC -> {w} ") for m in msgs) == 1, w
        assert new.stats and new.stats[0]["round"] == old.round + 1
        assert any(set(r["participants"]) == {"c", "d"} for r in new.stats)


class TestWorkerFailover:
    def test_failover_rotation_and_hello(self):
        fed = FederationService("127.0.0.1", 1, "w", 16,
                                fallback_hosts=[("127.0.0.1", 2), ("127.0.0.1", 3)],
                                failover_after=2)
        assert fed.endpoints == [("127.0.0.1", 1), ("127.0.0.1", 2), ("127.0.0.1", 3)]
        assert fed._connect_failed() is False and fed.leader == ("127.0.0.1", 1)
        assert fed._connect_failed() is True and fed.leader == ("127.0.0.1", 2)
        assert fed._hello()["last_committed"] == -1  # never synced with it
        fed._connect_failed()
        fed._connect_failed()
        fed._connect_failed()
        fed._connect_failed()
        assert fed.leader == ("127.0.0.1", 1)  # cyclic: back to the primary
        # synced with the primary: the HELLO carries our committed state
        fed.anchor = torch.zeros(4)
        fed._bn_anchor = torch.zeros(0)
        fed.committed_round, fed._synced_ep = 7, 0
        assert fed._hello()["last_committed"] == 7
        fed._connect_failed()
        fed._connect_failed()
        assert fed._hello()["last_committed"] == -1
        assert fed._hello()["anchor_hash16"] == "00" * 16

    def test_no_fallback_never_moves(self):
        fed = FederationService("127.0.0.1", 1, "w", 16, failover_after=1)
        assert not any(fed._connect_failed() for _ in range(5))
        assert fed.leader == ("127.0.0.1", 1)
        with pytest.raises(ValueError):
            FederationService("127.0.0.1", 1, "w", 16, failover_after=0)

    def test_workers_fail_over_to_other_region(self, caplog):
        """us's aggregator dies; c and d fail over to eu's aggregator, which
        admits them with its join token and FULL_SYNCs them. Their samples
        then count for eu (regional and root weight); both levels stay green."""
        caplog.set_level(logging.INFO, logger="sakura.federation")
        fail_kw = {"join_token": REGION_TOKEN, "fallback_regions": ["eu"],
                   "failover_after": 2, "reconnect_backoff_max_s": 1.0}
        h = Hier(codec="int8", root_rounds=10,
                 agg_kw={"eu": {"join_token": REGION_TOKEN}},
                 worker_kw={"c": dict(fail_kw), "d": dict(fail_kw)})
        h.wait_root(2)
        dead = h.kill_aggregator("us")
        cut = h.root.round
        h.finish()
        _assert_two_level_consensus(h)
        eu = h.aggs["eu"]
        assert {"c", "d"} <= set(eu.workers)
        assert any({"a", "b", "c", "d"} <= set(r["participants"]) for r in eu.stats)
        msgs = [r.getMessage() for r in caplog.records]
        for w in ("c", "d"):
            assert any(m.startswith(f"{w} failing over from ") for m in msgs), w
            assert any(m.startswith(f"FULL_SYNC -> {w} ") for m in msgs), w
            assert h.out[w].get("endpoint") == ("127.0.0.1", h.ports["eu"]), h.out[w]
        # the root kept running subset rounds with eu alone, all green, and
        # eu's weight grew with the extra workers' samples
        late = [r for r in h.root.stats if r["round"] > cut + 1]
        assert late and all(r["participants"] == ["agg-eu"] for r in late)
        assert dead.global_stats  # us took part before it died
