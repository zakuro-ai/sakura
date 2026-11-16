"""#91: sim_fed.py three-tier HALoS (workers -> synchronous regional
aggregators -> async delayed-Nesterov root) vs flat DiLoCo on the 3-GPU /
2-region skew topology."""
from __future__ import annotations

import importlib.util
import pathlib
import sys

import pytest

torch = pytest.importorskip("torch")

_ROOT = pathlib.Path(__file__).resolve().parents[2]


def _sim():
    if "sim_fed" in sys.modules:
        return sys.modules["sim_fed"]
    spec = importlib.util.spec_from_file_location("sim_fed", _ROOT / "sim_fed.py")
    m = importlib.util.module_from_spec(spec)
    sys.modules["sim_fed"] = m
    spec.loader.exec_module(m)
    return m


@pytest.fixture(scope="module")
def task():
    return _sim().make_task(n_train=1024, n_test=256)


@pytest.mark.parametrize("layout,regions", [("fleet", ["A", "A", "B"]),
                                            ("split", ["A", "B", "B"])])
def test_skew_topology(layout, regions):
    sim = _sim()
    topo = sim.skew_topology(layout)
    assert topo.regions == regions and topo.leader_region == "A"
    assert topo.speed == [140.0, 40.0, 10.0]  # 1:4:14 skew
    b = regions.index("B")
    # in-region legs are LAN, the B aggregator reaches the root over the WAN
    assert topo.region_xfer_s("B", 100_000) > 10 * topo.lan_xfer_s(b, 100_000)
    assert topo.region_xfer_s("A", 100_000) == topo.lan_xfer_s(0, 100_000)


def test_flat_round_is_gated_by_the_wan(task):
    sim = _sim()
    Xtr, ytr, Xte, yte = task
    topo = sim.skew_topology("fleet")
    res = sim.run_flat_timed("int8", "nesterov", Xtr, ytr, Xte, yte, topo,
                             budget_s=1.0, outer_lr=0.5)
    assert res["steps_per_round"] == [14, 4, 1]
    # a round costs compute (0.1 s) + the P620's WAN round trip (> 0.08 s)
    assert res["outer_rounds"] < 1.0 / 0.18
    assert res["wan_bytes"] > 0
    assert len(res["history"]) == 4 and 0.0 <= res["final_acc"] <= 1.0


@pytest.mark.parametrize("outer", ["plain", "delayed_nesterov"])
def test_halos_regions_never_wait_for_the_wan(task, outer):
    sim = _sim()
    Xtr, ytr, Xte, yte = task
    topo = sim.skew_topology("split")
    flat = sim.run_flat_timed("int8", "nesterov", Xtr, ytr, Xte, yte, topo,
                              budget_s=1.0, outer_lr=0.5)
    res = sim.run_halos("int8", outer, Xtr, ytr, Xte, yte, topo, budget_s=1.0,
                        outer_lr=0.7)
    # both regions committed at the root, asynchronously (some staleness)
    assert all(c > 0 for c in res["root_commits"].values())
    assert res["tau_max"] >= 1
    # region B keeps running regional rounds while its drift crosses the WAN
    assert res["regional_rounds"]["B"] > res["root_commits"]["B"]
    # no WAN idle: more samples per simulated second than the flat star
    assert res["tokens"] > flat["tokens"]
    assert 0.0 <= res["final_acc"] <= 1.0


def test_halos_lossless_single_region_matches_regional_progress(task):
    """fp32raw, root plain lr=1, no staleness discount, root and region
    co-located: the root anchor after every exchange is exactly the regional
    intended anchor at the snapshot, so the reconcile rule (anchor += U - D)
    leaves the region holding root + post-snapshot progress."""
    sim = _sim()
    Xtr, ytr, Xte, yte = task
    topo = sim.Topology(regions=["A", "A"], speed=[40.0, 40.0], leader_region="A",
                        links={("A", "A"): (0.0005, 1000.0)})
    res = sim.run_halos("fp32raw", "plain", Xtr, ytr, Xte, yte, topo, budget_s=0.5,
                        stale_alpha=0.0)
    flat = sim.run_flat_timed("fp32raw", "plain", Xtr, ytr, Xte, yte, topo,
                              budget_s=0.5, outer_lr=1.0)
    assert res["tau_max"] == 0  # one region: every drift is fresh
    # one region, lossless, plain lr=1: the hierarchy degenerates to the star
    # (the root lags the region by at most one exchange)
    assert abs(res["final_acc"] - flat["final_acc"]) < 0.1
