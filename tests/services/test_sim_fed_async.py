"""#90: sim_fed.py can run async local-SGD with delayed Nesterov and
staleness-weighted aggregation on a 2-region (fast/slow link) topology."""
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


def test_two_region_topology_slow_link_costs_more():
    sim = _sim()
    topo = sim.two_region_topology()
    assert sorted(set(topo.regions)) == ["A", "B"] and topo.W == 4
    a = topo.regions.index("A")
    b = topo.regions.index("B")
    assert topo.xfer_s(b, 100_000) > 10 * topo.xfer_s(a, 100_000)


@pytest.mark.parametrize("outer", ["nesterov", "delayed_nesterov"])
def test_async_run_completes_with_staleness(outer):
    sim = _sim()
    Xtr, ytr, Xte, yte = sim.make_task(n_train=1024, n_test=256)
    topo = sim.two_region_topology()
    res = sim.run_async_federation("int8", outer, Xtr, ytr, Xte, yte, topo,
                                   H=3, rounds=3, outer_lr=0.7, stale_alpha=0.5)
    assert res["history"] and 0.0 <= res["final_acc"] <= 1.0
    assert res["tau_max"] >= 1          # async: some uploads are stale
    assert res["sim_time_s"] > 0
