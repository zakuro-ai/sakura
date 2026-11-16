"""#94: bandwidth-reduction + compute-utilisation report vs data-parallel.

Loads the REAL federation modules as a synthetic `fed` package (no Rust
`sakura_wire` build needed), like test_codec_compression.py."""
from __future__ import annotations

import importlib.util
import json
import pathlib
import subprocess
import sys
import types

import pytest

_ROOT = pathlib.Path(__file__).resolve().parents[2]
_BASE = _ROOT / "sakura/services/federation"
_RESULTS = _ROOT / "results"


def _load(name: str):
    key = f"fed.{name}"
    if key in sys.modules:
        return sys.modules[key]
    if "fed" not in sys.modules:
        pkg = types.ModuleType("fed")
        pkg.__path__ = [str(_BASE)]
        sys.modules["fed"] = pkg
    spec = importlib.util.spec_from_file_location(key, _BASE / f"{name}.py")
    m = importlib.util.module_from_spec(spec)
    sys.modules[key] = m
    spec.loader.exec_module(m)
    return m


def _report():
    return _load("report")


def _write_run(d: pathlib.Path, rounds, workers, numel):
    d.mkdir(parents=True)
    (d / "leader.log").write_text(
        "noise line\n" + "".join("LEADER_ROUND " + json.dumps(r) + "\n" for r in rounds)
    )
    for w, (batch, img_s) in workers.items():
        (d / f"{w}.log").write_text(
            "WORKER_ROUND " + json.dumps({"round": 1, "shard": [0, numel]}) + "\n"
            + "ZKRESULT " + json.dumps(
                {"worker": w, "mode": "overlap", "batch": batch, "img_s": img_s}
            ) + "\n"
        )


def test_bandwidth_factor_is_steps_times_fp32_over_payload(tmp_path):
    rep = _report()
    numel = 1000
    rounds = [
        {"round": r, "n_samples": {"fast": 640, "slow": 16},
         "bytes_up": {"fast": 1000, "slow": 1000}, "bytes_down": 1000}
        for r in (1, 2)
    ]
    _write_run(tmp_path / "run", rounds, {"fast": (64, 100.0), "slow": (16, 2.0)}, numel)
    run = rep.parse_run_dir(tmp_path / "run")
    assert run["numel"] == numel and len(run["rounds"]) == 2
    bw = rep.bandwidth_reduction(run["rounds"], {"fast": 64, "slow": 16}, numel)
    # fast: 10 steps/round x 4000 B fp32 / 1000 B payload = 40x; slow: 1 step -> 4x
    assert bw["per_worker"]["fast"]["factor"] == 40.0
    assert bw["per_worker"]["fast"]["H_eff"] == 10.0
    assert bw["per_worker"]["slow"]["factor"] == 4.0
    assert bw["per_worker"]["slow"]["per_sync_ratio"] == 4.0
    assert bw["slowest_worker_factor"] == 4.0
    # fleet = (20 + 2) steps x 4000 / 4000 B shipped = 22x
    assert bw["fleet_factor"] == 22.0


def test_bandwidth_rejects_empty_model():
    with pytest.raises(ValueError):
        _report().bandwidth_reduction([], {}, 0)


@pytest.mark.skipif(not (_RESULTS / "e5" / "leader.log").exists(), reason="fleet logs absent")
def test_fleet_e5_measured_numbers():
    """The E5 fleet logs reproduce docs/internal/REPORT.md (97.6% of ceiling) and give a
    measured int8 bandwidth factor far above the codec's 4x alone."""
    rep = _report()
    rows = {r["run"]: r for r in rep.fleet_report(_RESULTS)}
    e5 = rows["e5"]
    assert e5["source"] == "measured"
    assert e5["compute"]["util"] == pytest.approx(0.976, abs=0.002)
    bw = e5["bandwidth"]
    assert bw["numel"] == 25_557_032
    for v in bw["per_worker"].values():
        assert v["per_sync_ratio"] == pytest.approx(3.996, abs=0.001)  # int8 + scales
    # the P620 is the slowest: ~74 steps/round -> ~300x; 4090 ~1000 -> ~4000x
    assert min(bw["per_worker"], key=lambda w: bw["per_worker"][w]["factor"]) == "p620"
    assert 250 < bw["slowest_worker_factor"] < 350
    assert bw["fleet_factor"] > 1000
    # E3 shipped fp32raw: the per-sync ratio is 1x, so the factor is H_eff alone
    for v in rows["e3"]["bandwidth"]["per_worker"].values():
        assert v["per_sync_ratio"] == pytest.approx(1.0, abs=0.001)
    # E5s streams K=4 shards: 4x fewer bytes per sync on top of int8
    for v in rows["e5s"]["bandwidth"]["per_worker"].values():
        assert v["per_sync_ratio"] == pytest.approx(4 * 3.996, rel=0.01)
    assert rows["e6"]["compute"] is None  # no R50 ceiling for CIFAR/R18


def test_ddp_lockstep_under_56_to_1_skew_is_skew_bound():
    rep = _report()
    m = rep.ddp_lockstep_util(rep.R50_SOLO_IMG_S, rep.R50_BATCH, 25_557_032)
    # even a free network leaves DDP waiting on the P620 every step
    assert m["util_free_network"] < 0.15
    assert m["util"] < 0.01
    blk = rep.federated_util(30.0, 25_584_000, 3, "block")
    ovl = rep.federated_util(30.0, 25_584_000, 3, "overlap")
    assert m["util"] < blk["util"] < ovl["util"] <= 1.0
    with pytest.raises(ValueError):
        rep.federated_util(30.0, 1, 3, "bogus")


def test_simulate_skew_counts_real_grad_and_payload_bytes():
    torch = pytest.importorskip("torch")
    rep, codec = _report(), _load("codec")

    def make_model():
        return torch.nn.Sequential(
            # > one 4096-element codec bucket, so padding does not dominate
            torch.nn.Linear(8, 64), torch.nn.BatchNorm1d(64), torch.nn.ReLU(),
            torch.nn.Linear(64, 64), torch.nn.ReLU(), torch.nn.Linear(64, 3),
        )

    g = torch.Generator().manual_seed(0)
    X = torch.randn(256, 8, generator=g)
    y = (X[:, 0] > 0).long() + (X[:, 1] > 0).long()
    steps = {"slow": 1, "mid": 4, "fast": 14}
    batch = {"slow": 16, "mid": 32, "fast": 32}
    sim = rep.simulate_skew(codec, make_model, (X, y, X, y), steps, batch, rounds=3)
    bw = sim["bandwidth"]
    assert bw["dp_bytes_measured"] == bw["dp_bytes_expected"] > 0
    pw = bw["per_worker"]
    assert pw["fast"]["H_eff"] == 14 and pw["slow"]["H_eff"] == 1
    # same payload size for everyone -> factor scales exactly with H_eff
    assert pw["fast"]["factor"] == pytest.approx(14 * pw["slow"]["per_sync_ratio"], rel=1e-2)
    assert pw["slow"]["factor"] == pytest.approx(pw["slow"]["per_sync_ratio"], rel=1e-2)
    assert 1.0 < pw["slow"]["per_sync_ratio"] <= 4.0
    assert 0.0 <= sim["final_acc"] <= 1.0
    md = rep.to_markdown({"fleet": [], "sim": sim})
    assert "Sim: skewed CPU fleet" in md


def test_r50_projection_matches_formula():
    rep = _report()
    numel, payload = 25_557_032, 25_584_000
    p = rep.r50_projection(payload, numel, period_s=30.0)
    assert p["codec_ratio"] == pytest.approx(3.996, abs=0.001)
    # P620: 29.75 img/s x 30 s / b16 = 55.8 steps per round
    assert p["H_eff"]["p620"] == pytest.approx(55.8, abs=0.1)
    assert p["per_worker_factor"]["p620"] == pytest.approx(55.8 * 3.9958, rel=1e-3)
    md = rep.to_markdown({"fleet": [], "projection": p})
    assert "INTELLECT-1 reference" in md


@pytest.mark.skipif(not (_RESULTS / "e5" / "leader.log").exists(), reason="fleet logs absent")
def test_sim_fed_report_flag_end_to_end(tmp_path):
    pytest.importorskip("torch")
    out = tmp_path / "rep"
    subprocess.run(
        [sys.executable, str(_ROOT / "sim_fed.py"), "--report", "--rounds", "1",
         "--numel", "100000", "--results", str(_RESULTS), "--out", str(out)],
        check=True, timeout=300, capture_output=True,
    )
    data = json.loads((out / "report.json").read_text())
    assert {r["run"] for r in data["fleet"]} >= {"e3", "e4", "e5", "e5s"}
    assert data["sim"]["source"] == "sim"
    assert (out / "report.md").read_text().startswith("# Bandwidth reduction")
