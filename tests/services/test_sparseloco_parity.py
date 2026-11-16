"""#89 / #50 SparseLoCo parity: outer='ef' (momentum folded into the codec's
error-feedback accumulator, no outer-momentum buffer) on the topk2 codec must
train the sim_fed.py synthetic task as well as a Nesterov outer step, at a
fraction of int8's bytes. Drives the real shipping primitives through
sim_fed.run_federation (block-mode rounds, CPU, no Rust build)."""
from __future__ import annotations

import importlib.util
import pathlib
import sys

import pytest

torch = pytest.importorskip("torch")

_ROOT = pathlib.Path(__file__).resolve().parents[2]
SEEDS = (1, 2)
ROUNDS = 30


@pytest.fixture(scope="module")
def sim():
    spec = importlib.util.spec_from_file_location("sim_fed_89", _ROOT / "sim_fed.py")
    m = importlib.util.module_from_spec(spec)
    saved = {k: v for k, v in sys.modules.items() if k == "fed" or k.startswith("fed.")}
    try:
        spec.loader.exec_module(m)
    finally:  # keep other tests' synthetic `fed` package intact
        for k in [k for k in sys.modules if k == "fed" or k.startswith("fed.")]:
            del sys.modules[k]
        sys.modules.update(saved)
    return m


@pytest.fixture(scope="module")
def task(sim):
    return sim.make_task(n_train=4096, n_test=1024)


def _mean_acc(sim, task, kind, outer, lr, mu=0.9):
    runs = [sim.run_federation(kind, outer, *task, rounds=ROUNDS, outer_lr=lr, mu=mu, seed=s)
            for s in SEEDS]
    return sum(r["final_acc"] for r in runs) / len(runs), runs[0]


def test_ef_topk2_parity_with_nesterov(sim, task):
    torch.set_num_threads(2)
    ef_acc, ef_run = _mean_acc(sim, task, "topk2", "ef", 0.5, 0.95)
    nes_acc, _ = _mean_acc(sim, task, "topk2", "nesterov", 0.5)
    base_acc, base_run = _mean_acc(sim, task, "int8", "plain", 1.0)
    chance = 1.0 / 8
    assert ef_acc > 3 * chance  # it actually learned
    assert ef_acc >= nes_acc - 0.03, (ef_acc, nes_acc)
    assert ef_acc >= base_acc - 0.03, (ef_acc, base_acc)
    # >= 4x fewer bytes per round than the shipped int8 (both directions)
    ef_bytes = ef_run["bytes_up"] + ef_run["bytes_down"]
    base_bytes = base_run["bytes_up"] + base_run["bytes_down"]
    assert base_bytes >= 4 * ef_bytes
