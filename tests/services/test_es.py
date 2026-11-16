"""#92: low-rank (EGGROLL) ES perturbations and the sim_fed ES round mode.

The ES primitives must be bit-deterministic in the seed (any node rebuilds a
peer's perturbation from 8 bytes), antithetic pairs must cancel exactly, and
the reconstructed gradient must point along the true one. Loaded by path (no
sakura_wire build), like test_outer_opt."""
from __future__ import annotations

import importlib.util
import math
import pathlib
import sys
import types

import pytest

torch = pytest.importorskip("torch")

_ROOT = pathlib.Path(__file__).resolve().parents[2]
_BASE = _ROOT / "sakura/services/federation"

SHAPES = [(32, 16), (32,), (8, 32), (8,), (4, 3, 5)]


def _es():
    if "fed.es" in sys.modules:
        return sys.modules["fed.es"]
    if "fed" not in sys.modules:
        pkg = types.ModuleType("fed")
        pkg.__path__ = [str(_BASE)]
        sys.modules["fed"] = pkg
    spec = importlib.util.spec_from_file_location("fed.es", _BASE / "es.py")
    m = importlib.util.module_from_spec(spec)
    sys.modules["fed.es"] = m
    spec.loader.exec_module(m)
    return m


def _numel(shapes):
    return sum(math.prod(s) for s in shapes)


# ------------------------------------------------------------- determinism
def test_same_seed_same_perturbation_bitwise():
    es = _es()
    a = es.lowrank_perturbation(1234, SHAPES, rank=2)
    b = es.lowrank_perturbation(1234, SHAPES, rank=2)
    assert a.shape == (_numel(SHAPES),)
    assert torch.equal(a, b)


def test_global_rng_state_does_not_leak_in():
    es = _es()
    torch.manual_seed(0)
    a = es.lowrank_perturbation(7, SHAPES, rank=1)
    torch.manual_seed(999)
    torch.randn(100)
    b = es.lowrank_perturbation(7, SHAPES, rank=1)
    assert torch.equal(a, b)


def test_different_seeds_differ():
    es = _es()
    a = es.lowrank_perturbation(1, SHAPES, rank=2)
    b = es.lowrank_perturbation(2, SHAPES, rank=2)
    assert not torch.allclose(a, b)


def test_member_seed_deterministic_distinct_and_fits_torch():
    es = _es()
    seeds = {es.member_seed(5, r, w, k) for r in range(3) for w in range(4) for k in range(8)}
    assert len(seeds) == 3 * 4 * 8
    assert es.member_seed(5, 1, 2, 3) == es.member_seed(5, 1, 2, 3)
    assert es.member_seed(5, 1, 2, 3) != es.member_seed(6, 1, 2, 3)
    assert all(0 <= s < 2**63 for s in seeds)
    torch.Generator().manual_seed(max(seeds))  # must not raise


# ------------------------------------------------------------- structure
def test_matrix_blocks_have_rank_at_most_r():
    es = _es()
    for r in (1, 3):
        e = es.lowrank_perturbation(11, SHAPES, rank=r)
        off = 0
        for s in SHAPES:
            n = math.prod(s)
            if len(s) >= 2:
                mat = e[off : off + n].reshape(s[0], -1)
                assert int(torch.linalg.matrix_rank(mat)) == min(r, *mat.shape)
            off += n


def test_entries_have_unit_variance():
    es = _es()
    e = torch.cat([es.lowrank_perturbation(s, [(64, 64)], rank=4) for s in range(40)])
    assert abs(float(e.mean())) < 0.05
    assert abs(float(e.var()) - 1.0) < 0.1


def test_rank_must_be_positive():
    es = _es()
    with pytest.raises(ValueError):
        es.lowrank_perturbation(1, SHAPES, rank=0)


# ------------------------------------------------------------- antithetic
def test_antithetic_pair_is_exactly_symmetric():
    es = _es()
    plus = es.antithetic_perturbation(42, 1, SHAPES, rank=2)
    minus = es.antithetic_perturbation(42, -1, SHAPES, rank=2)
    assert torch.equal(plus, -minus)
    assert torch.equal(plus + minus, torch.zeros_like(plus))
    with pytest.raises(ValueError):
        es.antithetic_perturbation(42, 0, SHAPES, rank=2)


def test_equal_pair_fitness_gives_zero_gradient():
    es = _es()
    pairs = [(es.member_seed(0, 1, 0, k), 0.3, 0.3) for k in range(6)]
    g = es.es_gradient(pairs, SHAPES, rank=2, sigma=0.1, shape_fitness=False)
    assert torch.equal(g, torch.zeros(_numel(SHAPES)))


def test_es_gradient_recovers_linear_objective_direction():
    """f(θ) = c·θ: the ES estimate must align with c (raw fitness, many pairs)."""
    es = _es()
    shapes = [(16, 8), (16,)]
    torch.manual_seed(3)
    c = torch.randn(_numel(shapes))
    sigma = 0.05
    pairs = []
    for k in range(400):
        sd = es.member_seed(9, 0, 0, k)
        e = es.lowrank_perturbation(sd, shapes, rank=1)
        pairs.append((sd, float(c @ (sigma * e)), float(c @ (-sigma * e))))
    g = es.es_gradient(pairs, shapes, rank=1, sigma=sigma, shape_fitness=False)
    cos = float(torch.nn.functional.cosine_similarity(g, c, dim=0))
    assert cos > 0.7, cos


def test_centered_ranks():
    es = _es()
    r = es.centered_ranks(torch.tensor([3.0, -1.0, 10.0, 0.0]))
    assert torch.allclose(r, torch.tensor([1 / 6, -0.5, 0.5, -1 / 6]))


# ------------------------------------------------------------- wire
def test_payload_roundtrip_and_size():
    es = _es()
    pairs = [(es.member_seed(1, 2, 3, k), -1.25 * k, 0.5 * k) for k in range(5)]
    buf = es.pack_es_payload(2, 3, pairs)
    assert len(buf) == 8 + es.PAIR_BYTES * 5 and es.PAIR_BYTES == 16
    rnd, worker, back = es.unpack_es_payload(buf)
    assert (rnd, worker) == (2, 3)
    assert back == pairs  # values chosen exactly representable in f32
    with pytest.raises(ValueError):
        es.unpack_es_payload(buf[:-1])


# ------------------------------------------------------------- sim_fed ES mode
def test_sim_fed_es_mode_runs_and_counts_bytes():
    spec = importlib.util.spec_from_file_location("sim_fed_es92", _ROOT / "sim_fed.py")
    sim = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(sim)
    Xtr, ytr, Xte, yte = sim.make_task(n_train=512, n_test=256)
    W, P, R = 4, 3, 2
    out = sim.run_es_federation(Xtr, ytr, Xte, yte, W=W, pairs=P, rounds=R, batch=32, eval_every=1)
    per_worker = 8 + 16 * P
    assert out["bytes_up"] == R * W * per_worker
    assert out["bytes_down"] == R * W * W * per_worker
    assert [h["round"] for h in out["history"]] == [1, 2]
    assert 0.0 <= out["final_acc"] <= 1.0 and math.isfinite(out["final_loss"])
    again = sim.run_es_federation(Xtr, ytr, Xte, yte, W=W, pairs=P, rounds=R, batch=32, eval_every=1)
    assert again["history"] == out["history"]  # seeded end to end
