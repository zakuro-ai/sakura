"""TDD for e7: the federation outer optimizer (plain / SGD-momentum / Nesterov)
applied to the aggregated pseudo-gradient dbar. Pure function, unit-tested in
isolation (this is the math that closes gap G1 — design-only until now)."""
from __future__ import annotations

import importlib.util
import pathlib
import sys
import types

import pytest

torch = pytest.importorskip("torch")

_BASE = pathlib.Path(__file__).resolve().parents[2] / "sakura/services/federation"


def _oo():
    if "fed.outer_opt" in sys.modules:
        return sys.modules["fed.outer_opt"]
    if "fed" not in sys.modules:
        pkg = types.ModuleType("fed")
        pkg.__path__ = [str(_BASE)]
        sys.modules["fed"] = pkg
    spec = importlib.util.spec_from_file_location("fed.outer_opt", _BASE / "outer_opt.py")
    m = importlib.util.module_from_spec(spec)
    sys.modules["fed.outer_opt"] = m
    spec.loader.exec_module(m)
    return m


def test_plain_is_scaled_passthrough():
    oo = _oo()
    g = torch.randn(100, generator=torch.Generator().manual_seed(0))
    assert torch.allclose(oo.outer_step(g, None, "plain", 1.0, 0.9), g)
    assert torch.allclose(oo.outer_step(g, None, "plain", 0.5, 0.9), 0.5 * g)


def test_sgdm_first_step_then_steady_state():
    oo = _oo()
    g = torch.randn(100, generator=torch.Generator().manual_seed(1))
    m = torch.zeros_like(g)
    u1 = oo.outer_step(g, m, "sgdm", 1.0, 0.9).clone()
    assert torch.allclose(u1, g, atol=1e-6)          # m = 0.9*0 + g = g ; update = m
    assert torch.allclose(m, g, atol=1e-6)           # buffer mutated in place
    for _ in range(400):
        u = oo.outer_step(g, m, "sgdm", 1.0, 0.9)
    assert torch.allclose(u, 10.0 * g, rtol=2e-3)    # steady m = g/(1-0.9) = 10g


def test_nesterov_first_step_is_lookahead():
    oo = _oo()
    g = torch.randn(100, generator=torch.Generator().manual_seed(2))
    m = torch.zeros_like(g)
    u1 = oo.outer_step(g, m, "nesterov", 1.0, 0.9)
    # m = g ; update = dbar + mu*m = g + 0.9g = 1.9g  (accelerated vs sgdm's g)
    assert torch.allclose(u1, 1.9 * g, atol=1e-6)


def test_outer_lr_scales_update():
    oo = _oo()
    g = torch.randn(50, generator=torch.Generator().manual_seed(3))
    m1, m2 = torch.zeros_like(g), torch.zeros_like(g)
    u_full = oo.outer_step(g, m1, "sgdm", 1.0, 0.9)
    u_half = oo.outer_step(g, m2, "sgdm", 0.5, 0.9)
    assert torch.allclose(u_half, 0.5 * u_full, atol=1e-6)


def test_ef_is_buffer_free_scaled_passthrough():
    """#89 SparseLoCo: outer='ef' keeps no momentum buffer; the step itself is
    lr·dbar and the momentum lives in the decayed error-feedback residual."""
    oo = _oo()
    assert "ef" in oo.OUTER_KINDS
    assert not oo.needs_momentum_buffer("ef")
    assert not oo.needs_momentum_buffer("plain")
    assert all(oo.needs_momentum_buffer(k) for k in ("sgdm", "slowmo", "nesterov"))
    g = torch.randn(100, generator=torch.Generator().manual_seed(4))
    assert torch.allclose(oo.outer_step(g, None, "ef", 1.0, 0.9), g)
    assert torch.allclose(oo.outer_step(g, None, "ef", 0.5, 0.9), 0.5 * g)


def test_ef_decay_only_for_ef():
    oo = _oo()
    assert oo.ef_decay("ef", 0.95) == 0.95
    for k in ("plain", "sgdm", "slowmo", "nesterov"):
        assert oo.ef_decay(k, 0.95) == 1.0  # classic, undecayed error feedback


def test_ef_residual_integrates_like_momentum():
    """A coordinate the sparsifier never sends accumulates lr·dbar in the
    decayed EF residual exactly like a heavy-ball buffer (steady 1/(1-μ)
    gain), and is released in full once it is sent — no separate buffer."""
    oo = _oo()
    g = torch.randn(64, generator=torch.Generator().manual_seed(5))
    ef = torch.zeros_like(g)
    m = torch.zeros_like(g)
    for _ in range(400):
        acc = oo.outer_step(g, None, "ef", 1.0, 0.9) + oo.ef_decay("ef", 0.9) * ef
        ef = acc  # codec sent nothing: the whole accumulator is the residual
        oo.outer_step(g, m, "sgdm", 1.0, 0.9)
    assert torch.allclose(ef, m, rtol=1e-5)
    assert torch.allclose(ef, 10.0 * g, rtol=2e-3)
    # lossless codec (residual always zero) -> ef degenerates to plain at lr
    assert torch.allclose(
        oo.outer_step(g, None, "ef", 0.7, 0.9) + oo.ef_decay("ef", 0.9) * torch.zeros_like(g),
        oo.outer_step(g, None, "plain", 0.7, 0.9),
    )


def test_leader_accepts_ef_without_momentum_buffer():
    from sakura.services.federation import Leader

    def tiny():
        return torch.nn.Sequential(torch.nn.Linear(4, 3))

    ld = Leader(model_fn=tiny, port=0, expected=["a"], codec="topk2", outer_opt="ef")
    assert ld.outer_kind == "ef" and ld.outer_m is None and ld.ef_bcast is not None
    ld = Leader(model_fn=tiny, port=0, expected=["a"], codec="topk2", outer_opt="nesterov")
    assert ld.outer_m is not None
    with pytest.raises(ValueError):
        Leader(model_fn=tiny, port=0, expected=["a"], outer_opt="eff")


# ---------------------------------------------------------------- #90: HALoS
def test_delayed_nesterov_with_delay_1_equals_nesterov():
    oo = _oo()
    gen = torch.Generator().manual_seed(4)
    m_n, m_d, buf = torch.zeros(40), torch.zeros(40), torch.zeros(40)
    for t in range(1, 8):
        g = torch.randn(40, generator=gen)
        u_n = oo.outer_step(g, m_n, "nesterov", 0.7, 0.9)
        u_d = oo.outer_step(g, m_d, "delayed_nesterov", 0.7, 0.9, buf=buf, step=t, delay=1)
        assert torch.allclose(u_n, u_d, atol=1e-6)
    assert torch.allclose(m_n, m_d, atol=1e-6)


def test_delayed_nesterov_buffers_n_updates_then_refreshes_momentum():
    oo = _oo()
    gen = torch.Generator().manual_seed(5)
    gs = [torch.randn(30, generator=gen) for _ in range(6)]
    m, buf = torch.zeros(30), torch.zeros(30)
    N, mu, lr = 3, 0.9, 0.5
    # steps 1..2: momentum frozen at 0, plain lr·dbar, buffer accumulates
    for t in (1, 2):
        u = oo.outer_step(gs[t - 1], m, "delayed_nesterov", lr, mu, buf=buf, step=t, delay=N)
        assert torch.allclose(u, lr * gs[t - 1], atol=1e-6)
        assert torch.count_nonzero(m) == 0
    assert torch.allclose(buf, gs[0] + gs[1], atol=1e-6)
    # step 3: m = mean of the 3 buffered deltas; buffer cleared; nesterov look-ahead
    u3 = oo.outer_step(gs[2], m, "delayed_nesterov", lr, mu, buf=buf, step=3, delay=N)
    mean3 = (gs[0] + gs[1] + gs[2]) / 3
    assert torch.allclose(m, mean3, atol=1e-6)
    assert torch.count_nonzero(buf) == 0
    assert torch.allclose(u3, lr * (gs[2] + mu * mean3), atol=1e-6)
    # steps 4..5 frozen, step 6 refreshes: m = μ·m + mean(g4..g6)
    m_before = m.clone()
    for t in (4, 5):
        u = oo.outer_step(gs[t - 1], m, "delayed_nesterov", lr, mu, buf=buf, step=t, delay=N)
        assert torch.allclose(m, m_before)
    oo.outer_step(gs[5], m, "delayed_nesterov", lr, mu, buf=buf, step=6, delay=N)
    assert torch.allclose(m, mu * mean3 + (gs[3] + gs[4] + gs[5]) / 3, atol=1e-6)


def test_delayed_nesterov_steady_state_gain_matches_nesterov_per_window():
    """Constant pseudo-gradient g: momentum converges to g/(1-μ) as for
    nesterov, but it is applied once per N updates instead of every update."""
    oo = _oo()
    g = torch.ones(10)
    m, buf = torch.zeros(10), torch.zeros(10)
    N = 4
    for t in range(1, 4 * 400 + 1):
        u = oo.outer_step(g, m, "delayed_nesterov", 1.0, 0.9, buf=buf, step=t, delay=N)
    assert torch.allclose(m, 10.0 * g, rtol=2e-3)
    assert torch.allclose(u, g + 0.9 * m, rtol=2e-3)  # t = 1600 is a refresh step


def test_delayed_nesterov_validates_inputs():
    oo = _oo()
    g = torch.ones(5)
    with pytest.raises(ValueError):
        oo.outer_step(g, torch.zeros(5), "delayed_nesterov", 1.0, 0.9)  # no buf
    with pytest.raises(ValueError):
        oo.outer_step(g, torch.zeros(5), "delayed_nesterov", 1.0, 0.9,
                      buf=torch.zeros(5), step=1, delay=0)
    with pytest.raises(ValueError):
        oo.outer_step(g, None, "delayed_nesterov", 1.0, 0.9, buf=torch.zeros(5))
    assert "delayed_nesterov" in oo.OUTER_KINDS


def test_staleness_weight_polynomial_decay():
    oo = _oo()
    assert oo.staleness_weight(0, 0.5) == 1.0
    assert oo.staleness_weight(7, 0.0) == 1.0          # alpha=0 disables
    assert oo.staleness_weight(3, 0.5) == pytest.approx(0.5)
    assert oo.staleness_weight(3, 1.0) == pytest.approx(0.25)
    ws = [oo.staleness_weight(t, 0.5) for t in range(10)]
    assert all(a > b for a, b in zip(ws, ws[1:]))       # strictly decreasing
    with pytest.raises(ValueError):
        oo.staleness_weight(-1, 0.5)
    with pytest.raises(ValueError):
        oo.staleness_weight(1, -0.1)


def test_staleness_weighted_delta_scales_share_without_renormalising():
    oo = _oo()
    d = torch.randn(20, generator=torch.Generator().manual_seed(6))
    # fresh upload: just the sample share
    assert torch.allclose(oo.staleness_weighted_delta(d, 0.25, 0, 0.5), 0.25 * d)
    # tau=3, alpha=0.5 -> share * 0.5
    assert torch.allclose(oo.staleness_weighted_delta(d, 0.25, 3, 0.5), 0.125 * d)
    # one fresh arrival per worker sums to the sync sample-weighted mean
    deltas = [torch.randn(20, generator=torch.Generator().manual_seed(10 + i)) for i in range(4)]
    shares = [0.1, 0.2, 0.3, 0.4]
    acc = sum(oo.staleness_weighted_delta(x, s, 0) for x, s in zip(deltas, shares))
    ref = sum(s * x for x, s in zip(deltas, shares))
    assert torch.allclose(acc, ref, atol=1e-6)
