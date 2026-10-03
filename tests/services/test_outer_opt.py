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
