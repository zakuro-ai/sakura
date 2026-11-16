"""Rejoin momentum policy (SAK-0014, #106).

Unit tests for ``apply_rejoin_policy``, the FederationService wiring (the
policy runs with the FULL_SYNC parameter hard-set on the training thread, and
not on other hard-sets), and a short smoke run of ``sim_rejoin.py``, the
long-run comparison whose numbers are recorded in SAK-0014.
No sockets.
"""

from __future__ import annotations

import importlib.util
import pathlib
import sys

import pytest

torch = pytest.importorskip("torch")

from sakura.services.federation.codec import FlatState  # noqa: E402
from sakura.services.federation.rejoin import (  # noqa: E402
    DEFAULT_REJOIN_POLICY,
    REJOIN_POLICIES,
    apply_rejoin_policy,
    check_policy,
)
from sakura.services.federation.service import FederationService  # noqa: E402

_ROOT = pathlib.Path(__file__).resolve().parents[2]


def _trained(opt_cls, **kw):
    torch.manual_seed(0)
    model = torch.nn.Sequential(torch.nn.Linear(4, 8), torch.nn.ReLU(), torch.nn.Linear(8, 2))
    opt = opt_cls(model.parameters(), **kw)
    for _ in range(3):
        opt.zero_grad()
        model(torch.randn(16, 4)).pow(2).mean().backward()
        opt.step()
    return model, opt


def _bufs(opt, key):
    return [st[key].clone() for st in opt.state.values() if key in st]


class TestPolicy:
    def test_default_is_noop(self):
        assert DEFAULT_REJOIN_POLICY == "noop"
        assert REJOIN_POLICIES == ("noop", "reset", "average")

    def test_noop_leaves_momentum(self):
        _, opt = _trained(torch.optim.SGD, lr=0.1, momentum=0.9)
        before = _bufs(opt, "momentum_buffer")
        assert apply_rejoin_policy(opt, "noop") == 0
        for a, b in zip(before, _bufs(opt, "momentum_buffer")):
            assert torch.equal(a, b)

    def test_reset_zeroes_sgd_momentum(self):
        _, opt = _trained(torch.optim.SGD, lr=0.1, momentum=0.9)
        assert all(b.abs().sum() > 0 for b in _bufs(opt, "momentum_buffer"))
        n = apply_rejoin_policy(opt, "reset")
        assert n == len(list(opt.state.values())) == 4
        assert all(torch.count_nonzero(b) == 0 for b in _bufs(opt, "momentum_buffer"))

    def test_average_scales_by_decay(self):
        _, opt = _trained(torch.optim.SGD, lr=0.1, momentum=0.9)
        before = _bufs(opt, "momentum_buffer")
        apply_rejoin_policy(opt, "average")  # default decay 0.5
        for a, b in zip(before, _bufs(opt, "momentum_buffer")):
            assert torch.allclose(b, 0.5 * a)
        apply_rejoin_policy(opt, "average", decay=0.25)
        for a, b in zip(before, _bufs(opt, "momentum_buffer")):
            assert torch.allclose(b, 0.125 * a)

    def test_adam_first_moment_only(self):
        _, opt = _trained(torch.optim.Adam, lr=1e-2)
        sq = _bufs(opt, "exp_avg_sq")
        steps = [st["step"].clone() for st in opt.state.values()]
        apply_rejoin_policy(opt, "reset")
        assert all(torch.count_nonzero(b) == 0 for b in _bufs(opt, "exp_avg"))
        for a, b in zip(sq, _bufs(opt, "exp_avg_sq")):
            assert torch.equal(a, b)  # second moments untouched
        for a, st in zip(steps, opt.state.values()):
            assert torch.equal(a, st["step"])

    def test_no_state_is_harmless(self):
        model = torch.nn.Linear(2, 2)
        fresh = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
        assert apply_rejoin_policy(fresh, "reset") == 0
        assert apply_rejoin_policy(None, "reset") == 0
        assert apply_rejoin_policy(object(), "average") == 0

    @pytest.mark.parametrize("policy,decay", [("zero", 0.5), ("average", 1.5), ("reset", -0.1)])
    def test_bad_arguments_raise(self, policy, decay):
        with pytest.raises(ValueError):
            check_policy(policy, decay)
        with pytest.raises(ValueError):
            FederationService("127.0.0.1", 1, "w", 8, rejoin_momentum=policy,
                              rejoin_momentum_decay=decay)


def _service(policy, opt_cls=torch.optim.SGD, **kw):
    model, opt = _trained(opt_cls, **kw)
    svc = FederationService("127.0.0.1", 1, "w", 8, rejoin_momentum=policy)
    svc.fs = FlatState(model)
    svc.anchor = torch.zeros(svc.fs.numel)
    svc._bn_pending = None
    svc._optimizer = opt
    return svc, model, opt


class TestServiceWiring:
    def test_service_default_is_noop(self):
        svc = FederationService("127.0.0.1", 1, "w", 8)
        assert svc.rejoin_momentum == "noop"

    @pytest.mark.parametrize("policy,factor", [("noop", 1.0), ("reset", 0.0), ("average", 0.5)])
    def test_fullsync_hard_set_applies_policy(self, policy, factor):
        svc, model, opt = _service(policy, lr=0.1, momentum=0.9)
        before = _bufs(opt, "momentum_buffer")
        svc._fullsync_pending = True
        svc._set_req.set()
        svc._service_round_requests()
        assert svc._set_done.is_set() and not svc._fullsync_pending
        assert all(torch.count_nonzero(p) == 0 for p in model.parameters())  # anchor = 0
        for a, b in zip(before, _bufs(opt, "momentum_buffer")):
            assert torch.allclose(b, factor * a)

    def test_resend_hard_set_keeps_momentum(self):
        # a retained-round resend also hard-sets params, but it is not a
        # FULL_SYNC: the policy must not run.
        svc, _, opt = _service("reset", lr=0.1, momentum=0.9)
        before = _bufs(opt, "momentum_buffer")
        svc._set_req.set()
        svc._service_round_requests()
        for a, b in zip(before, _bufs(opt, "momentum_buffer")):
            assert torch.equal(a, b)

    def test_on_train_begin_captures_optimizer(self, monkeypatch):
        model, opt = _trained(torch.optim.SGD, lr=0.1, momentum=0.9)
        svc = FederationService("127.0.0.1", 1, "w", 8, rejoin_momentum="reset")
        monkeypatch.setattr(svc, "_connect_and_bootstrap", lambda: None)
        monkeypatch.setattr(svc, "_bg_loop", lambda: None)

        class Ev:
            pass

        ev = Ev()
        ev.model, ev.optimizer, ev.train_loader = model, opt, None
        svc.on_train_begin(ev)
        svc._stop.set()
        assert svc._optimizer is opt


# ------------------------------------------------------------ sim smoke test


@pytest.fixture(scope="module")
def sim():
    spec = importlib.util.spec_from_file_location("sim_rejoin_106", _ROOT / "sim_rejoin.py")
    m = importlib.util.module_from_spec(spec)
    saved = {k: v for k, v in sys.modules.items() if k == "fed" or k.startswith("fed.")}
    try:
        spec.loader.exec_module(m)
    finally:
        for k in [k for k in sys.modules if k == "fed" or k.startswith("fed.")]:
            del sys.modules[k]
        sys.modules.update(saved)
    return m


def test_sim_compare_runs_single_threaded_and_restores(sim, monkeypatch):
    seen = []
    monkeypatch.setattr(sim, "_compare", lambda *a, **k: seen.append(torch.get_num_threads()) or {})
    prev = torch.get_num_threads()
    sim.compare(seeds=(1,), opts=("sgdm",), rounds=1)
    assert seen == [1]
    assert torch.get_num_threads() == prev


def test_sim_compares_three_policies(sim):
    res = sim.compare(seeds=(1,), opts=("sgdm",), rounds=12, schedule=((1, 4, 3),),
                      n_train=1024, n_test=256)
    summ = res["summary"]["sgdm"]
    assert set(summ) == {"noop", "reset", "average"}
    for s in summ.values():
        assert s["rejoin_d_loss_1"]["n"] == 1
        assert 0.0 <= s["final_acc"]["mean"] <= 1.0
    evs = {r["policy"]: r["events"][0] for r in res["runs"]}
    assert evs["noop"]["round"] == 7 and evs["noop"]["worker"] == 1
    assert evs["reset"]["momentum_norm_after"] == 0.0
    assert evs["average"]["momentum_norm_after"] == pytest.approx(
        0.5 * evs["average"]["momentum_norm_before"], rel=1e-3)
    assert evs["noop"]["momentum_norm_after"] == evs["noop"]["momentum_norm_before"]
    # paired data: identical before the rejoin round, so pre-rejoin history matches
    hist = {r["policy"]: r["history"] for r in res["runs"]}
    assert hist["noop"][:6] == hist["reset"][:6] == hist["average"][:6]
    assert "| reset |" in sim.to_markdown(res)
