"""SAK-0012 / #101: co-distillation round mode with a shared probe batch."""
from __future__ import annotations

import threading
import time

import pytest

torch = pytest.importorskip("torch")
nn = torch.nn

from sakura.services.federation.codistill import (  # noqa: E402
    CodistillLeader,
    CodistillWorker,
    decode_target,
    encode_target,
    kd_loss,
    leave_one_out_targets,
    make_probe,
    probe_indices,
)

MB = 2_000_000  # acceptance bound, decimal MB


def _wait(pred, timeout=30.0, what="condition"):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if pred():
            return
        time.sleep(0.02)
    raise AssertionError(f"timed out waiting for {what}")


# ------------------------------------------------------------------ units

def test_probe_is_deterministic_and_rng_independent():
    torch.manual_seed(1)
    a = make_probe(7, 16, (3, 4, 4))
    torch.manual_seed(2)
    b = make_probe(7, 16, (3, 4, 4))
    assert torch.equal(a, b) and a.shape == (16, 3, 4, 4)
    assert not torch.equal(a, make_probe(8, 16, (3, 4, 4)))
    idx = probe_indices(7, 10, 100)
    assert torch.equal(idx, probe_indices(7, 10, 100))
    assert len(set(idx.tolist())) == 10 and int(idx.max()) < 100
    with pytest.raises(ValueError):
        probe_indices(0, 11, 10)


def test_leave_one_out_targets():
    a, b, c = torch.full((2, 3), 1.0), torch.full((2, 3), 3.0), torch.full((2, 3), 8.0)
    t = leave_one_out_targets({"a": a, "b": b, "c": c}, ["a", "b", "c", "late"])
    assert torch.allclose(t["a"], torch.full((2, 3), 5.5))  # mean(b, c)
    assert torch.allclose(t["c"], torch.full((2, 3), 2.0))  # mean(a, b)
    assert torch.allclose(t["late"], torch.full((2, 3), 4.0))  # mean(a, b, c)
    # a lone contributor has no peer to learn from: no target, no KL term
    assert leave_one_out_targets({"a": a}, ["a"]) == {"a": None}
    assert leave_one_out_targets({}, ["a"]) == {"a": None}


@pytest.mark.parametrize("codec,tol", [("fp16", 1e-2), ("int8", 0.05)])
def test_target_codec_roundtrip(codec, tol):
    t = torch.randn(64, 100) * 4
    back = decode_target(encode_target(t, codec), 64, 100)
    assert back is not None and back.shape == t.shape
    assert float((back - t).abs().max()) <= tol * float(t.abs().max())
    assert decode_target(encode_target(None, codec), 64, 100) is None


def test_int8_target_bytes_for_the_adr_probe():
    body = encode_target(torch.zeros(512, 1000), "int8")
    assert len(body) == 1 + 512 * 4 + 512 * 1000
    assert len(encode_target(torch.zeros(512, 1000), "fp16")) == 1 + 512 * 1000 * 2
    with pytest.raises(ValueError):
        encode_target(torch.zeros(2, 2), "int4")


def test_kd_loss_zero_on_match_and_pulls_student():
    t = torch.randn(8, 5)
    assert float(kd_loss(t.clone(), t)) == pytest.approx(0.0, abs=1e-6)
    s = torch.zeros(8, 5, requires_grad=True)
    opt = torch.optim.SGD([s], lr=1.0)
    first = float(kd_loss(s, t).detach())
    for _ in range(50):
        opt.zero_grad()
        kd_loss(s, t).backward()
        opt.step()
    assert float(kd_loss(s, t).detach()) < 0.1 * first


# ------------------------------------------------------------------ e2e

class _MLP(nn.Module):
    def __init__(self, d_in, classes):
        super().__init__()
        self.net = nn.Sequential(nn.Flatten(), nn.Linear(d_in, 64), nn.ReLU(), nn.Linear(64, classes))

    def forward(self, x):
        return self.net(x)


class _CNN(nn.Module):
    def __init__(self, classes):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 8, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d(2), nn.Flatten()
        )
        self.fc = nn.Linear(32, classes)

    def forward(self, x):
        return self.fc(self.conv(x))


def _train(worker, model, steps_done, stop, alpha=1.0, rows_per_step=64):
    """Probe-only KL training loop: every step evaluates a probe slice with
    grad, adds the KL toward the latest ensemble target, publishes the full
    probe logits for the next round. Concurrent with the other worker."""
    probe = make_probe(worker.probe_seed, worker.probe_rows, (3, 8, 8))
    opt = torch.optim.SGD(model.parameters(), lr=0.5)
    g = torch.Generator().manual_seed(hash(worker.worker_id) & 0xFFFF)
    while not stop.is_set():
        idx = torch.randint(0, worker.probe_rows, (rows_per_step,), generator=g)
        tgt = worker.target()
        if tgt is not None:
            opt.zero_grad()
            (alpha * kd_loss(model(probe[idx]), tgt[idx])).backward()
            opt.step()
        with torch.no_grad():
            worker.publish(model(probe))
        steps_done[worker.worker_id] += 1
        time.sleep(0.005)


def _disagreement(m1, m2, probe):
    with torch.no_grad():
        return float(kd_loss(m1(probe), m2(probe), temperature=1.0))


def test_two_architectures_codistill_within_2mb_per_round():
    rows, classes = 512, 1000
    leader = CodistillLeader(
        0, ["mlp", "cnn"], probe_rows=rows, num_classes=classes, probe_seed=3,
        gap_s=0.3, host="127.0.0.1",
    ).start()
    workers = {
        "mlp": CodistillWorker("mlp", "127.0.0.1", leader.port, arch="mlp").start(),
        "cnn": CodistillWorker("cnn", "127.0.0.1", leader.port, arch="cnn").start(),
    }
    torch.manual_seed(0)
    models = {"mlp": _MLP(3 * 8 * 8, classes), "cnn": _CNN(classes)}
    with torch.no_grad():  # confident, disagreeing heads: something to transfer
        models["mlp"].net[-1].weight.mul_(20)
        models["cnn"].fc.weight.mul_(20)
    shapes = {k: sorted(tuple(p.shape) for p in m.parameters()) for k, m in models.items()}
    assert shapes["mlp"] != shapes["cnn"]  # weight averaging could not pair them
    probe = make_probe(3, rows, (3, 8, 8))
    before = _disagreement(models["mlp"], models["cnn"], probe)
    steps = {"mlp": 0, "cnn": 0}
    stop = threading.Event()
    threads = [
        threading.Thread(target=_train, args=(workers[k], models[k], steps, stop), daemon=True)
        for k in workers
    ]
    try:
        for t in threads:
            t.start()
        _wait(
            lambda: all(w.targets_received >= 4 for w in workers.values()),
            what="4 ensemble targets per worker",
        )
    finally:
        stop.set()
        for t in threads:
            t.join(10)
        for w in workers.values():
            w.close()
        leader.close()

    # both trained concurrently, and both learned from the other
    assert steps["mlp"] > 0 and steps["cnn"] > 0
    after = _disagreement(models["mlp"], models["cnn"], probe)
    assert after < 0.5 * before, (before, after)
    # every round any worker saw stayed within the bound, counted on the wire
    for w in workers.values():
        assert w.round_bytes
        for rb in w.round_bytes:
            assert rb["up"] + rb["down"] <= MB, rb
        full = [rb for rb in w.round_bytes if rb["up"] > 1_000_000]
        assert full and max(rb["up"] + rb["down"] for rb in full) == 1_538_097
    # leader's accounting agrees and both workers sat in the ensemble
    both = [st for st in leader.stats if st["ensemble"] == ["cnn", "mlp"]]
    assert both
    for st in leader.stats:
        for wid in st["participants"]:
            assert st["bytes_up"][wid] + st["bytes_down"][wid] <= MB


def test_target_is_peer_logits_and_stale_peer_still_teaches():
    rows, classes = 4, 3
    leader = CodistillLeader(
        0, ["a", "b"], probe_rows=rows, num_classes=classes, gap_s=3600,
        target_codec="fp16", max_stale_rounds=1, host="127.0.0.1",
    ).start()
    a = CodistillWorker("a", "127.0.0.1", leader.port).start()
    b = CodistillWorker("b", "127.0.0.1", leader.port).start()
    try:
        _wait(lambda: len(leader.conns) == 2, what="both workers connected")
        la, lb = torch.full((rows, classes), 1.0), torch.full((rows, classes), -2.0)
        a.publish(la)
        b.publish(lb)
        conns = lambda: list(leader.conns.values())  # noqa: E731
        st = leader.run_round(conns())
        assert st["fresh"] == ["a", "b"]
        _wait(lambda: a.target_round == 1 and b.target_round == 1, what="round 1 targets")
        assert torch.allclose(a.target(), lb) and torch.allclose(b.target(), la)
        # round 2: only `a` refreshes; b's round-1 logits are 1 round stale (kept)
        a.publish(la * 3)
        st = leader.run_round(conns())
        assert st["fresh"] == ["a"] and st["ensemble"] == ["a", "b"]
        _wait(lambda: a.target_round == 2 and b.target_round == 2, what="round 2 targets")
        assert torch.allclose(a.target(), lb) and torch.allclose(b.target(), la * 3)
        # an empty upload costs a header, not a matrix
        assert b.round_bytes[-1]["up"] == 16
        # round 4: b's logits are 3 rounds old > max_stale_rounds -> a gets none
        leader.run_round(conns())
        a.publish(la)
        st = leader.run_round(conns())
        assert st["ensemble"] == ["a"]
        _wait(lambda: len(a.round_bytes) == 4, what="round 4 reply")
        assert a.target_round == 2  # TGT_NONE leaves the previous target in place
    finally:
        a.close()
        b.close()
        leader.close()


def test_unknown_worker_and_wrong_shape_rejected():
    leader = CodistillLeader(0, ["a"], probe_rows=2, num_classes=2, gap_s=3600,
                             host="127.0.0.1").start()
    try:
        with pytest.raises(Exception):
            CodistillWorker("intruder", "127.0.0.1", leader.port).start()
        a = CodistillWorker("a", "127.0.0.1", leader.port).start()
        with pytest.raises(ValueError):
            a.publish(torch.zeros(3, 2))
        a.close()
    finally:
        leader.close()
