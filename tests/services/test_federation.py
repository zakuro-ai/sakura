"""Federation service: codec, BN merge, wire protocol, and loopback e2e tests.

CPU-only; run with /opt/zk/bin/python -m pytest or inside the container with
the sakura-ml package installed.
"""

from __future__ import annotations

import os
import socket
import threading
import time

import pytest

torch = pytest.importorskip("torch")

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from sakura.services.federation import Leader, FederationService  # noqa: E402
from sakura.services.federation.codec import (  # noqa: E402
    BUCKET,
    FlatState,
    anchor_hash16,
    apply_round_to_anchor,
    codec_decode,
    codec_encode,
    dequantize_int8,
    merge_bn_pooled,
    quantize_int8,
)
from sakura.services.federation.transport import (  # noqa: E402
    HDR,
    MAGIC,
    PROTO_VER,
    WireDead,
    recv_frame,
)


# ------------------------------------------------------------------ helpers


class TinyNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(32, 64),
            torch.nn.BatchNorm1d(64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 8),
        )

    def forward(self, x):
        return self.net(x)


def _worker_thread(wid, port, stop_evt, out, codec="int8", die_after=None, fed_kw=None):
    from sakura.runtime import SakuraRuntime
    from sakura.adapters.ddp import DDPAdapter

    torch.manual_seed(hash(wid) % 2**31)
    model = TinyNet()
    opt = torch.optim.SGD(model.parameters(), lr=0.05, momentum=0.9)
    loss_fn = torch.nn.CrossEntropyLoss()
    g = torch.Generator().manual_seed(hash(wid) % 1000)
    x = torch.randn(16, 32, generator=g)
    y = torch.randint(0, 8, (16,), generator=g)
    rt = SakuraRuntime()
    fed = FederationService("127.0.0.1", port, wid, 16, codec=codec, **(fed_kw or {}))
    rt.install(fed)
    adapter = DDPAdapter(rt, rank=0, world_size=3)
    t_die = time.monotonic() + die_after if die_after else None
    try:
        with rt:
            adapter.on_train_begin(model=model, optimizer=opt, train_loader=None)
            step = 0
            while not stop_evt.is_set():
                if t_die and time.monotonic() > t_die:
                    for s in [fed.ctl, *fed.data.values()]:
                        try:
                            s.close()
                        except OSError:
                            pass
                    out[wid] = {"died": True, "fed": fed}
                    return
                adapter.on_train_step_begin(model=model, batch=(x, y), step=step)
                opt.zero_grad()
                loss = loss_fn(model(x), y)
                loss.backward()
                adapter.on_optimizer_step(optimizer=opt)
                opt.step()
                if step % 10 == 0:
                    adapter.on_epoch_end(
                        0,
                        model=model,
                        optimizer=opt,
                        metrics={"loss": float(loss.item())},
                    )
                step += 1
                time.sleep(0.002)
            out[wid] = {
                "loss": float(loss.item()),
                "fed": fed,
                "steps": step,
                "anchor": fed.anchor.clone(),
                "bn": fed._bn_anchor.clone(),
            }
    except Exception as e:  # noqa: BLE001
        out[wid] = {"error": repr(e)}


def _start_leader(leader, warmup_s=1.0, timeout=10.0) -> int:
    """Start ``leader.serve_forever`` in a daemon thread and return its port.

    Construct the Leader with ``port=0``: the kernel picks a free port at
    bind time and we read it back from the listening socket, so there is no
    bind-close-reuse race between parallel suites on one host.
    """
    threading.Thread(
        target=leader.serve_forever, kwargs={"warmup_s": warmup_s}, daemon=True
    ).start()
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        srv = getattr(leader, "_srv", None)
        if srv is not None:
            return srv.getsockname()[1]
        time.sleep(0.01)
    raise RuntimeError("leader did not start listening")


def _run_e2e(codec="int8", n_rounds=3, die_one=False, shards=1, fed_kw=None,
             leader_kw=None, port=0):
    leader = Leader(
        model_fn=TinyNet,
        port=port,
        expected=["a", "b", "c"],
        codec=codec,
        gap_s=1.2,
        seed=42,
        max_rounds=n_rounds,
        shards=shards,
        **(leader_kw or {}),
    )
    port = _start_leader(leader, warmup_s=1.0)
    stop = threading.Event()
    out: dict = {}
    ths = []
    for wid in ["a", "b", "c"]:
        kw = {"die_after": 2.5} if (die_one and wid == "c") else {}
        t = threading.Thread(
            target=_worker_thread,
            args=(wid, port, stop, out),
            kwargs={"codec": codec, "fed_kw": fed_kw, **kw},
            daemon=True,
        )
        t.start()
        ths.append(t)
        time.sleep(0.1)
    t0 = time.monotonic()
    while leader.round < n_rounds and time.monotonic() - t0 < 60:
        time.sleep(0.25)
    leader.stop.set()
    stop.set()
    for t in ths:
        t.join(15)
    try:
        leader._srv.close()
    except Exception:
        pass
    return leader, out


# ------------------------------------------------------------------ codec


class TestCodec:
    def test_roundtrip_error_bounded(self):
        g = torch.Generator().manual_seed(0)
        x = torch.randn(BUCKET * 7 + 123, generator=g) * 1e-3
        idx = torch.randperm(x.numel(), generator=g)[:50]
        x[idx] *= 300.0
        payload, ef, _ = quantize_int8(x, None)
        deq = dequantize_int8(payload, x.numel())
        assert float((x - deq).abs().max()) <= float(x.abs().max()) / 127 + 1e-6

    def test_ef_equals_residual(self):
        g = torch.Generator().manual_seed(0)
        x = torch.randn(BUCKET * 7 + 123, generator=g) * 1e-3
        payload, ef, _ = quantize_int8(x, None)
        deq = dequantize_int8(payload, x.numel())
        assert torch.allclose(ef, x - deq, atol=1e-7)

    def test_ef_telescoping(self):
        g = torch.Generator().manual_seed(1)
        d = torch.randn(BUCKET * 3, generator=g) * 1e-2
        d[::997] *= 100
        applied = torch.zeros_like(d)
        ef_t = None
        for _ in range(30):
            p, ef_t, _ = quantize_int8(d, ef_t)
            applied += dequantize_int8(p, d.numel())
        err = float((applied - 30 * d).norm()) / float((30 * d).norm())
        assert err < 0.02, f"rel_err={err:.4f}"

    def test_fp32raw_lossless(self):
        g = torch.Generator().manual_seed(2)
        x = torch.randn(BUCKET * 2, generator=g)
        p, _, c = codec_encode(x, None, "fp32raw")
        assert torch.equal(codec_decode(p, x.numel(), "fp32raw"), x)
        assert c == 1.0

    def test_nonfinite_raises(self):
        g = torch.Generator().manual_seed(3)
        x = torch.randn(BUCKET, generator=g)
        x[5] = float("inf")
        with pytest.raises(FloatingPointError):
            quantize_int8(x, None)


# ------------------------------------------------------------------ BN merge


class TestBNMerge:
    def test_pooled_moment_math(self):
        import torch.nn as nn

        m = nn.Sequential(
            nn.Conv2d(3, 4, 1),
            nn.BatchNorm2d(4),
            nn.Conv2d(4, 2, 1),
            nn.BatchNorm2d(2),
        )
        fs = FlatState(m)
        assert fs.bn_numel == (4 + 4 + 2 + 2)
        var_mask = fs.bn_var_mask()
        mean_mask = ~var_mask

        v1 = torch.tensor([1.4] * 4 + [1.4] * 2)
        v2 = torch.tensor([0.2] * 4 + [0.2] * 2)
        m1 = torch.tensor([1.0] * 4 + [1.0] * 2)
        m2 = torch.tensor([-1.0] * 4 + [-1.0] * 2)

        def vec(mean, var):
            out = torch.empty(fs.bn_numel)
            out[mean_mask] = mean
            out[var_mask] = var
            return out

        merged = merge_bn_pooled(
            [vec(m1, v1), vec(m2, v2)], [0.5, 0.5], mean_mask, var_mask
        )
        exp_var = 0.5 * (1.4 + 1.0) + 0.5 * (0.2 + 1.0) - 0.0
        assert torch.allclose(merged[mean_mask], torch.zeros(6), atol=1e-6)
        assert torch.allclose(merged[var_mask], torch.full((6,), exp_var), atol=1e-5)
        assert bool((merged[var_mask] > 0).all())

    def test_extreme_skew_stays_positive(self):
        import torch.nn as nn

        m = nn.Sequential(nn.BatchNorm1d(6))
        fs = FlatState(m)
        var_mask = fs.bn_var_mask()
        mean_mask = ~var_mask

        def vec(mean_val, var_val):
            out = torch.empty(fs.bn_numel)
            out[mean_mask] = mean_val
            out[var_mask] = var_val
            return out

        merged = merge_bn_pooled(
            [vec(5.0, 1e-6), vec(-5.0, 1e-6)],
            [0.99, 0.01],
            mean_mask,
            var_mask,
        )
        assert bool((merged[var_mask] >= 1e-10).all())


# ------------------------------------------------------------------ wire


class TestWireProtocol:
    def test_partial_reads(self):
        a, b = socket.socketpair()
        a.settimeout(5)
        b.settimeout(5)
        payload = os.urandom(100_000)
        hdr_and_payload = HDR.pack(MAGIC, PROTO_VER, 5, 7, len(payload)) + payload

        def dribble():
            for i in range(0, len(hdr_and_payload), 1313):
                a.sendall(hdr_and_payload[i : i + 1313])
                time.sleep(0.0005)

        t = threading.Thread(target=dribble)
        t.start()
        ftype, rnd, got = recv_frame(b, time.monotonic() + 10)
        t.join()
        assert ftype == 5 and rnd == 7 and got == payload
        a.close()
        b.close()

    def test_bad_magic_raises(self):
        a, b = socket.socketpair()
        a.settimeout(5)
        b.settimeout(5)
        a.sendall(b"\x00" * 16)
        with pytest.raises(WireDead):
            recv_frame(b, time.monotonic() + 2)
        a.close()
        b.close()


# ------------------------------------------------------------------ anchor


class TestAnchor:
    def test_bit_identical_apply(self):
        g = torch.Generator().manual_seed(3)
        anchor_a = torch.randn(BUCKET * 2, generator=g)
        anchor_b = anchor_a.clone()
        delta = torch.randn(BUCKET * 2, generator=g) * 0.01
        payload, _, _ = quantize_int8(delta, None)
        ua = apply_round_to_anchor(anchor_a, payload, "int8")
        ub = apply_round_to_anchor(anchor_b, payload, "int8")
        assert torch.equal(anchor_a, anchor_b)
        assert torch.equal(ua, ub)


# ------------------------------------------------------------------ e2e


class TestE2EFp32raw:
    def test_rounds_complete_with_consensus(self):
        leader, out = _run_e2e(codec="fp32raw", n_rounds=3)
        assert leader.round >= 3
        assert all(r["consensus_ok"] for r in leader.stats)
        alive = [w for w in ("a", "b", "c") if "error" not in out.get(w, {})]
        assert len(alive) == 3
        hashes = {w: anchor_hash16(out[w]["anchor"], out[w]["bn"]).hex() for w in alive}
        lh = anchor_hash16(leader.anchor, leader.bn).hex()
        assert sum(h == lh for h in hashes.values()) >= 2


class TestE2EInt8:
    def test_rounds_complete_with_consensus(self):
        leader, out = _run_e2e(codec="int8", n_rounds=3)
        assert leader.round >= 3
        assert all(r["consensus_ok"] for r in leader.stats)
        alive = [w for w in ("a", "b", "c") if "error" not in out.get(w, {})]
        assert len(alive) == 3
        hashes = {w: anchor_hash16(out[w]["anchor"], out[w]["bn"]).hex() for w in alive}
        lh = anchor_hash16(leader.anchor, leader.bn).hex()
        assert sum(h == lh for h in hashes.values()) >= 2

    def test_worker_failover(self):
        leader, out = _run_e2e(codec="int8", n_rounds=3, die_one=True)
        assert leader.round >= 2
        later = [r for r in leader.stats if r["round"] >= 2]
        assert any(len(r["participants"]) == 2 for r in later)

    def test_streaming_shards(self):
        leader, out = _run_e2e(codec="int8", n_rounds=4, shards=3)
        assert leader.round >= 4
        assert all(r["consensus_ok"] for r in leader.stats)
        alive = [w for w in ("a", "b", "c") if "error" not in out.get(w, {})]
        assert len(alive) == 3
        hashes = {w: anchor_hash16(out[w]["anchor"], out[w]["bn"]).hex() for w in alive}
        lh = anchor_hash16(leader.anchor, leader.bn).hex()
        assert sum(h == lh for h in hashes.values()) >= 2


class TestE2EDelayedNesterov:
    """#90: the flat leader runs the buffered (Delayed) Nesterov outer step
    and keeps bit-identical consensus with its workers."""

    def test_rejects_bad_delay(self):
        with pytest.raises(ValueError):
            Leader(model_fn=TinyNet, port=0, expected=["a"],
                   outer_opt="delayed_nesterov", outer_delay=0)

    def test_rounds_complete_with_consensus(self):
        leader, out = _run_e2e(
            codec="int8", n_rounds=4,
            leader_kw={"outer_opt": "delayed_nesterov", "outer_lr": 0.7,
                       "outer_delay": 2},
        )
        assert leader.round >= 4
        assert all(r["consensus_ok"] for r in leader.stats)
        assert leader.outer_buf is not None and leader.outer_steps[0] >= 4
        assert torch.count_nonzero(leader.outer_m) > 0  # momentum refreshed
        alive = [w for w in ("a", "b", "c") if "error" not in out.get(w, {})]
        assert len(alive) == 3
        hashes = {w: anchor_hash16(out[w]["anchor"], out[w]["bn"]).hex() for w in alive}
        lh = anchor_hash16(leader.anchor, leader.bn).hex()
        assert sum(h == lh for h in hashes.values()) >= 2


class TestStreamingDiLoCo:
    """Issue #51: int4 on the streamed shard + bounded overlap staleness."""

    def test_int4_streaming_bounded_staleness(self):
        leader, out = _run_e2e(
            codec="int4", n_rounds=4, shards=3, fed_kw={"max_stale_steps": 3}
        )
        assert leader.round >= 4
        assert all(r["consensus_ok"] for r in leader.stats)
        alive = [w for w in ("a", "b", "c") if "error" not in out.get(w, {})]
        assert len(alive) == 3
        hashes = {w: anchor_hash16(out[w]["anchor"], out[w]["bn"]).hex() for w in alive}
        lh = anchor_hash16(leader.anchor, leader.bn).hex()
        assert sum(h == lh for h in hashes.values()) >= 2
        for w in alive:
            fed = out[w]["fed"]
            # worker-side error feedback is carried for int4, like int8
            assert fed.ef is not None and float(fed.ef.abs().sum()) > 0
            shards = {tuple(r["shard"]) for r in fed.records if "shard" in r}
            assert len(shards) >= 2  # round-robin over the shard slices

    def _svc(self, bound):
        return FederationService("127.0.0.1", 1, "w", 1, max_stale_steps=bound)

    def test_stale_bound_blocks_until_round_commits(self):
        svc = self._svc(2)
        svc._round_active.set()
        svc._snap_done.set()  # snapshot taken; round in flight
        svc.on_optimizer_step(None)
        svc.on_train_step_begin(None)  # 1 step since snapshot: no wait
        svc.on_optimizer_step(None)
        t = threading.Thread(target=svc.on_train_step_begin, args=(None,))
        t.start()
        t.join(0.3)
        assert t.is_alive()  # 2 steps since snapshot: bound reached
        svc._round_active.clear()  # round commits (apply done)
        t.join(2.0)
        assert not t.is_alive()

    def test_stale_bound_resets_at_snapshot(self):
        svc = self._svc(1)
        svc._steps_since_snap = 5
        svc._round_active.set()
        svc._snap_done.set()

        class _FS:
            bn_numel = 0

            def snapshot_params_to(self, _):
                pass

            def bn_vec(self):
                return torch.empty(0)

        svc.fs = _FS()
        svc.snap = torch.empty(0)
        svc._bn_snap = torch.empty(0)
        svc._snap_req.set()  # next round's snapshot is pending
        svc.on_train_step_begin(None)  # services the snapshot, resets, returns
        assert svc._steps_since_snap == 0

    def test_unbounded_overlap_never_waits(self):
        svc = self._svc(None)
        svc._round_active.set()
        svc._snap_done.set()
        for _ in range(50):
            svc.on_optimizer_step(None)
        t0 = time.monotonic()
        svc.on_train_step_begin(None)
        assert time.monotonic() - t0 < 0.1
