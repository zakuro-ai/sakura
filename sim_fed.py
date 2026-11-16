#!/usr/bin/env python3
"""e7+e8 CPU measurement harness.

Drives the REAL shipping federation primitives (FlatState, codec_encode/decode,
outer_step, apply_round_to_anchor, merge_bn_pooled) through a faithful
block-mode round on a CPU-simulated heterogeneous fleet, so the convergence and
compression numbers reflect the actual code — not a reimplementation. The
transport/consensus layers (validated separately by e3..e6 + the test suite)
are not exercised here; this isolates the *algorithmic* gains of:

  * e8 — compression codec:  int8 (shipped) vs int4 vs bucket-local top-k
  * e7 — outer optimizer:    plain (FedAvg, shipped) vs sgdm vs nesterov
  * #89 — SparseLoCo:        int2 / topk2 codecs + outer='ef' (momentum folded
    into the broadcast error-feedback accumulator) vs shipped int8/plain;
    ``python sim_fed.py --sparseloco`` runs only that sweep.
  * #90 (``--async-dn``) — async local-SGD on a 2-region fast/slow-link
    topology: staleness-weighted aggregation + delayed Nesterov vs nesterov
  * #91 (``--halos``) — three-tier HALoS (workers -> synchronous regional
    aggregators -> async staleness-weighted delayed-Nesterov root) vs flat
    DiLoCo on a 3-GPU (1:4:14) / 2-region skew topology

Loads the federation package as a synthetic module (no Rust sakura_wire build).
"""
from __future__ import annotations

import importlib.util
import json
import pathlib
import sys
import types

import torch

BASE = pathlib.Path(__file__).resolve().parent / "sakura/services/federation"


def _load():
    pkg = types.ModuleType("fed")
    pkg.__path__ = [str(BASE)]
    sys.modules["fed"] = pkg
    mods = {}
    for name in ["codec", "transport", "outer_opt", "leader"]:
        spec = importlib.util.spec_from_file_location(f"fed.{name}", BASE / f"{name}.py")
        m = importlib.util.module_from_spec(spec)
        sys.modules[f"fed.{name}"] = m
        spec.loader.exec_module(m)
        mods[name] = m
    return mods


FED = _load()
codec = FED["codec"]
oo = FED["outer_opt"]


# ----------------------------------------------------------------- model + data
def make_model(d_in=64, h=128, n_cls=8):
    return torch.nn.Sequential(
        torch.nn.Linear(d_in, h),
        torch.nn.BatchNorm1d(h),
        torch.nn.ReLU(),
        torch.nn.Linear(h, h),
        torch.nn.BatchNorm1d(h),
        torch.nn.ReLU(),
        torch.nn.Linear(h, n_cls),
    )


def make_task(d_in=64, n_cls=8, n_train=8192, n_test=2048, seed=0):
    """Fixed random teacher MLP generates noiseless argmax labels — learnable,
    deterministic, CPU-fast, and hard enough that the outer optimizer matters."""
    g = torch.Generator().manual_seed(seed)
    teacher = make_model(d_in, 128, n_cls)
    teacher.eval()
    for p in teacher.parameters():
        p.data = torch.randn(p.shape, generator=g)
    with torch.no_grad():
        Xtr = torch.randn(n_train, d_in, generator=g)
        Xte = torch.randn(n_test, d_in, generator=g)
        ytr = teacher(Xtr).argmax(1)
        yte = teacher(Xte).argmax(1)
    return Xtr, ytr, Xte, yte


def evaluate(model, fs, anchor, bn, Xte, yte):
    fs.set_params_from(anchor)
    fs.set_bn_from(bn)
    model.eval()
    with torch.no_grad():
        loss = torch.nn.functional.cross_entropy(model(Xte), yte).item()
        acc = (model(Xte).argmax(1) == yte).float().mean().item()
    return acc, loss


# ----------------------------------------------------------------- federation
def run_federation(kind, outer, Xtr, ytr, Xte, yte, W=4, H=10, rounds=60,
                   batch=64, lr=0.1, outer_lr=0.7, mu=0.9, seed=1):
    """Faithful block-mode federation using the shipping primitives. Returns
    history + total bytes. Heterogeneous shards (uneven sizes) like the real
    56:1-skew fleet, IID label distribution."""
    torch.manual_seed(seed)
    leader_model = make_model()
    fs = FED["codec"].FlatState(leader_model)
    anchor = torch.empty(fs.numel)
    fs.snapshot_params_to(anchor)
    bn = fs.bn_vec()
    var_mask = fs.bn_var_mask()
    mean_mask = ~var_mask
    lossy = codec.is_lossy(kind)
    ef_bcast = torch.zeros(fs.numel) if lossy else None
    outer_m = torch.zeros(fs.numel) if oo.needs_momentum_buffer(outer) else None

    # heterogeneous shards (sizes ~ 1:2:4:... ); IID labels
    g = torch.Generator().manual_seed(seed + 7)
    perm = torch.randperm(Xtr.shape[0], generator=g)
    sizes = torch.tensor([1.0, 2.0, 4.0, 8.0][:W])
    sizes = (sizes / sizes.sum() * Xtr.shape[0]).long()
    sizes[-1] += Xtr.shape[0] - int(sizes.sum())
    shards, off = [], 0
    for s in sizes:
        idx = perm[off:off + int(s)]
        shards.append((Xtr[idx], ytr[idx]))
        off += int(s)

    workers = [make_model() for _ in range(W)]
    wfs = [FED["codec"].FlatState(m) for m in workers]
    worker_ef = [None] * W
    ema_n = [0.0] * W
    eval_model = make_model()
    eval_fs = FED["codec"].FlatState(eval_model)

    bytes_up = bytes_down = 0
    hist = []
    flat = torch.empty(fs.numel)
    for r in range(1, rounds + 1):
        payloads, bnvecs, ns = [], [], []
        for w in range(W):
            wfs[w].set_params_from(anchor)   # block mode: start from consensus
            wfs[w].set_bn_from(bn)
            m, (Xs, ys) = workers[w], shards[w]
            opt = torch.optim.SGD(m.parameters(), lr=lr, momentum=0.9)
            m.train()
            for _ in range(H):
                bi = torch.randint(0, Xs.shape[0], (min(batch, Xs.shape[0]),))
                opt.zero_grad()
                loss = torch.nn.functional.cross_entropy(m(Xs[bi]), ys[bi])
                loss.backward()
                opt.step()
            wfs[w].snapshot_params_to(flat)
            delta = flat - anchor
            payload, worker_ef[w], _ = codec.codec_encode(delta, worker_ef[w], kind)
            payloads.append(payload)
            bnvecs.append(wfs[w].bn_vec())
            ns.append(float(Xs.shape[0]))
            bytes_up += len(payload)
        # ---- leader aggregation (mirrors leader.py) ----
        for w in range(W):
            ema_n[w] = ns[w] if ema_n[w] == 0 else 0.8 * ema_n[w] + 0.2 * ns[w]
        wsum = sum(ema_n)
        dbar = torch.zeros(fs.numel)
        for w in range(W):
            dbar.add_(codec.codec_decode(payloads[w], fs.numel, kind), alpha=ema_n[w] / wsum)
        m_slice = outer_m if outer_m is not None else None
        update = oo.outer_step(dbar, m_slice, outer, outer_lr if outer != "plain" else 1.0, mu)
        if ef_bcast is not None:
            update = update + oo.ef_decay(outer, mu) * ef_bcast  # outer='ef': decayed
            payload_b, ef_bcast, _ = codec.codec_encode(update, None, kind)
        else:
            payload_b, _, _ = codec.codec_encode(update, None, kind)
        bytes_down += len(payload_b) * W
        codec.apply_round_to_anchor(anchor, payload_b, kind)
        bn = codec.merge_bn_pooled(bnvecs, ema_n, mean_mask, var_mask)
        if r % 3 == 0 or r == rounds:
            acc, ls = evaluate(eval_model, eval_fs, anchor, bn, Xte, yte)
            hist.append({"round": r, "acc": round(acc, 4), "loss": round(ls, 4)})
    return {"history": hist, "final_acc": hist[-1]["acc"], "final_loss": hist[-1]["loss"],
            "bytes_up": bytes_up, "bytes_down": bytes_down}


def run_solo(Xtr, ytr, Xte, yte, steps, batch=64, lr=0.1, seed=1):
    """Single worker doing the same number of *sequential* SGD steps a federated
    worker does (H*rounds) — the e6-style honesty control."""
    torch.manual_seed(seed)
    m = make_model()
    fs = FED["codec"].FlatState(m)
    opt = torch.optim.SGD(m.parameters(), lr=lr, momentum=0.9)
    m.train()
    for _ in range(steps):
        bi = torch.randint(0, Xtr.shape[0], (batch,))
        opt.zero_grad()
        torch.nn.functional.cross_entropy(m(Xtr[bi]), ytr[bi]).backward()
        opt.step()
    m.eval()
    with torch.no_grad():
        acc = (m(Xte).argmax(1) == yte).float().mean().item()
    return round(acc, 4)


def compression_ratio_table(numel=25_500_000, seed=0):
    """Exact bytes-on-wire for a ResNet-50-sized (~25.5M) heavy-tailed delta."""
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(numel, generator=g) * 1e-3
    x[torch.randperm(numel, generator=g)[: numel // 500]] *= 200.0  # sparse outliers
    fp32 = numel * 4
    rows = []
    for kind in ["int8", "int4", "int2", "topk:0.1", "topk:0.05", "topk:0.01", "topk2"]:
        payload, _, contraction = codec.codec_encode(x, None, kind)
        rows.append({"codec": kind, "bytes": len(payload),
                     "MB": round(len(payload) / 1e6, 3),
                     "ratio_vs_fp32": round(fp32 / len(payload), 1),
                     "contraction": round(contraction, 4)})
    return {"numel": numel, "fp32_MB": round(fp32 / 1e6, 1), "rows": rows}


def _rounds_to(history, target):
    for h in history:
        if h["acc"] >= target:
            return h["round"]
    return None


def _mean_std(xs):
    xs = [x for x in xs if x is not None]
    if not xs:
        return None, None
    mu = sum(xs) / len(xs)
    sd = (sum((x - mu) ** 2 for x in xs) / len(xs)) ** 0.5
    return mu, sd


# ----------------------------------------------------- streaming DiLoCo (#51)
def run_streaming(kind, Xtr, ytr, Xte, yte, W=4, H=20, shards=1, tau=0,
                  total_steps=600, batch=64, lr=0.1, outer="plain", outer_lr=1.0,
                  mu=0.9, eval_every=20, seed=1):
    """Streaming-DiLoCo round loop with the SHIPPING worker semantics of
    FederationService (persistent local optimizer, live += new_anchor - snap
    on the synced slice) and leader semantics of Leader (codec_decode, ema_n
    weights, outer_step, broadcast-side EF, apply_round_to_anchor on [lo, hi)).

    * shards=K: a round runs every H/K inner steps and syncs only shard
      (r-1) mod K (codec.shard_bounds), so each shard is still synced every H
      steps and total bytes match block mode at equal codec — only the PEAK
      per round drops by K.
    * tau: overlap staleness. The merge of round r is applied tau inner steps
      into round r+1 (training continues meanwhile) — FederationService's
      max_stale_steps bound. tau=0 is block mode. tau must stay < H/K: at
      tau == H/K final acc dropped 6-9 pp (H=20, K=4, tau=5), while tau < H/K
      showed no loss. Likely cause: the apply (which hard-sets BN to the
      merge) lands right before the next snapshot, so every worker re-uploads
      the merged BN and the BN running stats stop moving.
    shards=1, tau=0 is exactly block mode.
    """
    stride = H // shards
    assert stride * shards == H, "H must be divisible by shards"
    assert tau == 0 or tau < stride, "overlap staleness must stay below H/shards"
    rounds = total_steps // stride
    torch.manual_seed(seed)
    leader_model = make_model()
    fs = codec.FlatState(leader_model)
    n = fs.numel
    anchor = torch.empty(n)
    fs.snapshot_params_to(anchor)
    bn = fs.bn_vec()
    bn_bytes = bn.numel() * 4
    var_mask = fs.bn_var_mask()
    mean_mask = ~var_mask
    lossy = codec.is_lossy(kind)
    ef_bcast = torch.zeros(n) if lossy else None
    outer_m = torch.zeros(n) if outer != "plain" else None

    g = torch.Generator().manual_seed(seed + 7)
    perm = torch.randperm(Xtr.shape[0], generator=g)
    sizes = torch.tensor([1.0, 2.0, 4.0, 8.0][:W])
    sizes = (sizes / sizes.sum() * Xtr.shape[0]).long()
    sizes[-1] += Xtr.shape[0] - int(sizes.sum())
    data, off = [], 0
    for sz in sizes:
        idx = perm[off:off + int(sz)]
        data.append((Xtr[idx], ytr[idx]))
        off += int(sz)

    workers = [make_model() for _ in range(W)]
    wfs = [codec.FlatState(m) for m in workers]
    for f in wfs:  # bootstrap FULL_SYNC
        f.set_params_from(anchor)
        f.set_bn_from(bn)
    opts = [torch.optim.SGD(m.parameters(), lr=lr, momentum=0.9) for m in workers]
    worker_ef = [torch.zeros(n) if lossy else None for _ in range(W)]
    ema_n = [0.0] * W
    eval_model = make_model()
    eval_fs = codec.FlatState(eval_model)

    def train(w, steps):
        m, (Xs, ys) = workers[w], data[w]
        m.train()
        for _ in range(steps):
            bi = torch.randint(0, Xs.shape[0], (min(batch, Xs.shape[0]),))
            opts[w].zero_grad()
            torch.nn.functional.cross_entropy(m(Xs[bi]), ys[bi]).backward()
            opts[w].step()

    pending = None  # (lo, hi, upd, [delta_slice per worker], bn_merged)
    apply_vec = torch.zeros(n)

    def apply_pending(w):
        lo, hi, upd, deltas, bn_m = pending
        apply_vec.zero_()
        torch.sub(upd, deltas[w], out=apply_vec[lo:hi])
        wfs[w].add_params_from(apply_vec)
        wfs[w].set_bn_from(bn_m)

    peak_round = bytes_total = 0
    hist = []
    flat = torch.empty(n)
    for r in range(1, rounds + 1):
        lo, hi = codec.shard_bounds(r, n, shards)
        payloads, deltas, bnvecs, ns = [], [], [], []
        for w in range(W):
            if pending is not None:
                train(w, tau)            # comm of round r-1 overlaps these
                apply_pending(w)
                train(w, stride - tau)
            else:
                train(w, stride)
            wfs[w].snapshot_params_to(flat)
            delta = (flat[lo:hi] - anchor[lo:hi]).clone()
            ef_w = worker_ef[w][lo:hi] if worker_ef[w] is not None else None
            payload, ef_new, _ = codec.codec_encode(delta, ef_w, kind)
            if worker_ef[w] is not None:
                worker_ef[w][lo:hi].copy_(ef_new)
            payloads.append(payload)
            deltas.append(delta)
            bnvecs.append(wfs[w].bn_vec())
            ns.append(float(data[w][0].shape[0]))
        for w in range(W):
            ema_n[w] = ns[w] if ema_n[w] == 0 else 0.8 * ema_n[w] + 0.2 * ns[w]
        wsum = sum(ema_n)
        dbar = torch.zeros(hi - lo)
        for w in range(W):
            dbar.add_(codec.codec_decode(payloads[w], hi - lo, kind), alpha=ema_n[w] / wsum)
        m_slice = outer_m[lo:hi] if outer_m is not None else None
        update = oo.outer_step(dbar, m_slice, outer, outer_lr, mu)
        if ef_bcast is not None:
            update = update + ef_bcast[lo:hi]
            payload_b, ef_new, _ = codec.codec_encode(update, None, kind)
            ef_bcast[lo:hi].copy_(ef_new)
        else:
            payload_b, _, _ = codec.codec_encode(update, None, kind)
        upd = codec.apply_round_to_anchor(anchor, payload_b, kind, lo, hi)
        bn = codec.merge_bn_pooled(bnvecs, ema_n, mean_mask, var_mask)
        pending = (lo, hi, upd, deltas, bn)
        # per-worker wire bytes this round: shard upload + BN sideband up,
        # shard broadcast + merged BN down (the BN sideband ships whole).
        per_worker = max(len(p) for p in payloads) + len(payload_b) + 2 * bn_bytes
        peak_round = max(peak_round, per_worker)
        bytes_total += sum(len(p) for p in payloads) + W * len(payload_b) + 2 * W * bn_bytes
        step = r * stride
        if step % eval_every == 0:
            acc, ls = evaluate(eval_model, eval_fs, anchor, bn, Xte, yte)
            hist.append({"step": step, "acc": round(acc, 4), "loss": round(ls, 4)})
    return {"history": hist, "final_acc": hist[-1]["acc"], "final_loss": hist[-1]["loss"],
            "peak_round_bytes": peak_round, "bytes_total": bytes_total,
            "rounds": rounds, "numel": n, "bn_bytes": bn_bytes}


def _steps_to(history, target):
    for h in history:
        if h["acc"] >= target:
            return h["step"]
    return None


def streaming_vs_block(seeds=(1, 2, 3, 4, 5), H=40, total_steps=800, eval_every=40):
    """Issue #51 acceptance: streaming (K=4 shards, int4 streamed shard,
    tau<=5 overlap) vs block (K=1, int8, tau=0) on the same task, seeds and
    compute budget. Peak bytes/round is exact; time-to-target is measured in
    inner steps (= wall time when the comm is fully hidden by the overlap)."""
    Xtr, ytr, Xte, yte = make_task()
    solo = _mean_std([run_solo(Xtr, ytr, Xte, yte, steps=total_steps, seed=s)
                      for s in seeds])[0]
    target = round(0.95 * solo, 3)
    configs = [
        ("block int8", "int8", 1, 0),
        ("stream int8 K=4 tau=5", "int8", 4, 5),
        ("stream int4 K=4 tau=0", "int4", 4, 0),
        ("stream int4 K=4 tau=5", "int4", 4, 5),
    ]
    rows = []
    print(f"=== #51 streaming vs block: 4 workers (1:2:4:8 shards), H={H}, "
          f"{total_steps} inner steps, {len(seeds)} seeds; target={target} acc "
          f"(0.95 x solo {solo:.4f}) ===")
    print(f"{'config':<24}{'final_acc':>14}{'steps->tgt':>16}{'peak_B/round':>14}"
          f"{'peak_x':>8}{'total_KB':>10}")
    base_peak = None
    for name, kind, K, tau in configs:
        runs = [run_streaming(kind, Xtr, ytr, Xte, yte, H=H, shards=K, tau=tau,
                              total_steps=total_steps, eval_every=eval_every, seed=s)
                for s in seeds]
        acc_m, acc_s = _mean_std([r["final_acc"] for r in runs])
        stt = [_steps_to(r["history"], target) for r in runs]
        stt_m, stt_s = _mean_std(stt)
        hit = sum(x is not None for x in stt)
        peak = runs[0]["peak_round_bytes"]
        base_peak = base_peak or peak
        bn_b = runs[0]["bn_bytes"]
        row = {"config": name, "codec": kind, "shards": K, "tau": tau,
               "final_acc_mean": round(acc_m, 4), "final_acc_std": round(acc_s, 4),
               "steps_to_target_mean": stt_m, "steps_to_target_std": stt_s,
               "steps_to_target": stt, "hit": hit,
               "peak_round_bytes": peak, "peak_reduction_x": round(base_peak / peak, 2),
               "peak_round_bytes_params_only": peak - 2 * bn_b,
               "bytes_total": runs[0]["bytes_total"]}
        rows.append(row)
        spd = (f"{stt_m:.0f}±{stt_s:.0f}({hit}/{len(seeds)})" if stt_m is not None
               else f"-({hit}/{len(seeds)})")
        print(f"{name:<24}{acc_m:>8.4f}±{acc_s:.3f}{spd:>16}{peak:>14}"
              f"{base_peak / peak:>7.2f}x{runs[0]['bytes_total'] / 1e3:>10.1f}")
    # the same codec/shard arithmetic at ResNet-50 scale (BN sideband ~0.2%)
    big = compression_peak_at_scale()
    print(f"ResNet-50-sized ({big['numel']:,} params) per-worker peak bytes/round: "
          f"block int8 {big['block_MB']} MB vs stream int4 K=4 {big['stream_MB']} MB "
          f"= {big['reduction_x']}x")
    return {"H": H, "total_steps": total_steps, "eval_every": eval_every,
            "seeds": list(seeds), "solo_acc": solo,
            "target": target, "rows": rows, "resnet50_scale": big}


def compression_peak_at_scale(numel=25_500_000, shards=4, seed=0):
    """Exact per-worker up+down payload bytes for one round at ResNet-50 size."""
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(numel, generator=g) * 1e-3
    block, _, _ = codec.codec_encode(x, None, "int8")
    lo, hi = codec.shard_bounds(1, numel, shards)
    stream, _, _ = codec.codec_encode(x[lo:hi].clone(), None, "int4")
    return {"numel": numel, "block_MB": round(2 * len(block) / 1e6, 2),
            "stream_MB": round(2 * len(stream) / 1e6, 2),
            "reduction_x": round(len(block) / len(stream), 2)}


# ------------------------------------------------- #90: async + delayed Nesterov
# HALoS (arXiv 2506.04531) flat-tier slice: asynchronous local-SGD where the
# leader commits each upload the moment it arrives, staleness-weighted, with a
# (Delayed) Nesterov outer step. Time is simulated (event queue), so the fast /
# slow links of a 2-region topology shape *which* deltas are stale and by how
# much, while the math runs through the shipping outer_opt / codec functions.
class Topology:
    """Workers tagged with a region and a local speed (steps per sim-second);
    links between regions as (one-way latency s, bandwidth MB/s). The leader
    sits in ``leader_region``; a worker's transfers use the link between its
    region and the leader's."""

    def __init__(self, regions, speed, leader_region, links):
        assert len(regions) == len(speed)
        self.regions = list(regions)
        self.speed = list(speed)
        self.leader_region = leader_region
        self.links = {tuple(sorted(k)): v for k, v in links.items()}

    @property
    def W(self):
        return len(self.regions)

    def link(self, w):
        return self.links[tuple(sorted((self.regions[w], self.leader_region)))]

    def xfer_s(self, w, nbytes):
        lat, mbps = self.link(w)
        return lat + nbytes / (mbps * 1e6)

    def lan_xfer_s(self, w, nbytes):
        """Worker ``w`` to an aggregator in its OWN region (#91)."""
        lat, mbps = self.links[(self.regions[w], self.regions[w])]
        return lat + nbytes / (mbps * 1e6)

    def region_xfer_s(self, region, nbytes):
        """Region aggregator to the root, which sits in ``leader_region``."""
        lat, mbps = self.links[tuple(sorted((region, self.leader_region)))]
        return lat + nbytes / (mbps * 1e6)


def two_region_topology():
    """Region A hosts the leader + 2 workers on a fast LAN; region B's 2
    workers reach it over a slow WAN link. Speeds are heterogeneous (one slow
    GPU per region) so arrivals interleave and staleness is uneven."""
    return Topology(
        regions=["A", "A", "B", "B"],
        speed=[100.0, 60.0, 100.0, 40.0],
        leader_region="A",
        links={("A", "A"): (0.0005, 1000.0), ("A", "B"): (0.04, 2.0)},
    )


def sync_round_time(topo, numel, up_bytes, H):
    """Block-mode round on the same topology: the slowest worker's
    download + H local steps + upload gates every round."""
    return max(topo.xfer_s(w, numel * 4) + H / topo.speed[w] + topo.xfer_s(w, up_bytes)
               for w in range(topo.W))


def run_async_federation(kind, outer, Xtr, ytr, Xte, yte, topo, H=10, rounds=60,
                         batch=64, lr=0.1, outer_lr=0.7, mu=0.9, delay=None,
                         stale_alpha=0.5, seed=1):
    """Async local-SGD: each worker pulls the fp32 anchor, runs H local steps,
    uploads its codec-compressed delta; the leader applies it on arrival as
    one global update (``staleness_weighted_delta`` -> ``outer_step``). Runs
    ``rounds * W`` global updates (= the sync run's upload count). ``delay``
    (N) defaults to W for ``delayed_nesterov``."""
    import heapq

    torch.manual_seed(seed)
    W = topo.W
    delay = delay or W
    leader_model = make_model()
    fs = FED["codec"].FlatState(leader_model)
    anchor = torch.empty(fs.numel)
    fs.snapshot_params_to(anchor)
    bn = fs.bn_vec()
    var_mask = fs.bn_var_mask()
    mean_mask = ~var_mask
    outer_m = torch.zeros(fs.numel) if oo.needs_momentum_buffer(outer) else None
    outer_buf = torch.zeros(fs.numel) if outer == "delayed_nesterov" else None

    g = torch.Generator().manual_seed(seed + 7)
    perm = torch.randperm(Xtr.shape[0], generator=g)
    sizes = torch.tensor([1.0, 2.0, 4.0, 8.0][:W])
    sizes = (sizes / sizes.sum() * Xtr.shape[0]).long()
    sizes[-1] += Xtr.shape[0] - int(sizes.sum())
    shards, off = [], 0
    for s_ in sizes:
        idx = perm[off:off + int(s_)]
        shards.append((Xtr[idx], ytr[idx]))
        off += int(s_)
    ns = [float(x.shape[0]) for x, _ in shards]
    nsum = sum(ns)

    workers = [make_model() for _ in range(W)]
    wfs = [FED["codec"].FlatState(m) for m in workers]
    worker_ef = [None] * W
    last_bn = [None] * W
    eval_model = make_model()
    eval_fs = FED["codec"].FlatState(eval_model)
    flat = torch.empty(fs.numel)

    def start(w, t_send, version):
        """Leader sends the anchor at t_send; worker trains; returns the
        upload's arrival event (payload computed now — the simulated clock
        only decides ordering and staleness)."""
        wfs[w].set_params_from(anchor)
        wfs[w].set_bn_from(bn)
        m, (Xs, ys) = workers[w], shards[w]
        opt = torch.optim.SGD(m.parameters(), lr=lr, momentum=0.9)
        m.train()
        for _ in range(H):
            bi = torch.randint(0, Xs.shape[0], (min(batch, Xs.shape[0]),))
            opt.zero_grad()
            torch.nn.functional.cross_entropy(m(Xs[bi]), ys[bi]).backward()
            opt.step()
        wfs[w].snapshot_params_to(flat)
        payload, worker_ef[w], _ = codec.codec_encode(flat - anchor, worker_ef[w], kind)
        t_arr = (t_send + topo.xfer_s(w, fs.numel * 4) + H / topo.speed[w]
                 + topo.xfer_s(w, len(payload)))
        return (t_arr, w, version, payload, wfs[w].bn_vec())

    version = 0
    q = []
    for w in range(W):
        ev = start(w, 0.0, version)
        heapq.heappush(q, (ev[0], ev[1], ev))
    hist, taus, bytes_up = [], [], 0
    t_now = 0.0
    for step in range(1, rounds * W + 1):
        t_now, w, ev = heapq.heappop(q)
        _, _, base, payload, wbn = ev
        bytes_up += len(payload)
        tau = version - base
        taus.append(tau)
        delta = codec.codec_decode(payload, fs.numel, kind)
        dbar = oo.staleness_weighted_delta(delta, ns[w] / nsum, tau, stale_alpha)
        update = oo.outer_step(dbar, outer_m, outer, outer_lr, mu,
                               buf=outer_buf, step=step, delay=delay)
        anchor.add_(update)
        version += 1
        last_bn[w] = wbn
        have = [i for i in range(W) if last_bn[i] is not None]
        bn = codec.merge_bn_pooled([last_bn[i] for i in have], [ns[i] for i in have],
                                   mean_mask, var_mask)
        if step % (3 * W) == 0 or step == rounds * W:
            acc, ls = evaluate(eval_model, eval_fs, anchor, bn, Xte, yte)
            hist.append({"round": step // W, "t": round(t_now, 3),
                         "acc": round(acc, 4), "loss": round(ls, 4)})
        ev = start(w, t_now, version)
        heapq.heappush(q, (ev[0], ev[1], ev))
    return {"history": hist, "final_acc": hist[-1]["acc"], "final_loss": hist[-1]["loss"],
            "sim_time_s": round(t_now, 3), "bytes_up": bytes_up,
            "tau_mean": round(sum(taus) / len(taus), 3), "tau_max": max(taus)}


def main_async(seeds=(1, 2, 3), H=10, R=60):
    """Sync Nesterov (block mode) vs async local-SGD with nesterov /
    delayed_nesterov on the 2-region topology; best outer-lr per arm."""
    out = pathlib.Path(__file__).resolve().parent / "results/e90_async_dn"
    out.mkdir(parents=True, exist_ok=True)
    topo = two_region_topology()
    Xtr, ytr, Xte, yte = make_task()
    numel = FED["codec"].FlatState(make_model()).numel
    up_bytes = len(codec.codec_encode(torch.zeros(numel), None, "int8")[0])
    t_round = sync_round_time(topo, numel, up_bytes, H)
    print(f"=== #90 async local-SGD, 2-region topology, {R}x{topo.W} global updates,"
          f" H={H}, int8, mean of {len(seeds)} seeds ===")
    print(f"regions={topo.regions} speed={topo.speed} links={topo.links}")
    print(f"sync round time = {t_round:.3f} sim-s -> {R} rounds = {R * t_round:.2f} sim-s")
    arms = [  # (label, mode, outer, stale_alpha, outer_lr grid)
        ("sync nesterov", "sync", "nesterov", 0.0, (0.3, 0.5, 0.7)),
        ("async plain a=0.5", "async", "plain", 0.5, (1.0, 2.0, 4.0)),
        ("async nesterov a=0", "async", "nesterov", 0.0, (0.3, 0.5, 0.7)),
        ("async nesterov a=0.5", "async", "nesterov", 0.5, (0.3, 0.5, 0.7)),
        ("async delayed_nesterov a=0", "async", "delayed_nesterov", 0.0, (0.7, 1.0, 2.0)),
        ("async delayed_nesterov a=0.5", "async", "delayed_nesterov", 0.5, (0.7, 1.0, 2.0)),
    ]
    results = {"topology": {"regions": topo.regions, "speed": topo.speed,
                            "links": {"-".join(k): v for k, v in topo.links.items()}},
               "H": H, "rounds": R, "seeds": list(seeds),
               "sync_round_time_s": t_round, "arms": []}
    print(f"\n{'arm':<31}{'olr':>5}{'final_acc':>14}{'sim_s':>8}{'tau_mean':>9}{'tau_max':>8}")
    for label, mode, outer, alpha, grid in arms:
        best = None
        for olr in grid:
            if mode == "sync":
                runs = [run_federation("int8", outer, Xtr, ytr, Xte, yte, H=H, rounds=R,
                                       outer_lr=olr, seed=s) for s in seeds]
                for r in runs:
                    r.update(sim_time_s=round(R * t_round, 3), tau_mean=0.0, tau_max=0)
            else:
                runs = [run_async_federation("int8", outer, Xtr, ytr, Xte, yte, topo, H=H,
                                             rounds=R, outer_lr=olr, stale_alpha=alpha,
                                             seed=s) for s in seeds]
            acc_m, acc_s = _mean_std([r["final_acc"] for r in runs])
            row = {"arm": label, "mode": mode, "outer": outer, "stale_alpha": alpha,
                   "outer_lr": olr, "final_acc_mean": round(acc_m, 4),
                   "final_acc_std": round(acc_s, 4),
                   "sim_time_s": _mean_std([r["sim_time_s"] for r in runs])[0],
                   "tau_mean": _mean_std([r["tau_mean"] for r in runs])[0],
                   "tau_max": max(r["tau_max"] for r in runs)}
            results["arms"].append(row)
            if best is None or row["final_acc_mean"] > best["final_acc_mean"]:
                best = row
        print(f"{label:<31}{best['outer_lr']:>5}{best['final_acc_mean']:>8.4f}"
              f"±{best['final_acc_std']:.3f}{best['sim_time_s']:>8.2f}"
              f"{best['tau_mean']:>9.2f}{best['tau_max']:>8}")
    (out / "results.json").write_text(json.dumps(results, indent=2))
    print(f"\nsaved -> {out/'results.json'}")


# ------------------------------- #91: three-tier HALoS vs flat DiLoCo (#52)
# Tier 1: workers. Tier 2: one regional aggregator per region running
# synchronous block-mode rounds on its in-region LAN with the shipping Leader
# round math (RegionAggregator / SAK-0011). Tier 3: a root that commits each
# region's drift the moment it arrives, staleness-weighted, through the
# delayed-Nesterov outer step (#90). A region never waits on the WAN: it keeps
# running regional rounds while its drift and the root's reply are in flight,
# then reconciles with the SAK-0011 rule (regional anchor += U - D, i.e.
# new global anchor + progress made since the upload snapshot).
# Flat DiLoCo is the star Leader in block mode on the same simulated clock:
# every round waits for the slowest worker's WAN round trip. Both arms are
# time-paced like FederationService (a worker runs speed x period local steps
# per round), so the 1:4:14 GPU skew sets each worker's sample share, not the
# round length. The real RegionAggregator is not run over sockets here: the
# transport and two-level consensus are covered by
# tests/services/test_federation_region.py; this isolates the algorithm.
SKEW_SPEED = {"4090": 140.0, "2080ti": 40.0, "p620": 10.0}  # steps/sim-s, R50 1:4:14


def skew_topology(layout="fleet"):
    """3-GPU skew fleet on 2 regions; the root sits in region A.

    ``fleet``: A = 4090 + 2080 Ti (one host), B = P620 behind the WAN.
    ``split``: A = 4090, B = 2080 Ti + P620 (a 2-worker region behind the WAN).
    Links: 1 GB/s / 0.5 ms in-region, 2 MB/s / 40 ms between regions (#90)."""
    regions = {"fleet": ["A", "A", "B"], "split": ["A", "B", "B"]}[layout]
    return Topology(regions=regions, speed=list(SKEW_SPEED.values()), leader_region="A",
                    links={("A", "A"): (0.0005, 1000.0), ("B", "B"): (0.0005, 1000.0),
                           ("A", "B"): (0.04, 2.0)})


class _Fleet:
    """Per-worker model, data shard (size proportional to speed, so a shard is
    covered at the same epoch rate on every GPU) and upload error feedback."""

    def __init__(self, topo, Xtr, ytr, kind, batch, lr, seed):
        self.kind, self.batch, self.lr = kind, batch, lr
        g = torch.Generator().manual_seed(seed + 7)
        perm = torch.randperm(Xtr.shape[0], generator=g)
        tot = sum(topo.speed)
        sizes = [int(Xtr.shape[0] * s / tot) for s in topo.speed]
        sizes[0] += Xtr.shape[0] - sum(sizes)
        self.shards, off = [], 0
        for sz in sizes:
            idx = perm[off:off + sz]
            self.shards.append((Xtr[idx], ytr[idx]))
            off += sz
        self.share = [s / tot for s in topo.speed]
        self.models = [make_model() for _ in range(topo.W)]
        self.fs = [codec.FlatState(m) for m in self.models]
        self.ef = [None] * topo.W
        self.flat = torch.empty(self.fs[0].numel)

    def train(self, w, anchor, bn, steps):
        """Block mode: start from ``anchor``, run ``steps`` local SGD steps,
        return the codec-encoded delta and the worker's BN vector."""
        f, m, (Xs, ys) = self.fs[w], self.models[w], self.shards[w]
        f.set_params_from(anchor)
        f.set_bn_from(bn)
        opt = torch.optim.SGD(m.parameters(), lr=self.lr, momentum=0.9)
        m.train()
        for _ in range(steps):
            bi = torch.randint(0, Xs.shape[0], (min(self.batch, Xs.shape[0]),))
            opt.zero_grad()
            torch.nn.functional.cross_entropy(m(Xs[bi]), ys[bi]).backward()
            opt.step()
        f.snapshot_params_to(self.flat)
        payload, self.ef[w], _ = codec.codec_encode(self.flat - anchor, self.ef[w], self.kind)
        return payload, f.bn_vec()


class _Group:
    """One synchronous block-mode aggregation group with the shipping Leader
    round math (ema_n sample weights, outer_step, broadcast error feedback,
    apply_round_to_anchor, pooled BN): the flat star, or one region."""

    def __init__(self, members, anchor, bn, kind, outer, outer_lr, mu, var_mask):
        self.members = list(members)
        self.anchor, self.bn = anchor.clone(), bn.clone()
        self.kind, self.outer, self.outer_lr, self.mu = kind, outer, outer_lr, mu
        self.var_mask, self.mean_mask = var_mask, ~var_mask
        n = anchor.numel()
        self.ef_bcast = torch.zeros(n) if codec.is_lossy(kind) else None
        self.m = torch.zeros(n) if oo.needs_momentum_buffer(outer) else None
        self.ema = {w: 0.0 for w in self.members}

    def intended(self):
        """SAK-0011 intended anchor: what workers would hold with a lossless
        broadcast (the regional anchor plus the undelivered residual)."""
        return self.anchor + self.ef_bcast if self.ef_bcast is not None else self.anchor.clone()

    def round(self, fleet, steps):
        payloads, bnvecs = [], []
        for w in self.members:
            p, b = fleet.train(w, self.anchor, self.bn, steps[w])
            payloads.append(p)
            bnvecs.append(b)
            ns = float(steps[w] * fleet.batch)  # samples since the last snapshot
            self.ema[w] = ns if self.ema[w] == 0 else 0.8 * self.ema[w] + 0.2 * ns
        wsum = sum(self.ema.values())
        dbar = torch.zeros(self.anchor.numel())
        for w, p in zip(self.members, payloads):
            dbar.add_(codec.codec_decode(p, dbar.numel(), self.kind), alpha=self.ema[w] / wsum)
        update = oo.outer_step(dbar, self.m, self.outer, self.outer_lr, self.mu)
        if self.ef_bcast is not None:
            update = update + oo.ef_decay(self.outer, self.mu) * self.ef_bcast
            payload_b, self.ef_bcast, _ = codec.codec_encode(update, None, self.kind)
        else:
            payload_b, _, _ = codec.codec_encode(update, None, self.kind)
        codec.apply_round_to_anchor(self.anchor, payload_b, self.kind)
        self.bn = codec.merge_bn_pooled(bnvecs, [self.ema[w] for w in self.members],
                                        self.mean_mask, self.var_mask)


class _TimeLog:
    """Evaluates the global model on a fixed simulated-time grid (k * dt for
    k = 1..budget/dt), so arms with different round lengths are compared at
    the same instants. ``until(t, ...)`` records every grid point < t with the
    state current just before the event at t."""

    def __init__(self, Xte, yte, dt, budget_s):
        self.Xte, self.yte, self.dt = Xte, yte, dt
        self.K = int(round(budget_s / dt))
        self.k = 1
        self.model = make_model()
        self.fs = codec.FlatState(self.model)
        self.hist = []

    def until(self, t, anchor, bn, **counters):
        while self.k <= self.K and self.k * self.dt < t - 1e-12:
            acc, ls = evaluate(self.model, self.fs, anchor, bn, self.Xte, self.yte)
            self.hist.append({"t": round(self.k * self.dt, 4), "acc": round(acc, 4),
                              "loss": round(ls, 4), **counters})
            self.k += 1


def _timed_steps(topo, period):
    return [max(1, int(round(s * period))) for s in topo.speed]


def _payload_len(kind, numel):
    """Codec payload bytes: fixed for a given codec and length, so the clock
    can be advanced before the round's math runs."""
    return len(codec.codec_encode(torch.zeros(numel), None, kind)[0])


def run_flat_timed(kind, outer, Xtr, ytr, Xte, yte, topo, period=0.1, budget_s=15.0,
                   batch=64, lr=0.1, outer_lr=0.7, mu=0.9, eval_dt=0.25, seed=1):
    """Flat DiLoCo: the star Leader in block mode. A round aggregates once
    every worker has downloaded the last broadcast, run its time-paced local
    steps and uploaded over its own link to the leader (region A)."""
    torch.manual_seed(seed)
    fs = codec.FlatState(make_model())
    anchor = torch.empty(fs.numel)
    fs.snapshot_params_to(anchor)
    fleet = _Fleet(topo, Xtr, ytr, kind, batch, lr, seed)
    grp = _Group(range(topo.W), anchor, fs.bn_vec(), kind, outer, outer_lr, mu, fs.bn_var_mask())
    steps = _timed_steps(topo, period)
    bn_bytes = grp.bn.numel() * 4
    leg = _payload_len(kind, fs.numel) + bn_bytes  # one upload or broadcast leg
    wan = [topo.regions[w] != topo.leader_region for w in range(topo.W)]
    t_ready = [topo.xfer_s(w, fs.numel * 4 + bn_bytes) for w in range(topo.W)]  # FULL_SYNC
    log = _TimeLog(Xte, yte, eval_dt, budget_s)
    tokens = rounds = wan_bytes = 0
    while True:
        t_agg = max(t_ready[w] + steps[w] / topo.speed[w] + topo.xfer_s(w, leg)
                    for w in range(topo.W))
        if t_agg > budget_s:
            break
        log.until(t_agg, grp.anchor, grp.bn, tokens=tokens, outer_rounds=rounds,
                  wan_bytes=wan_bytes)
        grp.round(fleet, steps)
        rounds += 1
        tokens += sum(steps) * batch
        wan_bytes += 2 * leg * sum(wan)
        t_ready = [t_agg + topo.xfer_s(w, leg) for w in range(topo.W)]
    log.until(float("inf"), grp.anchor, grp.bn, tokens=tokens, outer_rounds=rounds,
              wan_bytes=wan_bytes)
    h = log.hist
    return {"history": h, "final_acc": h[-1]["acc"], "final_loss": h[-1]["loss"],
            "tokens": tokens, "outer_rounds": rounds, "wan_bytes": wan_bytes,
            "steps_per_round": steps}


def run_halos(kind, outer, Xtr, ytr, Xte, yte, topo, period=0.1, budget_s=15.0,
              batch=64, lr=0.1, outer_lr=1.0, mu=0.9, delay=None, stale_alpha=0.5,
              push_every=1, region_outer="plain", region_outer_lr=1.0, eval_dt=0.25,
              seed=1):
    """Three-tier hierarchical async local-SGD (HALoS). Event-driven on the
    simulated clock:

    * ``round`` (region R aggregates at t): R's block-mode regional round
      runs; then any root reply that reached R by t is applied (SAK-0011:
      ``anchor += U - D``); then, if no drift is in flight and R ran
      ``push_every`` regional rounds since its last push, R snapshots
      ``D = intended - g`` (codec + upload error feedback) and sends it to
      the root; then the next regional round is scheduled. The WAN never
      gates a regional round.
    * ``arrive`` (root, drift from R at t): staleness tau = root versions
      since R's base; ``dbar = share_R * (1+tau)^-a * D``;
      ``outer_step(... delayed_nesterov, delay=N)``. The root replies with
      the codec-encoded ``G - mirror_R`` (what R has not seen yet), so a
      broadcast residual is resent next time instead of lost.

    ``delay`` (N) defaults to the number of regions."""
    import heapq
    import itertools

    torch.manual_seed(seed)
    fs = codec.FlatState(make_model())
    n = fs.numel
    G = torch.empty(n)
    fs.snapshot_params_to(G)
    Gbn = fs.bn_vec()
    var_mask = fs.bn_var_mask()
    fleet = _Fleet(topo, Xtr, ytr, kind, batch, lr, seed)
    steps = _timed_steps(topo, period)
    bn_bytes = Gbn.numel() * 4
    leg = _payload_len(kind, n) + bn_bytes
    regions = sorted(set(topo.regions))
    delay = delay or len(regions)
    m = torch.zeros(n) if oo.needs_momentum_buffer(outer) else None
    buf = torch.zeros(n) if outer == "delayed_nesterov" else None
    lossy = codec.is_lossy(kind)
    version = nstep = 0
    members = {R: [w for w in range(topo.W) if topo.regions[w] == R] for R in regions}
    share = {R: sum(fleet.share[w] for w in members[R]) for R in regions}
    mirror = {R: G.clone() for R in regions}
    region_bn = {}
    st = {}
    for R in regions:
        boot = topo.region_xfer_s(R, n * 4 + bn_bytes)  # root FULL_SYNC -> region
        st[R] = {"grp": _Group(members[R], G, Gbn, kind, region_outer, region_outer_lr, mu,
                               var_mask),
                 "g": G.clone(), "gver": 0, "ef_up": torch.zeros(n) if lossy else None,
                 "inflight": None, "since": 0, "down": [], "rounds": 0,
                 "t_ready": {w: boot + topo.lan_xfer_s(w, n * 4 + bn_bytes)
                             for w in members[R]}}
    q: list = []
    seq = itertools.count()

    def schedule_round(R):
        s = st[R]
        t_agg = max(s["t_ready"][w] + steps[w] / topo.speed[w] + topo.lan_xfer_s(w, leg)
                    for w in members[R])
        heapq.heappush(q, (t_agg, next(seq), "round", R, None))

    for R in regions:
        schedule_round(R)
    log = _TimeLog(Xte, yte, eval_dt, budget_s)
    tokens = wan_bytes = 0
    taus = []
    commits = {R: 0 for R in regions}

    def counters():
        return {"tokens": tokens, "outer_rounds": nstep, "wan_bytes": wan_bytes,
                "regional_rounds": sum(st[R]["rounds"] for R in regions)}

    while q:
        t, _, ev, R, data = heapq.heappop(q)
        if t > budget_s:
            break
        log.until(t, G, Gbn, **counters())
        s = st[R]
        if ev == "round":
            # Lazily computed at aggregation time: a region's anchor only
            # changes at its round boundaries, so it still equals the anchor
            # its workers started this round from.
            s["grp"].round(fleet, steps)
            s["rounds"] += 1
            s["since"] += 1
            tokens += sum(steps[w] for w in members[R]) * batch
            while s["down"] and s["down"][0][0] <= t:
                # SAK-0011 _commit_root: g += U; the regional BN is untouched
                # (only g_bn follows the root), like region.py.
                _, dec, ver = s["down"].pop(0)
                s["g"].add_(dec)
                s["gver"] = ver
                s["grp"].anchor.add_(dec - s["inflight"])  # pending += U - D
                s["inflight"] = None
            if s["inflight"] is None and s["since"] >= push_every:
                D = s["grp"].intended() - s["g"]
                payload, s["ef_up"], _ = codec.codec_encode(D, s["ef_up"], kind)
                s["inflight"], s["since"] = D, 0
                if R != topo.leader_region:
                    wan_bytes += len(payload) + bn_bytes
                t_arr = t + topo.region_xfer_s(R, len(payload) + bn_bytes)
                heapq.heappush(q, (t_arr, next(seq), "arrive", R,
                                   (payload, s["gver"], s["grp"].bn.clone())))
            s["t_ready"] = {w: t + topo.lan_xfer_s(w, leg) for w in members[R]}
            schedule_round(R)
        else:  # "arrive": the root commits R's drift on arrival
            payload, base, rbn = data
            tau = version - base
            taus.append(tau)
            dq = codec.codec_decode(payload, n, kind)
            dbar = oo.staleness_weighted_delta(dq, share[R], tau, stale_alpha)
            nstep += 1
            G.add_(oo.outer_step(dbar, m, outer, outer_lr, mu, buf=buf, step=nstep,
                                 delay=delay))
            version += 1
            commits[R] += 1
            region_bn[R] = rbn
            have = sorted(region_bn)
            Gbn = codec.merge_bn_pooled([region_bn[r] for r in have], [share[r] for r in have],
                                        ~var_mask, var_mask)
            pd, _, _ = codec.codec_encode(G - mirror[R], None, kind)
            dec = codec.codec_decode(pd, n, kind)
            mirror[R].add_(dec)
            if R != topo.leader_region:
                wan_bytes += len(pd) + bn_bytes
            s["down"].append((t + topo.region_xfer_s(R, len(pd) + bn_bytes), dec, version))
    log.until(float("inf"), G, Gbn, **counters())
    h = log.hist
    return {"history": h, "final_acc": h[-1]["acc"], "final_loss": h[-1]["loss"],
            "tokens": tokens, "outer_rounds": nstep, "wan_bytes": wan_bytes,
            "regional_rounds": {R: st[R]["rounds"] for R in regions},
            "root_commits": commits, "steps_per_round": steps,
            "tau_mean": round(sum(taus) / len(taus), 3) if taus else 0.0,
            "tau_max": max(taus) if taus else 0}


def _first_hit(history, target, key):
    for h in history:
        if h["acc"] >= target:
            return h[key]
    return None


def _at_tokens(history, tokens):
    """Accuracy at the last grid point whose token count is <= ``tokens``."""
    acc = None
    for h in history:
        if h["tokens"] <= tokens:
            acc = h["acc"]
    return acc


def _stat(xs):
    m, s = _mean_std(xs)
    hit = sum(x is not None for x in xs)
    return {"mean": None if m is None else round(m, 4),
            "std": None if s is None else round(s, 4), "hit": hit, "per_seed": xs}


def main_halos(argv=None):
    """#91: flat DiLoCo vs three-tier HALoS on the 3-GPU / 2-region skew
    topology. Best outer lr per arm by mean final accuracy; targets are
    fractions of the flat DiLoCo (nesterov) arm's mean final accuracy."""
    import argparse

    ap = argparse.ArgumentParser(prog="sim_fed.py --halos")
    ap.add_argument("--halos", action="store_true")
    ap.add_argument("--layout", choices=["fleet", "split", "both"], default="both")
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    ap.add_argument("--budget", type=float, default=15.0, help="simulated seconds")
    ap.add_argument("--period", type=float, default=0.1, help="compute period per round (s)")
    ap.add_argument("--eval-dt", type=float, default=0.25)
    ap.add_argument("--out", default=str(pathlib.Path(__file__).resolve().parent
                                         / "results/e91_halos"))
    a = ap.parse_args(argv)
    torch.set_num_threads(1)
    Xtr, ytr, Xte, yte = make_task()
    kw = {"period": a.period, "budget_s": a.budget, "eval_dt": a.eval_dt}
    arms = [  # (label, tier, outer, extra kwargs, outer-lr grid)
        ("flat DiLoCo nesterov", "flat", "nesterov", {}, (0.3, 0.5, 0.7)),
        ("flat plain (shipping default)", "flat", "plain", {}, (1.0,)),
        ("HALoS root plain a=0.5", "halos", "plain", {"stale_alpha": 0.5}, (1.0,)),
        ("HALoS root nesterov a=0.5", "halos", "nesterov", {"stale_alpha": 0.5},
         (0.3, 0.5, 0.7)),
        ("HALoS root delayed_nesterov a=0", "halos", "delayed_nesterov",
         {"stale_alpha": 0.0}, (0.3, 0.5, 0.7)),
        ("HALoS root delayed_nesterov a=0.5", "halos", "delayed_nesterov",
         {"stale_alpha": 0.5}, (0.3, 0.5, 0.7)),
    ]
    layouts = ["fleet", "split"] if a.layout == "both" else [a.layout]
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    report = {"issue": 91, "speed_steps_per_s": SKEW_SPEED, "seeds": a.seeds,
              "budget_s": a.budget, "period_s": a.period, "eval_dt": a.eval_dt,
              "codec": "int8", "layouts": {}}
    for layout in layouts:
        topo = skew_topology(layout)
        print(f"\n=== #91 layout={layout}: regions={topo.regions} speed={topo.speed} "
              f"links={topo.links}, budget {a.budget} sim-s, {len(a.seeds)} seeds ===")
        rows = []
        for label, tier, outer, extra, grid in arms:
            best = None
            for olr in grid:
                fn = run_flat_timed if tier == "flat" else run_halos
                runs = [fn("int8", outer, Xtr, ytr, Xte, yte, topo, outer_lr=olr, seed=s,
                           **kw, **extra) for s in a.seeds]
                acc_m, _ = _mean_std([r["final_acc"] for r in runs])
                print(f"  {label:<36} olr={olr:<4} final_acc={acc_m:.4f}", flush=True)
                if best is None or acc_m > best[1]:
                    best = (olr, acc_m, runs)
            olr, _, runs = best
            rows.append({"arm": label, "tier": tier, "outer": outer, "outer_lr": olr,
                         **extra, "runs": runs})
        base = rows[0]
        base_final = _mean_std([r["final_acc"] for r in base["runs"]])[0]
        base_tokens = _mean_std([r["tokens"] for r in base["runs"]])[0]
        targets = {"t95": round(0.95 * base_final, 4), "t98": round(0.98 * base_final, 4)}
        summary = []
        for row in rows:
            runs = row["runs"]
            accs = [r["final_acc"] for r in runs]
            acc_m, acc_s = _mean_std(accs)
            srow = {"arm": row["arm"], "tier": row["tier"], "outer": row["outer"],
                    "outer_lr": row["outer_lr"],
                    "final_acc_mean": round(acc_m, 4), "final_acc_std": round(acc_s, 4),
                    "final_acc_cv": round(acc_s / acc_m, 4) if acc_m else None,
                    "acc_at_flat_tokens": _stat([_at_tokens(r["history"], base_tokens)
                                                 for r in runs]),
                    "tokens_total": _mean_std([r["tokens"] for r in runs])[0],
                    "outer_rounds_total": _mean_std([r["outer_rounds"] for r in runs])[0],
                    "wan_MB": round(_mean_std([r["wan_bytes"] for r in runs])[0] / 1e6, 3)}
            if row["tier"] == "halos":
                srow["tau_mean"] = round(_mean_std([r["tau_mean"] for r in runs])[0], 3)
                srow["tau_max"] = max(r["tau_max"] for r in runs)
                srow["regional_rounds"] = runs[0]["regional_rounds"]
                srow["root_commits"] = runs[0]["root_commits"]
            for tn, tv in targets.items():
                for key, name in (("t", "time_s"), ("tokens", "tokens"),
                                  ("outer_rounds", "outer_rounds")):
                    srow[f"{name}_to_{tn}"] = _stat([_first_hit(r["history"], tv, key)
                                                     for r in runs])
            summary.append(srow)
        b = summary[0]
        print(f"targets (x flat DiLoCo final {base_final:.4f}): {targets}")
        print(f"{'arm':<36}{'olr':>5}{'final_acc':>15}{'CV':>7}{'acc@flatTok':>12}"
              f"{'t->t98':>12}{'tok->t98':>14}{'rnd->t98':>11}{'WAN_MB':>8}")
        for srow in summary:
            tt, tk, rr = (srow["time_s_to_t98"], srow["tokens_to_t98"],
                          srow["outer_rounds_to_t98"])
            for key, ref in (("time_s_to_t98", "time"), ("tokens_to_t98", "tokens")):
                if srow[key]["mean"] is not None and b[key]["mean"]:
                    srow[f"{ref}_vs_flat_t98"] = round(srow[key]["mean"] / b[key]["mean"], 3)
            cell = (lambda st: f"{st['mean']:.4g}({st['hit']})" if st["mean"] is not None
                    else f"-({st['hit']})")
            print(f"{srow['arm']:<36}{srow['outer_lr']:>5}{srow['final_acc_mean']:>9.4f}"
                  f"±{srow['final_acc_std']:.3f}{srow['final_acc_cv']:>7.3f}"
                  f"{srow['acc_at_flat_tokens']['mean'] or 0:>12.4f}{cell(tt):>12}"
                  f"{cell(tk):>14}{cell(rr):>11}{srow['wan_MB']:>8.2f}")
        report["layouts"][layout] = {
            "topology": {"regions": topo.regions, "speed": topo.speed,
                         "links": {"-".join(k): v for k, v in topo.links.items()}},
            "targets": targets, "flat_final_acc": base_final, "flat_tokens": base_tokens,
            "summary": summary,
            "runs": [{"arm": r["arm"], "outer_lr": r["outer_lr"], "runs": r["runs"]}
                     for r in rows]}
    (out / "results.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"\nsaved -> {out / 'results.json'}")
    return report


def main():
    out = pathlib.Path(__file__).resolve().parent / "results/e7e8_sim"
    out.mkdir(parents=True, exist_ok=True)
    SEEDS = [1, 2, 3]

    print("=== e8: compression ratio on a ~25.5M heavy-tailed delta (ResNet-50 sized) ===")
    comp = compression_ratio_table()
    print(f"fp32 = {comp['fp32_MB']} MB")
    for r in comp["rows"]:
        print(f"  {r['codec']:<11} {r['MB']:>8} MB   {r['ratio_vs_fp32']:>6}x   "
              f"contraction δ̂={r['contraction']}")

    Xtr, ytr, Xte, yte = make_task()
    H, R = 10, 60
    solo = _mean_std([run_solo(Xtr, ytr, Xte, yte, steps=H * R, seed=s) for s in SEEDS])[0]
    TARGET = round(0.95 * solo, 3)
    print(f"\n=== e7+e8 convergence: {R} rounds × {H} local steps, 4 workers (1:2:4:8 shards),"
          f" mean of {len(SEEDS)} seeds ===")
    print(f"solo control (1 worker, {H*R} steps): acc={solo:.4f}  | speed target = {TARGET} acc")

    # outer-lr per kind: momentum needs a calmer outer step than plain to avoid overshoot
    matrix = [
        ("int8", "plain", 1.0), ("int8", "sgdm", 0.5), ("int8", "nesterov", 0.5),
        ("fp32raw", "plain", 1.0),
        ("int4", "sgdm", 0.5), ("topk:0.1", "sgdm", 0.5),
        ("topk:0.05", "sgdm", 0.5), ("topk:0.01", "sgdm", 0.5),
    ]
    results = {"compression": comp, "solo_acc": solo, "target": TARGET,
               "H": H, "rounds": R, "seeds": SEEDS, "runs": []}
    print(f"\n{'codec':<11}{'outer':<9}{'final_acc':>12}{'rounds→tgt':>12}{'up_MB':>8}{'down_MB':>9}")
    for kind, outer, olr in matrix:
        runs = [run_federation(kind, outer, Xtr, ytr, Xte, yte, H=H, rounds=R,
                               outer_lr=olr, seed=s) for s in SEEDS]
        acc_m, acc_s = _mean_std([r["final_acc"] for r in runs])
        spd_m, _ = _mean_std([_rounds_to(r["history"], TARGET) for r in runs])
        hit = sum(_rounds_to(r["history"], TARGET) is not None for r in runs)
        up = runs[0]["bytes_up"] / 1e6
        down = runs[0]["bytes_down"] / 1e6
        spd = f"{spd_m:.0f}({hit}/{len(SEEDS)})" if spd_m else f"–({hit}/{len(SEEDS)})"
        results["runs"].append({"codec": kind, "outer": outer, "outer_lr": olr,
                                "final_acc_mean": round(acc_m, 4), "final_acc_std": round(acc_s, 4),
                                "rounds_to_target_mean": spd_m, "hit": hit,
                                "up_MB": round(up, 2), "down_MB": round(down, 2)})
        print(f"{kind:<11}{outer:<9}{acc_m:>8.4f}±{acc_s:.3f}{spd:>12}{up:>8.2f}{down:>9.2f}")

    results["sparseloco"] = sparseloco_sweep(Xtr, ytr, Xte, yte, TARGET, H=H, R=R)
    (out / "results.json").write_text(json.dumps(results, indent=2))
    print(f"\nsaved -> {out/'results.json'}")


# ----------------------------------------------------------------- ES mode (#92)
# Workers exchange (seed, fitness) per antithetic pair instead of deltas; every
# node regenerates the low-rank (EGGROLL) perturbations from the seeds. Run with
# `python sim_fed.py --es`. ES primitives live in sakura/services/federation/es.py.
def _load_es():
    spec = importlib.util.spec_from_file_location("fed.es", BASE / "es.py")
    m = importlib.util.module_from_spec(spec)
    sys.modules["fed.es"] = m
    spec.loader.exec_module(m)
    return m


def _recalibrate_bn(model, fs, anchor, X):
    """Recompute BN running stats for the anchor from local data (cumulative
    average, one pass). ES never ships BN stats: each node can do this itself,
    so it costs compute, not bytes."""
    fs.set_params_from(anchor)
    for mod in model.modules():
        if isinstance(mod, torch.nn.modules.batchnorm._BatchNorm):
            mod.reset_running_stats()
            mod.momentum = None
    model.train()
    with torch.no_grad():
        for i in range(0, X.shape[0], 512):
            model(X[i:i + 512])
    return fs.bn_vec()


def run_es_federation(Xtr, ytr, Xte, yte, W=4, pairs=16, rank=1, sigma=0.02,
                      rounds=60, batch=256, lr=0.01, outer="plain", mu=0.9,
                      seed=1, eval_every=3):
    """ES round mode on the same synthetic task and 1:2:4:8 shards as
    run_federation. Per round each worker evaluates `pairs` antithetic
    low-rank perturbations of the anchor on one minibatch of its shard
    (common random numbers within the round) and uploads (seed, f+, f-) per
    pair; the leader concatenates the populations, rebroadcasts them, and every
    node applies the same ES step through the shipping outer optimizer."""
    es = _load_es()
    torch.manual_seed(seed)
    model = make_model()
    fs = FED["codec"].FlatState(model)
    shapes = [tuple(p.shape) for p in fs.params]
    anchor = torch.empty(fs.numel)
    fs.snapshot_params_to(anchor)
    outer_m = torch.zeros(fs.numel) if outer != "plain" else None

    g = torch.Generator().manual_seed(seed + 7)
    perm = torch.randperm(Xtr.shape[0], generator=g)
    sizes = torch.tensor([1.0, 2.0, 4.0, 8.0][:W])
    sizes = (sizes / sizes.sum() * Xtr.shape[0]).long()
    sizes[-1] += Xtr.shape[0] - int(sizes.sum())
    shards, off = [], 0
    for s in sizes:
        idx = perm[off:off + int(s)]
        shards.append((Xtr[idx], ytr[idx]))
        off += int(s)
    calib = Xtr[perm[:2048]]

    bytes_up = bytes_down = 0
    hist = []
    for r in range(1, rounds + 1):
        msgs = []
        for w in range(W):
            Xs, ys = shards[w]
            bi = torch.randint(0, Xs.shape[0], (min(batch, Xs.shape[0]),))
            xb, yb = Xs[bi], ys[bi]
            recs = []
            model.train()
            with torch.no_grad():
                for k in range(pairs):
                    sd = es.member_seed(seed, r, w, k)
                    e = es.lowrank_perturbation(sd, shapes, rank)
                    f = []
                    for sign in (1.0, -1.0):
                        fs.set_params_from(anchor + (sign * sigma) * e)
                        f.append(-torch.nn.functional.cross_entropy(model(xb), yb).item())
                    recs.append((sd, f[0], f[1]))
            msg = es.pack_es_payload(r, w, recs)
            bytes_up += len(msg)
            msgs.append(msg)
        # leader: concatenate populations and rebroadcast them to every worker
        bytes_down += sum(len(m) for m in msgs) * W
        population = [p for m in msgs for p in es.unpack_es_payload(m)[2]]
        grad = es.es_gradient(population, shapes, rank, sigma)
        anchor.add_(oo.outer_step(grad, outer_m, outer, lr, mu))
        if r % eval_every == 0 or r == rounds:
            bn = _recalibrate_bn(model, fs, anchor, calib)
            acc, ls = evaluate(model, fs, anchor, bn, Xte, yte)
            hist.append({"round": r, "acc": round(acc, 4), "loss": round(ls, 4)})
    return {"history": hist, "final_acc": hist[-1]["acc"], "final_loss": hist[-1]["loss"],
            "bytes_up": bytes_up, "bytes_down": bytes_down}


def main_es():
    """ES (EGGROLL, low-rank) vs backprop-fed on the synthetic task: rounds and
    wall-clock to target, bytes/round, final accuracy, plus a modeled WAN time
    on the measured E5 link (11.5 MB/s striped TCP, 248 ms RTT)."""
    import time
    out = pathlib.Path(__file__).resolve().parent / "results/es_sim"
    out.mkdir(parents=True, exist_ok=True)
    SEEDS = [1, 2, 3]
    LINK_BPS, RTT = 11.5e6, 0.248
    Xtr, ytr, Xte, yte = make_task()
    H, R, R_ES = 10, 60, 300
    solo = _mean_std([run_solo(Xtr, ytr, Xte, yte, steps=H * R, seed=s) for s in SEEDS])[0]
    # t85 sits above the 0.46 majority-class rate; t95 is the e7/e8 speed target
    targets = {"t85": round(0.85 * solo, 3), "t95": round(0.95 * solo, 3)}
    print(f"solo control: acc={solo:.4f}  targets: {targets}")

    arms = [
        ("backprop int8+sgdm", R, lambda s: run_federation(
            "int8", "sgdm", Xtr, ytr, Xte, yte, H=H, rounds=R, outer_lr=0.5, seed=s)),
        ("backprop fp32raw+plain", R, lambda s: run_federation(
            "fp32raw", "plain", Xtr, ytr, Xte, yte, H=H, rounds=R, outer_lr=1.0, seed=s)),
        ("ES rank1 P=16/worker", R_ES, lambda s: run_es_federation(
            Xtr, ytr, Xte, yte, rounds=R_ES, rank=1, seed=s)),
        ("ES rank4 P=16/worker", R_ES, lambda s: run_es_federation(
            Xtr, ytr, Xte, yte, rounds=R_ES, rank=4, seed=s)),
    ]
    results = {"solo_acc": solo, "targets": targets, "seeds": SEEDS,
               "link_Bps": LINK_BPS, "rtt_s": RTT, "runs": []}
    print(f"\n{'arm':<24}{'final_acc':>14}{'rnd→t85':>9}{'s→t85':>8}{'rnd→t95':>9}"
          f"{'s→t95':>8}{'B/rnd up':>11}{'B/rnd down':>12}{'WAN s/rnd':>10}")
    for name, rounds, fn in arms:
        runs, secs = [], []
        for s in SEEDS:
            t0 = time.perf_counter()
            runs.append(fn(s))
            secs.append(time.perf_counter() - t0)
        acc_m, acc_s = _mean_std([r["final_acc"] for r in runs])
        sec_per_round = sum(secs) / len(secs) / rounds
        up_r = runs[0]["bytes_up"] / rounds
        down_r = runs[0]["bytes_down"] / rounds
        # one worker's leg per round: its upload + its share of the broadcast
        wan = sec_per_round + (up_r / 4 + down_r / 4) / LINK_BPS + RTT
        row = {"arm": name, "rounds": rounds, "final_acc_mean": round(acc_m, 4),
               "final_acc_std": round(acc_s, 4), "cpu_s_per_round": round(sec_per_round, 4),
               "bytes_up_per_round": up_r, "bytes_down_per_round": down_r,
               "wan_s_per_round": round(wan, 3)}
        cells = []
        for t, v in targets.items():
            rt, _ = _mean_std([_rounds_to(r["history"], v) for r in runs])
            hit = sum(_rounds_to(r["history"], v) is not None for r in runs)
            row[f"rounds_to_{t}"] = rt
            row[f"hit_{t}"] = hit
            row[f"cpu_s_to_{t}"] = round(rt * sec_per_round, 2) if rt else None
            row[f"wan_s_to_{t}"] = round(rt * wan, 1) if rt else None
            cells.append(f"{rt:>6.0f}({hit})" if rt else f"{'–':>6}({hit})")
            cells.append(f"{rt * sec_per_round:>8.1f}" if rt else f"{'–':>8}")
        results["runs"].append(row)
        print(f"{name:<24}{acc_m:>8.4f}±{acc_s:.3f}{cells[0]:>9}{cells[1]:>8}{cells[2]:>9}"
              f"{cells[3]:>8}{up_r:>11.0f}{down_r:>12.0f}{wan:>10.3f}")

    (out / "results.json").write_text(json.dumps(results, indent=2))
    print(f"\nsaved -> {out/'results.json'}")


# (codec, outer, outer_lr, mu): shipped baseline first; the rest are compared
# to it. For outer='ef', mu is the decay of the error-feedback accumulator
# (SparseLoCo uses 0.95).
SPARSELOCO_MATRIX = [
    ("int8", "plain", 1.0, 0.9),
    ("int8", "nesterov", 0.5, 0.9),
    ("int2", "plain", 1.0, 0.9),
    ("int2", "ef", 1.0, 0.9),
    ("topk2", "plain", 1.0, 0.9),
    ("topk2", "nesterov", 0.5, 0.9),
    ("topk2", "ef", 1.0, 0.9),
    ("topk2", "ef", 0.5, 0.95),
]
SPARSELOCO_SEEDS = [1, 2, 3, 4, 5]


def sparseloco_sweep(Xtr, ytr, Xte, yte, target, H=10, R=60, W=4,
                     seeds=SPARSELOCO_SEEDS, matrix=SPARSELOCO_MATRIX):
    """#89 / #50: bytes per round, rounds-to-target and final-acc CV for the
    SparseLoCo codecs and outer='ef' against the shipped int8/plain."""
    print(f"\n=== #89 SparseLoCo sweep: {R} rounds × {H} local steps, {W} workers,"
          f" {len(seeds)} seeds, target acc {target} ===")
    print(f"{'codec':<7}{'outer':<10}{'lr/mu':>9}{'B/rnd/wkr':>10}{'vs int8':>8}{'rounds→tgt':>14}"
          f"{'final_acc':>16}{'CV':>7}")
    rows = []
    base_bpr = None
    for kind, outer, olr, mu in matrix:
        runs = [run_federation(kind, outer, Xtr, ytr, Xte, yte, W=W, H=H, rounds=R,
                               outer_lr=olr, mu=mu, seed=s) for s in seeds]
        acc_m, acc_s = _mean_std([r["final_acc"] for r in runs])
        rtt = [_rounds_to(r["history"], target) for r in runs]
        spd_m, spd_s = _mean_std(rtt)
        hit = sum(x is not None for x in rtt)
        # up + down bytes per worker per round (both are one payload per worker)
        bpr = (runs[0]["bytes_up"] + runs[0]["bytes_down"]) / (R * W)
        base_bpr = base_bpr or bpr
        cv = acc_s / acc_m if acc_m else 0.0
        row = {"codec": kind, "outer": outer, "outer_lr": olr, "mu": mu,
               "bytes_per_round_per_worker": round(bpr, 1),
               "ratio_vs_int8": round(base_bpr / bpr, 2),
               "rounds_to_target_mean": spd_m, "rounds_to_target_std": spd_s,
               "hit": hit, "final_acc_mean": round(acc_m, 4),
               "final_acc_std": round(acc_s, 4), "final_acc_cv": round(cv, 4)}
        rows.append(row)
        spd = (f"{spd_m:.1f}±{spd_s:.1f}({hit}/{len(seeds)})" if spd_m
               else f"–({hit}/{len(seeds)})")
        print(f"{kind:<7}{outer:<10}{f'{olr}/{mu}':>9}{bpr:>10.0f}{row['ratio_vs_int8']:>7.1f}x{spd:>14}"
              f"{acc_m:>10.4f}±{acc_s:.3f}{cv:>7.3f}")
    return {"seeds": list(seeds), "target": target, "rows": rows}


def sparseloco_main():
    Xtr, ytr, Xte, yte = make_task()
    H, R = 10, 60
    solo = _mean_std([run_solo(Xtr, ytr, Xte, yte, steps=H * R, seed=s) for s in [1, 2, 3]])[0]
    target = round(0.95 * solo, 3)
    res = sparseloco_sweep(Xtr, ytr, Xte, yte, target, H=H, R=R)
    out = pathlib.Path(__file__).resolve().parent / "results/e7e8_sim"
    out.mkdir(parents=True, exist_ok=True)
    (out / "sparseloco.json").write_text(json.dumps(res, indent=2))
    print(f"\nsaved -> {out/'sparseloco.json'}")


def bandwidth_report(argv=None):
    """#94: bandwidth-reduction + compute-utilisation report vs data-parallel
    (INTELLECT-1-style). Logic lives in sakura/services/federation/report.py;
    this only wires the sim fixtures (model, task, shipping codec) into it."""
    import argparse

    ap = argparse.ArgumentParser(prog="sim_fed.py --report")
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--rounds", type=int, default=6, help="sim rounds")
    ap.add_argument("--period", type=float, default=30.0, help="round period T (s) for the R50 projection")
    ap.add_argument("--numel", type=int, default=25_557_032, help="params for the codec-bytes measurement (R50)")
    ap.add_argument("--results", default=str(pathlib.Path(__file__).resolve().parent / "results"))
    ap.add_argument("--out", default=None, help="output dir (default results/bandwidth_report)")
    a = ap.parse_args(argv)
    spec = importlib.util.spec_from_file_location("fed.report", BASE / "report.py")
    rep = importlib.util.module_from_spec(spec)
    sys.modules["fed.report"] = rep
    spec.loader.exec_module(rep)

    fleet = rep.fleet_report(a.results)
    # 56:1 img/s skew at the fleet's tuned batches: local steps per round ∝
    # solo img/s / batch, scaled so the P620 takes 1 step (P620:2080Ti:4090 ≈ 1:4:14).
    rate = {w: rep.R50_SOLO_IMG_S[w] / rep.R50_BATCH[w] for w in rep.R50_SOLO_IMG_S}
    steps = {w: max(1, round(rate[w] / rate["p620"])) for w in rate}
    sim = rep.simulate_skew(codec, make_model, make_task(), steps, rep.R50_BATCH, rounds=a.rounds)
    payload = rep.codec_payload_bytes(codec, a.numel)
    proj = rep.r50_projection(payload, a.numel, period_s=a.period)
    report = {"issue": 94, "fleet": fleet, "sim": sim, "projection": proj}

    out = pathlib.Path(a.out) if a.out else pathlib.Path(a.results) / "bandwidth_report"
    out.mkdir(parents=True, exist_ok=True)
    (out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    md = rep.to_markdown(report)
    (out / "report.md").write_text(md)
    print(md)
    print(f"saved -> {out/'report.json'}, {out/'report.md'}")
    return report


# ------------------------------------------------- dynamic membership (#95)
# `python sim_fed.py --dynamic`: workers join and leave mid-run through the
# SAK-0014 roster hooks. Unlike the modes above this drives the REAL Leader and
# FederationService over loopback (threads, port 0), so the join path, the
# FULL_SYNC bootstrap and the consensus hash are the shipping ones. Shared
# logic: sakura/services/federation/churn.py (also used by the test suite).
def dynamic_membership(argv=None):
    import argparse

    ap = argparse.ArgumentParser(prog="sim_fed.py --dynamic")
    ap.add_argument("--dynamic", action="store_true")
    ap.add_argument("--rounds", type=int, default=30)
    ap.add_argument("--gap", type=float, default=1.0, help="round period (s)")
    ap.add_argument("--roster", type=int, default=4, help="initial workers")
    ap.add_argument("--pool", type=int, default=4, help="extra workers that join later")
    ap.add_argument("--every", type=int, default=2, help="rounds between churn events")
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2])
    ap.add_argument("--out", default=str(pathlib.Path(__file__).resolve().parent
                                         / "results/dynamic_membership"))
    a = ap.parse_args(argv)

    from sakura.services.federation.churn import churn_schedule, run_fleet
    from sakura.services.federation.codec import FlatState

    torch.set_num_threads(1)  # one fleet of threads in one process
    Xtr, ytr, Xte, yte = make_task()
    roster = [f"w{i}" for i in range(a.roster)]
    pool = [f"w{i}" for i in range(a.roster, a.roster + a.pool)]
    everyone = roster + pool
    chunks = list(zip(Xtr.chunk(len(everyone)), ytr.chunk(len(everyone))))

    def data_fn(wid):
        return chunks[everyone.index(wid)]

    eval_model = make_model()
    eval_fs = FlatState(eval_model)

    def eval_fn(anchor, bn):
        acc, loss = evaluate(eval_model, eval_fs, anchor, bn, Xte, yte)
        return {"acc": round(acc, 4), "loss": round(loss, 4)}

    runs = []
    for seed in a.seeds:
        sched = churn_schedule(roster, pool, a.rounds, every=a.every, seed=seed)
        for label, sc in (("dynamic", sched), ("static", [])):
            r = run_fleet(make_model, data_fn, roster, sc, a.rounds, gap_s=a.gap,
                          lr=0.05, seed=seed, eval_fn=eval_fn)
            r.update(label=label, seed=seed)
            runs.append(r)
            print(f"seed={seed} {label:7s} rounds={r['rounds']} joins={r['n_join']} "
                  f"leaves={r['n_leave']} workers={r['distinct_workers']} "
                  f"discrepancies={r['hash_discrepancies']} acc={r['eval']['acc']}")

    def mean(label, key):
        xs = [r["eval"][key] for r in runs if r["label"] == label]
        return round(sum(xs) / len(xs), 4)

    summary = {
        "rounds_per_run": a.rounds,
        "gap_s": a.gap,
        "runs": len(runs),
        "join_events": sum(r["n_join"] for r in runs if r["label"] == "dynamic"),
        "leave_events": sum(r["n_leave"] for r in runs if r["label"] == "dynamic"),
        "hash_discrepancies": sum(r["hash_discrepancies"] for r in runs),
        "acc_dynamic_mean": mean("dynamic", "acc"),
        "acc_static_mean": mean("static", "acc"),
        "loss_dynamic_mean": mean("dynamic", "loss"),
        "loss_static_mean": mean("static", "loss"),
    }
    summary["acc_delta_dynamic_minus_static"] = round(
        summary["acc_dynamic_mean"] - summary["acc_static_mean"], 4)
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "results.json").write_text(json.dumps({"summary": summary, "runs": runs}, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    print(f"saved -> {out / 'results.json'}")
    return summary


if __name__ == "__main__":
    if "--sparseloco" in sys.argv[1:]:
        sparseloco_main()
    elif "--report" in sys.argv[1:]:
        bandwidth_report(sys.argv[1:])
    elif "--es" in sys.argv[1:]:
        main_es()
    elif "--async-dn" in sys.argv[1:]:  # #90: async local-SGD + delayed Nesterov
        main_async()
    elif "--halos" in sys.argv[1:]:  # #91: three-tier HALoS vs flat DiLoCo
        main_halos(sys.argv[1:])
    elif "--dynamic" in sys.argv[1:]:
        dynamic_membership(sys.argv[1:])
    elif "--streaming" in sys.argv:  # issue #51: streaming vs block comparison
        res = streaming_vs_block()
        out = pathlib.Path(__file__).resolve().parent / "results/streaming_sim"
        out.mkdir(parents=True, exist_ok=True)
        (out / "results.json").write_text(json.dumps(res, indent=2))
        print(f"saved -> {out / 'results.json'}")
    else:
        main()
