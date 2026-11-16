#!/usr/bin/env python3
"""Rejoin momentum policy comparison (SAK-0014, issue #106).

Compares what a FULL_SYNC does to a worker's inner-optimizer momentum
(``noop`` / ``reset`` / ``average``, implemented in
``sakura/services/federation/rejoin.py``) over a long run with a recorded
join/rejoin schedule, on the CPU synthetic task of ``sim_fed.py`` and the
shipping primitives (FlatState, int8 codec + error feedback, plain outer step,
pooled BN merge).

What it models, matching ``FederationService`` (not block-mode sim_fed):
  * every worker keeps ONE inner optimizer for the whole run, so its momentum
    survives rounds, as in a real training process;
  * a committed round sets an active worker's params to the new anchor (the
    block-mode apply) and leaves its optimizer untouched, as today;
  * an absent worker (ctl down) keeps training alone on its shard: its params
    and momentum follow a private trajectory and it uploads nothing;
  * when it comes back, the leader FULL_SYNCs it: params and BN hard-set to the
    anchor, error feedback zeroed, then the rejoin policy is applied to its
    optimizer; it trains and uploads in that same round.

Random batch draws do not depend on the policy, so for a given seed the three
policies see identical data and differences are paired.

    python sim_rejoin.py                  # full sweep -> results/rejoin_momentum_sim
    python sim_rejoin.py --rounds 30 --seeds 1   # quick look
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import pathlib
import statistics
import sys
import time
from typing import Any, Dict, List, Sequence, Tuple

import torch

ROOT = pathlib.Path(__file__).resolve().parent


def _load_sim_fed():
    """Load sim_fed.py without leaving its synthetic ``fed`` package behind
    (other test modules install their own)."""
    spec = importlib.util.spec_from_file_location("sim_fed_rejoin", ROOT / "sim_fed.py")
    m = importlib.util.module_from_spec(spec)
    saved = {k: v for k, v in sys.modules.items() if k == "fed" or k.startswith("fed.")}
    try:
        spec.loader.exec_module(m)
    finally:
        for k in [k for k in sys.modules if k == "fed" or k.startswith("fed.")]:
            del sys.modules[k]
        sys.modules.update(saved)
    rspec = importlib.util.spec_from_file_location(
        "sim_rejoin_policy", ROOT / "sakura/services/federation/rejoin.py"
    )
    r = importlib.util.module_from_spec(rspec)
    rspec.loader.exec_module(r)
    return m, r


SIM, REJOIN = _load_sim_fed()
codec = SIM.codec
oo = SIM.oo

POLICIES = REJOIN.REJOIN_POLICIES  # ("noop", "reset", "average")

# (worker, first absent round, rounds absent). The worker is back (FULL_SYNC)
# at round start + length. Every worker leaves at least once, worker 3 (the
# largest shard) for longest; one rejoin lands near the end of the run.
SCHEDULE: Tuple[Tuple[int, int, int], ...] = (
    (1, 20, 10),
    (3, 45, 15),
    (2, 75, 10),
    (0, 100, 20),
    (1, 130, 10),
)
OPTIMIZERS = {
    # name: (factory, lr)
    "sgdm": (lambda ps, lr: torch.optim.SGD(ps, lr=lr, momentum=0.9), 0.1),
    "adam": (lambda ps, lr: torch.optim.Adam(ps, lr=lr), 3e-3),
}


def _shards(Xtr, ytr, W, seed):
    g = torch.Generator().manual_seed(seed + 7)
    perm = torch.randperm(Xtr.shape[0], generator=g)
    sizes = torch.tensor([1.0, 2.0, 4.0, 8.0][:W])
    sizes = (sizes / sizes.sum() * Xtr.shape[0]).long()
    sizes[-1] += Xtr.shape[0] - int(sizes.sum())
    out, off = [], 0
    for s in sizes:
        idx = perm[off:off + int(s)]
        out.append((Xtr[idx], ytr[idx]))
        off += int(s)
    return out


def run_rejoin(policy: str, opt: str, task, seed: int = 1, W: int = 4, H: int = 10,
               rounds: int = 150, schedule: Sequence[Tuple[int, int, int]] = SCHEDULE,
               batch: int = 64, kind: str = "int8", decay: float = 0.5) -> Dict[str, Any]:
    """One long run under ``policy``. Returns the per-round anchor test
    loss/acc and one record per rejoin event."""
    Xtr, ytr, Xte, yte = task
    torch.manual_seed(seed)
    leader_model = SIM.make_model()
    fs = codec.FlatState(leader_model)
    anchor = torch.empty(fs.numel)
    fs.snapshot_params_to(anchor)
    bn = fs.bn_vec()
    var_mask = fs.bn_var_mask()
    mean_mask = ~var_mask
    ef_bcast = torch.zeros(fs.numel) if codec.is_lossy(kind) else None

    shards = _shards(Xtr, ytr, W, seed)
    workers = [SIM.make_model() for _ in range(W)]
    wfs = [codec.FlatState(m) for m in workers]
    make_opt, lr = OPTIMIZERS[opt]
    opts = [make_opt(m.parameters(), lr) for m in workers]
    worker_ef: List[Any] = [None] * W
    ema_n = [0.0] * W
    for w in range(W):  # bootstrap FULL_SYNC (fresh optimizer: policy is moot)
        wfs[w].set_params_from(anchor)
        wfs[w].set_bn_from(bn)
    eval_model = SIM.make_model()
    eval_fs = codec.FlatState(eval_model)

    def absent(w: int, r: int) -> bool:
        return any(w == sw and s <= r < s + n for sw, s, n in schedule)

    rejoin_at = {s + n: sw for sw, s, n in schedule if s + n <= rounds}
    hist: List[Dict[str, float]] = []
    events: List[Dict[str, Any]] = []
    flat = torch.empty(fs.numel)
    for r in range(1, rounds + 1):
        ev = None
        if r in rejoin_at:  # FULL_SYNC lands before the round's training
            w = rejoin_at[r]
            m_before = _first_moment_norm(opts[w])
            wfs[w].set_params_from(anchor)
            wfs[w].set_bn_from(bn)
            worker_ef[w] = None
            REJOIN.apply_rejoin_policy(opts[w], policy, decay)
            ev = {"worker": w, "round": r, "momentum_norm_before": m_before,
                  "momentum_norm_after": _first_moment_norm(opts[w])}
        payloads, bnvecs, ns, act = [], [], [], []
        for w in range(W):
            m, (Xs, ys) = workers[w], shards[w]
            m.train()
            losses = []
            for _ in range(H):
                bi = torch.randint(0, Xs.shape[0], (min(batch, Xs.shape[0]),))
                opts[w].zero_grad()
                loss = torch.nn.functional.cross_entropy(m(Xs[bi]), ys[bi])
                loss.backward()
                opts[w].step()
                losses.append(loss.item())
            if ev is not None and ev["worker"] == w:
                ev["first_window_train_loss"] = round(sum(losses) / len(losses), 4)
            if absent(w, r):
                continue  # trains alone, uploads nothing
            wfs[w].snapshot_params_to(flat)
            delta = flat - anchor
            if ev is not None and ev["worker"] == w:
                ev["first_drift_norm"] = round(float(delta.norm()), 4)
            payload, worker_ef[w], _ = codec.codec_encode(delta, worker_ef[w], kind)
            payloads.append(payload)
            bnvecs.append(wfs[w].bn_vec())
            ns.append(float(Xs.shape[0]))
            act.append(w)
        # ---- leader aggregation over the workers present (mirrors leader.py) ----
        for w, n in zip(act, ns):
            ema_n[w] = n if ema_n[w] == 0 else 0.8 * ema_n[w] + 0.2 * n
        wsum = sum(ema_n[w] for w in act)
        dbar = torch.zeros(fs.numel)
        for w, p in zip(act, payloads):
            dbar.add_(codec.codec_decode(p, fs.numel, kind), alpha=ema_n[w] / wsum)
        update = oo.outer_step(dbar, None, "plain", 1.0, 0.9)
        if ef_bcast is not None:
            update = update + oo.ef_decay("plain", 0.9) * ef_bcast
            payload_b, ef_bcast, _ = codec.codec_encode(update, None, kind)
        else:
            payload_b, _, _ = codec.codec_encode(update, None, kind)
        codec.apply_round_to_anchor(anchor, payload_b, kind)
        bn = codec.merge_bn_pooled(bnvecs, [ema_n[w] for w in act], mean_mask, var_mask)
        for w in act:  # committed round: params := anchor, optimizer untouched
            wfs[w].set_params_from(anchor)
            wfs[w].set_bn_from(bn)
        acc, ls = SIM.evaluate(eval_model, eval_fs, anchor, bn, Xte, yte)
        hist.append({"round": r, "acc": round(acc, 4), "loss": round(ls, 4)})
        if ev is not None:
            events.append(ev)
    loss_at = {h["round"]: h["loss"] for h in hist}
    for ev in events:
        r0 = ev["round"]
        pre = loss_at.get(r0 - 1)
        if pre is None:
            continue
        # transient: anchor test-loss change caused by the first round that
        # includes the rejoiner, and over the three rounds from the rejoin.
        ev["d_loss_1"] = round(loss_at[r0] - pre, 4)
        post = [loss_at[r] for r in range(r0, r0 + 3) if r in loss_at]
        ev["d_loss_3"] = round(sum(post) / len(post) - pre, 4)
    return {"policy": policy, "opt": opt, "seed": seed, "history": hist,
            "events": events, "final_acc": hist[-1]["acc"], "final_loss": hist[-1]["loss"]}


def _first_moment_norm(opt) -> float:
    sq = 0.0
    for st in opt.state.values():
        for k in REJOIN.FIRST_MOMENT_KEYS:
            t = st.get(k)
            if isinstance(t, torch.Tensor):
                sq += float(t.pow(2).sum())
    return round(sq ** 0.5, 4)


def _stats(xs: Sequence[float]) -> Dict[str, float]:
    mean = statistics.fmean(xs)
    sd = statistics.stdev(xs) if len(xs) > 1 else 0.0
    return {"mean": round(mean, 4), "std": round(sd, 4),
            "cv": round(sd / abs(mean), 4) if mean else 0.0, "n": len(xs)}


def compare(seeds: Sequence[int] = (1, 2, 3, 4, 5), opts: Sequence[str] = ("sgdm", "adam"),
            rounds: int = 150, schedule: Sequence[Tuple[int, int, int]] = SCHEDULE,
            decay: float = 0.5, n_train: int = 8192, n_test: int = 2048,
            threads: int = 1) -> Dict[str, Any]:
    """Run every policy x optimizer x seed and summarise.

    Runs with ``threads`` torch intra-op threads (default 1, like the
    ``sim_fed.py`` drivers) and restores the caller's setting afterwards. On a
    CPU-quota-limited CI pod torch otherwise sizes its pool from the host's
    core count, and the oversubscribed tiny matmuls stalled the test job past
    its timeout.
    """
    prev = torch.get_num_threads()
    torch.set_num_threads(threads)
    try:
        return _compare(seeds, opts, rounds, schedule, decay, n_train, n_test)
    finally:
        torch.set_num_threads(prev)


def _compare(seeds, opts, rounds, schedule, decay, n_train, n_test) -> Dict[str, Any]:
    task = SIM.make_task(n_train=n_train, n_test=n_test)
    out: Dict[str, Any] = {"issue": 106, "rounds": rounds, "seeds": list(seeds),
                           "schedule": [list(s) for s in schedule], "average_decay": decay,
                           "summary": {}, "runs": []}
    for opt in opts:
        per: Dict[str, List[Dict[str, Any]]] = {}
        for policy in POLICIES:
            per[policy] = [run_rejoin(policy, opt, task, seed=s, rounds=rounds,
                                      schedule=schedule, decay=decay) for s in seeds]
            out["runs"].extend(per[policy])
        summ: Dict[str, Any] = {}
        for policy in POLICIES:
            runs = per[policy]
            ev = [e for r in runs for e in r["events"] if "d_loss_1" in e]
            summ[policy] = {
                "final_acc": _stats([r["final_acc"] for r in runs]),
                "final_loss": _stats([r["final_loss"] for r in runs]),
                # mean anchor test loss over the last 10 rounds: steadier than
                # the single final value
                "tail_loss": _stats([statistics.fmean(h["loss"] for h in r["history"][-10:])
                                     for r in runs]),
                "rejoin_d_loss_1": _stats([e["d_loss_1"] for e in ev]) if ev else None,
                "rejoin_d_loss_3": _stats([e["d_loss_3"] for e in ev]) if ev else None,
                "rejoin_train_loss": _stats([e["first_window_train_loss"] for e in ev])
                if ev else None,
                "paired_final_acc_vs_noop": _stats(
                    [a["final_acc"] - b["final_acc"] for a, b in zip(runs, per["noop"])]),
                "paired_tail_loss_vs_noop": _stats(
                    [statistics.fmean(h["loss"] for h in a["history"][-10:])
                     - statistics.fmean(h["loss"] for h in b["history"][-10:])
                     for a, b in zip(runs, per["noop"])]),
            }
        out["summary"][opt] = summ
    return out


def to_markdown(res: Dict[str, Any]) -> str:
    lines = [f"# Rejoin momentum policy ({res['rounds']} rounds, seeds {res['seeds']}, "
             f"average decay {res['average_decay']})", ""]
    for opt, summ in res["summary"].items():
        lines += [f"## {opt}", "",
                  "| policy | final acc (mean ± sd, CV) | tail loss (mean ± sd) | "
                  "Δacc vs noop (paired) | rejoin Δloss r+0 | rejoin Δloss r+0..2 | "
                  "rejoiner train loss, 1st window |",
                  "|---|---|---|---|---|---|---|"]
        for p, s in summ.items():
            fa, tl, pa = s["final_acc"], s["tail_loss"], s["paired_final_acc_vs_noop"]
            d1, d3, tr = s["rejoin_d_loss_1"], s["rejoin_d_loss_3"], s["rejoin_train_loss"]
            lines.append(
                f"| {p} | {fa['mean']:.4f} ± {fa['std']:.4f} (CV {fa['cv']:.4f}) | "
                f"{tl['mean']:.4f} ± {tl['std']:.4f} | {pa['mean']:+.4f} ± {pa['std']:.4f} | "
                f"{d1['mean']:+.4f} ± {d1['std']:.4f} | {d3['mean']:+.4f} ± {d3['std']:.4f} | "
                f"{tr['mean']:.4f} ± {tr['std']:.4f} |")
        lines.append("")
    return "\n".join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="sim_rejoin.py")
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    ap.add_argument("--opts", nargs="+", default=["sgdm", "adam"], choices=sorted(OPTIMIZERS))
    ap.add_argument("--rounds", type=int, default=150)
    ap.add_argument("--decay", type=float, default=0.5, help="'average' policy factor")
    ap.add_argument("--threads", type=int, default=1, help="torch intra-op threads")
    ap.add_argument("--out", default=str(ROOT / "results/rejoin_momentum_sim"))
    a = ap.parse_args(argv)
    t0 = time.monotonic()
    res = compare(seeds=a.seeds, opts=a.opts, rounds=a.rounds, decay=a.decay,
                  threads=a.threads)
    res["wall_s"] = round(time.monotonic() - t0, 1)
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    summary = {k: v for k, v in res.items() if k != "runs"}
    (out / "results.json").write_text(json.dumps(summary, indent=2) + "\n")
    md = to_markdown(res)
    (out / "report.md").write_text(md + "\n")
    print(md)
    print(f"saved -> {out / 'results.json'}, {out / 'report.md'} ({res['wall_s']} s)")
    return res


if __name__ == "__main__":
    main()
