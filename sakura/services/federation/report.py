"""Bandwidth-reduction and compute-utilisation report vs data-parallel (#94).

INTELLECT-1 quotes "400x less bandwidth than data-parallel": DiLoCo syncs once
every H=100 inner steps (100x) and its all-reduce ships int8 instead of fp32
(4x). sakura's honest equivalent is the same ratio, counted per optimizer
step against a DDP baseline that all-reduces fp32 gradients every step::

    factor_w = steps_w x fp32_bytes(full model) / int8_payload_bytes_w

summed over the rounds of a run. ``H_eff`` (local steps between syncs) is not
configured in sakura: rounds are time-based, so each worker's ``H_eff`` is its
own throughput x round period / batch, and under a 56:1 skew it differs ~14x
between the fastest and the slowest GPU. The payload already includes the
per-bucket fp32 scales and padding, and with streaming shards (E5s) the 1/K
slice is in the measured bytes too, so nothing here is a formula plugged in
after the fact: every factor is computed from bytes that crossed the wire
(fleet logs) or that the shipping codec produced (sim).

Compute utilisation = achieved aggregate throughput / sum of the GPUs'
concurrent solo throughputs (docs/internal/REPORT.md's "% of ceiling"), i.e. productive
compute time / wall time weighted by each GPU's speed.

Three kinds of numbers come out of this module and are always labelled:

* ``measured`` — parsed from the E3-E6 fleet logs under ``results/``;
* ``sim``      — bytes produced by the real codec in the CPU skew simulation
  (``python sim_fed.py --report``);
* ``model``    — a wall-time model of DDP / block / overlapped rounds over the
  measured link (248 ms RTT, ~11.5 MB/s), for the regimes nobody ran.

Stdlib only at import time (the sim takes the codec module and torch
factories as arguments), so the parser and arithmetic are importable without
torch or the native ``sakura_wire`` build.
"""

from __future__ import annotations

import json
import pathlib
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence

FP32_BYTES = 4

# Concurrent tuned solo throughput per GPU on ResNet-50@224 (docs/internal/REPORT.md; the
# "ceiling" of every E3-E5 rung is their sum, 2 422 img/s), and their batches.
R50_SOLO_IMG_S: Dict[str, float] = {"p620": 29.75, "2080ti": 744.34, "4090": 1647.94}
R50_BATCH: Dict[str, int] = {"p620": 16, "2080ti": 96, "4090": 64}
# ResNet-50 float-param count, as logged by the E5+ runs' WORKER_ROUND shard
# field (E3/E4 predate it; their fp32raw payload, 102 228 128 B = 4 x this).
R50_NUMEL = 25_557_032
# Measured sync DDP (gloo) over the same mesh, docs/internal/REPORT.md baseline row.
DDP_MEASURED_IMG_S = 1.03
# Measured link (docs/internal/REPORT.md "Network findings").
LINK_MBPS = 11.5
LINK_RTT_S = 0.248
# INTELLECT-1 reference point (parent issue #54).
INTELLECT1 = {"factor": 400.0, "H": 100, "bits_ratio": 4.0, "compute_util": (0.83, 0.96)}

# Runs whose ceiling is the ResNet-50 solo sum above (E6 is CIFAR/ResNet-18).
R50_RUNS = ("e3", "e4", "e5", "e5s", "e5r", "e5r2")
DEFAULT_RUNS = ("e3", "e4", "e5", "e5s", "e5r2", "e6")


# ----------------------------------------------------------------- log parsing
def _records(path: pathlib.Path, tag: str) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    if not path.exists():
        return out
    for line in path.read_text(errors="replace").splitlines():
        if line.startswith(tag + " "):
            out.append(json.loads(line[len(tag) + 1 :]))
    return out


def parse_run_dir(run_dir: "str | pathlib.Path") -> Dict[str, Any]:
    """Collect LEADER_ROUND records (leader.log), each worker's ZKRESULT
    (batch, mode, img/s) and the full model size (max WORKER_ROUND shard end)."""
    d = pathlib.Path(run_dir)
    rounds = _records(d / "leader.log", "LEADER_ROUND")
    workers: Dict[str, Dict[str, Any]] = {}
    numel = 0
    for wlog in sorted(d.glob("*.log")):
        if wlog.name == "leader.log":
            continue
        res = _records(wlog, "ZKRESULT")
        for rec in _records(wlog, "WORKER_ROUND"):
            shard = rec.get("shard")
            if shard:
                numel = max(numel, int(shard[1]))
        if res:
            r = res[-1]
            workers[str(r["worker"])] = {
                "batch": int(r["batch"]),
                "mode": r.get("mode"),
                "img_s": float(r["img_s"]),
            }
    return {"name": d.name, "rounds": rounds, "workers": workers, "numel": numel}


# ----------------------------------------------------------------- arithmetic
def bandwidth_reduction(
    rounds: Sequence[Mapping[str, Any]],
    batch: Mapping[str, int],
    numel: int,
) -> Dict[str, Any]:
    """Per-worker and fleet bandwidth reduction vs fp32 DDP, from measured
    per-round sample counts and bytes.

    DDP moves ``FP32_BYTES * numel`` per worker per optimizer step (each
    direction); sakura moves ``bytes_up`` once per round in which the worker
    took ``n_samples / batch`` steps. The broadcast leg is symmetric
    (``bytes_down`` == one worker's upload), so the factor is the same per
    direction; it is reported on the upload leg.
    """
    if numel <= 0:
        raise ValueError("numel must be positive")
    dp_step = FP32_BYTES * numel
    steps: Dict[str, float] = {}
    up: Dict[str, int] = {}
    nrounds: Dict[str, int] = {}
    for rec in rounds:
        b_up = rec.get("bytes_up") or {}
        for w, n in (rec.get("n_samples") or {}).items():
            if w not in batch or w not in b_up:
                continue
            steps[w] = steps.get(w, 0.0) + float(n) / batch[w]
            up[w] = up.get(w, 0) + int(b_up[w])
            nrounds[w] = nrounds.get(w, 0) + 1
    per_worker: Dict[str, Dict[str, float]] = {}
    for w in sorted(steps, key=lambda k: steps[k]):
        if up[w] <= 0:
            continue
        h_eff = steps[w] / nrounds[w]
        per_round_payload = up[w] / nrounds[w]
        per_worker[w] = {
            "rounds": nrounds[w],
            "steps": round(steps[w], 1),
            "H_eff": round(h_eff, 1),
            "bytes_up": up[w],
            # fp32 bytes of the *full* model per round / payload actually sent:
            # = codec ratio x K for streaming shards.
            "per_sync_ratio": round(dp_step / per_round_payload, 3),
            "factor": round(steps[w] * dp_step / up[w], 2),
        }
    tot_steps = sum(steps[w] for w in per_worker)
    tot_up = sum(up[w] for w in per_worker)
    fleet = round(tot_steps * dp_step / tot_up, 2) if tot_up else None
    worst = min((v["factor"] for v in per_worker.values()), default=None)
    return {
        "numel": numel,
        "dp_bytes_per_step": dp_step,
        "per_worker": per_worker,
        "fleet_factor": fleet,
        "slowest_worker_factor": worst,
    }


def compute_utilisation(
    img_s: Mapping[str, float], solo_img_s: Mapping[str, float]
) -> Dict[str, Any]:
    """Achieved aggregate throughput / sum of concurrent solo throughputs."""
    ws = [w for w in solo_img_s if w in img_s]
    agg = sum(img_s[w] for w in ws)
    ceil = sum(solo_img_s[w] for w in ws)
    return {
        "aggregate_img_s": round(agg, 2),
        "ceiling_img_s": round(ceil, 2),
        "util": round(agg / ceil, 4) if ceil else None,
        "per_worker": {w: round(img_s[w] / solo_img_s[w], 4) for w in ws},
    }


def ddp_lockstep_util(
    solo_img_s: Mapping[str, float],
    batch: Mapping[str, int],
    numel: int,
    link_mbps: float = LINK_MBPS,
    rtt_s: float = LINK_RTT_S,
) -> Dict[str, float]:
    """Model: synchronous DDP with each GPU at its tuned batch. Every step
    waits for the slowest GPU, then a ring all-reduce of fp32 grads
    (2(N-1)/N x model bytes per worker + 2(N-1) latency hops). Also returns
    the skew-only bound with a free network — DDP's ceiling under 56:1 skew
    even on an infinitely fast link."""
    n = len(solo_img_s)
    t_compute = max(batch[w] / solo_img_s[w] for w in solo_img_s)
    vol = 2.0 * (n - 1) / n * FP32_BYTES * numel
    t_comm = vol / (link_mbps * 1e6) + 2 * (n - 1) * rtt_s
    samples = sum(batch[w] for w in solo_img_s)
    ceil = sum(solo_img_s.values())
    return {
        "t_compute_s": round(t_compute, 4),
        "t_allreduce_s": round(t_comm, 3),
        "img_s": round(samples / (t_compute + t_comm), 3),
        "util": round(samples / (t_compute + t_comm) / ceil, 6),
        "util_free_network": round(samples / t_compute / ceil, 4),
    }


def federated_util(
    period_s: float,
    payload_bytes: int,
    n_workers: int,
    mode: str,
    link_mbps: float = LINK_MBPS,
    rtt_s: float = LINK_RTT_S,
    pause_s: float = 0.035,
) -> Dict[str, float]:
    """Model: time-based rounds of ``period_s`` training seconds. The leader
    gathers ``n_workers`` payloads and broadcasts as many over the one link.

    * ``block``  : workers idle during the exchange -> T / (T + t_comm);
    * ``overlap``: workers keep training; they only pause ``pause_s`` per round
      for snapshot + apply (E5 measured 9-46 ms) -> 1 - pause / max(T, t_comm).

    Pessimistic for the measured fleet: 2 of 3 workers sit on the leader's LAN.
    """
    t_comm = 2.0 * n_workers * payload_bytes / (link_mbps * 1e6) + 2 * rtt_s
    if mode == "block":
        util = period_s / (period_s + t_comm)
    elif mode == "overlap":
        util = 1.0 - pause_s / max(period_s, t_comm)
    else:
        raise ValueError(mode)
    return {"t_comm_s": round(t_comm, 3), "util": round(util, 4)}


def profile_h_eff(
    solo_img_s: Mapping[str, float], batch: Mapping[str, int], period_s: float
) -> Dict[str, float]:
    """Local steps per round each GPU takes in a ``period_s`` round."""
    return {w: round(solo_img_s[w] * period_s / batch[w], 1) for w in solo_img_s}


# ----------------------------------------------------------------- fleet report
def fleet_report(
    results_dir: "str | pathlib.Path", runs: Sequence[str] = DEFAULT_RUNS
) -> List[Dict[str, Any]]:
    """Measured bandwidth reduction + compute utilisation for each fleet run."""
    out: List[Dict[str, Any]] = []
    for name in runs:
        run = parse_run_dir(pathlib.Path(results_dir) / name)
        numel_src = "logged"
        if not run["numel"] and name in R50_RUNS:
            run["numel"], numel_src = R50_NUMEL, "R50_NUMEL"
        if not run["rounds"] or not run["workers"] or not run["numel"]:
            continue
        batch = {w: v["batch"] for w, v in run["workers"].items()}
        row: Dict[str, Any] = {
            "run": name,
            "source": "measured",
            "mode": sorted({str(v["mode"]) for v in run["workers"].values()}),
            "numel_source": numel_src,
            "bandwidth": bandwidth_reduction(run["rounds"], batch, run["numel"]),
            "compute": None,
        }
        if name in R50_RUNS:
            img_s = {w: v["img_s"] for w, v in run["workers"].items()}
            row["compute"] = compute_utilisation(img_s, R50_SOLO_IMG_S)
        out.append(row)
    return out


# ----------------------------------------------------------------- sim
def simulate_skew(
    codec: Any,
    make_model: Callable[[], Any],
    data: Sequence[Any],
    steps_per_round: Mapping[str, int],
    batch: Mapping[str, int],
    rounds: int = 6,
    kind: str = "int8",
    lr: float = 0.05,
    seed: int = 1,
) -> Dict[str, Any]:
    """Time-based heterogeneous rounds on the CPU with the shipping codec.

    Each worker takes ``steps_per_round[w]`` local SGD steps per round (the
    skew: proportional to its throughput), uploads an EF-compressed delta,
    and the leader aggregates by sample count, encodes the broadcast and
    applies it with ``apply_round_to_anchor`` — the same primitives
    ``leader.py`` uses. DDP bytes are counted from the actual fp32 gradient
    tensors produced at every step, codec bytes from the payloads produced.
    """
    import torch  # local: keep the module importable without torch

    Xtr, ytr, Xte, yte = data
    torch.manual_seed(seed)
    names = list(steps_per_round)
    leader = make_model()
    fs = codec.FlatState(leader)
    anchor = torch.empty(fs.numel)
    fs.snapshot_params_to(anchor)
    bn = fs.bn_vec()
    var_mask = fs.bn_var_mask()
    mean_mask = ~var_mask

    g = torch.Generator().manual_seed(seed + 7)
    perm = torch.randperm(Xtr.shape[0], generator=g)
    shards = {w: (Xtr[idx], ytr[idx]) for w, idx in zip(names, perm.chunk(len(names)))}
    models = {w: make_model() for w in names}
    wfs = {w: codec.FlatState(models[w]) for w in names}
    ef: Dict[str, Any] = {w: None for w in names}
    ef_b: Optional[Any] = None
    flat = torch.empty(fs.numel)

    grad_bytes = {w: 0 for w in names}
    steps = {w: 0 for w in names}
    sim_rounds: List[Dict[str, Any]] = []
    for r in range(1, rounds + 1):
        payloads: Dict[str, bytes] = {}
        ns: Dict[str, float] = {}
        bnvecs = []
        for w in names:
            m = models[w]
            wfs[w].set_params_from(anchor)
            wfs[w].set_bn_from(bn)
            opt = torch.optim.SGD(m.parameters(), lr=lr, momentum=0.9)
            m.train()
            Xs, ys = shards[w]
            for _ in range(steps_per_round[w]):
                bi = torch.randint(0, Xs.shape[0], (batch[w],))
                opt.zero_grad()
                loss: Any = torch.nn.functional.cross_entropy(m(Xs[bi]), ys[bi])
                loss.backward()
                # what a DDP all-reduce of this step would carry: the fp32
                # grads of every float parameter (FlatState's quantized path)
                grad_bytes[w] += sum(
                    p.grad.numel() * FP32_BYTES for p in wfs[w].params
                )
                opt.step()
                steps[w] += 1
            wfs[w].snapshot_params_to(flat)
            payloads[w], ef[w], _ = codec.codec_encode(flat - anchor, ef[w], kind)
            ns[w] = float(steps_per_round[w] * batch[w])
            bnvecs.append(wfs[w].bn_vec())
        tot = sum(ns.values())
        dbar = torch.zeros(fs.numel)
        for w in names:
            dbar.add_(codec.codec_decode(payloads[w], fs.numel, kind), alpha=ns[w] / tot)
        if ef_b is not None:
            dbar = dbar + ef_b
        payload_b, ef_b, _ = codec.codec_encode(dbar, None, kind)
        codec.apply_round_to_anchor(anchor, payload_b, kind)
        bn = codec.merge_bn_pooled(bnvecs, [ns[w] for w in names], mean_mask, var_mask)
        sim_rounds.append(
            {
                "round": r,
                "n_samples": {w: int(ns[w]) for w in names},
                "bytes_up": {w: len(payloads[w]) for w in names},
                "bytes_down": len(payload_b),
            }
        )

    eval_model = make_model()
    efs = codec.FlatState(eval_model)
    efs.set_params_from(anchor)
    efs.set_bn_from(bn)
    eval_model.eval()
    with torch.no_grad():
        acc = float((eval_model(Xte).argmax(1) == yte).float().mean().item())
    bw = bandwidth_reduction(sim_rounds, batch, fs.numel)
    # cross-check: DDP bytes counted from the real gradients == steps x 4 x numel
    bw["dp_bytes_measured"] = sum(grad_bytes.values())
    bw["dp_bytes_expected"] = sum(steps.values()) * FP32_BYTES * fs.numel
    return {"source": "sim", "kind": kind, "rounds": rounds, "bandwidth": bw,
            "final_acc": round(acc, 4)}


def codec_payload_bytes(codec: Any, numel: int, kind: str = "int8", seed: int = 0) -> int:
    """Bytes the shipping codec emits for a heavy-tailed ``numel`` delta."""
    import torch

    g = torch.Generator().manual_seed(seed)
    x = torch.randn(numel, generator=g) * 1e-3
    x[torch.randperm(numel, generator=g)[: max(1, numel // 500)]] *= 200.0
    payload, _, _ = codec.codec_encode(x, None, kind)
    return len(payload)


def r50_projection(
    payload_bytes: int,
    numel: int,
    period_s: float = 30.0,
    shards: int = 1,
) -> Dict[str, Any]:
    """Fleet-profile projection at ResNet-50 size: H_eff from the measured
    solo throughputs x round period, codec bytes measured on a real encode."""
    h = profile_h_eff(R50_SOLO_IMG_S, R50_BATCH, period_s)
    dp = FP32_BYTES * numel
    per_sync = dp / (payload_bytes / shards)
    tot_h = sum(h.values())
    n = len(h)
    return {
        "source": "sim",
        "period_s": period_s,
        "shards": shards,
        "numel": numel,
        "payload_bytes": payload_bytes,
        "codec_ratio": round(dp / payload_bytes, 4),
        "H_eff": h,
        "per_worker_factor": {w: round(hw * per_sync, 1) for w, hw in h.items()},
        "fleet_factor": round(tot_h * per_sync / n, 1),
        "util_model": {
            "ddp": ddp_lockstep_util(R50_SOLO_IMG_S, R50_BATCH, numel),
            "block": federated_util(period_s, payload_bytes // shards, n, "block"),
            "overlap": federated_util(period_s, payload_bytes // shards, n, "overlap"),
        },
    }


# ----------------------------------------------------------------- markdown
def _fmt(x: Optional[float], unit: str = "x") -> str:
    return "n/a" if x is None else f"{x:,.1f}{unit}"


def _pct(x: Optional[float]) -> str:
    if x is None:
        return "n/a"
    return f"{100 * x:.2f}%" if x < 0.01 else f"{100 * x:.1f}%"


def to_markdown(report: Mapping[str, Any]) -> str:
    """Render the JSON report as the Markdown table committed under results/."""
    lines = [
        "# Bandwidth reduction and compute utilisation vs data-parallel",
        "",
        "Generated by `python sim_fed.py --report` (#94). Factor = fp32 bytes a DDP",
        "all-reduce would ship per optimizer step x steps taken / bytes actually",
        "shipped. Labels: **measured** = fleet logs in `results/`; **sim** = bytes",
        "from the shipping codec on CPU; **model** = wall-time model over the",
        f"measured link ({LINK_RTT_S * 1000:.0f} ms RTT, {LINK_MBPS} MB/s).",
        "",
        "## Measured on the fleet (ResNet-50 unless noted; 56:1 img/s skew)",
        "",
        "| Run | Mode | Rounds | H_eff P620 / 2080Ti / 4090 | Bytes/sync vs fp32 "
        "| Factor slowest | Factor fleet | Compute util |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for row in report.get("fleet", []):
        bw = row["bandwidth"]
        pw = bw["per_worker"]
        h = " / ".join(
            f"{pw[w]['H_eff']:.0f}" if w in pw else "-" for w in ("p620", "2080ti", "4090")
        )
        ratio = sorted({v["per_sync_ratio"] for v in pw.values()})
        nr = max((v["rounds"] for v in pw.values()), default=0)
        util = row["compute"]["util"] if row["compute"] else None
        name = row["run"] + (" (CIFAR R18)" if row["run"] == "e6" else "")
        lines.append(
            f"| {name} | {','.join(row['mode'])} | {nr} | {h} | "
            f"{'/'.join(f'{r:.2f}x' for r in ratio)} | {_fmt(bw['slowest_worker_factor'])} | "
            f"{_fmt(bw['fleet_factor'])} | {_pct(util)} |"
        )
    lines += [
        "",
        f"Sync DDP (gloo) over the same mesh, measured: {DDP_MEASURED_IMG_S} img/s "
        f"= {_pct(DDP_MEASURED_IMG_S / sum(R50_SOLO_IMG_S.values()))} compute util.",
    ]
    sim = report.get("sim")
    if sim:
        bw = sim["bandwidth"]
        lines += [
            "",
            "## Sim: skewed CPU fleet, shipping codec end-to-end",
            "",
            f"{sim['rounds']} rounds, codec `{sim['kind']}`, model numel {bw['numel']:,}; "
            f"final anchor acc {sim['final_acc']}. DDP bytes counted from real grads: "
            f"{bw['dp_bytes_measured']:,} (expected {bw['dp_bytes_expected']:,}).",
            "",
            "| Worker | H_eff | Bytes/sync vs fp32 | Factor |",
            "|---|---|---|---|",
        ]
        for w, v in bw["per_worker"].items():
            lines.append(
                f"| {w} | {v['H_eff']:.0f} | {v['per_sync_ratio']:.2f}x | {_fmt(v['factor'])} |"
            )
        lines.append(f"| **fleet** | | | **{_fmt(bw['fleet_factor'])}** |")
    proj = report.get("projection")
    if proj:
        um = proj["util_model"]
        h = proj["H_eff"]
        lines += [
            "",
            f"## Sim + model: ResNet-50 size, T={proj['period_s']:.0f} s rounds, "
            f"K={proj['shards']}",
            "",
            f"Measured int8 payload {proj['payload_bytes']:,} B for {proj['numel']:,} params "
            f"-> codec ratio {proj['codec_ratio']}x (scales + padding included). "
            f"H_eff from solo throughput: P620 {h['p620']:.0f}, 2080Ti {h['2080ti']:.0f}, "
            f"4090 {h['4090']:.0f}.",
            "",
            "| Worker | Factor |",
            "|---|---|",
        ]
        for w, f in proj["per_worker_factor"].items():
            lines.append(f"| {w} | {_fmt(f)} |")
        lines += [
            f"| **fleet (mean)** | **{_fmt(proj['fleet_factor'])}** |",
            "",
            "| Regime (model) | Compute util |",
            "|---|---|",
            f"| sync DDP, fp32 ring all-reduce | {_pct(um['ddp']['util'])} |",
            f"| sync DDP, free network (skew only) | {_pct(um['ddp']['util_free_network'])} |",
            f"| sakura block rounds | {_pct(um['block']['util'])} |",
            f"| sakura overlapped rounds | {_pct(um['overlap']['util'])} |",
        ]
    lines += [
        "",
        f"INTELLECT-1 reference: {INTELLECT1['factor']:.0f}x = H {INTELLECT1['H']} x "
        f"{INTELLECT1['bits_ratio']:.0f}x (fp32->int8), compute util 83-96%.",
        "",
    ]
    return "\n".join(lines)
