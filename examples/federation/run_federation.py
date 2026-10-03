#!/usr/bin/env python3
"""Cross-region local-SGD federation example (ResNet-50, synthetic data).

One CPU-only leader process owns the canonical anchor and drives consensus
rounds; one worker process per GPU trains at full local speed and exchanges
compressed weight deltas with the leader. No dataset is required — workers
train on a synthetic ImageNet-shaped batch, so this runs anywhere torch +
torchvision are installed.

Three terminals, two regions (replace <leader-host> with the leader's
address as seen from each worker):

    # terminal 1 — leader (CPU-only, reachable by both workers)
    python examples/federation/run_federation.py leader \\
        --port 29500 --expected tokyo-a,paris-b --codec int8 --link-duty 0.5

    # terminal 2 — worker in region A
    python examples/federation/run_federation.py worker \\
        --leader-host <leader-host> --port 29500 --id tokyo-a \\
        --batch 96 --amp bf16 --region tokyo

    # terminal 3 — worker in region B
    python examples/federation/run_federation.py worker \\
        --leader-host <leader-host> --port 29500 --id paris-b \\
        --batch 64 --amp fp16 --region paris

Each committed round prints a LEADER_ROUND line (leader) and WORKER_ROUND
lines (workers); `hash16` is the consensus hash and must agree everywhere.
See docs/services/federation.md for the full protocol description.
"""

from __future__ import annotations

import argparse
import json
import logging
import time

import torch

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s"
)
log = logging.getLogger("examples.federation")


def build_model(device: torch.device) -> torch.nn.Module:
    """ResNet-50; the leader and every worker must build the same architecture."""
    import torchvision

    return torchvision.models.resnet50(num_classes=1000).to(device)


def run_leader(args: argparse.Namespace) -> None:
    from sakura.services.federation import Leader

    torch.set_num_threads(4)
    leader = Leader(
        model_fn=lambda: build_model(torch.device("cpu")),
        port=args.port,
        expected=[w.strip() for w in args.expected.split(",") if w.strip()],
        codec=args.codec,
        gap_s=args.gap,
        max_rounds=args.rounds or 10**9,
        shards=args.shards,
        link_duty=args.link_duty,
    )
    leader.serve_forever()  # blocks; rounds start once all workers connect
    print(
        "LEADER_DONE "
        + json.dumps({"rounds": leader.round, "overruns": leader.overruns}),
        flush=True,
    )


def run_worker(args: argparse.Namespace) -> None:
    from sakura.adapters.ddp import DDPAdapter
    from sakura.runtime import SakuraRuntime
    from sakura.services.federation import FederationService

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(device)
    opt = torch.optim.SGD(model.parameters(), lr=0.05, momentum=0.9)
    loss_fn = torch.nn.CrossEntropyLoss()

    amp = args.amp if device.type == "cuda" else "off"
    if amp != args.amp:
        log.warning("no CUDA device found; --amp %s disabled", args.amp)
    amp_dtype = {"off": None, "fp16": torch.float16, "bf16": torch.bfloat16}[amp]
    scaler = torch.amp.GradScaler("cuda", enabled=amp == "fp16")

    # Synthetic ImageNet-shaped batch: no dataset on disk required.
    x = torch.randn(args.batch, 3, 224, 224, device=device)
    y = torch.randint(0, 1000, (args.batch,), device=device)

    rt = SakuraRuntime()
    fed = FederationService(
        args.leader_host,
        args.port,
        args.id,
        args.batch,
        codec=args.codec,
        region=args.region,
    )
    rt.install(fed)
    adapter = DDPAdapter(rt, rank=0, world_size=1)

    step = 0
    losses: list[float] = []
    t0 = time.monotonic()
    with rt:
        adapter.on_train_begin(model=model, optimizer=opt, train_loader=None)
        adapter.on_epoch_begin(0)
        try:
            while args.steps == 0 or step < args.steps:
                adapter.on_train_step_begin(model=model, batch=(x, y), step=step)
                opt.zero_grad(set_to_none=True)
                if amp_dtype is not None:
                    with torch.autocast(device_type="cuda", dtype=amp_dtype):
                        loss = loss_fn(model(x), y)
                    if amp == "fp16":
                        scaler.scale(loss).backward()
                        scale_before = scaler.get_scale()
                        scaler.step(opt)
                        scaler.update()
                        # A scale drop means the step was SKIPPED (inf/nan
                        # grads) — don't count its samples toward the
                        # round's n_i weighting.
                        if scaler.get_scale() >= scale_before:
                            adapter.on_optimizer_step(optimizer=opt)
                    else:
                        loss.backward()
                        adapter.on_optimizer_step(optimizer=opt)
                        opt.step()
                else:
                    loss = loss_fn(model(x), y)
                    loss.backward()
                    adapter.on_optimizer_step(optimizer=opt)
                    opt.step()
                step += 1
                if step % 10 == 0:
                    losses.append(float(loss.item()))
                    del losses[:-50]
                    adapter.on_epoch_end(
                        0, model=model, optimizer=opt, metrics={"loss": losses[-1]}
                    )
        except KeyboardInterrupt:
            log.info("interrupted; shutting down cleanly")
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        adapter.on_train_end(model=model)

    elapsed = time.monotonic() - t0
    # Snapshot: the bg thread may still be appending if its shutdown join
    # expired; json.dumps over a live list raises mid-iteration.
    records = list(fed.records)
    print(
        "WORKER_DONE "
        + json.dumps(
            {
                "worker": args.id,
                "device": str(device),
                "steps": step,
                "img_s": round(step * args.batch / max(elapsed, 1e-9), 2),
                "rounds": len(records),
                "losses_tail": [round(v, 4) for v in losses[-5:]],
            }
        ),
        flush=True,
    )


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest="role", required=True)

    lp = sub.add_parser("leader", help="CPU-only aggregator process")
    lp.add_argument("--port", type=int, default=29500)
    lp.add_argument(
        "--expected",
        default="w0,w1",
        help="comma-separated worker ids that must connect before round 1",
    )
    lp.add_argument("--codec", default="int8", choices=["int8", "fp32raw"])
    lp.add_argument("--gap", type=float, default=30.0, help="seconds between rounds")
    lp.add_argument(
        "--link-duty",
        type=float,
        default=0.0,
        help=">0 = self-paced gap targeting this inter-region link occupancy "
        "(e.g. 0.5); overrides --gap after the first round",
    )
    lp.add_argument("--shards", type=int, default=1, help="K-way streaming shards")
    lp.add_argument(
        "--rounds", type=int, default=0, help="stop after N rounds (0=run forever)"
    )

    wp = sub.add_parser("worker", help="one process per GPU")
    wp.add_argument("--leader-host", default="127.0.0.1")
    wp.add_argument("--port", type=int, default=29500)
    wp.add_argument("--id", default="w0", help="must appear in the leader's --expected")
    wp.add_argument("--batch", type=int, default=32)
    wp.add_argument("--amp", default="off", choices=["off", "fp16", "bf16"])
    wp.add_argument("--region", default="default")
    wp.add_argument(
        "--codec",
        default="int8",
        choices=["int8", "fp32raw"],
        help="must match the leader",
    )
    wp.add_argument(
        "--steps", type=int, default=0, help="stop after N steps (0=run forever)"
    )

    args = p.parse_args()
    run_leader(args) if args.role == "leader" else run_worker(args)


if __name__ == "__main__":
    main()
