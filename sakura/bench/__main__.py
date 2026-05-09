"""`sakura-bench` CLI.

Subcommands:
  run      — run a Workload via BaselineRunner or SakuraRunner; write RunReport JSON.
  compare  — compare two RunReport JSON files; print a speedup summary.
  export   — render a markdown table from one or more RunReport JSON files.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Optional

from sakura.bench.harness import BaselineRunner, RunReport, SakuraRunner, Workload


_WORKLOAD_REGISTRY = {
    "mnist-mlp": "sakura.bench.workloads.mnist:make_workload",
    "mnist-mlp-multi": "sakura.bench.workloads.mnist:make_workload_multi",
    "cifar10-resnet50": "sakura.bench.workloads.cifar:make_workload",
    "distilbert-sst2": "sakura.bench.workloads.distilbert:make_workload",
    "distilbert-sst2-hf": "sakura.bench.workloads.distilbert_hf:make_workload",
    "llama3-1b-finetune": "sakura.bench.workloads.llama:make_workload",
    "mistral-7b-lora": "sakura.bench.workloads.mistral:make_workload",
    "distilbert-glue": "sakura.bench.workloads.glue:make_workload",
}


def _resolve_workload(name: str) -> Workload:
    if name not in _WORKLOAD_REGISTRY:
        raise ValueError(
            f"unknown workload {name!r}; available: {sorted(_WORKLOAD_REGISTRY)}"
        )
    spec = _WORKLOAD_REGISTRY[name]
    module_path, attr = spec.split(":")
    import importlib
    mod = importlib.import_module(module_path)
    factory = getattr(mod, attr)
    return factory()


def _make_async_eval(kwarg, workload):
    """Build an AsyncEval bridged to the workload's eval_fn.

    The harness loop populates `_bench_snapshot["state_dict"]` after each
    epoch; the bridged eval_fn rebuilds the model from that state_dict and
    runs the eval over a *materialized* val tensor pair (cached on the
    first call). The DataLoader iterator's per-batch Python overhead would
    otherwise hold the GIL and defeat the thread-overlap win — slicing
    through a flat tensor pair is GIL-free for the inner forward pass.

    `kwarg` selects the dispatcher: `thread` (default), `local` (subprocess
    over QUIC), or `in_thread` (synchronous, debug only — no overlap).
    """
    from sakura.services.async_eval import AsyncEval

    kind = (kwarg or "thread").lower()
    if kind == "thread":
        from sakura.dispatch import ThreadDispatcher
        dispatcher = ThreadDispatcher(max_workers=1)
    elif kind == "local":
        from sakura.dispatch import LocalDispatcher
        dispatcher = LocalDispatcher()
    elif kind == "in_thread":
        from sakura.dispatch import InThreadDispatcher
        dispatcher = InThreadDispatcher()
    else:
        raise ValueError(
            f"async_eval dispatcher kind must be one of "
            f"{{'thread', 'local', 'in_thread'}}, got {kwarg!r}"
        )

    # Bridge state: state_dict snapshot updated each epoch by the harness;
    # val_x / val_y materialized lazily on first eval call from val_loader.
    snapshot: dict = {"state_dict": None, "val_loader": None,
                      "val_x": None, "val_y": None}

    def _materialize_val(loader):
        import torch
        xs, ys = [], []
        for batch in loader:
            if isinstance(batch, (tuple, list)) and len(batch) == 2:
                xs.append(batch[0]); ys.append(batch[1])
            else:
                # Bail to the loader path for non-(x, y) workloads.
                return None, None
        if not xs:
            return None, None
        return torch.cat(xs), torch.cat(ys)

    def bridged_eval_fn(epoch: int, payload):
        import torch
        sd = snapshot["state_dict"]
        if sd is None:
            return {"epoch": epoch, "skipped": True, "reason": "no-snapshot"}
        # Materialize val tensors once. The eval thread does this on its
        # first call so the data lives in shared memory and the inner loop
        # is pure tensor slicing — torch matmul releases the GIL.
        if snapshot["val_x"] is None:
            x, y = _materialize_val(snapshot["val_loader"])
            snapshot["val_x"], snapshot["val_y"] = x, y
        val_x, val_y = snapshot["val_x"], snapshot["val_y"]
        if val_x is None:
            # Workload's batches aren't (x, y) tuples — fall back to the
            # DataLoader path. Loses some overlap but stays correct.
            m = workload.make_model()
            m.load_state_dict(sd)
            return workload.eval_fn(m, snapshot["val_loader"])
        # Tensor-slice eval path.
        m = workload.make_model()
        m.load_state_dict(sd)
        m.eval()
        bs = 64
        correct = total = 0
        loss_sum = 0.0
        with torch.no_grad():
            for i in range(0, val_x.shape[0], bs):
                xb = val_x[i:i+bs]; yb = val_y[i:i+bs]
                logits = m(xb)
                loss_sum += float(torch.nn.functional.cross_entropy(logits, yb, reduction="sum"))
                correct += int((logits.argmax(dim=-1) == yb).sum())
                total += int(yb.numel())
        return {"val_loss": loss_sum / max(total, 1),
                "val_acc": correct / max(total, 1),
                "epoch": epoch}

    svc = AsyncEval(
        eval_fn=bridged_eval_fn,
        eval_payload=None,
        dispatcher=dispatcher,
        max_pending=1,
        on_backpressure="block",
    )
    svc._bench_snapshot = snapshot  # bridge surface read by the harness loop
    return svc


_SERVICE_FACTORIES = {
    "telemetry": lambda kw, wl: __import__(
        "sakura.services.telemetry", fromlist=["Telemetry"]
    ).Telemetry(output=lambda _r: None),
    "mixed_precision": lambda kw, wl: __import__(
        "sakura.services.mixed_precision", fromlist=["MixedPrecision"]
    ).MixedPrecision(dtype=kw or "auto"),
    "compile": lambda kw, wl: __import__(
        "sakura.services.compile", fromlist=["Compile"]
    ).Compile(mode=kw or "default"),
    "activation_checkpoint": lambda kw, wl: __import__(
        "sakura.services.activation_checkpoint", fromlist=["ActivationCheckpoint"]
    ).ActivationCheckpoint(target_types=()),
    "zero1": lambda kw, wl: __import__(
        "sakura.services.zero1", fromlist=["ZeRO1"]
    ).ZeRO1(),
    "async_eval": _make_async_eval,
}


def _build_services(specs: list[str], workload) -> list:
    """Parse `--service` specs into Service instances.

    Each spec is `name` or `name:kwarg` (one positional kwarg). Examples:
      --service telemetry
      --service mixed_precision:bf16
      --service compile:reduce-overhead
      --service async_eval:thread

    Workload is passed because some factories (notably async_eval) bridge
    workload.eval_fn to a service-specific signature.
    """
    services = []
    for s in specs:
        name, _, kwarg = s.partition(":")
        factory = _SERVICE_FACTORIES.get(name)
        if factory is None:
            raise ValueError(
                f"unknown service {name!r}; available: {sorted(_SERVICE_FACTORIES)}"
            )
        services.append(factory(kwarg or None, workload))
    return services


def _cmd_run(args) -> int:
    wl = _resolve_workload(args.workload)
    if args.runner == "baseline":
        runner = BaselineRunner(framework=args.framework)
    elif args.runner == "sakura":
        # Build services from --service flags. Default to telemetry-only if none given.
        specs = list(args.service) if args.service else ["telemetry"]
        services = _build_services(specs, wl)
        runner = SakuraRunner(framework=args.framework, services=services)
    else:
        raise ValueError(f"unknown runner {args.runner!r}")

    report = runner.run(wl)

    out_path = args.output
    if os.path.isdir(out_path):
        # Auto-name: <workload>-<runner>-<framework>.json
        out_path = os.path.join(out_path, f"{args.workload}-{args.runner}-{args.framework}.json")
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(report.to_json())
    print(f"wrote: {out_path}")
    print(f"elapsed: {report.elapsed_secs:.2f}s  samples/sec: {report.samples_per_sec:.1f}")
    if report.final_metrics:
        print(f"final: {json.dumps(report.final_metrics)}")
    return 0


def _cmd_compare(args) -> int:
    from sakura.bench.compare import load_reports, speedup_summary

    reports = load_reports(args.reports)
    if len(reports) < 2:
        print("compare needs at least 2 reports", file=sys.stderr)
        return 2
    # Pair them by index; first is baseline, second is sakura, etc.
    for i in range(0, len(reports) - 1, 2):
        print(speedup_summary(reports[i], reports[i + 1]))
    return 0


def _cmd_export(args) -> int:
    from sakura.bench.compare import load_reports, render_markdown_table

    reports = load_reports(args.reports)
    print(render_markdown_table(reports))
    return 0


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(prog="sakura-bench")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_run = sub.add_parser("run", help="run a workload + write a RunReport JSON")
    p_run.add_argument("--workload", required=True, choices=sorted(_WORKLOAD_REGISTRY))
    p_run.add_argument("--runner", choices=["baseline", "sakura"], default="baseline")
    p_run.add_argument("--framework", choices=["pytorch-ddp", "lightning", "hf-trainer"],
                        default="pytorch-ddp")
    p_run.add_argument("--output", default=".",
                        help="Output JSON path (or directory; auto-names if dir).")
    p_run.add_argument("--service", action="append", default=[],
                        help=(
                            "Service to install on the SakuraRuntime (repeatable). "
                            "Format: name[:kwarg]. Available: "
                            f"{sorted(_SERVICE_FACTORIES)}. Examples: "
                            "--service mixed_precision:bf16 --service compile:reduce-overhead"
                        ))
    p_run.set_defaults(func=_cmd_run)

    p_cmp = sub.add_parser("compare", help="compare RunReport JSON files (pairs: baseline, sakura)")
    p_cmp.add_argument("reports", nargs="+", help="paths to JSON RunReport files")
    p_cmp.set_defaults(func=_cmd_compare)

    p_exp = sub.add_parser("export", help="render markdown table from RunReport JSON files")
    p_exp.add_argument("reports", nargs="+", help="paths to JSON RunReport files")
    p_exp.set_defaults(func=_cmd_export)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
