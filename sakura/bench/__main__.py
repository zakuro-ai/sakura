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
    "cifar10-resnet50": "sakura.bench.workloads.cifar:make_workload",
    "distilbert-sst2": "sakura.bench.workloads.distilbert:make_workload",
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


_SERVICE_FACTORIES = {
    "telemetry": lambda kw: __import__("sakura.services.telemetry", fromlist=["Telemetry"])
                              .Telemetry(output=lambda _r: None),
    "mixed_precision": lambda kw: __import__("sakura.services.mixed_precision",
                                              fromlist=["MixedPrecision"])
                              .MixedPrecision(dtype=kw or "auto"),
    "compile": lambda kw: __import__("sakura.services.compile", fromlist=["Compile"])
                              .Compile(mode=kw or "default"),
    "activation_checkpoint": lambda kw: __import__(
        "sakura.services.activation_checkpoint", fromlist=["ActivationCheckpoint"]
    ).ActivationCheckpoint(target_types=()),
    "zero1": lambda kw: __import__("sakura.services.zero1", fromlist=["ZeRO1"]).ZeRO1(),
}


def _build_services(specs: list[str]) -> list:
    """Parse `--service` specs into Service instances.

    Each spec is `name` or `name:kwarg` (one positional kwarg). Examples:
      --service telemetry
      --service mixed_precision:bf16
      --service compile:reduce-overhead
    """
    services = []
    for s in specs:
        name, _, kwarg = s.partition(":")
        factory = _SERVICE_FACTORIES.get(name)
        if factory is None:
            raise ValueError(
                f"unknown service {name!r}; available: {sorted(_SERVICE_FACTORIES)}"
            )
        services.append(factory(kwarg or None))
    return services


def _cmd_run(args) -> int:
    wl = _resolve_workload(args.workload)
    if args.runner == "baseline":
        runner = BaselineRunner(framework=args.framework)
    elif args.runner == "sakura":
        # Build services from --service flags. Default to telemetry-only if none given.
        specs = list(args.service) if args.service else ["telemetry"]
        services = _build_services(specs)
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
