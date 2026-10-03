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
from typing import Any, Callable, Optional, cast

from sakura.bench.harness import BaselineRunner, SakuraRunner, Workload


_WORKLOAD_REGISTRY = {
    "mnist-mlp": "sakura.bench.workloads.mnist:make_workload",
    "mnist-mlp-multi": "sakura.bench.workloads.mnist:make_workload_multi",
    "cifar10-resnet50": "sakura.bench.workloads.cifar:make_workload",
    "distilbert-sst2": "sakura.bench.workloads.distilbert:make_workload",
    "distilbert-sst2-hf": "sakura.bench.workloads.distilbert_hf:make_workload",
    "llama3-1b-finetune": "sakura.bench.workloads.llama:make_workload",
    "mistral-7b-lora": "sakura.bench.workloads.mistral:make_workload",
    "distilbert-glue": "sakura.bench.workloads.glue:make_workload",
    "gpt2-124m": "sakura.bench.workloads.gpt2:make_workload",
    "asr-deepspeech": "sakura.bench.workloads.asr:make_workload",
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
    return cast(Workload, factory())


def _make_async_eval(kwarg: Optional[str], workload: Workload) -> Any:
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

    dispatcher = _resolve_dispatcher(kwarg)

    # Bridge state: state_dict snapshot updated each epoch by the harness;
    # val_x / val_y materialized lazily on first eval call from val_loader.
    snapshot: dict[str, Any] = {"state_dict": None, "val_loader": None,
                                "val_x": None, "val_y": None}

    def _materialize_val(loader: Any) -> tuple[Any, Any]:
        import torch
        xs, ys = [], []
        for batch in loader:
            if isinstance(batch, (tuple, list)) and len(batch) == 2:
                xs.append(batch[0])
                ys.append(batch[1])
            else:
                # Bail to the loader path for non-(x, y) workloads.
                return None, None
        if not xs:
            return None, None
        return torch.cat(xs), torch.cat(ys)

    def bridged_eval_fn(epoch: int, payload: Any) -> dict[str, Any]:
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
                xb = val_x[i:i+bs]
                yb = val_y[i:i+bs]
                logits = m(xb)
                loss_sum += float(torch.nn.functional.cross_entropy(logits, yb, reduction="sum"))
                correct += int((logits.argmax(dim=-1) == yb).sum())
                total += int(yb.numel())
        return {"val_loss": loss_sum / max(total, 1),
                "val_acc": correct / max(total, 1),
                "epoch": epoch}

    def sync_eval_fn(model: Any, val_loader: Any) -> dict[str, Any]:
        # Adaptive-gate sync/calibration path: evaluate a CPU copy of the live
        # model so eval works regardless of the training model's device.  Using
        # a CPU copy:
        # (a) avoids device-placement mismatches when training on GPU but the
        #     workload's eval_fn loads data onto CPU (the common bench case);
        # (b) mirrors the async path which also CPU-materialises a state_dict
        #     snapshot in the eval thread — so calibration cost is an
        #     apples-to-apples comparison and the A/B gate is fair.
        import torch
        sd = {k: v.detach().cpu() if hasattr(v, "detach") else v
              for k, v in model.state_dict().items()}
        cpu_model = workload.make_model()
        cpu_model.load_state_dict(sd)
        cpu_model.eval()
        with torch.no_grad():
            return workload.eval_fn(cpu_model, val_loader)

    def device_eval_fn(model: Any, val_loader: Any) -> dict[str, Any]:
        # On-device eval path: evaluate the LIVE model where it trains. The
        # workload's eval_fn moves batches to the model's device, so GPU training
        # -> GPU eval (far faster than the CPU copy for compute-bound models like
        # an RNN). The warmup profiler uses this to decide whether overlapping a
        # CPU eval is even worth it; if not, eval runs here synchronously.
        import torch
        model.eval()
        with torch.no_grad():
            return workload.eval_fn(model, val_loader)

    def cpu_probe_fn(model: Any) -> float:
        # Cheap estimate of the *full* CPU-eval cost: time ONE val batch on a CPU
        # copy and scale by the batch count, so the warmup profiler doesn't pay a
        # whole (possibly slow) CPU eval just to measure it. Returns milliseconds.
        import time
        import torch
        loader = snapshot["val_loader"]
        batches = list(loader) if loader is not None else []
        if not batches:
            return 0.0
        sd = {k: v.detach().cpu() if hasattr(v, "detach") else v
              for k, v in model.state_dict().items()}
        cpu_model = workload.make_model()
        cpu_model.load_state_dict(sd)
        cpu_model.eval()
        t0 = time.perf_counter()
        with torch.no_grad():
            workload.eval_fn(cpu_model, [batches[0]])
        per_batch_ms = (time.perf_counter() - t0) * 1e3
        return per_batch_ms * len(batches)

    svc = AsyncEval(
        eval_fn=bridged_eval_fn,
        eval_payload=None,
        dispatcher=dispatcher,
        sync_eval_fn=sync_eval_fn,   # bug #1 fix: arms the adaptive gate
        device_eval_fn=device_eval_fn,  # warmup profiler: measure on-device eval
        cpu_probe_fn=cpu_probe_fn,      # warmup profiler: cheap CPU-eval estimate
        adaptive=True,
        calibration_epochs=2,
        trial_epochs=2,
        total_epochs=workload.epochs,
        max_pending=1,
        on_backpressure="skip",  # never stall training; skip eval if it can't keep up
    )
    svc._bench_snapshot = snapshot  # type: ignore[attr-defined]  # bridge surface read by the harness loop
    return svc


def _resolve_dispatcher(kind: Optional[str]) -> Any:
    """Map a CLI dispatcher kind string to a Dispatcher instance.

    Shared by `async_eval` and `async_checkpoint` factories.
    """
    kind = (kind or "thread").lower()
    if kind == "thread":
        from sakura.dispatch import ThreadDispatcher
        return ThreadDispatcher(max_workers=1)
    if kind == "local":
        from sakura.dispatch import LocalDispatcher
        return LocalDispatcher()
    if kind == "in_thread":
        from sakura.dispatch import InThreadDispatcher
        return InThreadDispatcher()
    raise ValueError(
        f"async dispatcher kind must be one of "
        f"{{'thread', 'local', 'in_thread'}}, got {kind!r}"
    )


def _make_async_checkpoint(kwarg: Optional[str], workload: Workload) -> Any:
    """Build an AsyncCheckpoint that snapshots the model each epoch.

    The harness loop populates `_bench_snapshot["state_dict"]` after each
    epoch; AsyncCheckpoint's state_provider reads it. Writes go to a
    per-run temp directory under the system tempdir; for production use
    callers should construct AsyncCheckpoint directly with a real `dir`.

    `kwarg` selects the dispatcher: `thread` (default), `local` (subprocess
    over QUIC), or `in_thread` (synchronous, debug only — no overlap).
    """
    import tempfile
    from sakura.services.async_checkpoint import AsyncCheckpoint

    dispatcher = _resolve_dispatcher(kwarg)
    snapshot: dict[str, Any] = {"state_dict": None, "val_loader": None,
                                "val_x": None, "val_y": None}
    out_dir = tempfile.mkdtemp(prefix=f"sakura-bench-ckpt-{workload.name}-")

    def state_provider() -> Any:
        return snapshot["state_dict"]

    svc = AsyncCheckpoint(
        dir=out_dir,
        dispatcher=dispatcher,
        state_provider=state_provider,
        every="epoch",
        keep=None,  # bench mode: don't bother rotating; tempdir is disposable
    )
    svc._bench_snapshot = snapshot  # type: ignore[attr-defined]  # bridge surface — same shape as async_eval
    svc._bench_out_dir = out_dir  # type: ignore[attr-defined]  # exposed for tests / cleanup
    return svc


def _make_activation_checkpoint(kwarg: Optional[str], workload: Workload) -> Any:
    """Build an ActivationCheckpoint targeting the workload's transformer
    block class(es).

    The CLI cannot know which submodules to checkpoint — that is a property
    of the model, declared by the Workload via ``block_types`` (e.g. the
    GPT-2 ``Block``). Fail loudly here if a workload asks for activation
    checkpointing without declaring any block types, so the misuse is obvious
    at build time instead of as an opaque ValueError deep inside the service.
    """
    from sakura.services.activation_checkpoint import ActivationCheckpoint

    if not workload.block_types:
        raise ValueError(
            f"workload {workload.name!r} does not declare any block_types; "
            "activation_checkpoint needs at least one transformer block class "
            "to wrap (set Workload.block_types in the workload factory)"
        )
    return ActivationCheckpoint(target_types=workload.block_types)


def _make_kernel_opt(kwarg: Optional[str], workload: Workload) -> Any:
    """Build a KernelOpt (cuDNN/TF32/channels_last backend knobs).

    kwarg presets: None -> defaults (cudnn_benchmark + tf32 + flatten_rnn);
    "channels_last"/"all" -> also enable channels_last; "no-benchmark" ->
    leave cudnn.benchmark off.
    """
    from sakura.services.kernel_opt import KernelOpt

    kw = (kwarg or "").lower()
    return KernelOpt(
        cudnn_benchmark="no-benchmark" not in kw,
        channels_last=kw in ("channels_last", "all"),
    )


_SERVICE_FACTORIES: dict[str, Callable[[Optional[str], Workload], Any]] = {
    "telemetry": lambda kw, wl: __import__(
        "sakura.services.telemetry", fromlist=["Telemetry"]
    ).Telemetry(output=lambda _r: None),
    "kernel_opt": _make_kernel_opt,
    "mixed_precision": lambda kw, wl: __import__(
        "sakura.services.mixed_precision", fromlist=["MixedPrecision"]
    ).MixedPrecision(dtype=kw or "auto"),
    "compile": lambda kw, wl: __import__(
        "sakura.services.compile", fromlist=["Compile"]
    ).Compile(mode=kw or "default"),
    "activation_checkpoint": _make_activation_checkpoint,
    "zero1": lambda kw, wl: __import__(
        "sakura.services.zero1", fromlist=["ZeRO1"]
    ).ZeRO1(),
    "async_eval": _make_async_eval,
    "async_checkpoint": _make_async_checkpoint,
}


def _build_services(specs: list[str], workload: Workload) -> list[Any]:
    """Parse `--service` specs into Service instances.

    Each spec is `name` or `name:kwarg` (one positional kwarg). Examples:
      --service telemetry
      --service mixed_precision:bf16
      --service compile:reduce-overhead
      --service async_eval:thread

    Workload is passed because some factories (notably async_eval) bridge
    workload.eval_fn to a service-specific signature.
    """
    services: list[Any] = []
    for s in specs:
        name, _, kwarg = s.partition(":")
        factory = _SERVICE_FACTORIES.get(name)
        if factory is None:
            raise ValueError(
                f"unknown service {name!r}; available: {sorted(_SERVICE_FACTORIES)}"
            )
        services.append(factory(kwarg or None, workload))
    return services


def _cmd_run(args: argparse.Namespace) -> int:
    wl = _resolve_workload(args.workload)
    if args.mode == "time-to-target":
        # CLI target overrides the workload's configured metric_target.
        if args.target_metric is not None and args.target_value is not None:
            wl.metric_target = (args.target_metric, args.target_value)
        if wl.metric_target is None:
            raise ValueError(
                "time-to-target mode requires a target: pass --target-metric and "
                "--target-value, or choose a workload with metric_target set."
            )
    if args.runner == "baseline":
        runner = BaselineRunner(
            framework=args.framework, mode=args.mode, max_epochs=args.max_epochs,
        )
    elif args.runner == "sakura":
        # Build services from --service flags. Default to telemetry-only if none given.
        specs = list(args.service) if args.service else ["telemetry"]
        services = _build_services(specs, wl)
        runner = SakuraRunner(
            framework=args.framework, services=services,
            mode=args.mode, max_epochs=args.max_epochs,
        )
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
    if report.reached_target is not None:
        print(f"reached_target: {report.reached_target}  "
              f"epochs_to_target: {report.epochs_to_target}")
    if report.final_metrics:
        print(f"final: {json.dumps(report.final_metrics)}")
    return 0


def _cmd_compare(args: argparse.Namespace) -> int:
    from sakura.bench.compare import load_reports, speedup_summary

    reports = load_reports(args.reports)
    if len(reports) < 2:
        print("compare needs at least 2 reports", file=sys.stderr)
        return 2
    # Pair them by index; first is baseline, second is sakura, etc.
    for i in range(0, len(reports) - 1, 2):
        print(speedup_summary(reports[i], reports[i + 1]))
    return 0


def _cmd_export(args: argparse.Namespace) -> int:
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
    p_run.add_argument("--mode", choices=["fixed", "time-to-target"], default="fixed",
                        help=("Run mode. 'time-to-target' stops after the workload's "
                              "metric_target is reached (pytorch-ddp framework only)."))
    p_run.add_argument("--target-metric", default=None,
                        help=("Metric name for time-to-target mode (e.g. val_loss, "
                              "perplexity, val_acc). Overrides workload.metric_target."))
    p_run.add_argument("--target-value", type=float, default=None,
                        help="Target value paired with --target-metric.")
    p_run.add_argument("--max-epochs", type=int, default=None,
                        help=("Epoch cap for time-to-target mode (defaults to the "
                              "workload's configured epochs)."))
    p_run.set_defaults(func=_cmd_run)

    p_cmp = sub.add_parser("compare", help="compare RunReport JSON files (pairs: baseline, sakura)")
    p_cmp.add_argument("reports", nargs="+", help="paths to JSON RunReport files")
    p_cmp.set_defaults(func=_cmd_compare)

    p_exp = sub.add_parser("export", help="render markdown table from RunReport JSON files")
    p_exp.add_argument("reports", nargs="+", help="paths to JSON RunReport files")
    p_exp.set_defaults(func=_cmd_export)

    args = parser.parse_args(argv)
    return cast(int, args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
