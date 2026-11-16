"""``python -m sakura.atelier run|validate|presets|predict`` -- the whole
interface a runner needs (CONTRACTS §2). ``run`` orchestrates materialise ->
backend.train -> export -> runtime_eval -> report.json; ``predict`` is a thin
CLI wrapper over ``Backend.predict`` that the runner's ``POST /infer/{id}``
calls through. Neither this module nor the registry imports a training stack
itself (each backend does, lazily, inside its own venv) -- ``predict``
dispatches to whichever backend produced the run's artifact, generically, so
it works for every track's backend without this file knowing their internals.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any

# Must be set before ANY backend lazily imports torch: CUDA's default device
# order is not nvidia-smi's PCI order, so a bare CUDA_VISIBLE_DEVICES=0 can
# pick the wrong physical card on a multi-GPU host (a resident service's GPU
# was picked by mistake once -- hit this for real).
os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
# cuBLAS picks non-deterministic reduction algorithms unless this is set
# BEFORE the CUDA context exists; it is what lets a seeded replay land on the
# same numbers instead of "close" (a CIFAR replay was 0.018 off without it).
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
# Fewer, larger-grained CUDA allocations: the allocator can grow its segments instead of splitting memory it can
# no longer join, which is what torch's own out-of-memory message recommends. Only when the operator set nothing.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

from sakura.atelier.backends import load_backend
from sakura.atelier.data import materialise
from sakura.atelier.evaluation import build_evaluation
from sakura.atelier.registry import list_presets, resolve, resolved_yaml_text
from sakura.atelier.spec import AtelierSpec, load_spec
from sakura.atelier.types import Artifact, Checkpoint, MetricsSink, RunDir
from sakura.atelier.wrap_up import REASON, REASON_MAX_STEPS, TimeBudget


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _write_report(out: RunDir, report: dict[str, Any]) -> None:
    (out.root / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n",
                                           encoding="utf-8")


def _artifact_entry(a: Artifact, out: RunDir, ev: dict[str, Any] | None = None) -> dict[str, Any]:
    """The artifact's ``report.json`` row (CONTRACTS §2.1). A backend's ``extras``
    (e.g. the whisper ggml export's ``footprint_bytes``) are added as further keys;
    they never shadow the contract keys.

    ``ev`` is the artifact's own ``runtime_eval`` row. ``validated`` says whether
    THIS file was re-scored after export (``eval_rows`` > 0 rows with a number),
    so a reader never takes an untested format for a tested one (BE-9);
    ``eval_acc`` is its accuracy (null for a non-accuracy metric, whose number
    is ``eval_metric`` / ``eval_value``)."""
    entry: dict[str, Any] = {"path": str(a.path.relative_to(out.root)), "format": a.format,
                             "sha256": a.sha256, "bytes": a.bytes, "size_bytes": a.bytes}
    value = ev.get("value") if ev else None
    rows = int(ev.get("n") or 0) if ev else 0
    validated = value is not None and rows > 0
    entry.update({
        "validated": validated, "eval_rows": rows if validated else 0,
        "eval_acc": value if validated and ev and ev.get("metric") == "accuracy" else None,
        "eval_metric": ev.get("metric") if validated and ev else None,
        "eval_value": value if validated else None,
        "matches_training": ev.get("matches_training") if validated and ev else None,
    })
    if not validated:
        entry["validation_reason"] = (ev or {}).get("reason") or "not re-scored after export"
    for k, v in (a.extras or {}).items():
        entry.setdefault(k, v)
    return entry


def _matches_training(ev: Any, train_metrics: dict[str, float]) -> bool | None:
    """CONTRACTS §2.3: a runtime_eval format's own metric vs the training
    metric of the SAME name, within tolerance (fp32 onnx: ±0.02; a backend
    that already knows its tolerance -- e.g. gguf/mlx's ±0.05 -- sets
    ev.matches_training itself and this is never called)."""
    train_value = train_metrics.get(ev.metric)
    if train_value is None:
        return None
    tolerance = 0.02 if ev.format == "onnx" else 0.05
    return bool(abs(ev.value - train_value) <= tolerance)


def _free_accelerator(backend: Any) -> None:
    """Let go of the accelerator memory this process holds once training, the exports and their re-scoring are
    done: what the backend still has loaded (its ``release()``, when it has one), then the allocator's cache.
    A 4,096-token text-generation run kept about 8 GiB of an 11 GiB card reserved here, and the notebook check,
    which re-scores the exports from its own process, ran out of memory. Never raises: a finished run is not
    lost over memory that could not be given back."""
    try:
        release = getattr(backend, "release", None)
        if callable(release):
            release()
        import gc

        gc.collect()
        torch = sys.modules.get("torch")  # only when the backend imported it: the engine core does not
        if torch is not None and torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception as exc:  # noqa: BLE001 -- see the docstring: this must never fail the run
        print(f"could not free the accelerator before the notebook check: {type(exc).__name__}: {exc}",
              file=sys.stderr)


def _limits_applied(spec: AtelierSpec, data: Any, result: Any, budget: TimeBudget) -> dict[str, Any]:
    """``report.json`` ``limits_applied`` (CONTRACTS §2.1): only the limits that changed this run. Row caps
    come from materialisation (with the row counts before and after), ``max_steps`` from the backend (its
    loop stopped on the cap, or it shortened a training that has no loop), ``time_budget_s`` from the
    budget's own wrap-up signal having stopped the loop. Empty when no limit changed anything."""
    limits = spec.limits
    if limits is None:
        return {}
    applied: dict[str, Any] = dict(data.extra.get("limits_applied") or {})
    applied.update(getattr(result, "limits_applied", None) or {})
    reason = (getattr(result, "stopped_early", None) or {}).get("reason")
    if reason == REASON_MAX_STEPS and limits.max_steps is not None:
        applied["max_steps"] = limits.max_steps
    if reason == REASON and budget.fired:
        applied["time_budget_s"] = limits.time_budget_s
    return applied


def _load_checkpoint(resume_from: Any, model: str, task: str) -> Checkpoint | None:
    if resume_from is None:
        return None
    uri = resume_from.uri
    # forge://<job_id> is resolved by the runner before it ever reaches the
    # engine (CONTRACTS §3.2); the engine only ever sees a local path here.
    path = Path(uri.removeprefix("file://"))
    if not path.exists():
        raise ValueError(f"resume_from.uri {uri!r} does not exist locally: {path}")
    if resume_from.sha256:
        # resume_from pins a single checkpoint file or archive; for a
        # directory checkpoint there is nothing single to hash, so only
        # verify when it is a file.
        if path.is_file() and _sha256_file(path) != resume_from.sha256:
            raise ValueError(f"resume_from.sha256 mismatch for {path}")
    return Checkpoint(path=path, model=model, task=task)


def cmd_run(args: argparse.Namespace) -> int:
    if args.device:
        # No `device` parameter on Backend.train (types.py's Protocol is
        # fixed, E2 codes against it) -- backends read this instead.
        os.environ["SAKURA_ATELIER_DEVICE"] = args.device
    # Absolute from the start: backends record paths (checkpoint, sidecars) that the
    # notebook's kernel later reads from INSIDE this directory, where a relative
    # `--out` would resolve to runs/x/runs/x. The runner always passes an absolute
    # path; a person typing `--out runs/x` must get the same notebook.
    out = RunDir(Path(args.out).resolve()).ensure()
    start = time.time()
    status = "done"
    error: str | None = None
    report: dict[str, Any] = {}

    spec_text = Path(args.spec).read_text(encoding="utf-8")
    try:
        spec = load_spec(spec_text)
    except Exception as exc:  # a spec that does not even parse still gets a report.json
        _write_report(out, {
            "status": "failed", "atelier": None, "task": None, "backend": None,
            "model": None, "hp": None, "spec_sha256": _sha256_text(spec_text),
            "data_sha256": None, "billable_seconds": round(time.time() - start, 3),
            "train_seconds": 0.0, "metrics": {}, "runtime_eval": [], "artifacts": [],
            "resumed_from": None, "error": f"{type(exc).__name__}: {exc}",
        })
        print(f"FAILED: {exc}", file=sys.stderr)
        return 1

    if spec.unconfirmed_guesses:
        _write_report(out, {
            "status": "failed", "atelier": spec.atelier, "task": spec.task, "backend": None,
            "model": spec.model, "hp": spec.hp, "spec_sha256": _sha256_text(spec_text),
            "data_sha256": None, "billable_seconds": round(time.time() - start, 3),
            "train_seconds": 0.0, "metrics": {}, "runtime_eval": [], "artifacts": [],
            "resumed_from": None,
            "error": f"unconfirmed guessed fields: {spec.unconfirmed_guesses}",
        })
        print("FAILED: unconfirmed guessed fields", file=sys.stderr)
        return 1

    data_dir = Path(args.data_dir).resolve() if args.data_dir else out.root / "data"
    backend_name = None

    try:
        resolved = resolve(spec)  # checks model/task/preset against the registry
        backend_name = resolved.backend
        backend = load_backend(backend_name)
        data = materialise(spec, data_dir)

        resolved_text = resolved_yaml_text(resolved, data.manifest_sha256)
        (out.root / "resolved.yaml").write_text(resolved_text, encoding="utf-8")
        spec_sha256 = _sha256_text(resolved_text)

        resume_checkpoint = _load_checkpoint(spec.resume_from, spec.model, spec.task)
        resumed_from = spec.resume_from.uri if spec.resume_from else None

        with MetricsSink(out.metrics_path) as metrics:
            _deterministic(resolved.spec.seed, getattr(backend, "deterministic_algorithms", True))
            # limits.time_budget_s: the budget raises the ordinary wrap-up signal, counted from here
            with TimeBudget(out.root, spec.limits.time_budget_s if spec.limits else None) as budget:
                result = backend.train(resolved, data, out, metrics, resume_checkpoint)

        artifacts: list[Artifact] = []
        for fmt in spec.export:
            artifacts.append(backend.export(result, fmt, out))
        # Files a backend delivers beside its exports and does not re-score (the decision task's
        # decision_config.json): listed with a path, format, sha256 and size, like the notebook.
        extras: list[Artifact] = list(getattr(backend, "extra_artifacts", lambda *_: [])(result, out))

        runtime_eval: list[dict[str, Any]] = []
        for artifact in artifacts:
            try:
                ev = backend.runtime_eval(artifact, data)
            except Exception as exc:  # noqa: BLE001 -- a crashing check must never lose a finished training
                # KeyboardInterrupt and SystemExit are not Exceptions and still stop the run; the wrap-up and
                # time-budget signals are files, not exceptions. This artifact's row says why it was not scored.
                print(traceback.format_exc(), file=sys.stderr)
                reason = f"{type(exc).__name__}: {exc}".replace("\r", " ").replace("\n", " ")
                runtime_eval.append({
                    "format": artifact.format, "runtime": None, "metric": None, "value": None, "n": 0,
                    "matches_training": None, "reason": reason if len(reason) <= 300 else reason[:297] + "...",
                    "provider": None, "training_delta": None,
                })
                continue
            matches_training = ev.matches_training
            # a check on fewer rows than the final evaluation is judged against the training score on those rows
            train_value = getattr(ev, "train_value", None)
            train_metrics = result.metrics if train_value is None else {**result.metrics, ev.metric: train_value}
            # a reason beside a value says the two were not comparable (a capped check with no training score on its rows)
            if matches_training is None and ev.value is not None and ev.reason is None:
                matches_training = _matches_training(ev, train_metrics)
            # how far the re-scored file is from the training metric of the same name (None when there is
            # none): `matches_training` says whether that is within the tolerance, this says by how much
            trained = train_metrics.get(ev.metric)
            delta = None
            if ev.value is not None and isinstance(trained, (int, float)) and not isinstance(trained, bool):
                delta = round(float(ev.value) - float(trained), 6)
                delta = delta if delta == delta else None  # a NaN training metric has no distance
            runtime_eval.append({
                "format": ev.format, "runtime": ev.runtime, "metric": ev.metric,
                "value": ev.value, "n": ev.n, "matches_training": matches_training,
                "reason": ev.reason, "provider": getattr(ev, "provider", None), "training_delta": delta,
            })
            if train_value is not None:  # how many rows each side used, and the training-side score on them
                runtime_eval[-1].update(training_value=train_value, training_n=getattr(ev, "train_n", None))
            if getattr(ev, "n_total", None):  # capped: how many rows it was capped from
                runtime_eval[-1]["n_total"] = ev.n_total
            if getattr(ev, "rows_eval_only", 0):  # rows with no target: not re-scored, said so
                runtime_eval[-1]["rows_eval_only"] = ev.rows_eval_only
            if getattr(ev, "rows_skipped", 0):  # rows the runtime could not take: left out, said so
                runtime_eval[-1]["rows_skipped"] = ev.rows_skipped
        onnx_provider = next((r["provider"] for r in runtime_eval if r["format"] == "onnx"), None)
        evaluation = build_evaluation(backend, result, resolved, data)

        from sakura.atelier.licence import collect_licence
        licence = collect_licence(spec.model, resolved.hyperparameters.get("base_model"), out,
                                  base_revision=(result.provenance or {}).get("base_commit")
                                  or resolved.hyperparameters.get("base_revision"))
        for rel in licence["files"]:
            p = out.root / rel
            extras.append(Artifact(path=p, format="license", sha256=hashlib.sha256(p.read_bytes()).hexdigest(),
                                   bytes=p.stat().st_size))

        curation = data.extra.get("curation")
        if curation:
            # the curated manifest: every kept and dropped sample id with the rule that dropped it (CONTRACTS §1.6)
            from sakura.atelier.curation import MANIFEST_NAME

            manifest_path = out.artifacts / MANIFEST_NAME
            manifest_path.parent.mkdir(parents=True, exist_ok=True)
            manifest_path.write_text(json.dumps(curation["manifest"], indent=2, sort_keys=True) + "\n", encoding="utf-8")
            extras.append(Artifact(path=manifest_path, format="curated_manifest",
                                   sha256=_sha256_file(manifest_path), bytes=manifest_path.stat().st_size))

        billable_seconds = round(time.time() - start, 3)
        report = {
            "status": "done", "atelier": spec.atelier, "task": spec.task,
            "backend": backend_name, "model": spec.model, "hp": spec.hp,
            "spec_sha256": spec_sha256, "data_sha256": data.manifest_sha256,
            "billable_seconds": billable_seconds, "train_seconds": round(result.train_seconds, 3),
            "metrics": result.metrics,
            "runtime_eval": runtime_eval, "onnx_provider": onnx_provider,
            "evaluation": evaluation,
            "artifacts": [*(_artifact_entry(a, out, ev) for a, ev in zip(artifacts, runtime_eval)),
                          *({"path": str(x.path.relative_to(out.root)), "format": x.format, "sha256": x.sha256,
                             "bytes": x.bytes} for x in extras)],
            "resumed_from": resumed_from, "error": None,
            "license": licence,
        }
        if data.extra.get("data_split"):
            report["data_split"] = data.extra["data_split"]
        if data.extra.get("data_encoding"):
            report["data_encoding"] = data.extra["data_encoding"]
        if curation:
            report["curation"] = {**curation["report"], "manifest": str(manifest_path.relative_to(out.root)),
                                  "manifest_sha256": _sha256_file(manifest_path)}
        limits_applied = _limits_applied(spec, data, result, budget)
        if limits_applied:
            report["limits_applied"] = limits_applied
        if result.provenance:
            report["provenance"] = result.provenance
        if getattr(result, "calibration", None):
            report["calibration"] = result.calibration
        if getattr(result, "early_stop", None):
            report["early_stop"] = result.early_stop
        if getattr(result, "stopped_early", None):
            report["stopped_early"] = result.stopped_early
        if getattr(result, "epoch_metrics", None):
            report["epoch_metrics"] = result.epoch_metrics
        if getattr(result, "augmentation", None):
            report["augmentation"] = result.augmentation
        for key, block in (getattr(result, "report", None) or {}).items():
            report.setdefault(key, block)  # a backend's own blocks (CONTRACTS §2.1): never over a contract key
        if os.environ.get("SAKURA_ATELIER_NOTEBOOK") != "0" and not getattr(args, "no_notebook", False):
            # Lazy: nbformat/nbclient/matplotlib live in the backend venv, and
            # the engine core must import without them (CONTRACTS §2).
            from sakura.atelier.notebook.bundle import write_notebook_bundle

            # The check replays the run in another process (the notebook's kernel) on the same card: this one
            # must not sit on the memory it trained and re-scored with while that process runs.
            _free_accelerator(backend)
            nb_artifact, check = write_notebook_bundle(out, resolved, resolved_text, data, backend,
                                                       backend_name, report)
            if nb_artifact is not None:
                report["artifacts"].append({"path": str(nb_artifact.path.relative_to(out.root)),
                                            "format": nb_artifact.format, "sha256": nb_artifact.sha256,
                                            "bytes": nb_artifact.bytes})
            report["notebook_check"] = check
            report["billable_seconds"] = round(time.time() - start, 3)  # the check is machine time too (§2.1)
        _write_report(out, report)
        return 0
    except Exception as exc:
        status = "failed"
        error = f"{type(exc).__name__}: {exc}"
        print(traceback.format_exc(), file=sys.stderr)
        _write_report(out, {
            "status": status, "atelier": spec.atelier, "task": spec.task,
            "backend": backend_name, "model": spec.model, "hp": spec.hp,
            "spec_sha256": None, "data_sha256": None,
            "billable_seconds": round(time.time() - start, 3), "train_seconds": 0.0,
            "metrics": {}, "runtime_eval": [], "artifacts": [],
            "resumed_from": spec.resume_from.uri if spec.resume_from else None,
            "error": error,
        })
        return 1


def cmd_validate(args: argparse.Namespace) -> int:
    spec_text = Path(args.spec).read_text(encoding="utf-8")
    try:
        spec = load_spec(spec_text)
        resolve(spec)  # also checks model/task/preset against the registry
    except Exception as exc:
        print(f"INVALID: {exc}", file=sys.stderr)
        return 1
    if spec.unconfirmed_guesses:
        print(f"INVALID: unconfirmed guessed fields: {spec.unconfirmed_guesses}", file=sys.stderr)
        return 1
    print("OK")
    return 0


def cmd_presets(args: argparse.Namespace) -> int:
    for p in list_presets(args.task):
        print(p)
    return 0


def _delivered_artifact(run_dir: Path, fmt: str | None) -> tuple[dict[str, Any], Artifact]:
    """`(report, artifact)` for a finished run: the requested format, else
    the first one exported."""
    report = json.loads((run_dir / "report.json").read_text(encoding="utf-8"))
    if report.get("status") != "done":
        raise ValueError(f"run status is {report.get('status')!r}, not 'done': {report.get('error')}")
    entries = report.get("artifacts") or []
    if fmt:
        entry = next((e for e in entries if e["format"] == fmt), None)
        if entry is None:
            raise ValueError(f"no {fmt!r} artifact in this run (have: {[e['format'] for e in entries]})")
    elif entries:
        entry = entries[0]
    else:
        raise ValueError("this run has no artifacts to predict from")
    artifact = Artifact(path=run_dir / entry["path"], format=entry["format"],
                        sha256=entry["sha256"], bytes=entry["bytes"])
    return report, artifact


def cmd_routes(args: argparse.Namespace) -> int:
    """Print the routing table (JSON), optionally for one image family or a backend list."""
    from sakura.atelier.registry.routes import families, routes
    backends = None
    if args.family:
        backends = families()[args.family]
    if args.backends:
        backends = [b for b in args.backends.split(",") if b]
    print(json.dumps(routes(backends), sort_keys=True))
    return 0


def cmd_serve(args: argparse.Namespace) -> int:
    """Answer predictions for one delivered run, one JSON line in, one JSON
    line out, until stdin closes -- the runner's warm try-it worker.

    Request: ``{"input": "<path to inputs JSON>"}`` or ``{"file": "<path>"}``
    (the same two shapes as ``predict --input`` / ``--file``). The model is
    loaded on the first request and kept (the backends memoise their
    loaders), so only that one pays the load. Library chatter goes to stderr;
    stdout carries nothing but answers, so the protocol cannot desync.
    """
    out = sys.stdout
    try:
        report, artifact = _delivered_artifact(Path(args.run_dir), args.format)
        if report.get("task") == "decision":
            # The warm worker answers on the CPU and leaves the GPU to the next training job. Set before
            # the backend imports torch, which reads it once. Other tasks' workers are not touched.
            os.environ["CUDA_VISIBLE_DEVICES"] = ""
        backend = load_backend(report["backend"])
    except Exception as exc:
        out.write(json.dumps({"error": f"{type(exc).__name__}: {exc}"}) + "\n")
        out.flush()
        return 1
    for line in sys.stdin:
        if not line.strip():
            continue
        try:
            req = json.loads(line)
            if "input" in req:
                inputs: Any = json.loads(Path(req["input"]).read_text(encoding="utf-8"))
            elif "file" in req:
                inputs = {"file": str(req["file"])}
            else:
                raise ValueError("a request needs 'input' or 'file'")
            with contextlib.redirect_stdout(sys.stderr):
                result = backend.predict(artifact, inputs)
            out.write(json.dumps(result) + "\n")
        except Exception as exc:
            out.write(json.dumps({"error": f"{type(exc).__name__}: {exc}"}) + "\n")
        out.flush()
    return 0


def cmd_predict(args: argparse.Namespace) -> int:
    """Load the delivered artifact from RUN_DIR (a ``run --out`` directory)
    and print one JSON object on stdout in the per-task shape of CONTRACTS
    §3.1. Dispatches to whichever backend produced the run (read from its
    report.json), generically -- this file never imports a backend itself."""
    run_dir = Path(args.run_dir)
    try:
        report, artifact = _delivered_artifact(run_dir, args.format)

        if args.input and args.file:
            raise ValueError("pass --input or --file, not both")
        if args.input:
            inputs: Any = json.loads(Path(args.input).read_text(encoding="utf-8"))
        elif args.file:
            inputs = {"file": str(Path(args.file))}
        else:
            raise ValueError("predict needs --input FILE.json or --file FILE")

        backend = load_backend(report["backend"])
        # stdout is the response channel: libraries that log to it (Ultralytics
        # announces "Loading model.onnx for ONNX Runtime inference...") would
        # corrupt the JSON the runner parses, so they talk to stderr instead.
        with contextlib.redirect_stdout(sys.stderr):
            result = backend.predict(artifact, inputs)
        print(json.dumps(result))
        return 0
    except Exception as exc:
        print(json.dumps({"error": f"{type(exc).__name__}: {exc}"}))
        return 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m sakura.atelier")
    sub = parser.add_subparsers(dest="command", required=True)

    p_run = sub.add_parser("run", help="materialise, train, export, evaluate, report")
    p_run.add_argument("spec")
    p_run.add_argument("--out", required=True)
    p_run.add_argument("--data-dir", default=None)
    p_run.add_argument("--device", default=None)
    p_run.add_argument("--no-notebook", action="store_true",
                       help="skip the notebook bundle (CONTRACTS §2.5)")
    p_run.set_defaults(func=cmd_run)

    p_validate = sub.add_parser("validate", help="schema + registry check, no run")
    p_validate.add_argument("spec")
    p_validate.set_defaults(func=cmd_validate)

    p_presets = sub.add_parser("presets", help="list preset ids")
    p_presets.add_argument("--task", default=None)
    p_presets.set_defaults(func=cmd_presets)

    p_predict = sub.add_parser("predict", help="run Backend.predict on a run's delivered artifact")
    p_predict.add_argument("run_dir")
    p_predict.add_argument("--input", default=None, help="JSON file holding the request's `inputs`")
    p_predict.add_argument("--file", default=None, help="an uploaded image/audio/CSV file")
    p_predict.add_argument("--format", default=None, help="which exported format to use (default: first)")
    p_predict.set_defaults(func=cmd_predict)

    p_routes = sub.add_parser("routes", help="print the routing table generated from models.yaml")
    p_routes.add_argument("--family", help="only the backends of this image family")
    p_routes.add_argument("--backends", help="comma-separated backends")
    p_routes.set_defaults(func=cmd_routes)

    p_serve = sub.add_parser("serve", help="answer predictions for a delivered run, JSON lines on stdin/stdout")
    p_serve.add_argument("run_dir")
    p_serve.add_argument("--format", default=None, help="which exported format to use (default: first)")
    p_serve.set_defaults(func=cmd_serve)

    return parser


def _deterministic(seed: int, algorithms: bool = True) -> None:
    """Seed every RNG and ask torch for deterministic kernels.

    ``algorithms=False`` (a backend's ``deterministic_algorithms = False``) keeps the seeds
    and cuDNN determinism but leaves ``torch.use_deterministic_algorithms`` off. CTC training
    needs that: with it on, PyTorch routes CTC loss to cuDNN, whose backward returns NaN for
    a whole batch when one sample is unalignable (it ignores ``zero_infinity``).

    `warn_only=True`: an op with no deterministic implementation warns
    instead of failing the job -- the replay test, not a crash, is what says
    whether that op mattered. Imported lazily: the engine must import without
    torch (CONTRACTS.md §2)."""
    import random

    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:  # pragma: no cover
        pass
    try:
        import torch
    except ImportError:  # pragma: no cover -- a backend without torch
        return
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if algorithms:
        torch.use_deterministic_algorithms(True, warn_only=True)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result: int = args.func(args)
    return result


if __name__ == "__main__":
    raise SystemExit(main())
