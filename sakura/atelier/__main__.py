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
# pick the wrong physical card on a multi-GPU host (e.g. a GPU already occupied by
# another service).
os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
# cuBLAS picks non-deterministic reduction algorithms unless this is set
# BEFORE the CUDA context exists; it is what lets a seeded replay land on the
# same numbers instead of "close" (a CIFAR replay was 0.018 off without it).
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

from sakura.atelier.backends import load_backend
from sakura.atelier.data import materialise
from sakura.atelier.registry import list_presets, resolve, resolved_yaml_text
from sakura.atelier.spec import AtelierSpec, load_spec
from sakura.atelier.types import Artifact, Checkpoint, MetricsSink, RunDir


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
    out = RunDir(Path(args.out)).ensure()
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

    data_dir = Path(args.data_dir) if args.data_dir else out.root / "data"
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
            result = backend.train(resolved, data, out, metrics, resume_checkpoint)

        artifacts: list[Artifact] = []
        for fmt in spec.export:
            artifacts.append(backend.export(result, fmt, out))

        runtime_eval = []
        for artifact in artifacts:
            ev = backend.runtime_eval(artifact, data)
            matches_training = ev.matches_training
            if matches_training is None and ev.value is not None:
                matches_training = _matches_training(ev, result.metrics)
            runtime_eval.append({
                "format": ev.format, "runtime": ev.runtime, "metric": ev.metric,
                "value": ev.value, "n": ev.n, "matches_training": matches_training,
                "reason": ev.reason,
            })

        billable_seconds = round(time.time() - start, 3)
        report = {
            "status": "done", "atelier": spec.atelier, "task": spec.task,
            "backend": backend_name, "model": spec.model, "hp": spec.hp,
            "spec_sha256": spec_sha256, "data_sha256": data.manifest_sha256,
            "billable_seconds": billable_seconds, "train_seconds": round(result.train_seconds, 3),
            "metrics": result.metrics,
            "runtime_eval": runtime_eval,
            "artifacts": [{"path": str(a.path.relative_to(out.root)), "format": a.format,
                           "sha256": a.sha256, "bytes": a.bytes} for a in artifacts],
            "resumed_from": resumed_from, "error": None,
        }
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
