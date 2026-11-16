"""Directory -> agent -> tar -> Artifact. The one entry point the CLI calls
(CONTRACTS §2.5), and it never raises: a broken bundle is a `notebook_check`
with the error on stderr and in the report, and the model is still delivered.

Order matters inside: the environment files and the notebook are built first
(a build error is reported before any kernel starts), then `report.json` is
written beside the run because the replay path's Training and Export cells
read it, then the agent runs with the run directory as its cwd, and only then
is the notebook written -- the delivered file carries the agent's outputs and
its Repairs section, not the pre-execution copy.

Only the caller's `environ` (default `os.environ`) is read, and only for the
knobs named below; `SAKURA_ATELIER_LLM_KEY` is consumed by `agent.repairer_from_env`
and stripped from the kernel's environment there.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import sys
import tarfile
import traceback
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from sakura.atelier.notebook import agent as A
from sakura.atelier.notebook import cells as C
from sakura.atelier.notebook import environment as E
from sakura.atelier.notebook.build import BuildContext, build_notebook
from sakura.atelier.types import Artifact, MaterialisedData, ResolvedSpec, RunDir

#: The run files copied into the bundle so it is self-describing offline.
_COPIES = ("resolved.yaml", "report.json", "metrics.jsonl")

_README = """# L'Atelier run: @@TITLE@@

`notebook.ipynb` is the pipeline the runner executed for this job, with the outputs it measured.

Run it three ways:

1. **The runner image** (exact environment): `docker build -t atelier-run environment/ && docker run --gpus all -p 8888:8888 -v "$PWD":/work atelier-run`
   (the Dockerfile pins the image by digest when the run was containerised, else the CUDA base it reported).
2. **A bare machine with uv**: `bash environment/init.sh` then `../.venv/bin/jupyter lab notebook.ipynb`.
3. **Replay check** (no training, verifies everything from this run's checkpoint): `ATELIER_REPLAY=check ATELIER_RUN_DIR=<run dir> jupyter nbconvert --execute notebook.ipynb`.

Files: `resolved.yaml` (the replayable spec), `report.json`, `metrics.jsonl`, `environment/` (image.json, requirements.lock.txt, apt-packages.txt, init.sh, Dockerfile).
Notebook check by the agent: @@CHECK@@
"""


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()


def _failed(error: str, seconds: float = 0.0) -> dict[str, Any]:
    """A `notebook_check` for a bundle that broke before the agent could judge it."""
    return A.CheckResult(False, 0, 0, seconds, None, error).as_report()


def _recorded_outputs(nb: dict[str, Any]) -> dict[int, list[dict[str, Any]]]:
    """The engine-recorded outputs, keyed by cell identity (a patch replaces a
    cell's source in place and inserts new dicts, so identity survives both)."""
    return {id(c): list(c["outputs"]) for c in nb["cells"] if c.get("cell_type") == "code" and c.get("outputs")}


def _restore_unreached(nb: dict[str, Any], recorded: dict[int, list[dict[str, Any]]]) -> None:
    """After a failed check the cells past the failure never ran and the agent
    left them empty; the delivered file shows the run's own values there (the
    design: "the recorded outputs from the run and the agent's last attempt")."""
    for c in nb["cells"]:
        if c.get("cell_type") == "code" and c.get("execution_count") is None and not c.get("outputs"):
            if id(c) in recorded:
                c["outputs"] = recorded[id(c)]


def write_notebook_bundle(
    run_dir: RunDir,
    resolved: ResolvedSpec,
    resolved_yaml: str,
    data: MaterialisedData,
    backend: Any,
    backend_name: str,
    report: dict[str, Any],
    *,
    environ: Mapping[str, str] | None = None,
    log: Callable[[str], Any] = print,
) -> tuple[Artifact | None, dict[str, Any]]:
    """Write ``artifacts/notebook/`` (notebook, README, copies, environment),
    run the agent on it, tar it to ``artifacts/notebook.tar``. Returns the
    ``notebook`` Artifact and the ``notebook_check`` dict; on any exception
    ``(None, {"ok": False, ..., "error": ...})`` with the traceback on stderr."""
    e: Mapping[str, str] = os.environ if environ is None else environ
    bundle_dir = run_dir.artifacts / "notebook"
    try:
        shutil.rmtree(bundle_dir, ignore_errors=True)
        env_dir = bundle_dir / "environment"
        env_dir.mkdir(parents=True)
        env = E.capture_environment(backend_name, e)
        E.write_environment(env_dir, env, E.requirements_lock(), E.apt_packages())

        nb = build_notebook(BuildContext(run_dir, resolved, resolved_yaml, data, backend, backend_name, report, env))
        problems = C.check_notebook(nb)
        if problems:
            raise AssertionError("built an invalid notebook: " + "; ".join(problems))

        # The replay path reads report.json beside the run (Training and Export
        # cells); it is not written by the CLI yet, so write it here, then copy.
        (run_dir.root / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n",
                                                   encoding="utf-8")
        for name in _COPIES:
            src = run_dir.root / name
            if src.exists():
                shutil.copy(src, bundle_dir / name)

        recorded = _recorded_outputs(nb)
        # Absolute on purpose: the kernel's cwd IS the run dir, so a relative
        # `--out` (runs/x) would resolve inside it (runs/x/runs/x). Found by the
        # agent on the first real run -- it repaired the cell; this is the fix.
        run_root = run_dir.root.resolve()
        check = A.repair_loop(
            nb,
            run_root,
            {
                "ATELIER_REPLAY": "check",
                "ATELIER_RUN_DIR": str(run_root),
                "MPLBACKEND": "Agg",
                "ATELIER_REPLAY_EVAL_ROWS": e.get("ATELIER_REPLAY_EVAL_ROWS", "256"),
            },
            A.repairer_from_env(e),
            attempts=int(e.get("ATELIER_NOTEBOOK_REPAIR_ATTEMPTS", "3")),
            budget_seconds=float(e.get("ATELIER_NOTEBOOK_CHECK_SECONDS", "600")),
            log=log,
        )
        verdict = check.as_report()
        _restore_unreached(nb, recorded)

        title = str(nb["metadata"]["atelier"]["title"])
        (bundle_dir / "notebook.ipynb").write_text(json.dumps(nb, indent=1) + "\n", encoding="utf-8")
        (bundle_dir / "README.md").write_text(C.fill(_README, TITLE=title, CHECK=json.dumps(verdict)),
                                              encoding="utf-8")
        tar_path = run_dir.artifacts / "notebook.tar"
        with tarfile.open(tar_path, "w") as tf:
            tf.add(bundle_dir, arcname="notebook")
        return Artifact(tar_path, "notebook", _sha256(tar_path), tar_path.stat().st_size), verdict
    except Exception as exc:  # the bundle never fails the job (CONTRACTS §2.5)
        print(traceback.format_exc(), file=sys.stderr)
        log(f"notebook bundle failed: {type(exc).__name__}: {exc}")
        return None, _failed(f"{type(exc).__name__}: {exc}")
