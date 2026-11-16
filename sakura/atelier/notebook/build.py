"""The seven sections. Code cells are the source of what ran; driver cells are
templates that call it, with `ATELIER_REPLAY=check` short-circuiting Training
and Export to the run's checkpoint and artifacts (that is what the agent runs).

A backend's `NOTEBOOK_CELLS` tuple lists, in pipeline order, the functions the
notebook shows by source. Its `train`, `export` and `runtime_eval` split the
tuple into sections: helpers before `train` are the Model section (how the
model is built), helpers after `train` belong to Training, after `export` to
Export and after `runtime_eval` to Evaluation. A backend without the tuple
shows just those three.

The Evaluation driver cell consumes `artifacts` from the Export driver, so it
is emitted after it; both carry their section in `metadata.atelier`, and the
headings themselves stay in contract order (the static check reads headings).
"""

from __future__ import annotations

import json
import logging
import os
import textwrap
from dataclasses import dataclass
from typing import Any

from sakura.atelier.notebook import cells as C
from sakura.atelier.notebook.recorder import NotebookRecorder
from sakura.atelier.types import MaterialisedData, ResolvedSpec, RunDir

#: The three methods every backend has; they anchor the helpers to sections.
_STEPS = ("train", "export", "runtime_eval")
_STEP_SECTION = {"train": "## Training", "export": "## Export", "runtime_eval": "## Evaluation"}


@dataclass
class BuildContext:
    run_dir: RunDir
    resolved: ResolvedSpec
    resolved_yaml: str
    data: MaterialisedData
    #: The loaded backend object (`load_backend(backend_name)`).
    backend: Any
    backend_name: str
    #: The report so far: metrics, train_seconds, runtime_eval, artifacts.
    report: dict[str, Any]
    #: `environment.capture_environment()`; inlined into the Setup cell.
    environment: dict[str, Any]


def backend_cells(backend: Any) -> list[tuple[str, Any]]:
    """``[(label, function)]`` from the backend's ``NOTEBOOK_CELLS``, with
    `train`, `export` and `runtime_eval` appended (in that order) when the
    tuple leaves any of them out, so every notebook shows the three steps."""
    fns = list(getattr(backend, "NOTEBOOK_CELLS", ()) or ())
    named = [(_label(f), f) for f in fns]
    have = {n for n, _ in named}
    for step in _STEPS:
        if step not in have and hasattr(backend, step):
            named.append((step, getattr(backend, step)))
    return named


def _label(fn: Any) -> str:
    name = getattr(fn, "__name__", None)
    return str(name) if name else repr(fn)


def _sections_of(fns: list[tuple[str, Any]]) -> dict[str, list[tuple[str, Any]]]:
    """Split the pipeline-ordered list by the step each helper follows."""
    out: dict[str, list[tuple[str, Any]]] = {h: [] for h in C.SECTION_HEADINGS}
    section = "## Model"
    for label, fn in fns:
        if label in _STEP_SECTION:
            section = _STEP_SECTION[label]
        out[section].append((label, fn))
    return out


#: Chaos knob for the repair loop: `SAKURA_ATELIER_NOTEBOOK_FAULT=<template>` ships
#: that driver cell with a realistic bug (a wrong file name, a misspelt attribute),
#: so the first execution fails and the agent has to repair it. For verifying the
#: agent on a real fleet only -- never set it where a customer notebook is built.
FAULT_ENV = "SAKURA_ATELIER_NOTEBOOK_FAULT"
FAULTS: dict[str, tuple[str, str]] = {
    # A wrong file name is NOT a fault here: `curves` reports a missing
    # metrics.jsonl as text and the cell passes (staging job 0dd5a18a). A
    # wrong method name raises whatever the run recorded.
    "curves": ("NotebookRecorder().curves(", "NotebookRecorder().plot_curves("),
    "eval": ('"value": ev.value', '"value": ev.score'),
}


def requested_fault() -> str | None:
    """The template named by `FAULT_ENV`, validated; `None` when unset."""
    name = os.environ.get(FAULT_ENV, "").strip()
    if not name:
        return None
    if name not in FAULTS:
        raise ValueError(f"{FAULT_ENV}={name!r}: unknown template, choose one of {sorted(FAULTS)}")
    return name


def _with_fault(name: str, template: str) -> str:
    if requested_fault() != name:
        return template
    old, new = FAULTS[name]
    assert old in template, f"fault for {name!r} no longer matches its template"
    logging.getLogger(__name__).warning("%s=%s: shipping the %r cell with an injected bug", FAULT_ENV, name, name)
    return template.replace(old, new, 1)


def _code(template: str, section: str, name: str, outputs: list[dict[str, Any]] | None = None,
          **values: str) -> dict[str, Any]:
    return C.code(C.fill(_with_fault(name, template), **values), outputs,
                  atelier={"template": name, "section": section})


def _source_cells(section: str, fns: list[tuple[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for label, fn in fns:
        src, note = C.source_of(fn)
        if note:
            out.append(C.markdown(f"_{label}_: {note}", atelier={"template": f"source:{label}", "section": section}))
        else:
            # A method's source comes back indented by its class; a cell must parse on
            # its own. Its annotations name types its module imported (`AtelierSpec`,
            # `MaterialisedData`); without the future import the `def` itself would
            # raise a NameError in the kernel, and the drivers call the module's real
            # functions anyway, so the cell only has to define.
            out.append(C.code("from __future__ import annotations\n" + textwrap.dedent(src),
                              atelier={"template": f"source:{label}", "section": section}))
    return out


def _inline(text: str) -> str:
    """Make ``text`` safe inside a triple-double-quoted Python literal: a
    backslash must survive as itself, and a `\"\"\"` would end the literal."""
    return text.replace("\\", "\\\\").replace('"""', '\\"\\"\\"')


SETUP = '''import os, json, sys
from pathlib import Path

# ATELIER_REPLAY=check: Training and Export reuse this run's checkpoint and artifacts,
# everything else runs for real. Unset it to train again from scratch.
REPLAY = os.environ.get("ATELIER_REPLAY", "") == "check"
RUN_DIR = Path(os.environ.get("ATELIER_RUN_DIR", "."))
SPEC_YAML = """@@SPEC_YAML@@"""
ENVIRONMENT = json.loads("""@@ENVIRONMENT_JSON@@""")
print("backend:", ENVIRONMENT.get("backend"), "| image:", ENVIRONMENT.get("image"), "| python:", ENVIRONMENT.get("python"))
print("replay mode:", REPLAY, "| run dir:", RUN_DIR.resolve())
print("nothing is installed here: environment/init.sh (uv), environment/Dockerfile and")
print("environment/requirements.lock.txt rebuild the exact environment this ran in")
'''

RESOLVE = '''from sakura.atelier.spec import load_spec
from sakura.atelier.registry import resolve
from sakura.atelier.backends import load_backend
spec = load_spec(SPEC_YAML)
resolved = resolve(spec)
backend = load_backend("@@BACKEND@@")
print("task:", spec.task, "| model:", spec.model, "| preset:", spec.hp, "| seed:", spec.seed)
'''

DATA = '''from sakura.atelier.data import materialise
from sakura.atelier.notebook.recorder import frame_summary
data = materialise(spec, RUN_DIR / "data")
print(frame_summary(data.train, "train")); print(); print(frame_summary(data.validation, "validation"))
print("labels:", data.labels)
print("manifest sha256:", data.manifest_sha256, "| spec data.sha256:", spec.data.sha256 or "(unpinned)")
'''

MODEL = '''import json
print(json.dumps(resolved.hyperparameters, indent=2, sort_keys=True))
'''

TRAIN = '''import json
from sakura.atelier.types import RunDir, MetricsSink
from sakura.atelier.__main__ import _deterministic
out = RunDir(RUN_DIR).ensure()
if REPLAY:
    report = json.loads((RUN_DIR / "report.json").read_text())
    print("replay: using this run's recorded training result;", "train_seconds =", report["train_seconds"])
    print(json.dumps(report["metrics"], indent=2))
    result = None
else:
    _deterministic(spec.seed, getattr(backend, "deterministic_algorithms", True))
    with MetricsSink(out.metrics_path) as metrics:
        result = backend.train(resolved, data, out, metrics, None)
    print("train_seconds =", round(result.train_seconds, 1)); print(json.dumps(result.metrics, indent=2))
'''

CURVES = '''import base64
from IPython.display import display, Image
from sakura.atelier.notebook.recorder import NotebookRecorder
for o in NotebookRecorder().curves(RUN_DIR / "metrics.jsonl"):
    if o["output_type"] == "display_data":
        display(Image(data=base64.b64decode(o["data"]["image/png"])))
    else:
        print(o["text"], end="")
'''

EXPORT = '''from pathlib import Path
from sakura.atelier.types import Artifact
if REPLAY:
    artifacts = [Artifact(path=RUN_DIR / a["path"], format=a["format"], sha256=a["sha256"], bytes=a["bytes"])
                 for a in report["artifacts"] if a["format"] != "notebook"]
    print("replay: using this run's exported artifacts")
else:
    artifacts = [backend.export(result, fmt, out) for fmt in spec.export]
for a in artifacts:
    print(a.format, a.path, a.sha256[:12], a.bytes, "bytes")
'''

EVAL = '''import os
rows = []
eval_data = data
if REPLAY:
    cap = int(os.environ.get("ATELIER_REPLAY_EVAL_ROWS", "256"))
    try:
        eval_data = type(data)(root=data.root, train=data.train, validation=data.validation[:cap],
                               manifest_sha256=data.manifest_sha256, labels=data.labels, extra=data.extra)
    except Exception:
        eval_data = data  # a split that cannot be sliced is scored whole
for a in artifacts:
    ev = backend.runtime_eval(a, eval_data)
    rows.append({"format": ev.format, "runtime": ev.runtime, "metric": ev.metric, "value": ev.value,
                 "n": ev.n, "matches_training": ev.matches_training, "reason": ev.reason,
                 "provider": getattr(ev, "provider", None)})
    if REPLAY and ev.value is None:
        # The run scored this export; a replay that cannot is a broken notebook, not a result.
        recorded = next((r for r in report.get("runtime_eval", []) if r["format"] == ev.format), None)
        if recorded and recorded.get("value") is not None:
            raise RuntimeError(f"replay could not re-score {ev.format} ({ev.reason}); the run scored it {recorded['value']}")
for r in rows:
    print(r)
onnx_provider = next((r["provider"] for r in rows if r["format"] == "onnx"), None)
print("onnx provider (what the session actually ran on):", onnx_provider)
if onnx_provider == "CPUExecutionProvider":
    print("note: ran on CPU; a GPU build of onnxruntime falls back to CPU when a CUDA library cannot be loaded")
'''

EVALUATION = '''import base64
from IPython.display import display, Image, HTML
from sakura.atelier.notebook.recorder import NotebookRecorder
# The evaluation block (CONTRACTS section 2.6) is computed in the run and recorded in report.json.
if REPLAY:
    evaluation_report = json.loads((RUN_DIR / "report.json").read_text())
else:
    from sakura.atelier.evaluation import build_evaluation
    evaluation_report = {"evaluation": build_evaluation(backend, result, resolved, data)}
for o in NotebookRecorder().evaluation_view(evaluation_report):
    if o["output_type"] == "display_data" and "image/png" in o["data"]:
        display(Image(data=base64.b64decode(o["data"]["image/png"])))
    elif o["output_type"] == "display_data":
        display(HTML(o["data"]["text/html"]))
    else:
        print(o["text"], end="")
'''

PACKAGE = '''import json
from pathlib import Path
def _rel(p):
    p = Path(p)
    return str(p.relative_to(RUN_DIR)) if p.is_relative_to(RUN_DIR) else str(p)
manifest = {"spec": SPEC_YAML,
            "artifacts": [{"path": _rel(a.path), "format": a.format, "sha256": a.sha256, "bytes": a.bytes} for a in artifacts],
            "runtime_eval": rows, "environment": ENVIRONMENT}
for name in ("labels.json", "preprocess.json"):
    p = RUN_DIR / "artifacts" / name
    print(f"{name}:", p.read_text()[:2000] if p.exists() else "not produced")
Path("package-manifest.json").write_text(json.dumps(manifest, indent=2, default=str))
print("wrote package-manifest.json")
env_dir = next((d for d in (Path("environment"), RUN_DIR / "artifacts" / "notebook" / "environment") if d.is_dir()), None)
if env_dir is None:
    print("environment files: (see environment/ next to this notebook)")
else:
    for p in sorted(env_dir.iterdir()):
        print(f"environment/{p.name}: {p.stat().st_size} bytes")
'''


def build_notebook(ctx: BuildContext, recorder: NotebookRecorder | None = None) -> dict[str, Any]:
    requested_fault()
    rec = recorder or NotebookRecorder()
    by_section = _sections_of(backend_cells(ctx.backend))
    spec = ctx.resolved.spec
    title = f"L'Atelier: {spec.name} ({spec.task}, {spec.model})"
    env_json = _inline(json.dumps(ctx.environment, sort_keys=True, default=str))
    body: list[dict[str, Any]] = [
        C.markdown(f"# {title}\n\nThis notebook is the pipeline the runner executed for this job: the code "
                   "below is the code that ran (shown by source), and the outputs are the values it "
                   "measured. Run it top to bottom with `ATELIER_REPLAY=check` to verify everything from "
                   "the run's checkpoint, or unset to train again."),
        C.markdown("## Setup\n\nThe resolved spec (preset expanded), the execution environment, and the "
                   "replay switch. Nothing is installed from here: `environment/init.sh` does that."),
        _code(SETUP, "## Setup", "setup", SPEC_YAML=_inline(ctx.resolved_yaml), ENVIRONMENT_JSON=env_json),
        _code(RESOLVE, "## Setup", "resolve", BACKEND=ctx.backend_name),
        C.markdown("## Data\n\nLoad the dataset the spec names, verify its manifest digest, and explore it. "
                   "`materialise` is the engine's own loader, shown here by source."),
        *_source_cells("## Data", [("materialise", _materialise_fn())]),
        _code(DATA, "## Data", "data", rec.data_summary(ctx.data, spec)),
        C.markdown("## Model\n\nHow the backend builds the model from the resolved hyper-parameters."),
        *_source_cells("## Model", by_section["## Model"]),
        _code(MODEL, "## Model", "model", rec.hyperparameters(ctx.resolved)),
        C.markdown("## Training\n\nThe backend's `train`, by source, then the driver and the curves from "
                   "`metrics.jsonl`."),
        *_source_cells("## Training", by_section["## Training"]),
        _code(TRAIN, "## Training", "train", rec.training_summary(ctx.report)),
        _code(CURVES, "## Training", "curves", rec.curves(ctx.run_dir.metrics_path)),
        C.markdown("## Evaluation\n\nEvery exported format re-scored in its own runtime on the validation "
                   "split, against the training metric (CONTRACTS §2.3 tolerances). The driver runs after "
                   "the Export driver below, which produces the artifacts it scores."),
        *_source_cells("## Evaluation", by_section["## Evaluation"]),
        C.markdown("## Export\n\nThe backend's `export`, by source, and the artifacts it produced."),
        *_source_cells("## Export", by_section["## Export"]),
        _code(EXPORT, "## Export", "export", rec.artifact_table(ctx.report)),
        _code(EVAL, "## Evaluation", "eval", rec.eval_table(ctx.report)),
        _code(EVALUATION, "## Evaluation", "evaluation", rec.evaluation_view(ctx.report)),
        C.markdown("## Package\n\nThe manifest (artifacts with sha256, labels, preprocessing) and the "
                   "environment files."),
        _code(PACKAGE, "## Package", "package", rec.manifest_view(ctx.run_dir)),
        C.markdown(C.REPAIRS_HEADING + "\n\nNo repairs were needed; the first execution passed.",
                   atelier={"repairs": True}),
    ]
    meta = {"atelier": {"title": title, "backend": ctx.backend_name, "task": spec.task, "model": spec.model}}
    return C.notebook(body, title=title, metadata=meta)


def _materialise_fn() -> Any:
    from sakura.atelier.data import materialise

    return materialise
