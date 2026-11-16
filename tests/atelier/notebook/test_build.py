"""The seven sections are built from the code that ran and the values it
measured; no kernel is started here, and no training stack is imported."""

import importlib

import pandas as pd

from sakura.atelier.notebook import build, cells
from sakura.atelier.spec import load_spec
from sakura.atelier.types import MaterialisedData, ResolvedSpec, RunDir

SPEC = """
atelier: 1
name: t
task: tabular_classification
data: {uri: file:///tmp/x.csv, format: csv, split: {validation_fraction: 0.2}}
features: {inputs: [{name: x, type: number}], output: {name: y, type: category}}
model: ludwig_ecd
hp: tabular/ecd@1
export: [onnx]
"""


class FakeBackend:
    name = "fake"
    NOTEBOOK_CELLS = ()

    def train(self, spec, data, out, metrics, resume): return None

    def export(self, result, fmt, out): return None

    def runtime_eval(self, artifact, data): return None


def _ctx(tmp_path, backend=None):
    spec = load_spec(SPEC)
    rd = RunDir(tmp_path / "run").ensure()
    (rd.root / "metrics.jsonl").write_text("")
    df = pd.DataFrame({"x": [1.0, 2.0], "y": ["a", "b"]})
    data = MaterialisedData(root=tmp_path, train=df, validation=df, manifest_sha256="00" * 32,
                            labels=["a", "b"])
    resolved = ResolvedSpec(spec=spec, hyperparameters={"epochs": 2}, backend="fake")
    report = {"metrics": {"accuracy": 1.0}, "train_seconds": 1.0, "runtime_eval": [], "artifacts": []}
    return build.BuildContext(run_dir=rd, resolved=resolved, resolved_yaml=SPEC, data=data,
                              backend=backend or FakeBackend(), backend_name="fake", report=report,
                              environment={"image": None, "python": "3.11", "backend": "fake"})


def _headings(nb):
    return [cells.source_text(c) for c in nb["cells"]
            if c["cell_type"] == "markdown" and cells.source_text(c).startswith("## ")]


def test_build_passes_the_static_check_and_has_every_section_in_order(tmp_path):
    nb = build.build_notebook(_ctx(tmp_path))
    assert cells.check_notebook(nb) == []
    heads = [h.split("\n", 1)[0] for h in _headings(nb)]
    assert heads[:7] == list(cells.SECTION_HEADINGS) and heads[-1] == cells.REPAIRS_HEADING


def test_code_cells_carry_template_and_section_metadata_and_the_spec_is_inlined(tmp_path):
    nb = build.build_notebook(_ctx(tmp_path))
    codes = [c for c in nb["cells"] if c["cell_type"] == "code"]
    assert codes
    assert all(c["metadata"]["atelier"]["template"] and c["metadata"]["atelier"]["section"] for c in codes)
    setup = "\n".join(cells.source_text(c) for c in codes if c["metadata"]["atelier"]["section"] == "## Setup")
    assert "task: tabular_classification" in setup and "ATELIER_REPLAY" in setup
    assert '"backend": "fake"' in setup  # the environment JSON is inlined too


def test_backend_functions_appear_by_source(tmp_path):
    nb = build.build_notebook(_ctx(tmp_path))
    text = "\n".join(cells.source_text(c) for c in nb["cells"])
    assert "def train(self, spec, data, out, metrics, resume)" in text
    assert "def materialise(spec" in text  # the engine's own loader, by source


def test_eval_driver_runs_after_the_export_driver_but_keeps_its_section(tmp_path):
    nb = build.build_notebook(_ctx(tmp_path))
    templates = [c["metadata"]["atelier"]["template"] for c in nb["cells"] if c["cell_type"] == "code"]
    assert templates.index("export") < templates.index("eval")
    by_t = {c["metadata"]["atelier"]["template"]: c for c in nb["cells"] if c["cell_type"] == "code"}
    assert by_t["eval"]["metadata"]["atelier"]["section"] == "## Evaluation"
    assert by_t["export"]["metadata"]["atelier"]["section"] == "## Export"


def test_recorded_outputs_are_attached_to_the_driver_cells(tmp_path):
    nb = build.build_notebook(_ctx(tmp_path))
    by_t = {c["metadata"]["atelier"]["template"]: c for c in nb["cells"] if c["cell_type"] == "code"}
    assert "train: 2 rows x 2 columns" in by_t["data"]["outputs"][0]["text"]
    assert "metrics.jsonl is empty" in by_t["curves"]["outputs"][0]["text"]
    assert "accuracy" in by_t["train"]["outputs"][0]["data"]["text/plain"]


def test_triple_quotes_and_backslashes_in_the_spec_survive_inlining(tmp_path):
    ctx = _ctx(tmp_path)
    ctx.resolved_yaml = SPEC + 'description: "a \\"quoted\\" """ thing"\n'
    ctx.environment = {"image": 'sha256:"""x', "python": "3.11", "backend": "fake", "path": "C:\\x"}
    nb = build.build_notebook(ctx)
    assert cells.check_notebook(nb) == []


def test_unreadable_source_becomes_a_note_not_a_crash(tmp_path):
    class Builtin:
        name = "b"
        NOTEBOOK_CELLS = (len,)
        train = staticmethod(len)
        export = staticmethod(len)
        runtime_eval = staticmethod(len)

    nb = build.build_notebook(_ctx(tmp_path, backend=Builtin()))
    assert cells.check_notebook(nb) == []
    text = "\n".join(cells.source_text(c) for c in nb["cells"])
    assert "source unavailable" in text


def test_helpers_are_placed_in_the_section_of_the_step_they_follow(tmp_path):
    def build_model(): ...

    def train(spec, data, out, metrics, resume): ...

    def train_helper(): ...

    def export(result, fmt, out): ...

    def runtime_eval(artifact, data): ...

    def eval_helper(): ...

    class B:
        name = "b"
        NOTEBOOK_CELLS = (build_model, train, train_helper, export, runtime_eval, eval_helper)

    for f in B.NOTEBOOK_CELLS:
        setattr(B, f.__name__, staticmethod(f))
    assert [n for n, _ in build.backend_cells(B())] == [
        "build_model", "train", "train_helper", "export", "runtime_eval", "eval_helper"]
    nb = build.build_notebook(_ctx(tmp_path, backend=B()))
    sec = {c["metadata"]["atelier"]["template"]: c["metadata"]["atelier"]["section"]
           for c in nb["cells"] if c["cell_type"] == "code"}
    assert sec["source:build_model"] == "## Model"
    assert sec["source:train_helper"] == "## Training"
    assert sec["source:eval_helper"] == "## Evaluation"


def test_backend_cells_defaults_to_train_export_runtime_eval():
    class NoCells:
        def train(self): ...

        def export(self): ...

        def runtime_eval(self): ...

    assert [n for n, _ in build.backend_cells(NoCells())] == ["train", "export", "runtime_eval"]


def test_every_real_backend_declares_notebook_cells():
    for name in ("ludwig", "open_clip", "hf_detection", "peft", "tabpfn", "whisper", "deepspeech"):
        mod = importlib.import_module(f"sakura.atelier.backends.{name}")
        assert isinstance(mod.NOTEBOOK_CELLS, tuple) and mod.NOTEBOOK_CELLS, name
        assert all(callable(f) for f in mod.NOTEBOOK_CELLS), name
        assert all(getattr(f, "__module__", None) == mod.__name__ for f in mod.NOTEBOOK_CELLS), name
        assert mod.BACKEND.NOTEBOOK_CELLS is mod.NOTEBOOK_CELLS, name
        names = [f.__name__ for f in mod.NOTEBOOK_CELLS]
        assert {"train", "export", "runtime_eval"} <= set(names), name
        assert names.index("train") < names.index("export") < names.index("runtime_eval"), name


def test_every_backend_source_cell_defines_in_a_fresh_kernel():
    # A source cell is `from __future__ import annotations` + the function's source, run in a
    # kernel where the module's other names are NOT bound. Anything evaluated at def time (a
    # default argument reading a module constant, a decorator, a base class) is a NameError
    # there -- peft's export_mlx defaulted `quant=DEFAULT_MLX_QUANT` and needed an LLM repair.
    import textwrap

    for name in ("ludwig", "open_clip", "hf_detection", "peft", "tabpfn", "whisper", "deepspeech"):
        mod = importlib.import_module(f"sakura.atelier.backends.{name}")
        for fn in mod.NOTEBOOK_CELLS:
            src, note = cells.source_of(fn)
            if note:
                continue
            ns: dict = {}
            try:
                exec("from __future__ import annotations\n" + textwrap.dedent(src), ns)  # noqa: S102
            except NameError as exc:
                raise AssertionError(f"{name}.{fn.__name__} cannot be defined on its own: {exc}") from exc


def _cell(nb, template):
    return next(c for c in nb["cells"] if c["metadata"].get("atelier", {}).get("template") == template)


def test_fault_knob_ships_one_driver_cell_with_a_bug(tmp_path, monkeypatch):
    monkeypatch.setenv(build.FAULT_ENV, "curves")
    nb = build.build_notebook(_ctx(tmp_path))
    src = cells.source_text(_cell(nb, "curves"))
    assert "NotebookRecorder().plot_curves(" in src and "NotebookRecorder().curves(" not in src
    # Only the named cell is touched; the notebook still parses.
    assert '"value": ev.value' in cells.source_text(_cell(nb, "eval"))
    assert cells.check_notebook(nb) == []


def test_fault_knob_is_off_by_default_and_rejects_unknown_names(tmp_path, monkeypatch):
    monkeypatch.delenv(build.FAULT_ENV, raising=False)
    assert "NotebookRecorder().curves(" in cells.source_text(_cell(build.build_notebook(_ctx(tmp_path)), "curves"))
    monkeypatch.setenv(build.FAULT_ENV, "package")
    import pytest
    with pytest.raises(ValueError, match="unknown template"):
        build.build_notebook(_ctx(tmp_path))


def test_every_fault_actually_fails_when_the_cell_runs(tmp_path, monkeypatch):
    """A fault that the cell tolerates is no fault: the knob exists to make the
    first execution fail. Each faulty cell is executed on its own in a kernel."""
    from sakura.atelier.notebook import agent
    for name in build.FAULTS:
        monkeypatch.setenv(build.FAULT_ENV, name)
        nb = build.build_notebook(_ctx(tmp_path))
        faulty = _cell(nb, name)
        prelude = "from pathlib import Path\nRUN_DIR = Path('.')\nREPLAY = False\nartifacts = []\nreport = {}\n"
        if name == "eval":
            prelude += ("class _Ev:\n    format='onnx'; runtime='r'; metric='m'; value=1.0; n=1; matches_training=True; reason=None\n"
                        "class _B:\n    def runtime_eval(self, a, d): return _Ev()\n"
                        "backend = _B(); data = None; artifacts = [object()]\n")
        probe = cells.notebook([cells.code(prelude), cells.code(cells.source_text(faulty))], title="probe")
        failure = agent.execute(probe, tmp_path, {}, 60)
        assert failure is not None and failure.cell_index == 1, f"fault {name!r} did not fail"
