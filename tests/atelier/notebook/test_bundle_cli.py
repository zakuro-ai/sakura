"""`run` ships its notebook bundle (CONTRACTS §2.5): the tar is listed as the
`notebook` artifact, `notebook_check` is in the report, the loss curve is in
the delivered file, and the agent's end-to-end execution passes against a
backend the kernel resolves through the `SAKURA_ATELIER_BACKEND_MODULE_*`
hook. A broken builder never fails the job; `--no-notebook` and
`SAKURA_ATELIER_NOTEBOOK=0` skip the bundle without a `notebook_check`."""

from __future__ import annotations

import json
import tarfile
from pathlib import Path

import pandas as pd
import pytest

import sakura.atelier.__main__ as cli
from sakura.atelier.backends import load_backend
from tests.atelier.notebook.fake_backend import FakeBackend

ROOT = Path(__file__).resolve().parents[3]
FAKE_MODULE = "tests.atelier.notebook.fake_backend"

SPEC = """
atelier: 1
name: t
task: tabular_classification
data: {uri: file://@@CSV@@, format: csv, split: {validation_fraction: 0.5}}
features: {inputs: [{name: x, type: number}], output: {name: y, type: category}}
model: ludwig_ecd
hp: tabular/ecd@1
seed: 1
export: [onnx]
"""


@pytest.fixture
def spec_path(tmp_path: Path) -> Path:
    csv = tmp_path / "d.csv"
    pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0], "y": ["a", "b", "a", "b"]}).to_csv(csv, index=False)
    p = tmp_path / "spec.yaml"
    p.write_text(SPEC.replace("@@CSV@@", str(csv)))
    return p


@pytest.fixture
def fake_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    """The CLI and the agent's kernel both resolve `ludwig` to the fake: the
    parent through `load_backend`'s env hook, the kernel (a fresh process that
    inherits the environment) through the same variable plus PYTHONPATH."""
    monkeypatch.setenv("SAKURA_ATELIER_BACKEND_MODULE_LUDWIG", FAKE_MODULE)
    monkeypatch.setenv("PYTHONPATH", str(ROOT))
    monkeypatch.delenv("SAKURA_ATELIER_LLM_URL", raising=False)
    monkeypatch.delenv("SAKURA_ATELIER_NOTEBOOK", raising=False)


def test_load_backend_env_hook_resolves_a_module_by_name(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SAKURA_ATELIER_BACKEND_MODULE_LUDWIG", FAKE_MODULE)
    assert isinstance(load_backend("ludwig"), FakeBackend)
    monkeypatch.delenv("SAKURA_ATELIER_BACKEND_MODULE_LUDWIG")
    with pytest.raises(ModuleNotFoundError):
        load_backend("no-such-backend")


def test_run_writes_the_bundle_and_the_check_and_lists_the_artifact(
    tmp_path: Path, spec_path: Path, fake_backend: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A RELATIVE --out, as a person types it: the kernel's cwd is the run dir, so the
    # agent must receive an absolute ATELIER_RUN_DIR or the replay cells look in
    # runs/x/runs/x (the first real run hit this and the agent had to repair it).
    monkeypatch.chdir(tmp_path)
    out = tmp_path / "out"
    assert cli.main(["run", str(spec_path), "--out", "out"]) == 0
    report = json.loads((out / "report.json").read_text())
    assert report["status"] == "done"
    formats = [a["format"] for a in report["artifacts"]]
    assert formats == ["onnx", "license", "notebook"] and report["artifacts"][-1]["path"] == "artifacts/notebook.tar"
    check = report["notebook_check"]
    assert set(check) == {"ok", "attempts", "repairs", "seconds", "failed_cell", "error", "templates_touched"}
    assert check["attempts"] == 1 and check["repairs"] == 0
    assert report["billable_seconds"] >= check["seconds"], "the check is machine time too"
    with tarfile.open(out / "artifacts" / "notebook.tar") as tf:
        names = set(tf.getnames())
    for n in ("notebook/notebook.ipynb", "notebook/README.md", "notebook/resolved.yaml", "notebook/report.json",
              "notebook/metrics.jsonl", "notebook/environment/init.sh", "notebook/environment/requirements.lock.txt",
              "notebook/environment/Dockerfile", "notebook/environment/image.json",
              "notebook/environment/apt-packages.txt"):
        assert n in names, n
    bundle = out / "artifacts" / "notebook"
    nb = json.loads((bundle / "notebook.ipynb").read_text())
    pngs = [o for c in nb["cells"] if c["cell_type"] == "code" for o in c["outputs"]
            if o.get("output_type") == "display_data" and "image/png" in o.get("data", {})]
    assert pngs, "the loss curve must be in the delivered notebook"
    readme = (bundle / "README.md").read_text()
    assert "ATELIER_REPLAY=check" in readme and '"ok": true' in readme
    # the copies are the run's own files
    assert (bundle / "resolved.yaml").read_text() == (out / "resolved.yaml").read_text()
    assert (bundle / "metrics.jsonl").read_text() == (out / "metrics.jsonl").read_text()
    assert check["ok"], check


def test_a_failed_check_still_delivers_the_bundle_with_the_recorded_outputs(
    tmp_path: Path, spec_path: Path, fake_backend: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The parent resolves the fake; the kernel is pointed at a module that does
    not exist, so its RESOLVE cell fails. The job still succeeds, the tar is
    listed, the verdict names the cell, and the cells the agent never reached
    keep the run's recorded outputs (the curve among them)."""
    monkeypatch.setattr(cli, "load_backend", lambda name: FakeBackend())
    monkeypatch.setenv("SAKURA_ATELIER_BACKEND_MODULE_LUDWIG", "tests.atelier.notebook.no_such_module")
    out = tmp_path / "out"
    assert cli.main(["run", str(spec_path), "--out", str(out)]) == 0
    report = json.loads((out / "report.json").read_text())
    assert report["status"] == "done" and [a["format"] for a in report["artifacts"]] == ["onnx", "license", "notebook"]
    check = report["notebook_check"]
    assert check["ok"] is False and check["attempts"] == 1 and "no_such_module" in check["error"]
    nb = json.loads((out / "artifacts" / "notebook" / "notebook.ipynb").read_text())
    assert nb["cells"][check["failed_cell"]]["metadata"]["atelier"]["template"] == "resolve"
    assert any(o["output_type"] == "error" for o in nb["cells"][check["failed_cell"]]["outputs"])
    pngs = [o for c in nb["cells"] if c["cell_type"] == "code" for o in c["outputs"]
            if o.get("output_type") == "display_data" and "image/png" in o.get("data", {})]
    assert pngs, "the recorded curve survives a failed check"
    assert '"ok": false' in (out / "artifacts" / "notebook" / "README.md").read_text()


def test_the_bundle_never_fails_the_job(tmp_path: Path, spec_path: Path, fake_backend: None,
                                        monkeypatch: pytest.MonkeyPatch) -> None:
    import sakura.atelier.notebook.bundle as bundle

    monkeypatch.setattr(bundle, "build_notebook", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("builder bug")))
    out = tmp_path / "out"
    assert cli.main(["run", str(spec_path), "--out", str(out)]) == 0
    report = json.loads((out / "report.json").read_text())
    assert report["status"] == "done" and [a["format"] for a in report["artifacts"]] == ["onnx", "license"]
    assert (out / "artifacts" / "model.onnx").exists()
    check = report["notebook_check"]
    assert set(check) == {"ok", "attempts", "repairs", "seconds", "failed_cell", "error", "templates_touched"}
    assert check["ok"] is False and "builder bug" in check["error"]
    assert not (out / "artifacts" / "notebook.tar").exists()


def test_no_notebook_flag_and_env_skip_the_bundle(tmp_path: Path, spec_path: Path, fake_backend: None,
                                                  monkeypatch: pytest.MonkeyPatch) -> None:
    out = tmp_path / "o1"
    assert cli.main(["run", str(spec_path), "--out", str(out), "--no-notebook"]) == 0
    r = json.loads((out / "report.json").read_text())
    assert "notebook_check" not in r and [a["format"] for a in r["artifacts"]] == ["onnx", "license"]
    assert not (out / "artifacts" / "notebook").exists()
    monkeypatch.setenv("SAKURA_ATELIER_NOTEBOOK", "0")
    out2 = tmp_path / "o2"
    assert cli.main(["run", str(spec_path), "--out", str(out2)]) == 0
    assert "notebook_check" not in json.loads((out2 / "report.json").read_text())


def _stub_bundle(monkeypatch: pytest.MonkeyPatch, seen: list[str]) -> None:
    """Replace the bundle (and its kernel) with a stub that records when it ran."""
    import sakura.atelier.notebook.bundle as bundle

    def write(*_args: object, **_kwargs: object) -> tuple[None, dict[str, object]]:
        seen.append("notebook")
        return None, {"ok": True}

    monkeypatch.setattr(bundle, "write_notebook_bundle", write)


def test_the_backend_lets_go_of_the_accelerator_before_the_notebook_check(
    tmp_path: Path, spec_path: Path, fake_backend: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The check re-scores the exports from another process on the same card: a 4,096-token text-generation run
    # (staging job 64a2d0bb) kept the card's memory reserved in this process and the check ran out of memory.
    seen: list[str] = []
    monkeypatch.setattr(FakeBackend, "release", lambda self: seen.append("release"), raising=False)
    _stub_bundle(monkeypatch, seen)
    out = tmp_path / "out"
    assert cli.main(["run", str(spec_path), "--out", str(out)]) == 0
    assert seen == ["release", "notebook"]
    assert json.loads((out / "report.json").read_text())["notebook_check"] == {"ok": True}


def test_a_release_that_fails_does_not_lose_the_run(
    tmp_path: Path, spec_path: Path, fake_backend: None, monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str]
) -> None:
    def release(self: object) -> None:
        raise RuntimeError("the driver went away")

    seen: list[str] = []
    monkeypatch.setattr(FakeBackend, "release", release, raising=False)
    _stub_bundle(monkeypatch, seen)
    out = tmp_path / "out"
    assert cli.main(["run", str(spec_path), "--out", str(out)]) == 0
    assert seen == ["notebook"] and json.loads((out / "report.json").read_text())["status"] == "done"
    assert "could not free the accelerator before the notebook check: RuntimeError: the driver went away" in (
        capsys.readouterr().err)


def test_a_backend_without_a_release_runs_as_before(
    tmp_path: Path, spec_path: Path, fake_backend: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    assert not hasattr(FakeBackend, "release")
    seen: list[str] = []
    _stub_bundle(monkeypatch, seen)
    assert cli.main(["run", str(spec_path), "--out", str(tmp_path / "out")]) == 0
    assert seen == ["notebook"]


def test_run_report_carries_the_evaluation_block_and_the_onnx_provider(
    tmp_path: Path, spec_path: Path, fake_backend: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("sklearn")  # the evaluation metrics need it; CI installs it
    monkeypatch.chdir(tmp_path)
    assert cli.main(["run", str(spec_path), "--out", "out"]) == 0
    report = json.loads((tmp_path / "out" / "report.json").read_text())
    assert report["onnx_provider"] == "CPUExecutionProvider"
    assert report["runtime_eval"][0]["provider"] == "CPUExecutionProvider"
    ev = report["evaluation"]
    assert ev["splits"]["validation"]["n"] == 4 and ev["splits"]["validation"]["accuracy"] == 0.75
    assert ev["confusion_matrix"]["matrix"] == [[1, 1], [0, 2]]
    assert [m["predicted"] for m in ev["misclassified"]] == ["b"]
    assert any("validation split has only 4 rows" in w for w in ev["warnings"])
    assert any("validation_fraction" in w for w in ev["warnings"])
    # the bundle's copy of report.json and the executed notebook carry it too
    with tarfile.open(tmp_path / "out" / "artifacts" / "notebook.tar") as tar:
        names = tar.getnames()
        copy = json.loads(tar.extractfile(next(n for n in names if n.endswith("report.json"))).read())
        nb = json.loads(tar.extractfile(next(n for n in names if n.endswith("notebook.ipynb"))).read())
    assert copy["evaluation"]["splits"]["validation"]["accuracy"] == 0.75
    assert report["notebook_check"]["ok"] is True
    text = json.dumps(nb)
    assert "onnx provider" in text and "confusion matrix" in text


TEXTGEN_SPEC = """
atelier: 1
name: t
task: text_generation
data: {uri: file://@@JSONL@@, format: jsonl, split: {validation_fraction: 0.25}}
features: {inputs: [{name: prompt, type: text}], output: {name: completion, type: text}}
model: lfm2-350m
hp: text_generation/lora-fast@1
seed: 1
export: [safetensors]
"""


def _run_textgen(tmp_path: Path, rows: list[dict[str, str]], monkeypatch: pytest.MonkeyPatch) -> dict:
    monkeypatch.setenv("SAKURA_ATELIER_BACKEND_MODULE_PEFT", "tests.atelier.notebook.fake_textgen_backend")
    monkeypatch.setenv("PYTHONPATH", str(ROOT))
    monkeypatch.delenv("SAKURA_ATELIER_LLM_URL", raising=False)
    monkeypatch.delenv("SAKURA_ATELIER_NOTEBOOK", raising=False)
    jsonl = tmp_path / "rows.jsonl"
    jsonl.write_text("".join(json.dumps(r) + "\n" for r in rows))  # json.dumps writes a lone surrogate as \udce1
    spec = tmp_path / "spec.yaml"
    spec.write_text(TEXTGEN_SPEC.replace("@@JSONL@@", str(jsonl)))
    out = tmp_path / "out"
    assert cli.main(["run", str(spec), "--out", str(out)]) == 0
    return dict(json.loads((out / "report.json").read_text()))


def _rows(bad: bool) -> list[dict[str, str]]:
    rows = [{"prompt": f"{i}+1=", "completion": str(i + 1)} for i in range(8)]
    if bad:
        rows[2]["completion"] = "three \udce1"
    return rows


def test_a_lone_surrogate_in_the_data_trains_passes_the_check_and_is_counted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    report = _run_textgen(tmp_path, _rows(bad=True), monkeypatch)
    assert report["status"] == "done"
    assert report["notebook_check"]["ok"], report["notebook_check"]
    assert report["data_encoding"] == {"rows_with_invalid_unicode": 1, "characters_replaced": 1,
                                       "columns": ["completion"]}


def test_a_clean_text_generation_table_checks_as_before_and_has_no_data_encoding(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    report = _run_textgen(tmp_path, _rows(bad=False), monkeypatch)
    assert report["notebook_check"]["ok"], report["notebook_check"]
    assert report["notebook_check"]["attempts"] == 1 and "data_encoding" not in report
