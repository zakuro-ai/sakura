"""Every release ships its model's licence (plan Task 2)."""

import json

from sakura.atelier.__main__ import main
from sakura.atelier.hf_ref import HfRef
from sakura.atelier.licence import collect_licence
from sakura.atelier.types import RunDir

SHA = "0123456789abcdef0123456789abcdef01234567"


def test_licence_files_ship_in_artifacts(tmp_path):
    out = RunDir(tmp_path).ensure()

    def fetch(ref):
        assert ref == HfRef("acme/tiny", SHA)
        return {"license": "apache-2.0", "files": {"LICENSE": b"Apache License 2.0 ...", "NOTICE": b"(c) acme"}}

    lic = collect_licence("hf:acme/tiny@" + SHA, None, out, fetch=fetch)
    # ONE file, whatever the repo ships: the hub fetches artifacts by format, so one `license` artifact.
    assert lic == {"spdx": "apache-2.0", "files": ["artifacts/LICENSE.txt"],
                   "source": f"hf://acme/tiny@{SHA}", "included": ["LICENSE", "NOTICE"]}
    text = (out.artifacts / "LICENSE.txt").read_text()
    assert "Apache License 2.0" in text and "(c) acme" in text
    assert text.index("== LICENSE ==") < text.index("== NOTICE ==")


def test_a_repo_with_no_licence_file_still_ships_one(tmp_path):
    out = RunDir(tmp_path).ensure()
    lic = collect_licence("hf:acme/tiny@" + SHA, None, out, fetch=lambda ref: {"license": "mit", "files": {}})
    assert lic["files"] == ["artifacts/LICENSE.txt"] and lic["spdx"] == "mit"
    assert "mit" in (out.artifacts / "LICENSE.txt").read_text() and "acme/tiny" in (out.artifacts / "LICENSE.txt").read_text()


def test_licence_fetch_failure_is_recorded_not_fatal(tmp_path):
    out = RunDir(tmp_path).ensure()

    def fetch(ref):
        raise OSError("hub unreachable")

    lic = collect_licence("hf:acme/tiny@" + SHA, None, out, fetch=fetch)
    assert lic["spdx"] is None and lic["files"] == ["artifacts/LICENSE.txt"] and "hub unreachable" in lic["error"]
    note = (out.artifacts / "LICENSE.txt").read_text()
    assert f"hf://acme/tiny@{SHA}" in note and "https://huggingface.co/acme/tiny" in note


def test_engine_model_licence_from_registry(tmp_path):
    out = RunDir(tmp_path).ensure()
    lic = collect_licence("resnet18", None, out)
    assert lic["spdx"] == "BSD-3-Clause" and lic["source"] == "torchvision://resnet18"
    assert (out.artifacts / "LICENSE.txt").read_text().startswith("BSD-3-Clause")


def test_run_report_carries_the_licence(tmp_path, monkeypatch):
    import pandas as pd
    csv = tmp_path / "t.csv"
    pd.DataFrame({"x": range(20), "y": ["a", "b"] * 10}).to_csv(csv, index=False)
    spec = tmp_path / "spec.yaml"
    spec.write_text(f"""
atelier: 1
name: t
task: tabular_classification
data: {{uri: "file://{csv}", format: csv, split: {{validation_fraction: 0.25}}}}
features:
  inputs: [{{name: x, type: number}}]
  output: {{name: y, type: category}}
model: ludwig_ecd
hp: tabular/ecd@1
export: [onnx]
""")
    monkeypatch.setenv("SAKURA_ATELIER_BACKEND_MODULE_LUDWIG", "tests.atelier.fake_train_backend")
    monkeypatch.setenv("SAKURA_ATELIER_NOTEBOOK", "0")
    assert main(["run", str(spec), "--out", str(tmp_path / "out")]) == 0
    report = json.loads((tmp_path / "out" / "report.json").read_text())
    assert report["license"]["spdx"] == "Apache-2.0"
    assert {"path": "artifacts/LICENSE.txt"}.items() <= next(
        a for a in report["artifacts"] if a["path"] == "artifacts/LICENSE.txt").items()
    formats = [a["format"] for a in report["artifacts"]]
    assert len(formats) == len(set(formats)), formats  # the hub fetches each artifact by its format


def test_per_base_model_licence_comes_from_the_base(tmp_path):
    """hf-causal-lm / lfm2-350m / hf-decision-lm say `license: per-base-model`:
    the release carries the base repo's own licence, at the pinned revision."""
    out = RunDir(tmp_path).ensure()
    seen = []

    def fetch(ref):
        seen.append(ref)
        return {"license": "apache-2.0", "files": {"LICENSE": b"Apache"}}

    lic = collect_licence("hf-causal-lm", "HuggingFaceTB/SmolLM2-135M", out, fetch=fetch, base_revision=SHA)
    assert seen == [HfRef("HuggingFaceTB/SmolLM2-135M", SHA)]
    assert lic == {"spdx": "apache-2.0", "files": ["artifacts/LICENSE.txt"],
                   "source": f"hf://HuggingFaceTB/SmolLM2-135M@{SHA}", "included": ["LICENSE"]}
