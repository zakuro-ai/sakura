"""End to end on real hardware: a Ludwig run ships a notebook whose check passes
without repairs. Marked gpu -- runs on the GPU runners, deselected elsewhere."""

import json

import pytest

import sakura.atelier.__main__ as cli

pytestmark = pytest.mark.gpu

SPEC = """
atelier: 1
name: nb-gpu
task: tabular_classification
data: {uri: file://@@CSV@@, format: csv, split: {validation_fraction: 0.2}}
features: {inputs: [{name: x, type: number}, {name: z, type: number}], output: {name: y, type: category}}
model: ludwig_ecd
hp: tabular/ecd@1
overrides: {epochs: 1}
seed: 7
export: [onnx]
"""


def test_ludwig_run_ships_a_notebook_whose_check_passes(tmp_path):
    import numpy as np
    import pandas as pd

    rng = np.random.default_rng(0)
    df = pd.DataFrame({"x": rng.normal(size=200), "z": rng.normal(size=200)})
    df["y"] = np.where(df.x + df.z > 0, "p", "n")
    csv = tmp_path / "d.csv"
    df.to_csv(csv, index=False)
    spec = tmp_path / "spec.yaml"
    spec.write_text(SPEC.replace("@@CSV@@", str(csv)))
    out = tmp_path / "out"

    assert cli.main(["run", str(spec), "--out", str(out)]) == 0

    report = json.loads((out / "report.json").read_text())
    assert report["notebook_check"]["ok"], report["notebook_check"]
    assert [a["format"] for a in report["artifacts"]] == ["onnx", "notebook"]
