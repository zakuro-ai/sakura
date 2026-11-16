"""A decision run still ships its notebook bundle, and the agent's replay of it passes (CONTRACTS §2.5).

The kernel is a fresh process that resolves the backend by name, so the fake LM reaches it through the same
`SAKURA_ATELIER_BACKEND_MODULE_*` hook the other notebook tests use, plus PYTHONPATH."""

from __future__ import annotations

import json
import tarfile
from pathlib import Path

import pytest

pytest.importorskip("torch")
for _mod in ("nbformat", "nbclient", "ipykernel"):
    pytest.importorskip(_mod)

import sakura.atelier.__main__ as cli  # noqa: E402
from tests.atelier.decision_fakes import write_dataset  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
SPEC = """
atelier: 1
name: ticket-triage
task: decision
data: {uri: "data.jsonl", format: jsonl}
model: hf-decision-lm
hp: decision/lora-forge@1
overrides: {base_model: Qwen/Qwen3.5-2B-Base, epochs: 1, batch_size: 8, learning_rate: 0.05, max_seq_length: 512, precision: fp32}
export: [safetensors]
seed: 7
"""


def test_a_decision_run_ships_a_notebook_that_replays(tmp_path, monkeypatch):
    # The run directory is the working directory: the kernel starts in it, and the spec's relative data path
    # must resolve from there as it does here.
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("SAKURA_ATELIER_BACKEND_MODULE_PEFT", "tests.atelier.fake_decision_backend")
    monkeypatch.setenv("PYTHONPATH", str(ROOT))
    monkeypatch.delenv("SAKURA_ATELIER_NOTEBOOK", raising=False)
    monkeypatch.delenv("SAKURA_ATELIER_LLM_URL", raising=False)
    write_dataset(tmp_path / "data.jsonl", n=90)
    (tmp_path / "spec.yaml").write_text(SPEC)

    assert cli.main(["run", "spec.yaml", "--out", "."]) == 0

    report = json.loads((tmp_path / "report.json").read_text())
    assert [a["format"] for a in report["artifacts"]] == ["safetensors", "decision_config", "license", "notebook"]
    check = report["notebook_check"]
    assert check["ok"], check
    assert check["attempts"] == 1 and check["repairs"] == 0
    with tarfile.open(tmp_path / "artifacts" / "notebook.tar") as tf:
        assert "notebook/notebook.ipynb" in tf.getnames()
    nb = json.loads((tmp_path / "artifacts" / "notebook" / "notebook.ipynb").read_text())
    text = json.dumps(nb)
    assert "task: decision" in text and "hf-decision-lm" in text
