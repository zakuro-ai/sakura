"""`model: hf:<repo>@<sha>` -- fine-tune any Hugging Face repo, pinned (plan Task 2)."""

import pytest

from sakura.atelier.hf_ref import HfRef, parse_hf_model
from sakura.atelier.registry import resolve
from sakura.atelier.spec import load_spec

SHA = "0123456789abcdef0123456789abcdef01234567"


def test_parse_hf_model_requires_pinned_sha():
    assert parse_hf_model(f"hf:HuggingFaceTB/SmolLM2-135M@{SHA}") == HfRef("HuggingFaceTB/SmolLM2-135M", SHA)
    assert parse_hf_model("ludwig_ecd") is None
    for bad in ["hf:HuggingFaceTB/SmolLM2-135M", "hf:HuggingFaceTB/SmolLM2-135M@main",
                f"hf:noslash@{SHA}", f"hf:a/b/c@{SHA}", f"hf:../x@{SHA}", "hf:@" + SHA]:
        with pytest.raises(ValueError):
            parse_hf_model(bad)


SPEC = f"""
atelier: 1
name: smol
task: text_generation
data: {{uri: "file:///tmp/x.jsonl", format: jsonl, split: {{validation_fraction: 0.2}}}}
features:
  inputs: [{{name: prompt, type: text}}]
  output: {{name: completion, type: text}}
model: "hf:HuggingFaceTB/SmolLM2-135M@{SHA}"
hp: text_generation/lora-fast@1
export: [safetensors]
"""


def test_hf_model_resolves_to_the_tasks_backend_with_the_pinned_base():
    r = resolve(load_spec(SPEC))
    assert r.backend == "peft"
    assert r.hyperparameters["base_model"] == "HuggingFaceTB/SmolLM2-135M"
    assert r.hyperparameters["base_revision"] == SHA


def test_hf_model_on_a_task_with_no_hf_backend_is_refused():
    spec = (SPEC.replace("task: text_generation", "task: tabular_regression")
                .replace("text_generation/lora-fast@1", "tabular/ecd@1")
                .replace("type: text}}]", "type: number}}]").replace("type: text}}", "type: number}}"))
    with pytest.raises(ValueError, match="no Hugging Face backend"):
        resolve(load_spec(spec))
