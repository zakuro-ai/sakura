"""The real peft backend with its four model-loading seams swapped for a tiny fake LM, so the whole decision
pipeline (prepare, train, calibrate, report, artifacts, predict, serve) runs without a GPU or `transformers`.

Reached through `SAKURA_ATELIER_BACKEND_MODULE_PEFT=tests.atelier.fake_decision_backend` (the hook
`load_backend` documents for tests); everything else is `PEFTBackend`'s own code."""

from __future__ import annotations

import io
from pathlib import Path
from typing import Any

import torch

from sakura.atelier.backends.peft import PEFTBackend
from tests.atelier.decision_fakes import CharTokenizer, FakeLM


def _write(state: dict[str, Any], path: Path) -> None:
    # through a file object: torch.load/save refuse a PATH that ends in .safetensors unless safetensors is installed
    buf = io.BytesIO()
    torch.save(state, buf)
    path.write_bytes(buf.getvalue())


def _read(path: Path) -> dict[str, Any]:
    state: dict[str, Any] = torch.load(io.BytesIO(path.read_bytes()))
    return state


class FakePEFT(PEFTBackend):
    #: (source, revision, device) of every base the run asked to load: lets a test see a resume.
    loads: list[tuple[str, str | None, str]] = []
    #: confidence of the fake model: > 1 sharpens its logits, so it comes out overconfident
    scale = 1.0

    def _decision_load(self, opt: Any, source: str, revision: str | None, device: str) -> tuple[Any, Any, str | None]:
        FakePEFT.loads.append((source, revision, device))
        model = FakeLM(seed=0)
        model.scale = FakePEFT.scale
        saved = Path(source) / "model.safetensors"
        if saved.is_file():  # a resume: the previous run's merged weights
            model.load_state_dict(_read(saved))
        return CharTokenizer(), model, "fake-commit"

    def _decision_adapt(self, model: Any, opt: Any) -> Any:
        return model  # no LoRA on the fake: every parameter trains

    def _decision_merge(self, model: Any, opt: Any, merged_dir: Path, tokenizer: Any, source: str,
                        revision: str | None) -> Any:
        merged_dir.mkdir(parents=True, exist_ok=True)
        _write(model.state_dict(), merged_dir / "model.safetensors")  # not real safetensors: it is a fake
        (merged_dir / "config.json").write_text("{}")
        return model

    def _decision_load_saved(self, model_dir: Path, cfg: dict[str, Any], device: str) -> tuple[Any, Any]:
        model = FakeLM(seed=0)
        model.scale = FakePEFT.scale
        model.load_state_dict(_read(model_dir / "model.safetensors"))
        return CharTokenizer(), model


BACKEND = FakePEFT()
