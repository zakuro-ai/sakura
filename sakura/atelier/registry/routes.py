"""The routing table, generated from models.yaml (the only hand-kept copy).

`routes()` is what `backend_name` routes by, what each runner image carries in its
`ai.zakuro.atelier.routes` label, and what the hub vendors. Contract 2: adds `hf:` model
ids routed by task (`hf_tasks`). `devices` (model -> cpu|gpu) is additive within contract 2: a
reader that does not know the key ignores it."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable

import yaml

ROUTES_CONTRACT = 2
_DIR = Path(__file__).parent

#: Where a model trains at full speed: `cpu` = a CPU-only machine is fine, `gpu` = it needs a GPU.
DEVICES = ("cpu", "gpu")


def check_models(models: dict[str, Any]) -> dict[str, Any]:
    """Raise unless every models.yaml entry names a valid `device`; return `models`."""
    bad = {k: (m or {}).get("device") for k, m in models.items()
           if (m or {}).get("device") not in DEVICES}
    if bad:
        raise ValueError(f"models.yaml: `device` must be one of {list(DEVICES)}; got {bad}")
    return models

#: task -> the backend that fine-tunes ANY Hugging Face repo for it (`model: hf:<repo>@<sha>`).
#: Only entries whose backend exists (is in a family) are routed.
HF_TASK_BACKENDS: dict[str, str] = {
    "image_classification": "hf_image_classification",
    "audio_classification": "hf_audio_classification",
    "text_classification": "hf_text_classification",
    "object_detection": "hf_detection",
    "speech_recognition": "whisper",
    "text_generation": "peft",
}


#: task -> the registry model whose exports / hp_defaults / hp_required an
#: `hf:` model of that task inherits (its backend must be HF_TASK_BACKENDS[task]).
HF_TASK_TEMPLATES: dict[str, str] = {
    "image_classification": "hf-image-classifier",
    "audio_classification": "hf-audio-classifier",
    "text_classification": "hf-text-classifier",
    "object_detection": "rtdetr_r18",
    "speech_recognition": "whisper-small",
    "text_generation": "hf-causal-lm",
}


@lru_cache(maxsize=1)
def _models() -> dict[str, Any]:
    models: dict[str, Any] = yaml.safe_load((_DIR / "models.yaml").read_text())
    return check_models(models)


@lru_cache(maxsize=1)
def families() -> dict[str, list[str]]:
    """family -> backends (registry/families.yaml)."""
    return {k: list(v) for k, v in yaml.safe_load((_DIR / "families.yaml").read_text()).items()}


@lru_cache(maxsize=1)
def _full() -> dict[str, Any]:
    models = _models()
    tasks: dict[str, str] = {}
    for model_id, m in models.items():
        if not m.get("task_default"):
            continue
        for task in m["tasks"]:
            if task in tasks and tasks[task] != m["backend"]:
                raise ValueError(f"models.yaml: two task_default models for {task!r}")
            tasks[task] = m["backend"]
    known = set(families()["all"])
    return {
        "contract": ROUTES_CONTRACT,
        "models": {k: m["backend"] for k, m in models.items()},
        "tasks": tasks,
        "exports": {k: list(m.get("exports") or []) for k, m in models.items()},
        "model_tasks": {k: list(m["tasks"]) for k, m in models.items()},
        "devices": {k: m["device"] for k, m in models.items()},
        "hf_tasks": {t: b for t, b in HF_TASK_BACKENDS.items() if b in known},
        "hf_templates": {t: m for t, m in HF_TASK_TEMPLATES.items()
                         if HF_TASK_BACKENDS.get(t) in known and m in models},
    }


def routes(backends: Iterable[str] | None = None) -> dict[str, Any]:
    """The routing table, restricted to `backends` when given."""
    full = _full()
    if backends is None:
        return {k: (dict(v) if isinstance(v, dict) else v) for k, v in full.items()}
    keep = set(backends)
    models = {k: b for k, b in full["models"].items() if b in keep}
    return {
        "contract": full["contract"],
        "models": models,
        "tasks": {t: b for t, b in full["tasks"].items() if b in keep},
        "exports": {k: full["exports"][k] for k in models},
        "model_tasks": {k: full["model_tasks"][k] for k in models},
        "devices": {k: full["devices"][k] for k in models},
        "hf_tasks": {t: b for t, b in full["hf_tasks"].items() if b in keep},
        "hf_templates": {t: m for t, m in full["hf_templates"].items() if m in models},
    }
