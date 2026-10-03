"""Backend registry. Imported by name and lazily, because each backend's
training stack (ludwig, ultralytics, tabpfn, peft) is installed only in that
backend's own venv: importing this package must never import any of them."""

from __future__ import annotations

import importlib

from sakura.atelier.types import Backend

#: task -> backend module under sakura.atelier.backends. Contract §2.2.
TASK_BACKENDS: dict[str, str] = {
    "image_classification": "ludwig",
    "audio_classification": "ludwig",
    "tabular_classification": "ludwig",
    "tabular_regression": "ludwig",
    "object_detection": "ultralytics",
    "text_generation": "peft",
    "speech_recognition": "deepspeech",
}

#: Models served by a backend other than their task's default.
MODEL_BACKENDS: dict[str, str] = {"tabpfn_v2": "tabpfn"}


def backend_name(task: str, model: str) -> str:
    if model in MODEL_BACKENDS:
        return MODEL_BACKENDS[model]
    try:
        return TASK_BACKENDS[task]
    except KeyError:
        raise ValueError(f"no backend for task {task!r}") from None


def load_backend(name: str) -> Backend:
    module = importlib.import_module(f"sakura.atelier.backends.{name}")
    backend = module.BACKEND
    if not isinstance(backend, Backend):
        raise TypeError(f"sakura.atelier.backends.{name}.BACKEND does not implement Backend")
    return backend
