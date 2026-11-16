"""Backend registry. Imported by name and lazily, because each backend's
training stack (ludwig, ultralytics, tabpfn, peft) is installed only in that
backend's own venv: importing this package must never import any of them."""

from __future__ import annotations

import importlib
import os
from typing import Any

from sakura.atelier.types import Backend

# The routing tables are generated from registry/models.yaml (registry/routes.py), never
# hand-kept here. `TASK_BACKENDS` / `MODEL_BACKENDS` stay importable (lazily, PEP 562):
# TASK_BACKENDS = each task's default backend; MODEL_BACKENDS = the models whose backend is
# not their task's default (or that set `model_route: true`).


def _routes() -> dict[str, Any]:
    from sakura.atelier.registry.routes import routes
    return routes()


def __getattr__(name: str) -> Any:
    if name == "TASK_BACKENDS":
        return _routes()["tasks"]
    if name == "MODEL_BACKENDS":
        r = _routes()
        from sakura.atelier.registry.routes import _models
        models = _models()
        return {m: b for m, b in r["models"].items()
                if models[m].get("model_route") or any(r["tasks"].get(t) != b for t in models[m]["tasks"])}
    raise AttributeError(name)


def backend_name(task: str, model: str) -> str:
    """The backend for a spec: its model's, else (`hf:<repo>@<sha>`) the task's HF backend,
    else the task's default."""
    r = _routes()
    if model in r["models"]:
        return str(r["models"][model])
    if model.startswith("hf:") and task in r["hf_tasks"]:
        return str(r["hf_tasks"][task])
    try:
        return str(r["tasks"][task])
    except KeyError:
        raise ValueError(f"no backend for task {task!r}") from None


def load_backend(name: str) -> Backend:
    """``sakura.atelier.backends.<name>.BACKEND``, unless
    ``SAKURA_ATELIER_BACKEND_MODULE_<NAME>`` names another module to take it
    from. The override exists for tests: the notebook agent's kernel is a
    fresh process that resolves the backend by name, and this is how a fake
    reaches it without a training stack installed."""
    override = os.environ.get(f"SAKURA_ATELIER_BACKEND_MODULE_{name.upper().replace('-', '_')}")
    module = importlib.import_module(override or f"sakura.atelier.backends.{name}")
    backend = module.BACKEND
    if not isinstance(backend, Backend):
        raise TypeError(f"sakura.atelier.backends.{name}.BACKEND does not implement Backend")
    return backend
