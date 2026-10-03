"""Preset + model registry: expands an AtelierSpec's ``hp`` id into explicit
hyperparameters and resolves ``model`` against ``registry/models.yaml``.

Presets are immutable once released (CONTRACTS §1.1): a preset file's content
must never change after it ships, only grow a new ``@version``. ``resolve``
is what makes ``resolved.yaml`` a faithful, idempotent expansion -- running
either the original spec or ``resolved.yaml`` must produce the same run, so
a spec that already carries ``hyperparameters`` (i.e. it IS a resolved.yaml)
is used as-is rather than re-expanded from the preset file.

Contract: docs/atelier/CONTRACTS.md §1, §2.1.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from sakura.atelier.backends import backend_name
from sakura.atelier.spec import AtelierSpec, dump_spec
from sakura.atelier.types import ResolvedSpec

REGISTRY_DIR = Path(__file__).parent
PRESETS_DIR = REGISTRY_DIR / "presets"
MODELS_PATH = REGISTRY_DIR / "models.yaml"


def load_models() -> dict[str, Any]:
    with open(MODELS_PATH, encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def preset_path(hp: str) -> Path:
    # hp is "<family>/<name>@<version>"; spec.py's PRESET_ID regex already
    # enforces that shape before this ever runs.
    family, rest = hp.split("/", 1)
    return PRESETS_DIR / family / f"{rest}.yaml"


def load_preset(hp: str) -> dict[str, Any]:
    path = preset_path(hp)
    if not path.exists():
        raise ValueError(f"unknown preset {hp!r} (no file at {path})")
    with open(path, encoding="utf-8") as fh:
        doc = yaml.safe_load(fh) or {}
    if "hyperparameters" not in doc:
        raise ValueError(f"preset {hp!r} has no hyperparameters block")
    hp_values = doc["hyperparameters"]
    if not isinstance(hp_values, dict):
        raise ValueError(f"preset {hp!r} hyperparameters must be a mapping")
    return dict(hp_values)


def list_presets(task: str | None = None) -> list[str]:
    """Preset ids, optionally filtered to those usable for ``task`` (same
    family-matching rule as AtelierSpec's hp/task validator)."""
    ids: list[str] = []
    if not PRESETS_DIR.exists():
        return ids
    for family_dir in sorted(p for p in PRESETS_DIR.iterdir() if p.is_dir()):
        family = family_dir.name
        if task is not None and not (task.startswith(family) or family == task):
            continue
        for f in sorted(family_dir.glob("*.yaml")):
            ids.append(f"{family}/{f.stem}")
    return ids


def resolve(spec: AtelierSpec) -> ResolvedSpec:
    """Expand ``spec.hp`` (or reuse ``spec.hyperparameters`` on a replay) and
    check ``spec.model`` against the registry. Raises ``ValueError`` listing
    what is wrong; never silently drops an unknown model or task mismatch."""
    models = load_models()
    if spec.model not in models:
        raise ValueError(f"unknown model {spec.model!r} (not in registry/models.yaml)")
    tasks = models[spec.model].get("tasks", [])
    if spec.task not in tasks:
        raise ValueError(
            f"model {spec.model!r} does not support task {spec.task!r} (supports {tasks})"
        )
    exports = models[spec.model].get("exports", [])
    bad = [f for f in spec.export if f not in exports]
    if bad:
        raise ValueError(
            f"model {spec.model!r} cannot export {bad} (it exports {exports or 'nothing: evaluation-only'})"
        )
    if not spec.export and exports:
        raise ValueError(f"export is empty; model {spec.model!r} exports {exports}")

    if spec.hyperparameters is not None:
        # Already resolved (this spec IS a resolved.yaml, or a hand-authored
        # spec that set hyperparameters directly): use as-is -- re-expanding
        # from the preset file here would defeat the replay guarantee if the
        # preset were ever edited in place.
        hyperparameters = dict(spec.hyperparameters)
    else:
        hyperparameters = load_preset(spec.hp)
        hyperparameters.update(spec.overrides)  # shallow per-key override (CONTRACTS §1)

    backend = backend_name(spec.task, spec.model)
    return ResolvedSpec(spec=spec, hyperparameters=hyperparameters, backend=backend)


def resolved_yaml_text(resolved: ResolvedSpec, data_sha256: str) -> str:
    """``resolved.yaml``: the spec with its preset expanded into an explicit
    ``hyperparameters`` block, ``hp`` kept for provenance, and ``data.sha256``
    filled in when the input spec left it empty (CONTRACTS §1.1, §1.2)."""
    data = resolved.spec.model_dump(mode="json")
    data["hyperparameters"] = resolved.hyperparameters
    if not data["data"]["sha256"]:
        data["data"]["sha256"] = data_sha256
    elif data["data"]["sha256"] != data_sha256:
        # materialise() already raises on this before we get here; this is a
        # defensive double-check so resolved.yaml never silently disagrees
        # with the manifest it is written alongside.
        raise ValueError("data.sha256 does not match the materialised manifest")
    return dump_spec(data)


__all__ = [
    "list_presets", "load_models", "load_preset", "preset_path", "resolve",
    "resolved_yaml_text",
]
