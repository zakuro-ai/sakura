"""Lazy loader for optional (extra) dependencies.

The core sakura surface (dispatch / runtime / events / wire) installs without
the heavy ML stack. Training features import their dependencies through
``load`` so that a missing dependency surfaces as an actionable install hint
rather than a bare ``ModuleNotFoundError``.

    torch = load("torch", extra="training")
"""
from __future__ import annotations

import importlib
from types import ModuleType


def load(module: str, *, extra: str) -> ModuleType:
    """Import ``module`` or raise with a ``pip install 'sakura-ml[extra]'`` hint.

    ``extra`` is the optional-dependency group that provides ``module`` (e.g.
    ``"training"`` for torch, ``"lightning"`` for lightning).
    """
    try:
        return importlib.import_module(module)
    except ImportError as exc:
        raise ModuleNotFoundError(
            f"sakura: this feature requires the optional dependency '{module}', "
            f"which is not installed. Install it with: "
            f"pip install 'sakura-ml[{extra}]'"
        ) from exc


__all__ = ["load"]
