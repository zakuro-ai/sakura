"""The notebook tests need the optional `notebook` extra (nbformat, nbclient, matplotlib).

Without it they are skipped, with the reason, instead of failing: a lean install (and the public repo's CI) does
not have the extra. CI here installs it, so these tests still run on every PR.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

_HERE = Path(__file__).resolve().parent
_MISSING = [m for m in ("nbformat", "nbclient", "matplotlib") if importlib.util.find_spec(m) is None]


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    if not _MISSING:
        return
    skip = pytest.mark.skip(
        reason=f"needs the notebook extra (pip install 'sakura-ml[notebook]'); missing: {', '.join(_MISSING)}"
    )
    for item in items:
        if _HERE in Path(str(item.fspath)).resolve().parents:
            item.add_marker(skip)
