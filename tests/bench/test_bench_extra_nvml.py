"""The NVML in-process backend must be declared as a bench dependency, and
`pynvml` must be allow-listed for mypy (no type stubs ship with it)."""
from __future__ import annotations

import pathlib
import sys

if sys.version_info >= (3, 11):
    import tomllib
else:  # tomllib is stdlib only from 3.11; skip rather than fail collection
    import pytest
    tomllib = pytest.importorskip("tomllib")


def _pyproject() -> dict:
    root = pathlib.Path(__file__).resolve().parents[2]
    return tomllib.loads((root / "pyproject.toml").read_text())


def test_bench_extra_declares_nvidia_ml_py():
    bench = _pyproject()["project"]["optional-dependencies"]["bench"]
    assert any(dep.startswith("nvidia-ml-py") for dep in bench), bench


def test_mypy_overrides_allowlist_pynvml():
    overrides = _pyproject()["tool"]["mypy"]["overrides"]
    assert any("pynvml" in o.get("module", []) for o in overrides), overrides
