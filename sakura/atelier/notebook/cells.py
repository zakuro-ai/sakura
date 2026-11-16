"""nbformat-4 cells as plain dicts, without importing nbformat.

Why dicts: the engine core must import with none of nbformat/nbclient/
matplotlib installed (plan: Global Constraints), and a notebook is JSON --
building it needs no library. `nbformat` is only needed to execute or
validate, and those happen inside functions in `agent.py`.

Why `@@NAME@@` and never f-strings: a driver cell is Python source destined
for the customer's kernel. An f-string template with a forgotten field would
raise a NameError *there*, at replay time; a `@@NAME@@` left unfilled is a
`TemplateError` at build time, caught by `check_notebook` as well, so the
notebook never ships half-rendered. Templates written as Python string
literals must double their backslashes (`\\n`), which the test suite's
`python -W error` run enforces.

`source_of` is how "code by source" degrades: a function whose source cannot
be read (a compiled extension, a builtin) is cited by module path in a
markdown note instead of crashing the build (Review Focus 3).
"""

from __future__ import annotations

import ast
import base64
import inspect
import re
from typing import Any

#: The seven sections, in the order the contract fixes them (CONTRACTS §2.5).
SECTION_HEADINGS: tuple[str, ...] = (
    "## Setup",
    "## Data",
    "## Model",
    "## Training",
    "## Evaluation",
    "## Export",
    "## Package",
)
#: Appended last by the agent; not required by the static check because the
#: first copy of the notebook has not been repaired yet.
REPAIRS_HEADING = "## Repairs"

_PLACEHOLDER = re.compile(r"@@([A-Z0-9_]+)@@")
# IPython line/cell magics and shell escapes are not Python; `ast.parse`
# would reject them, so the static check swaps each one for a `pass`.
_MAGIC = re.compile(r"^\s*[%!].*$", re.MULTILINE)


class TemplateError(ValueError):
    """A template placeholder was left unfilled."""


def fill(template: str, **values: str) -> str:
    """Substitute every `@@NAME@@` in ``template`` with ``values[NAME]``.

    Substituted values are not rescanned, so a value that happens to contain
    `@@...@@` is data. A placeholder with no value raises `TemplateError`
    naming it, which is the build error the design calls for.
    """
    missing: list[str] = []

    def _sub(m: re.Match[str]) -> str:
        name = m.group(1)
        if name in values:
            return values[name]
        missing.append(name)
        return m.group(0)

    out = _PLACEHOLDER.sub(_sub, template)
    if missing:
        names = ", ".join(f"@@{n}@@" for n in dict.fromkeys(missing))
        raise TemplateError(f"unfilled placeholder(s): {names}")
    return out


def source_of(obj: Any) -> tuple[str, str | None]:
    """Return ``(source, None)`` or, when the source is unreadable, ``("", note)``
    where ``note`` cites the object by module path for a markdown cell."""
    try:
        return inspect.getsource(obj), None
    except (OSError, TypeError):
        mod = getattr(obj, "__module__", None) or type(obj).__module__
        name = getattr(obj, "__qualname__", None) or getattr(obj, "__name__", None) or repr(obj)
        return "", f"source unavailable; this step ran `{mod}.{name}`"


def markdown(text: str, **metadata: Any) -> dict[str, Any]:
    return {"cell_type": "markdown", "metadata": dict(metadata), "source": text}


def code(
    source: str, outputs: list[dict[str, Any]] | None = None, **metadata: Any
) -> dict[str, Any]:
    return {
        "cell_type": "code",
        "metadata": dict(metadata),
        "source": source,
        "execution_count": None,
        "outputs": list(outputs or []),
    }


def stream_output(text: str, name: str = "stdout") -> dict[str, Any]:
    return {"output_type": "stream", "name": name, "text": text}


def html_output(html: str, text: str = "") -> dict[str, Any]:
    return {
        "output_type": "display_data",
        "metadata": {},
        "data": {"text/html": html, "text/plain": text},
    }


def png_output(png: bytes, text: str = "") -> dict[str, Any]:
    """A `display_data` output carrying a base64 PNG, as nbformat stores it."""
    return {
        "output_type": "display_data",
        "metadata": {},
        "data": {"image/png": base64.b64encode(png).decode("ascii"), "text/plain": text},
    }


def source_text(cell: dict[str, Any]) -> str:
    """nbformat allows ``source`` as a string or a list of lines; normalise."""
    src = cell.get("source", "")
    return "".join(src) if isinstance(src, list) else str(src)


def notebook(
    cells: list[dict[str, Any]], title: str, metadata: dict[str, Any] | None = None
) -> dict[str, Any]:
    meta: dict[str, Any] = {
        "kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"},
        "language_info": {"name": "python"},
        "atelier": {"title": title},
    }
    meta.update(metadata or {})
    return {"nbformat": 4, "nbformat_minor": 5, "metadata": meta, "cells": list(cells)}


def check_notebook(nb: dict[str, Any]) -> list[str]:
    """Static check: nbformat 4, every section heading present, no unfilled
    placeholder, every code cell parses. Returns the problems, ``[]`` when
    valid. Runs on the first copy and on every agent patch, before any kernel
    is started."""
    problems: list[str] = []
    if nb.get("nbformat") != 4:
        problems.append("nbformat must be 4")
    cells_: list[dict[str, Any]] = nb.get("cells") or []
    headings = "\n".join(source_text(c) for c in cells_ if c.get("cell_type") == "markdown")
    for h in SECTION_HEADINGS:
        if h not in headings:
            problems.append(f"missing section: {h!r}")
    for i, cell in enumerate(cells_):
        src = source_text(cell)
        if _PLACEHOLDER.search(src):
            problems.append(f"cell {i}: unfilled template placeholder")
        if cell.get("cell_type") == "code":
            try:
                ast.parse(_MAGIC.sub("pass  # magic", src))
            except SyntaxError as e:
                problems.append(f"cell {i}: syntax error at line {e.lineno}: {e.msg}")
    return problems
