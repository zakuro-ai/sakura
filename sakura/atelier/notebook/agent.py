"""The notebook agent: execute the notebook where the run happened, and when a
cell fails ask the model for one patched cell, check it, apply it, run again
from a fresh kernel. A notebook that does not run is not a deliverable; this
is what makes "verified" true (CONTRACTS §2.5).

The loop is held to a time budget as well as an attempt count (Review Focus
4): each cell's nbclient timeout is the budget left, so a hanging cell is
stopped and named instead of eating the job. Every patch goes through
`validate_patch` before it touches the notebook (Review Focus 5): headings
are read-only, a cell cannot be deleted, `pip install` and network access are
refused, and a patch that does not parse is refused; the refusal becomes the
next turn of the conversation and counts against the attempts.

The LLM key reaches exactly one place, the Authorization header of the
request. It is stripped from the kernel's environment, never written to the
notebook, the prompt, or a log line.
"""

from __future__ import annotations

import ast
import difflib
import json
import math
import os
import re
import time
import urllib.request
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from sakura.atelier.notebook import cells as C

DEFAULT_MODEL = "claude-opus-5-5"
#: Env vars that must never reach a kernel a patched cell could print from.
_SECRET_ENV = ("SAKURA_ATELIER_LLM_KEY",)
_FORBIDDEN: tuple[tuple[re.Pattern[str], str], ...] = (
    (
        # `pip install`, `!pip install`, `python -m pip install`, `pip.main([...])`,
        # and the subprocess spelling `['pip', 'install', ...]`.
        re.compile(r"pip\s+install|pip\.main\s*\(|['\"]pip['\"]\s*,\s*['\"]install['\"]|uv\s+pip|conda\s+install"),
        "pip install is refused: a missing package is an environment bug, fix the lock file",
    ),
    (
        re.compile(r"^\s*(import|from)\s+(requests|httpx|urllib|socket|aiohttp)\b", re.M),
        "network access is refused in a notebook cell",
    ),
    (re.compile(r"spec\.yaml|resolved\.yaml"), "the spec files are read-only"),
)
_ANSI = re.compile(r"\x1b\[[0-9;]*m")
_MAGIC_LINE = re.compile(r"^\s*[%!].*$", re.M)

PostFn = Callable[[str, dict[str, str], dict[str, Any], float], dict[str, Any]]


@dataclass
class Failure:
    """One failing cell, as the repairer sees it."""

    cell_index: int
    source: str
    traceback: str
    outputs_text: str
    cwd: Path | None = None


@dataclass
class Patch:
    """One proposed replacement (or, with `insert_before`, a new cell placed
    before `cell_index`)."""

    cell_index: int
    new_source: str
    reason: str
    insert_before: bool = False


@dataclass
class CheckResult:
    """The verdict written to `report.json.notebook_check`."""

    ok: bool
    attempts: int
    repairs: int
    seconds: float
    failed_cell: int | None
    error: str | None
    templates_touched: list[str] = field(default_factory=list)

    def as_report(self) -> dict[str, Any]:
        # Keys in the order the Global Constraints fix them.
        return {
            "ok": self.ok,
            "attempts": self.attempts,
            "repairs": self.repairs,
            "seconds": self.seconds,
            "failed_cell": self.failed_cell,
            "error": self.error,
            "templates_touched": list(self.templates_touched),
        }


class Repairer(Protocol):
    def propose(self, nb: dict[str, Any], failure: Failure, rejected: str | None) -> Patch | None: ...


# --------------------------------------------------------------------------- execution


def _outputs_text(cell: Mapping[str, Any]) -> str:
    parts: list[str] = []
    for o in cell.get("outputs") or []:
        if o.get("output_type") == "stream":
            parts.append("".join(o.get("text", "")))
        elif o.get("output_type") == "error":
            parts.append("\n".join(o.get("traceback") or []))
        elif "data" in o:
            parts.append("".join(o["data"].get("text/plain", "")))
    return _ANSI.sub("", "\n".join(parts))


def _kernel_env(env: Mapping[str, str]) -> dict[str, str]:
    """The kernel's environment: the process env plus the caller's overrides,
    minus anything secret (a patched cell could `print(os.environ)`)."""
    full = {**os.environ, **env}
    for name in _SECRET_ENV:
        full.pop(name, None)
    return full


def execute(nb: dict[str, Any], cwd: Path, env: dict[str, str], timeout_seconds: float) -> Failure | None:
    """Run `nb` top to bottom in a fresh `python3` kernel with `cwd` as the
    working directory. Returns None on a clean run, else the first failure.

    Outputs and execution counts are written back onto `nb`, so the delivered
    file shows the agent's run. `timeout_seconds` bounds every single cell
    (nbclient has no whole-notebook timeout), so a cell that hangs is stopped
    and named within the caller's budget.
    """
    import nbformat
    from nbclient import NotebookClient
    from nbclient.exceptions import CellExecutionError, CellTimeoutError, DeadKernelError

    node = nbformat.from_dict(nb)  # type: ignore[no-untyped-call, unused-ignore]
    for cell in node.cells:
        if cell.cell_type == "code":
            cell.outputs = []
            cell.execution_count = None

    started: dict[str, int] = {"index": -1}

    def on_cell_start(cell: Any, cell_index: int) -> None:
        started["index"] = cell_index

    # nbclient 0.11: start_kernel kwargs travel through `execute(**kw)` ->
    # `setup_kernel(**kw)` -> `km.start_kernel(**kw)`; `env` is one of them.
    client = NotebookClient(
        node,
        timeout=max(1, math.ceil(timeout_seconds)),
        kernel_name="python3",
        resources={"metadata": {"path": str(cwd)}},
        allow_errors=False,
        interrupt_on_timeout=False,
        shutdown_kernel="immediate",  # a stuck kernel must not outlive the budget
        on_cell_start=on_cell_start,
    )
    failure: Failure | None = None
    try:
        client.execute(env=_kernel_env(env))
    except CellTimeoutError:
        # nbclient's message repeats the cell source between dashed rules; the
        # source and outputs travel in their own fields, so one line is enough.
        idx = _failed_index(node, started["index"])
        failure = Failure(
            idx,
            C.source_text(node.cells[idx]),
            f"cell {idx} timed out after {timeout_seconds:.0f}s (the time budget left for the check)",
            _outputs_text(node.cells[idx]),
            cwd,
        )
    except CellExecutionError as e:
        idx = _failed_index(node, started["index"])
        failure = Failure(
            idx, C.source_text(node.cells[idx]), _ANSI.sub("", str(e))[-6000:], _outputs_text(node.cells[idx]), cwd
        )
    except DeadKernelError as e:
        idx = _failed_index(node, started["index"])
        failure = Failure(
            idx, C.source_text(node.cells[idx]), f"the kernel died while running cell {idx}: {e}", _outputs_text(node.cells[idx]), cwd
        )
    finally:
        for i, cell in enumerate(node.cells):
            if cell.cell_type == "code":
                nb["cells"][i]["outputs"] = [dict(o) for o in cell.outputs]
                nb["cells"][i]["execution_count"] = cell.get("execution_count")
    return failure


def _failed_index(node: Any, started: int) -> int:
    """The failing cell: the one nbclient last started, or, if the hook did not
    fire, the first cell with an error output / the first unexecuted code cell."""
    if 0 <= started < len(node.cells):
        return started
    last = -1
    for i, cell in enumerate(node.cells):
        if cell.cell_type != "code":
            continue
        if any(o.get("output_type") == "error" for o in cell.outputs):
            return i
        if cell.get("execution_count") is not None or cell.outputs:
            last = i
    for i, cell in enumerate(node.cells):
        if i > last and cell.cell_type == "code":
            return i
    return max(last, 0)


# --------------------------------------------------------------------------- patches


def validate_patch(nb: dict[str, Any], patch: Patch) -> str | None:
    """None when the patch may be applied, else the reason it is refused (the
    reason is sent back to the repairer verbatim as its next turn)."""
    cells_ = nb["cells"]
    if not 0 <= patch.cell_index < len(cells_):
        return f"cell index {patch.cell_index} is out of range (0..{len(cells_) - 1})"
    if cells_[patch.cell_index]["cell_type"] != "code" and not patch.insert_before:
        return "only code cells may be patched; markdown cells (headings) are read-only"
    if not patch.new_source.strip():
        return "an empty patch deletes the cell, which is refused"
    for rx, why in _FORBIDDEN:
        if rx.search(patch.new_source):
            return why
    try:
        ast.parse(_MAGIC_LINE.sub("pass", patch.new_source))
    except SyntaxError as e:
        return f"patch has a syntax error at line {e.lineno}: {e.msg}"
    return None


def apply_patch(nb: dict[str, Any], patch: Patch, attempt: int) -> str:
    """Apply an accepted patch to `nb` and return the template it touched."""
    if patch.insert_before:
        section = nb["cells"][patch.cell_index].get("metadata", {}).get("atelier", {}).get("section", "")
        new = C.code(
            patch.new_source,
            atelier={
                "template": "agent:inserted",
                "section": section,
                "repair": {"attempt": attempt, "original": None, "reason": patch.reason},
            },
        )
        nb["cells"].insert(patch.cell_index, new)
        return "agent:inserted"
    cell = nb["cells"][patch.cell_index]
    meta = cell.setdefault("metadata", {}).setdefault("atelier", {})
    meta["repair"] = {"attempt": attempt, "original": C.source_text(cell), "reason": patch.reason}
    cell["source"] = patch.new_source
    cell["outputs"] = []
    cell["execution_count"] = None
    return str(meta.get("template") or f"cell:{patch.cell_index}")


def _write_repairs_section(nb: dict[str, Any], entries: list[tuple[str, str, str, str]]) -> None:
    idx = next(
        (i for i, c in enumerate(nb["cells"]) if c.get("metadata", {}).get("atelier", {}).get("repairs")),
        None,
    )
    if not entries:
        text = C.REPAIRS_HEADING + "\n\nNo repairs were needed; the first execution passed."
    else:
        blocks = [C.REPAIRS_HEADING, "", f"The notebook agent repaired {len(entries)} cell(s) so the full cycle runs:"]
        for template, original, new, reason in entries:
            diff = "\n".join(
                difflib.unified_diff(original.splitlines(), new.splitlines(), "before", "after", lineterm="")
            )
            blocks += ["", f"**{template}** — {reason}", "", "```diff", diff, "```"]
        text = "\n".join(blocks)
    cell = C.markdown(text, atelier={"repairs": True})
    if idx is None:
        nb["cells"].append(cell)
    else:
        nb["cells"][idx] = cell


#: A failure that names text UTF-8 cannot hold. No edit of a template changes what the dataset contains,
#: so the repair loop does not spend attempts on it (the engine already replaces such characters).
_DATA_ENCODING = re.compile(r"Unicode(?:Encode|Decode)Error:.*(?:surrogates not allowed|can't (?:en|de)code)", re.S)


def is_data_problem(failure: Failure) -> bool:
    return bool(_DATA_ENCODING.search(failure.traceback))


def _last_line(text: str) -> str:
    lines = [ln for ln in text.strip().splitlines() if ln.strip()]
    return lines[-1][:500] if lines else text[:500]


def repair_loop(
    nb: dict[str, Any],
    cwd: Path,
    env: dict[str, str],
    repairer: Repairer | None,
    attempts: int,
    budget_seconds: float,
    log: Callable[[str], Any] = print,
) -> CheckResult:
    """Execute, repair, execute again, until clean or the budget is spent.
    Mutates `nb` (patches, outputs, and the final Repairs section).

    An attempt is one round of the conversation: an execution (the first one,
    and one after every applied patch) or, when the previous patch was
    refused, one more proposal against the unchanged notebook. A refused
    patch therefore costs an attempt without paying for a re-execution that
    could only fail the same way.
    """
    start = time.monotonic()
    entries: list[tuple[str, str, str, str]] = []
    templates: list[str] = []
    rejected: str | None = None
    failure: Failure | None = None
    n = 0
    need_execute = True
    data_problem = False
    while n < max(1, attempts):
        remaining = budget_seconds - (time.monotonic() - start)
        if remaining <= 0:
            log(f"notebook check: budget of {budget_seconds:.0f}s exhausted after {n} attempt(s)")
            break
        n += 1
        if need_execute:
            failure = execute(nb, cwd, env, remaining)
            log(f"notebook check attempt {n}: " + ("ok" if failure is None else f"cell {failure.cell_index} failed"))
            if failure is None:
                _write_repairs_section(nb, entries)
                return CheckResult(True, n, len(entries), round(time.monotonic() - start, 1), None, None, templates)
            if repairer is None or n >= attempts:
                break
            if is_data_problem(failure):
                log("notebook check: the failure is about the data's text encoding, not the notebook; no repair attempted")
                data_problem = True
                break
        assert failure is not None and repairer is not None
        patch = repairer.propose(nb, failure, rejected)
        if patch is None:
            log("notebook agent proposed no patch")
            break
        why = validate_patch(nb, patch)
        if why:
            rejected = f"your previous patch was rejected: {why}"
            need_execute = False
            log(f"notebook check attempt {n}: patch rejected: {why}")
            continue
        rejected = None
        need_execute = True
        original = "" if patch.insert_before else C.source_text(nb["cells"][patch.cell_index])
        template = apply_patch(nb, patch, n)
        log(f"patch applied to {template}: {patch.reason[:200]}")
        entries.append((template, original, patch.new_source, patch.reason))
        templates.append(template)
    _write_repairs_section(nb, entries)
    if failure is None:
        err: str | None = "budget exhausted before the first execution"
    else:
        err = _last_line(failure.traceback)
        if data_problem:
            err = "data problem (a template edit cannot fix it): " + err
    return CheckResult(
        False, n, len(entries), round(time.monotonic() - start, 1), failure.cell_index if failure else None, err, templates
    )


# --------------------------------------------------------------------------- the LLM repairer

_SYSTEM = (
    "You repair one failing cell of a Jupyter notebook that reproduces a machine-learning training run. "
    "Rules: the spec and hyper-parameters are read-only; never remove or skip a section; never access the network; "
    "never pip install (a missing package is an environment bug: say so and return no patch); keep the ATELIER_REPLAY "
    "switch's meaning. Reply with one line starting with 'reason:' then exactly one ```python fenced block holding the "
    "full replacement source of the failing cell. If a new cell must run before it, start the reason with 'insert-before:'."
)
_FENCE = re.compile(r"```(?:python)?\n(.*?)```", re.S)


def _post_json(url: str, headers: dict[str, str], body: dict[str, Any], timeout: float) -> dict[str, Any]:
    req = urllib.request.Request(
        url, data=json.dumps(body).encode(), headers={**headers, "Content-Type": "application/json"}
    )
    # The url is the operator-configured LLM endpoint (LLMRepairer base_url), not request input.
    # nosemgrep: python.lang.security.audit.dynamic-urllib-use-detected.dynamic-urllib-use-detected
    with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310 - the url is operator configuration
        return dict(json.loads(resp.read().decode()))


def _listing(cwd: Path | None, limit: int = 200) -> list[str]:
    if cwd is None or not cwd.is_dir():
        return []
    try:
        return sorted(str(p.relative_to(cwd)) for p in cwd.rglob("*"))[:limit]
    except OSError:
        return []


class LLMRepairer:
    """An OpenAI-compatible chat-completions repairer (the hub's LiteLLM proxy
    in production). The key is held privately and only ever placed in the
    Authorization header; `post` is injectable so tests never touch the network."""

    def __init__(
        self, url: str, key: str | None, model: str = DEFAULT_MODEL, timeout: float = 120.0, post: PostFn | None = None
    ):
        base = url.rstrip("/")
        self.url = base if base.endswith("/chat/completions") else base + "/chat/completions"
        self._key = key
        self.model = model
        self.timeout = timeout
        self._post: PostFn = post or _post_json

    def __repr__(self) -> str:  # never show the key
        return f"LLMRepairer(url={self.url!r}, model={self.model!r})"

    def propose(self, nb: dict[str, Any], failure: Failure, rejected: str | None) -> Patch | None:
        before = []
        for i, c in enumerate(nb["cells"][: failure.cell_index]):
            if c["cell_type"] == "code":
                before.append(f"# --- cell {i} ---\n{C.source_text(c)}\n# outputs:\n{_outputs_text(c)[-1500:]}")
        parts: list[str | None] = [
            rejected,
            "Previous cells (source and outputs):\n" + "\n\n".join(before)[-20000:],
            f"Failing cell {failure.cell_index}:\n```python\n{failure.source}\n```",
            f"Traceback:\n{failure.traceback[-6000:]}",
            f"Cell output:\n{failure.outputs_text[-2000:]}",
            "Files in the working directory:\n" + "\n".join(_listing(failure.cwd)),
        ]
        user = "\n\n".join(p for p in parts if p)
        headers = {"Authorization": f"Bearer {self._key}"} if self._key else {}
        body = {
            "model": self.model,
            "temperature": 0,
            "messages": [{"role": "system", "content": _SYSTEM}, {"role": "user", "content": user}],
        }
        try:
            resp = self._post(self.url, headers, body, self.timeout)
            content = str(resp["choices"][0]["message"]["content"])
        except Exception as e:  # the agent failing is a check failure, never a job failure
            # Rejected as empty by validate_patch; the exception text is not
            # repeated (an HTTP error body could echo the request headers).
            return Patch(failure.cell_index, "", f"llm call failed: {type(e).__name__}")
        m = _FENCE.search(content)
        if not m:
            return None
        reason_line = next((ln for ln in content.splitlines() if ln.lower().startswith("reason:")), "reason: (none given)")
        reason = reason_line.split(":", 1)[1].strip()
        insert = reason.lower().startswith("insert-before:")
        return Patch(failure.cell_index, m.group(1).rstrip("\n"), reason, insert_before=insert)


def repairer_from_env(environ: Mapping[str, str] | None = None) -> Repairer | None:
    """The configured repairer, or None (execute-only) when
    `SAKURA_ATELIER_LLM_URL` is unset."""
    e = os.environ if environ is None else environ
    url = e.get("SAKURA_ATELIER_LLM_URL")
    if not url:
        return None
    return LLMRepairer(url, e.get("SAKURA_ATELIER_LLM_KEY") or None, e.get("SAKURA_ATELIER_LLM_MODEL") or DEFAULT_MODEL)
