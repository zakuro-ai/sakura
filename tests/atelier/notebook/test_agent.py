"""`agent.py` is what makes "verified" true (CONTRACTS §2.5): it executes the
notebook in a fresh kernel, and when a cell fails it asks a repairer for one
patched cell, checks the patch, applies it and runs again. These tests pin the
contract with a scripted repairer and no model: a hanging cell is stopped
within the budget and named; a patch that touches a heading, deletes a cell,
installs packages or does not parse is refused and the refusal goes back as
the next turn; the LLM key reaches the Authorization header and nothing else."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from sakura.atelier.notebook import agent, cells


def _nb(*sources: str) -> dict[str, Any]:
    body = [cells.markdown(h) for h in cells.SECTION_HEADINGS]
    body += [
        cells.code(s, atelier={"template": f"t{i}", "section": "## Data"})
        for i, s in enumerate(sources)
    ]
    body.append(
        cells.markdown(
            cells.REPAIRS_HEADING + "\n\nNo repairs were needed; the first execution passed.",
            atelier={"repairs": True},
        )
    )
    return cells.notebook(body, title="t")


def test_execute_returns_none_on_a_clean_notebook_and_a_failure_with_traceback_otherwise(
    tmp_path: Path,
) -> None:
    assert agent.execute(_nb("x = 1", "print(x)"), tmp_path, {}, 60) is None
    f = agent.execute(_nb("x = 1", "raise ValueError('boom')"), tmp_path, {}, 60)
    assert f is not None
    assert f.cell_index == 8 and "boom" in f.traceback and "raise ValueError" in f.source


def test_execute_runs_in_the_given_cwd_with_the_given_env(tmp_path: Path) -> None:
    nb = _nb("import os, pathlib; print(pathlib.Path.cwd().resolve()); print(os.environ['ATELIER_REPLAY'])")
    assert agent.execute(nb, tmp_path, {"ATELIER_REPLAY": "check"}, 60) is None
    text = "".join(o.get("text", "") for o in nb["cells"][7]["outputs"])
    assert str(tmp_path.resolve()) in text and "check" in text


def test_execute_times_out_a_hanging_cell_and_names_it(tmp_path: Path) -> None:
    f = agent.execute(_nb("import time; time.sleep(30)"), tmp_path, {}, 3)
    assert f is not None and f.cell_index == 7 and "timed out" in f.traceback.lower()


def test_validate_patch_rejects_pip_network_markdown_and_syntax() -> None:
    nb = _nb("x = 1")
    bad = [
        ("import subprocess; subprocess.run(['pip', 'install', 'x'])", "pip install"),
        ("!pip install x", "pip install"),
        ("import requests", "network"),
        ("x = (", "syntax"),
    ]
    for src, why in bad:
        r = agent.validate_patch(nb, agent.Patch(cell_index=7, new_source=src, reason="r"))
        assert r and why in r
    heading = agent.validate_patch(nb, agent.Patch(cell_index=0, new_source="x", reason="r"))
    assert heading is not None and "markdown" in heading
    empty = agent.validate_patch(nb, agent.Patch(cell_index=7, new_source="  \n", reason="r"))
    assert empty is not None and "delete" in empty
    out = agent.validate_patch(nb, agent.Patch(cell_index=99, new_source="x", reason="r"))
    assert out is not None and "out of range" in out
    assert agent.validate_patch(nb, agent.Patch(cell_index=7, new_source="x = 2", reason="r")) is None


def test_repair_loop_with_a_scripted_repairer_fixes_the_cell_and_records_it(tmp_path: Path) -> None:
    nb = _nb("x = 1", "y = x + undefined_name")

    class Scripted:
        calls = 0

        def propose(self, nb: dict[str, Any], failure: agent.Failure, rejected: str | None) -> agent.Patch | None:
            self.calls += 1
            return agent.Patch(
                cell_index=failure.cell_index, new_source="y = x + 1", reason="undefined_name was a typo for 1"
            )

    res = agent.repair_loop(nb, tmp_path, {}, Scripted(), attempts=3, budget_seconds=120, log=lambda *_: None)
    assert res.ok and res.attempts == 2 and res.repairs == 1 and res.templates_touched == ["t1"]
    patched = nb["cells"][8]
    assert cells.source_text(patched) == "y = x + 1"
    assert patched["metadata"]["atelier"]["repair"]["original"] == "y = x + undefined_name"
    assert patched["metadata"]["atelier"]["repair"]["attempt"] == 1
    repairs_md = cells.source_text(nb["cells"][-1])
    assert "typo" in repairs_md and "-y = x + undefined_name" in repairs_md and "+y = x + 1" in repairs_md
    assert res.as_report() == {
        "ok": True,
        "attempts": 2,
        "repairs": 1,
        "seconds": res.seconds,
        "failed_cell": None,
        "error": None,
        "templates_touched": ["t1"],
    }
    assert cells.check_notebook(nb) == []


def test_repair_loop_sends_a_rejection_back_and_stops_at_the_budget(tmp_path: Path) -> None:
    nb = _nb("raise RuntimeError('x')")
    seen: list[str | None] = []

    class Stubborn:
        def propose(self, nb: dict[str, Any], failure: agent.Failure, rejected: str | None) -> agent.Patch | None:
            seen.append(rejected)
            return agent.Patch(cell_index=failure.cell_index, new_source="!pip install fixit", reason="install")

    res = agent.repair_loop(nb, tmp_path, {}, Stubborn(), attempts=2, budget_seconds=120, log=lambda *_: None)
    assert not res.ok and res.attempts == 2 and res.repairs == 0 and res.failed_cell == 7
    assert seen[0] is None and seen[1] is not None and "pip install" in seen[1]
    assert "No repairs were needed" not in cells.source_text(nb["cells"][-1]) or res.repairs == 0


def test_repair_loop_stops_within_the_time_budget_when_a_cell_hangs(tmp_path: Path) -> None:
    nb = _nb("import time; time.sleep(30)")

    class Unhelpful:
        def propose(self, nb: dict[str, Any], failure: agent.Failure, rejected: str | None) -> agent.Patch | None:
            return agent.Patch(cell_index=failure.cell_index, new_source="import time; time.sleep(30)  # again", reason="retry")

    res = agent.repair_loop(nb, tmp_path, {}, Unhelpful(), attempts=5, budget_seconds=4, log=lambda *_: None)
    assert not res.ok and res.failed_cell == 7 and res.error is not None and "timed out" in res.error.lower()
    assert res.seconds < 30


def test_without_a_repairer_only_the_first_execution_runs(tmp_path: Path) -> None:
    res = agent.repair_loop(
        _nb("raise RuntimeError('x')"), tmp_path, {}, None, attempts=3, budget_seconds=60, log=lambda *_: None
    )
    assert res.attempts == 1 and res.repairs == 0 and not res.ok
    assert res.error is not None and "RuntimeError" in res.error


def test_llm_repairer_parses_a_fenced_patch_and_never_logs_the_key() -> None:
    posted: dict[str, Any] = {}

    def fake_post(url: str, headers: dict[str, str], body: dict[str, Any], timeout: float) -> dict[str, Any]:
        posted.update(url=url, headers=headers, body=body)
        return {"choices": [{"message": {"content": "reason: use 1\n```python\ny = x + 1\n```"}}]}

    r = agent.LLMRepairer("http://llm/v1", "secret-key", "claude-opus-5-5", post=fake_post)
    p = r.propose(_nb("y = x + undefined_name"), agent.Failure(7, "y = x + undefined_name", "NameError", ""), None)
    assert p is not None
    assert p.new_source == "y = x + 1" and p.cell_index == 7 and "use 1" in p.reason and not p.insert_before
    assert posted["url"] == "http://llm/v1/chat/completions"
    assert posted["headers"]["Authorization"] == "Bearer secret-key" and posted["body"]["model"] == "claude-opus-5-5"
    assert "secret-key" not in json.dumps(posted["body"])


def test_llm_repairer_returns_none_without_a_fence_and_an_empty_patch_when_the_call_fails() -> None:
    def no_fence(url: str, headers: dict[str, str], body: dict[str, Any], timeout: float) -> dict[str, Any]:
        return {"choices": [{"message": {"content": "a missing package is an environment bug"}}]}

    def boom(url: str, headers: dict[str, str], body: dict[str, Any], timeout: float) -> dict[str, Any]:
        raise OSError("connection refused")

    failure = agent.Failure(7, "x", "ModuleNotFoundError", "")
    assert agent.LLMRepairer("http://llm/v1", None, post=no_fence).propose(_nb("x"), failure, None) is None
    p = agent.LLMRepairer("http://llm/v1", None, post=boom).propose(_nb("x"), failure, None)
    assert p is not None and p.new_source == "" and "OSError" in p.reason
    assert agent.validate_patch(_nb("x"), p) is not None


def test_repairer_from_env_is_none_without_a_url() -> None:
    assert agent.repairer_from_env({}) is None
    r = agent.repairer_from_env({"SAKURA_ATELIER_LLM_URL": "http://x", "SAKURA_ATELIER_LLM_KEY": "k"})
    assert isinstance(r, agent.LLMRepairer) and r.model == agent.DEFAULT_MODEL


def test_repair_loop_does_not_ask_for_a_patch_when_the_data_cannot_be_encoded(tmp_path: Path) -> None:
    """A character the data holds that UTF-8 cannot write is a data problem: no edit of a
    template fixes it, so the repairer is never asked and no attempt is spent on it."""
    nb = _nb("x = 1", "('a' + chr(0xDCE1)).encode('utf-8')")

    class Never:
        def propose(self, nb: dict[str, Any], failure: agent.Failure, rejected: str | None) -> agent.Patch | None:
            raise AssertionError("a data problem must not be sent to the repairer")

    res = agent.repair_loop(nb, tmp_path, {}, Never(), attempts=3, budget_seconds=120, log=lambda *_: None)
    assert res.ok is False and res.attempts == 1 and res.repairs == 0
    assert res.failed_cell == 8 and res.templates_touched == []
    assert res.error is not None and res.error.startswith("data problem")
    assert "surrogates not allowed" in res.error
