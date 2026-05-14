import builtins
import importlib.metadata
import json
import runpy
import sys
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

import pytest
from typer.testing import CliRunner

from zakuro_poc.cli import app
from zakuro_poc.models import ExecutionResult

runner = CliRunner()


def _result(
    *,
    status: str = "succeeded",
    stderr: str = "",
    stdout: str = "ok",
) -> ExecutionResult:
    now = datetime.now(UTC)
    return ExecutionResult(
        job_id="job-test",
        job_name="test",
        backend="noop",
        status=status,  # type: ignore[arg-type]
        stdout=stdout,
        stderr=stderr,
        exit_code=0 if status == "succeeded" else 1 if status == "failed" else None,
        duration_ms=1,
        artifact_dir="/tmp/job-test",
        started_at=now,
        finished_at=now,
        error_message=stderr or None,
    )


def test_version_exits_0():
    result = runner.invoke(app, ["version"])
    assert result.exit_code == 0
    assert "zakuro-poc-plugin" in result.stdout


def test_version_reports_unknown_when_package_is_missing(monkeypatch):
    def fake_version(_name: str) -> str:
        raise importlib.metadata.PackageNotFoundError

    monkeypatch.setattr(importlib.metadata, "version", fake_version)

    result = runner.invoke(app, ["version"])

    assert result.exit_code == 0
    assert "version unknown" in result.stdout


def test_validate_valid_plan_exits_0(tmp_path):
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(
        json.dumps({"job_name": "test", "image": "python:3.11", "command": ["ls"]})
    )
    result = runner.invoke(app, ["validate", "--plan", str(plan_path)])
    assert result.exit_code == 0


def test_validate_invalid_plan_exits_nonzero(tmp_path):
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(
        json.dumps({"job_name": "test", "image": "python:3.11", "command": ["bash", "-c", "ls"]})
    )
    result = runner.invoke(app, ["validate", "--plan", str(plan_path)])
    assert result.exit_code == 1


def test_validate_missing_plan_fails_cleanly(tmp_path):
    result = runner.invoke(app, ["validate", "--plan", str(tmp_path / "missing.json")])

    assert result.exit_code == 1
    assert "Plan file not found" in result.stdout


def test_validate_invalid_json_fails_cleanly(tmp_path):
    plan_path = tmp_path / "plan.json"
    plan_path.write_text("{ invalid json }")

    result = runner.invoke(app, ["validate", "--plan", str(plan_path)])

    assert result.exit_code == 1
    assert "Invalid JSON in plan" in result.stdout


def test_validate_schema_error_fails_cleanly(tmp_path):
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(json.dumps({"job_name": "test", "image": "python:3.11"}))

    result = runner.invoke(app, ["validate", "--plan", str(plan_path)])

    assert result.exit_code == 1
    assert "does not match schema" in result.stdout


def test_plan_show_does_not_execute(tmp_path):
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(
        json.dumps({"job_name": "test", "image": "python:3.11", "command": ["ls"]})
    )
    result = runner.invoke(app, ["plan-show", "--plan", str(plan_path)])
    assert result.exit_code == 0
    assert "Job Name:" in result.stdout


def test_execute_yes_with_noop_succeeds(tmp_path):
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(
        json.dumps(
            {
                "job_name": "test",
                "backend": "noop",
                "image": "python:3.11",
                "command": ["ls"],
            }
        )
    )
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps({"artifact_root": str(tmp_path)}))

    result = runner.invoke(
        app, ["execute", "--plan", str(plan_path), "--config", str(config_path), "--yes"]
    )
    assert result.exit_code == 0
    assert "Status: succeeded" in result.stdout


def test_execute_prompts_and_accepts_yes(tmp_path):
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(
        json.dumps(
            {
                "job_name": "test",
                "backend": "noop",
                "image": "python:3.11",
                "command": ["ls"],
            }
        )
    )
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps({"artifact_root": str(tmp_path)}))

    result = runner.invoke(
        app,
        ["execute", "--plan", str(plan_path), "--config", str(config_path)],
        input="yes\n",
    )

    assert result.exit_code == 0
    assert "Type 'yes' to execute" in result.stdout


def test_execute_without_yes_aborts(tmp_path):
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(
        json.dumps(
            {
                "job_name": "test",
                "backend": "noop",
                "image": "python:3.11",
                "command": ["ls"],
            }
        )
    )
    result = runner.invoke(app, ["execute", "--plan", str(plan_path)], input="no\n")
    assert result.exit_code == 1
    assert "Execution aborted" in result.stdout


def test_execute_prints_json_output(monkeypatch, tmp_path):
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(
        json.dumps(
            {
                "job_name": "test",
                "backend": "noop",
                "image": "python:3.11",
                "command": ["ls"],
            }
        )
    )
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps({"artifact_root": str(tmp_path)}))

    monkeypatch.setattr("zakuro_poc.cli.execute_plan", lambda *_args, **_kwargs: _result())

    result = runner.invoke(
        app,
        ["execute", "--plan", str(plan_path), "--config", str(config_path), "--yes", "--json"],
    )

    assert result.exit_code == 0
    assert '"status": "succeeded"' in result.stdout


def test_execute_prints_rejection_as_json(tmp_path):
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(
        json.dumps(
            {
                "job_name": "test",
                "backend": "noop",
                "image": "python:3.11",
                "command": ["bash", "-c", "ls"],
            }
        )
    )
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps({"artifact_root": str(tmp_path)}))

    result = runner.invoke(
        app,
        ["execute", "--plan", str(plan_path), "--config", str(config_path), "--json"],
    )

    assert result.exit_code == 2
    assert '"status": "rejected"' in result.stdout


@pytest.mark.parametrize(
    ("status", "expected_exit"),
    [("failed", 1), ("timed_out", 124), ("rejected", 2)],
)
def test_execute_maps_result_status_to_exit_codes(monkeypatch, tmp_path, status, expected_exit):
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(
        json.dumps(
            {
                "job_name": "test",
                "backend": "noop",
                "image": "python:3.11",
                "command": ["ls"],
            }
        )
    )
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps({"artifact_root": str(tmp_path)}))

    monkeypatch.setattr(
        "zakuro_poc.cli.execute_plan",
        lambda *_args, **_kwargs: _result(status=status, stderr="details"),
    )

    result = runner.invoke(
        app, ["execute", "--plan", str(plan_path), "--config", str(config_path), "--yes"]
    )

    assert result.exit_code == expected_exit
    assert "details" in result.stdout


def test_execute_prints_stderr_when_present(monkeypatch, tmp_path):
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(
        json.dumps(
            {
                "job_name": "test",
                "backend": "noop",
                "image": "python:3.11",
                "command": ["ls"],
            }
        )
    )
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps({"artifact_root": str(tmp_path)}))

    monkeypatch.setattr(
        "zakuro_poc.cli.execute_plan",
        lambda *_args, **_kwargs: _result(stderr="diagnostic", stdout="payload"),
    )

    result = runner.invoke(
        app, ["execute", "--plan", str(plan_path), "--config", str(config_path), "--yes"]
    )

    assert result.exit_code == 0
    assert "payload" in result.stdout
    assert "diagnostic" in result.stdout


def test_execute_rejects_invalid_plan_before_prompt(tmp_path):
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(
        json.dumps(
            {
                "job_name": "test",
                "backend": "noop",
                "image": "python:3.11",
                "command": ["bash", "-c", "ls"],
            }
        )
    )
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps({"artifact_root": str(tmp_path)}))

    result = runner.invoke(app, ["execute", "--plan", str(plan_path), "--config", str(config_path)])

    assert result.exit_code == 2
    assert "Validation rejected" in result.stdout
    assert "Type 'yes' to execute" not in result.stdout
    assert "Status: rejected" in result.stdout


def test_doctor_returns_useful_output():
    result = runner.invoke(app, ["doctor"])
    assert "Python >= 3.11" in result.stdout


def test_doctor_reports_all_ok(monkeypatch, tmp_path):
    from zakuro_poc import cli

    monkeypatch.setattr(cli.sys, "version_info", SimpleNamespace(major=3, minor=11))
    monkeypatch.setattr(cli, "docker_available", lambda: True)
    monkeypatch.setattr(
        cli,
        "load_config",
        lambda _path=None: cli.ZakuroPocConfig(artifact_root=str(tmp_path)),
    )

    result = runner.invoke(app, ["doctor"])

    assert result.exit_code == 0
    assert "[OK] Docker CLI available" in result.stdout


def test_doctor_reports_version_failure(monkeypatch):
    from zakuro_poc import cli

    monkeypatch.setattr(cli.sys, "version_info", SimpleNamespace(major=3, minor=10))
    monkeypatch.setattr(cli, "load_config", lambda _path=None: cli.ZakuroPocConfig())
    monkeypatch.setattr(cli, "docker_available", lambda: False)

    result = runner.invoke(app, ["doctor"])

    assert result.exit_code == 1
    assert "Python >= 3.11" in result.stdout
    assert "[FAIL]" in result.stdout


def test_doctor_reports_import_failure(monkeypatch):
    from zakuro_poc import cli

    monkeypatch.setattr(cli.sys, "version_info", SimpleNamespace(major=3, minor=11))
    monkeypatch.setattr(cli, "load_config", lambda _path=None: cli.ZakuroPocConfig())
    monkeypatch.setattr(cli, "docker_available", lambda: False)

    original_import = builtins.__import__

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):  # noqa: ANN001
        if name == "zakuro_poc":
            raise ImportError("boom")
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    result = runner.invoke(app, ["doctor"])

    assert result.exit_code == 1
    assert "Package import failed" in result.stdout


def test_doctor_reports_config_load_failure(monkeypatch):
    from zakuro_poc import cli

    monkeypatch.setattr(cli.sys, "version_info", SimpleNamespace(major=3, minor=11))
    monkeypatch.setattr(cli, "docker_available", lambda: False)
    monkeypatch.setattr(
        cli,
        "load_config",
        lambda _path=None: (_ for _ in ()).throw(RuntimeError("bad config")),
    )

    result = runner.invoke(app, ["doctor"])

    assert result.exit_code == 1
    assert "Config load failed" in result.stdout


def test_doctor_reports_artifact_root_failure(monkeypatch, tmp_path):
    from zakuro_poc import cli

    monkeypatch.setattr(cli.sys, "version_info", SimpleNamespace(major=3, minor=11))
    monkeypatch.setattr(cli, "docker_available", lambda: False)
    monkeypatch.setattr(
        cli,
        "load_config",
        lambda _path=None: cli.ZakuroPocConfig(artifact_root=str(tmp_path)),
    )

    def fake_touch(*_args, **_kwargs):  # noqa: ANN001, ANN002
        raise OSError("read-only")

    monkeypatch.setattr(Path, "touch", fake_touch)

    result = runner.invoke(app, ["doctor"])

    assert result.exit_code == 1
    assert "Artifact root not writable" in result.stdout


def test_main_module_entrypoint_executes(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["zakuro-poc-plugin", "version"])
    try:
        runpy.run_module("zakuro_poc.__main__", run_name="__main__")
    except SystemExit as exc:
        assert exc.code in (0, None)
