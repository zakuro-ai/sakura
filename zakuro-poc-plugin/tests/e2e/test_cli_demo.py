import json

from typer.testing import CliRunner

from zakuro_poc.cli import app

runner = CliRunner()


def test_version_exits_0():
    result = runner.invoke(app, ["version"])
    assert result.exit_code == 0
    assert "zakuro-poc-plugin" in result.stdout


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


def test_doctor_returns_useful_output():
    result = runner.invoke(app, ["doctor"])
    assert "Python >= 3.11" in result.stdout
