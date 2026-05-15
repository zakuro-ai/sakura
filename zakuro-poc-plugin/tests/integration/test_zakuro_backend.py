import subprocess

from zakuro_poc.backends.zakuro_backend import ZakuroBackend, build_zakuro_command
from zakuro_poc.config import ZakuroPocConfig
from zakuro_poc.execution.artifacts import create_artifact_dir, write_text_artifact
from zakuro_poc.execution.ids import new_job_id
from zakuro_poc.models import ExecutionPlan


def test_build_zakuro_command_uses_configured_contract(tmp_path):
    plan_file = tmp_path / "plan.json"
    config = ZakuroPocConfig(
        zakuro={
            "executable": "zc-dev",
            "execute_args": ["execute", "--local"],
            "plan_arg": "--plan-file",
            "json_arg": "--output-json",
        }
    )

    command = build_zakuro_command(plan_file, config)

    assert command == [
        "zc-dev",
        "execute",
        "--local",
        "--plan-file",
        str(plan_file),
        "--output-json",
    ]


def test_zakuro_backend_reports_missing_executable(tmp_path):
    job_id = new_job_id()
    artifact_dir = create_artifact_dir(tmp_path, job_id)
    plan = ExecutionPlan(job_name="test-zakuro", backend="zakuro", command=["echo", "hello"])
    write_text_artifact(artifact_dir, "plan.json", plan.model_dump_json(indent=2))
    config = ZakuroPocConfig(zakuro={"executable": "definitely-missing-zc"})

    result = ZakuroBackend().run(plan, artifact_dir, config)

    assert result.status == "failed"
    assert result.exit_code is None
    assert "Zakuro executable not found" in result.stderr


def test_zakuro_backend_normalises_subprocess_success(monkeypatch, tmp_path):
    job_id = new_job_id()
    artifact_dir = create_artifact_dir(tmp_path, job_id)
    plan = ExecutionPlan(job_name="test-zakuro", backend="zakuro", command=["echo", "hello"])
    write_text_artifact(artifact_dir, "plan.json", plan.model_dump_json(indent=2))
    config = ZakuroPocConfig()

    def fake_run(*args, **_kwargs):  # noqa: ANN001, ANN202
        return subprocess.CompletedProcess(
            args=args[0], returncode=0, stdout='{"ok": true}', stderr=""
        )

    monkeypatch.setattr(subprocess, "run", fake_run)

    result = ZakuroBackend().run(plan, artifact_dir, config)

    assert result.status == "succeeded"
    assert result.exit_code == 0
    assert result.stdout == '{"ok": true}'
    assert result.backend == "zakuro"


def test_zakuro_backend_normalises_timeout(monkeypatch, tmp_path):
    job_id = new_job_id()
    artifact_dir = create_artifact_dir(tmp_path, job_id)
    plan = ExecutionPlan(
        job_name="test-zakuro-timeout", backend="zakuro", command=["echo", "hello"]
    )
    plan.resource_limits.timeout_seconds = 1
    write_text_artifact(artifact_dir, "plan.json", plan.model_dump_json(indent=2))
    config = ZakuroPocConfig()

    def fake_run(*args, **_kwargs):  # noqa: ANN001, ANN202
        raise subprocess.TimeoutExpired(cmd=args[0], timeout=1, output="partial", stderr="late")

    monkeypatch.setattr(subprocess, "run", fake_run)

    result = ZakuroBackend().run(plan, artifact_dir, config)

    assert result.status == "timed_out"
    assert result.exit_code is None
    assert result.stdout == "partial"
    assert "Zakuro execution timed out" in result.stderr


def test_zakuro_backend_reports_non_zero_exit(monkeypatch, tmp_path):
    job_id = new_job_id()
    artifact_dir = create_artifact_dir(tmp_path, job_id)
    plan = ExecutionPlan(job_name="test-zakuro-fail", backend="zakuro", command=["echo", "hello"])
    write_text_artifact(artifact_dir, "plan.json", plan.model_dump_json(indent=2))
    config = ZakuroPocConfig()

    def fake_run(*_args, **_kwargs):  # noqa: ANN001, ANN202
        return subprocess.CompletedProcess(args=["zc"], returncode=7, stdout="", stderr="oops")

    monkeypatch.setattr(subprocess, "run", fake_run)

    result = ZakuroBackend().run(plan, artifact_dir, config)

    assert result.status == "failed"
    assert result.exit_code == 7
    assert result.error_message == "zc execute returned a non-zero exit code"


def test_zakuro_backend_reports_unexpected_exception(monkeypatch, tmp_path):
    job_id = new_job_id()
    artifact_dir = create_artifact_dir(tmp_path, job_id)
    plan = ExecutionPlan(job_name="test-zakuro-error", backend="zakuro", command=["echo", "hello"])
    write_text_artifact(artifact_dir, "plan.json", plan.model_dump_json(indent=2))
    config = ZakuroPocConfig()

    def fake_run(*_args, **_kwargs):  # noqa: ANN001, ANN202
        raise RuntimeError("kaboom")

    monkeypatch.setattr(subprocess, "run", fake_run)

    result = ZakuroBackend().run(plan, artifact_dir, config)

    assert result.status == "failed"
    assert "Failed to execute Zakuro command" in result.stderr
    assert result.error_message == "kaboom"


def test_zakuro_backend_reports_segmentation_fault(monkeypatch, tmp_path):
    job_id = new_job_id()
    artifact_dir = create_artifact_dir(tmp_path, job_id)
    plan = ExecutionPlan(
        job_name="test-zakuro-segfault", backend="zakuro", command=["echo", "hello"]
    )
    write_text_artifact(artifact_dir, "plan.json", plan.model_dump_json(indent=2))
    config = ZakuroPocConfig()

    def fake_run(*_args, **_kwargs):  # noqa: ANN001, ANN202
        return subprocess.CompletedProcess(
            args=["zc"], returncode=-11, stdout="", stderr="Segmentation fault (core dumped)"
        )

    monkeypatch.setattr(subprocess, "run", fake_run)

    result = ZakuroBackend().run(plan, artifact_dir, config)

    assert result.status == "failed"
    assert result.exit_code == -11
    assert result.error_message == "zc execute returned a non-zero exit code"
    assert "Segmentation fault" in result.stderr


def test_zakuro_backend_handles_malformed_json_output(monkeypatch, tmp_path):
    job_id = new_job_id()
    artifact_dir = create_artifact_dir(tmp_path, job_id)
    plan = ExecutionPlan(
        job_name="test-zakuro-badjson", backend="zakuro", command=["echo", "hello"]
    )
    write_text_artifact(artifact_dir, "plan.json", plan.model_dump_json(indent=2))
    config = ZakuroPocConfig()

    def fake_run(*_args, **_kwargs):  # noqa: ANN001, ANN202
        return subprocess.CompletedProcess(
            args=["zc"], returncode=0, stdout='{"incomplete": true', stderr=""
        )

    monkeypatch.setattr(subprocess, "run", fake_run)

    result = ZakuroBackend().run(plan, artifact_dir, config)

    # Currently it just passes through stdout without parsing it
    assert result.status == "succeeded"
    assert result.stdout == '{"incomplete": true'
    assert result.exit_code == 0


def test_real_zakuro_backend_is_selected_by_runner(tmp_path):
    from zakuro_poc.execution.runner import execute_plan

    config = ZakuroPocConfig(
        artifact_root=str(tmp_path),
        zakuro={"executable": "definitely-missing-zc"},
    )
    plan = ExecutionPlan(job_name="test-zakuro-runner", backend="zakuro", command=["echo", "hello"])

    result = execute_plan(plan, config)

    assert result.backend == "zakuro"
    assert result.status == "failed"
    assert "Zakuro executable not found" in result.stderr
    assert (tmp_path / result.job_id / "result.json").exists()
