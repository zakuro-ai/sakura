import subprocess

import pytest

from zakuro_poc.backends.docker_backend import DockerBackend, build_docker_command
from zakuro_poc.config import ZakuroPocConfig
from zakuro_poc.execution.artifacts import create_artifact_dir
from zakuro_poc.execution.ids import new_job_id
from zakuro_poc.models import ExecutionPlan


@pytest.mark.docker
def test_docker_simple_python_command_succeeds(tmp_path):
    job_id = new_job_id()
    artifact_dir = create_artifact_dir(tmp_path, job_id)
    plan = ExecutionPlan(
        job_name="test-docker",
        image="python:3.11-slim",
        command=["python", "-c", "print('hello from docker')"],
    )
    config = ZakuroPocConfig()

    backend = DockerBackend()
    result = backend.run(plan, artifact_dir, config)

    assert result.status == "succeeded"
    assert result.exit_code == 0
    assert "hello from docker" in result.stdout


@pytest.mark.docker
def test_docker_command_returning_nonzero_fails(tmp_path):
    job_id = new_job_id()
    artifact_dir = create_artifact_dir(tmp_path, job_id)
    plan = ExecutionPlan(
        job_name="test-docker-fail",
        image="python:3.11-slim",
        command=[
            "python",
            "-c",
            "import sys; print('error output', file=sys.stderr); sys.exit(42)",
        ],
    )
    config = ZakuroPocConfig()

    backend = DockerBackend()
    result = backend.run(plan, artifact_dir, config)

    assert result.status == "failed"
    assert result.exit_code == 42
    assert "error output" in result.stderr


@pytest.mark.docker
def test_docker_timeout_produces_timed_out(tmp_path):
    job_id = new_job_id()
    artifact_dir = create_artifact_dir(tmp_path, job_id)
    plan = ExecutionPlan(
        job_name="test-docker-timeout",
        image="python:3.11-slim",
        command=["python", "-c", "import time; time.sleep(10)"],
    )
    plan.resource_limits.timeout_seconds = 1
    config = ZakuroPocConfig()

    backend = DockerBackend()
    result = backend.run(plan, artifact_dir, config)

    assert result.status == "timed_out"
    assert result.exit_code is None
    assert "Job timed out after" in result.stderr


def test_docker_command_does_not_use_shell(tmp_path):
    job_id = new_job_id()
    artifact_dir = create_artifact_dir(tmp_path, job_id)
    plan = ExecutionPlan(
        job_name="test-docker",
        image="python:3.11-slim",
        command=["python", "-c", "print('hello')"],
    )
    config = ZakuroPocConfig()

    cmd = build_docker_command(plan, artifact_dir, config)
    assert not any("shell=True" in str(arg) for arg in cmd)


def test_network_defaults_to_none(tmp_path):
    job_id = new_job_id()
    artifact_dir = create_artifact_dir(tmp_path, job_id)
    plan = ExecutionPlan(
        job_name="test-docker",
        image="python:3.11-slim",
        command=["python", "-c", "print('hello')"],
    )
    config = ZakuroPocConfig()

    cmd = build_docker_command(plan, artifact_dir, config)
    assert "--network" in cmd
    idx = cmd.index("--network")
    assert cmd[idx + 1] == "none"


def test_memory_and_cpu_flags_are_present(tmp_path):
    job_id = new_job_id()
    artifact_dir = create_artifact_dir(tmp_path, job_id)
    plan = ExecutionPlan(
        job_name="test-docker",
        image="python:3.11-slim",
        command=["python", "-c", "print('hello')"],
    )
    plan.resource_limits.cpu_count = 2.5
    plan.resource_limits.memory_mb = 1024
    config = ZakuroPocConfig()

    cmd = build_docker_command(plan, artifact_dir, config)

    assert "--cpus" in cmd
    cpu_idx = cmd.index("--cpus")
    assert cmd[cpu_idx + 1] == "2.5"

    assert "--memory" in cmd
    mem_idx = cmd.index("--memory")
    assert cmd[mem_idx + 1] == "1024m"


def test_docker_command_uses_hardening_flags(tmp_path):
    job_id = new_job_id()
    artifact_dir = create_artifact_dir(tmp_path, job_id)
    plan = ExecutionPlan(
        job_name="test-docker",
        image="python:3.11-slim",
        command=["python", "-c", "print('hello')"],
    )
    config = ZakuroPocConfig()

    cmd = build_docker_command(plan, artifact_dir, config)

    assert "--security-opt" in cmd
    assert "no-new-privileges" in cmd
    assert "--cap-drop" in cmd
    assert "ALL" in cmd
    assert "--pids-limit" in cmd
    assert "--user" in cmd
    user_idx = cmd.index("--user")
    assert cmd[user_idx + 1] == "65532:65532"


def test_docker_available_returns_false_when_cli_is_missing(monkeypatch):
    from zakuro_poc.backends import docker_backend

    def fake_run(*_args, **_kwargs):  # noqa: ANN001, ANN202
        raise FileNotFoundError("docker")

    monkeypatch.setattr(subprocess, "run", fake_run)

    assert docker_backend.docker_available() is False


def test_docker_user_is_configurable(tmp_path):
    job_id = new_job_id()
    artifact_dir = create_artifact_dir(tmp_path, job_id)
    plan = ExecutionPlan(
        job_name="test-docker",
        image="python:3.11-slim",
        command=["python", "-c", "print('hello')"],
    )
    config = ZakuroPocConfig(docker={"user": "1000:1000"})

    cmd = build_docker_command(plan, artifact_dir, config)

    user_idx = cmd.index("--user")
    assert cmd[user_idx + 1] == "1000:1000"


def test_docker_command_honours_env_and_working_dir(tmp_path):
    job_id = new_job_id()
    artifact_dir = create_artifact_dir(tmp_path, job_id)
    plan = ExecutionPlan(
        job_name="test-docker",
        image="python:3.11-slim",
        command=["python", "-c", "print('hello')"],
        env={"SAKURA_MODE": "test"},
        working_dir="repo",
    )
    config = ZakuroPocConfig()

    cmd = build_docker_command(plan, artifact_dir, config)

    assert "--env" in cmd
    assert "SAKURA_MODE=test" in cmd
    workdir_idx = cmd.index("-w")
    assert cmd[workdir_idx + 1] == "/workspace/repo"


def test_read_only_root_flag_is_configurable(tmp_path):
    job_id = new_job_id()
    artifact_dir = create_artifact_dir(tmp_path, job_id)
    plan = ExecutionPlan(
        job_name="test-docker",
        image="python:3.11-slim",
        command=["python", "-c", "print('hello')"],
    )
    config = ZakuroPocConfig(docker={"read_only_root": True})

    cmd = build_docker_command(plan, artifact_dir, config)

    assert "--read-only" in cmd
    assert "--tmpfs" in cmd


def test_network_mode_uses_config_when_network_is_allowed(tmp_path):
    job_id = new_job_id()
    artifact_dir = create_artifact_dir(tmp_path, job_id)
    plan = ExecutionPlan(
        job_name="test-docker",
        image="python:3.11-slim",
        command=["python", "-c", "print('hello')"],
        repo_url="https://github.com/octocat/Hello-World.git",
        network_enabled=True,
    )
    config = ZakuroPocConfig(allow_network=True, docker={"network_mode": "bridge"})

    cmd = build_docker_command(plan, artifact_dir, config)

    network_idx = cmd.index("--network")
    assert cmd[network_idx + 1] == "bridge"


def test_docker_backend_reports_unavailable_docker(tmp_path, monkeypatch):
    job_id = new_job_id()
    artifact_dir = create_artifact_dir(tmp_path, job_id)
    plan = ExecutionPlan(job_name="test-docker-missing", image="python:3.11-slim", command=["ls"])
    config = ZakuroPocConfig()

    monkeypatch.setattr("zakuro_poc.backends.docker_backend.docker_available", lambda: False)

    result = DockerBackend().run(plan, artifact_dir, config)

    assert result.status == "failed"
    assert result.exit_code is None
    assert "Docker is not available" in result.stderr
    assert (artifact_dir / "result.json").exists()


def test_docker_backend_normalises_unexpected_errors(tmp_path, monkeypatch):
    job_id = new_job_id()
    artifact_dir = create_artifact_dir(tmp_path, job_id)
    plan = ExecutionPlan(job_name="test-docker-error", image="python:3.11-slim", command=["ls"])
    config = ZakuroPocConfig()

    monkeypatch.setattr("zakuro_poc.backends.docker_backend.docker_available", lambda: True)

    def fake_run(*_args, **_kwargs):  # noqa: ANN001, ANN202
        raise RuntimeError("boom")

    monkeypatch.setattr(subprocess, "run", fake_run)

    result = DockerBackend().run(plan, artifact_dir, config)

    assert result.status == "failed"
    assert "Failed to execute docker command" in result.stderr
    assert result.error_message == "boom"


@pytest.mark.docker
def test_docker_artifact_files_created(tmp_path):
    job_id = new_job_id()
    artifact_dir = create_artifact_dir(tmp_path, job_id)
    plan = ExecutionPlan(
        job_name="test-docker-artifacts",
        image="python:3.11-slim",
        command=["python", "-c", "print('hello')"],
    )
    config = ZakuroPocConfig()

    backend = DockerBackend()
    backend.run(plan, artifact_dir, config)

    assert (artifact_dir / "stdout.txt").exists()
    assert (artifact_dir / "stderr.txt").exists()
    assert (artifact_dir / "result.json").exists()
    assert (artifact_dir / "metadata.json").exists()
    assert (artifact_dir / "plan.json").exists()
    assert (artifact_dir / "workspace").exists()
    assert (artifact_dir / "workspace").is_dir()


@pytest.mark.docker
def test_docker_non_root_user_can_write_workspace(tmp_path):
    job_id = new_job_id()
    artifact_dir = create_artifact_dir(tmp_path, job_id)
    plan = ExecutionPlan(
        job_name="test-docker-non-root-write",
        image="python:3.11-slim",
        command=["python", "-c", "open('/workspace/non-root.txt', 'w').write('ok')"],
    )
    config = ZakuroPocConfig()

    backend = DockerBackend()
    result = backend.run(plan, artifact_dir, config)

    assert result.status == "succeeded"
    assert (artifact_dir / "workspace" / "non-root.txt").read_text(encoding="utf-8") == "ok"


def test_docker_backend_mock_success(tmp_path, monkeypatch):
    import subprocess

    job_id = new_job_id()
    artifact_dir = create_artifact_dir(tmp_path, job_id)
    plan = ExecutionPlan(
        job_name="test-mock-success",
        image="python:3.11-slim",
        command=["echo", "hello"],
    )
    config = ZakuroPocConfig()

    monkeypatch.setattr("zakuro_poc.backends.docker_backend.docker_available", lambda: True)

    def fake_run(*args, **kwargs):
        if args[0][1] == "rm":
            return subprocess.CompletedProcess(args=args[0], returncode=0, stdout="", stderr="")
        return subprocess.CompletedProcess(
            args=args[0], returncode=0, stdout="mock stdout", stderr="mock stderr"
        )

    monkeypatch.setattr(subprocess, "run", fake_run)

    backend = DockerBackend()
    result = backend.run(plan, artifact_dir, config)

    assert result.status == "succeeded"
    assert result.exit_code == 0
    assert result.stdout == "mock stdout"
    assert result.stderr == "mock stderr"


def test_docker_backend_mock_timeout_bytes(tmp_path, monkeypatch):
    import subprocess

    job_id = new_job_id()
    artifact_dir = create_artifact_dir(tmp_path, job_id)
    plan = ExecutionPlan(
        job_name="test-mock-timeout",
        image="python:3.11-slim",
        command=["sleep", "10"],
    )
    plan.resource_limits.timeout_seconds = 1
    config = ZakuroPocConfig()

    monkeypatch.setattr("zakuro_poc.backends.docker_backend.docker_available", lambda: True)

    def fake_run(*args, **kwargs):
        if args[0][1] == "rm":
            return subprocess.CompletedProcess(args=args[0], returncode=0, stdout="", stderr="")
        raise subprocess.TimeoutExpired(
            cmd=args[0], timeout=1, output=b"bytes stdout", stderr=b"bytes stderr"
        )

    monkeypatch.setattr(subprocess, "run", fake_run)

    backend = DockerBackend()
    result = backend.run(plan, artifact_dir, config)

    assert result.status == "timed_out"
    assert result.exit_code is None
    assert result.stdout == "bytes stdout"
    assert "bytes stderr" in result.stderr
    assert "Job timed out after" in result.stderr
