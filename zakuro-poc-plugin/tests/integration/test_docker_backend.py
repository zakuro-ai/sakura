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
