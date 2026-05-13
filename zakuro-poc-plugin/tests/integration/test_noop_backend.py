import json

from zakuro_poc.backends.noop_backend import NoopBackend
from zakuro_poc.config import ZakuroPocConfig
from zakuro_poc.execution.artifacts import create_artifact_dir
from zakuro_poc.execution.ids import new_job_id
from zakuro_poc.models import ExecutionPlan


def test_noop_backend_returns_success(tmp_path):
    job_id = new_job_id()
    artifact_dir = create_artifact_dir(tmp_path, job_id)
    plan = ExecutionPlan(
        job_name="test-noop",
        command=["echo", "hello"],
    )
    config = ZakuroPocConfig()

    backend = NoopBackend()
    result = backend.run(plan, artifact_dir, config)

    assert result.status == "succeeded"
    assert result.exit_code == 0
    assert result.backend == "noop"


def test_noop_backend_writes_stdout(tmp_path):
    job_id = new_job_id()
    artifact_dir = create_artifact_dir(tmp_path, job_id)
    plan = ExecutionPlan(
        job_name="test-noop",
        command=["echo", "hello"],
    )
    config = ZakuroPocConfig()

    backend = NoopBackend()
    result = backend.run(plan, artifact_dir, config)

    assert "NoopBackend: Executing plan test-noop" in result.stdout
    assert "echo" in result.stdout

    stdout_file = artifact_dir / "stdout.txt"
    assert stdout_file.read_text(encoding="utf-8") == result.stdout


def test_noop_backend_preserves_job_name(tmp_path):
    job_id = new_job_id()
    artifact_dir = create_artifact_dir(tmp_path, job_id)
    plan = ExecutionPlan(
        job_name="specific-job-name",
        command=["echo", "hello"],
    )
    config = ZakuroPocConfig()

    backend = NoopBackend()
    result = backend.run(plan, artifact_dir, config)

    assert result.job_name == "specific-job-name"


def test_noop_backend_returns_duration(tmp_path):
    job_id = new_job_id()
    artifact_dir = create_artifact_dir(tmp_path, job_id)
    plan = ExecutionPlan(
        job_name="test-noop",
        command=["echo", "hello"],
    )
    config = ZakuroPocConfig()

    backend = NoopBackend()
    result = backend.run(plan, artifact_dir, config)

    assert result.duration_ms >= 0


def test_noop_backend_creates_result_artifact(tmp_path):
    job_id = new_job_id()
    artifact_dir = create_artifact_dir(tmp_path, job_id)
    plan = ExecutionPlan(
        job_name="test-noop",
        command=["echo", "hello"],
    )
    config = ZakuroPocConfig()

    backend = NoopBackend()
    backend.run(plan, artifact_dir, config)

    result_file = artifact_dir / "result.json"
    assert result_file.exists()

    data = json.loads(result_file.read_text(encoding="utf-8"))
    assert data["job_name"] == "test-noop"
    assert data["status"] == "succeeded"
