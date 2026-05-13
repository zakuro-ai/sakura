import pytest
from pydantic import ValidationError

from zakuro_poc.config import ZakuroPocConfig
from zakuro_poc.execution.runner import execute_plan
from zakuro_poc.models import ExecutionPlan


def test_noop_plan_executes(tmp_path):
    config = ZakuroPocConfig(artifact_root=str(tmp_path))
    plan = ExecutionPlan(job_name="test", backend="noop", command=["ls"])
    result = execute_plan(plan, config)
    assert result.status == "succeeded"


def test_invalid_plan_rejected(tmp_path):
    config = ZakuroPocConfig(artifact_root=str(tmp_path))  # noqa: F841
    # This plan has an empty command, which is rejected by Pydantic
    with pytest.raises(ValidationError):
        ExecutionPlan(job_name="test", backend="noop", command=[])


def test_invalid_plan_rejected_by_policy(tmp_path):
    config = ZakuroPocConfig(artifact_root=str(tmp_path))
    # This plan uses bash, which is rejected by security policy
    plan = ExecutionPlan(job_name="test", backend="noop", command=["bash", "-c", "ls"])
    with pytest.raises(ValueError, match="Security policy violations"):
        execute_plan(plan, config)


def test_unknown_backend_rejected(tmp_path):
    config = ZakuroPocConfig(artifact_root=str(tmp_path))
    plan = ExecutionPlan(job_name="test", command=["ls"])
    # Bypass pydantic validation for testing the runner's exception
    plan.backend = "unknown"
    with pytest.raises(ValueError, match="Unknown backend: unknown"):
        execute_plan(plan, config)


def test_artifacts_are_written(tmp_path):
    config = ZakuroPocConfig(artifact_root=str(tmp_path))
    plan = ExecutionPlan(job_name="test", backend="noop", command=["ls"])
    result = execute_plan(plan, config)

    artifact_dir = tmp_path / result.job_id
    assert (artifact_dir / "plan.json").exists()
    assert (artifact_dir / "result.json").exists()


def test_result_contains_job_id(tmp_path):
    config = ZakuroPocConfig(artifact_root=str(tmp_path))
    plan = ExecutionPlan(job_name="test", backend="noop", command=["ls"])
    result = execute_plan(plan, config)
    assert result.job_id.startswith("job-")


def test_duration_is_non_negative(tmp_path):
    config = ZakuroPocConfig(artifact_root=str(tmp_path))
    plan = ExecutionPlan(job_name="test", backend="noop", command=["ls"])
    result = execute_plan(plan, config)
    assert result.duration_ms >= 0
