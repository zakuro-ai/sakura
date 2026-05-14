import pytest
from pydantic import ValidationError

from zakuro_poc.models import ExecutionPlan, ResourceLimits


def test_valid_minimal_plan():
    plan = ExecutionPlan(
        job_name="test-job", image="python:3.11", command=["python", "-c", "print('hello')"]
    )
    assert plan.job_name == "test-job"
    assert plan.image == "python:3.11"
    assert plan.command == ["python", "-c", "print('hello')"]
    assert plan.backend == "docker"


def test_zakuro_backend_plan_is_valid():
    plan = ExecutionPlan(job_name="test-zakuro", backend="zakuro", command=["echo", "hello"])
    assert plan.backend == "zakuro"


def test_empty_command_rejected():
    with pytest.raises(ValidationError):
        ExecutionPlan(job_name="test", image="python:3.11", command=[])


def test_empty_image_rejected():
    with pytest.raises(ValidationError):
        ExecutionPlan(job_name="test", image="", command=["ls"])


def test_invalid_cpu_rejected():
    with pytest.raises(ValidationError):
        ResourceLimits(cpu_count=0.0)
    with pytest.raises(ValidationError):
        ResourceLimits(cpu_count=-1.0)


def test_memory_below_minimum_rejected():
    with pytest.raises(ValidationError):
        ResourceLimits(memory_mb=127)


def test_timeout_above_maximum_rejected():
    with pytest.raises(ValidationError):
        ResourceLimits(timeout_seconds=3601)


def test_gpu_negative_rejected():
    with pytest.raises(ValidationError):
        ResourceLimits(gpu_count=-1)


def test_env_var_invalid_name_rejected():
    with pytest.raises(ValidationError):
        ExecutionPlan(job_name="test", command=["ls"], env={"1INVALID": "value"})


def test_https_repo_url_accepted():
    plan = ExecutionPlan(
        job_name="test", command=["ls"], repo_url="https://github.com/octocat/Hello-World.git"
    )
    assert plan.repo_url == "https://github.com/octocat/Hello-World.git"


def test_non_https_repo_url_rejected():
    with pytest.raises(ValidationError):
        ExecutionPlan(
            job_name="test", command=["ls"], repo_url="git@github.com:octocat/Hello-World.git"
        )
    with pytest.raises(ValidationError):
        ExecutionPlan(
            job_name="test", command=["ls"], repo_url="http://github.com/octocat/Hello-World.git"
        )
