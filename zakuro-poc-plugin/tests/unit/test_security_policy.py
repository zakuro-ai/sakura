import pytest
from pydantic import ValidationError

from zakuro_poc.models import ExecutionPlan
from zakuro_poc.security.policy import validate_security_policy
from zakuro_poc.validation import validate_plan_or_raise


def test_safe_python_command_accepted():
    plan = ExecutionPlan(job_name="test", command=["python", "-c", "print('hello')"])
    violations = validate_security_policy(plan)
    assert not violations
    validate_plan_or_raise(plan)


def test_bash_c_rejected():
    plan = ExecutionPlan(job_name="test", command=["bash", "-c", "ls"])
    violations = validate_security_policy(plan)
    assert any("shell interpreters" in v for v in violations)


def test_sh_c_rejected():
    plan = ExecutionPlan(job_name="test", command=["sh", "-c", "ls"])
    violations = validate_security_policy(plan)
    assert any("shell interpreters" in v for v in violations)


def test_image_using_latest_rejected():
    plan = ExecutionPlan(job_name="test", command=["ls"], image="ubuntu:latest")
    violations = validate_security_policy(plan)
    assert any("latest" in v for v in violations)


def test_env_var_api_token_rejected():
    plan = ExecutionPlan(job_name="test", command=["ls"], env={"API_TOKEN": "value"})
    violations = validate_security_policy(plan)
    assert any("suggests secret" in v for v in violations)


def test_env_var_password_rejected():
    plan = ExecutionPlan(job_name="test", command=["ls"], env={"DB_PASSWORD": "value"})
    violations = validate_security_policy(plan)
    assert any("suggests secret" in v for v in violations)


def test_artifact_path_traversal_rejected():
    plan = ExecutionPlan(job_name="test", command=["ls"], artifact_dir="../../etc/shadow")
    violations = validate_security_policy(plan)
    assert any("path traversal" in v for v in violations)


def test_network_enabled_without_repo_url_rejected():
    plan = ExecutionPlan(job_name="test", command=["ls"], network_enabled=True)
    violations = validate_security_policy(plan)
    assert any("network enabled without repo_url" in v for v in violations)


def test_network_enabled_with_https_repo_url_accepted():
    plan = ExecutionPlan(
        job_name="test",
        command=["ls"],
        network_enabled=True,
        repo_url="https://github.com/octocat/Hello-World.git",
    )
    violations = validate_security_policy(plan)
    assert not violations


def test_non_https_repo_rejected():
    with pytest.raises(ValidationError):
        ExecutionPlan(job_name="test", command=["ls"], repo_url="http://github.com")
