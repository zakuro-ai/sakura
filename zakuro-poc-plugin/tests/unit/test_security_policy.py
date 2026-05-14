import pytest
from pydantic import ValidationError

from zakuro_poc.config import ZakuroPocConfig
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
    violations = validate_security_policy(plan, ZakuroPocConfig(allow_network=True))
    assert not violations


def test_non_https_repo_rejected():
    with pytest.raises(ValidationError):
        ExecutionPlan(job_name="test", command=["ls"], repo_url="http://github.com")


def test_network_requires_config_approval():
    plan = ExecutionPlan(
        job_name="test",
        command=["ls"],
        network_enabled=True,
        repo_url="https://github.com/octocat/Hello-World.git",
    )
    violations = validate_security_policy(plan)
    assert any("config does not allow network access" in v for v in violations)


def test_resource_limits_must_fit_configured_maxima():
    plan = ExecutionPlan(job_name="test", command=["ls"])
    plan.resource_limits.timeout_seconds = 120
    plan.resource_limits.memory_mb = 2048
    plan.resource_limits.cpu_count = 2.0

    config = ZakuroPocConfig(max_timeout_seconds=60, max_memory_mb=1024, max_cpu_count=1.0)
    violations = validate_security_policy(plan, config)

    assert any("timeout exceeds configured maximum" in v for v in violations)
    assert any("memory exceeds configured maximum" in v for v in violations)
    assert any("CPU count exceeds configured maximum" in v for v in violations)


def test_config_can_explicitly_allow_shell_and_latest_image():
    plan = ExecutionPlan(job_name="test", command=["bash", "-lc", "true"], image="ubuntu:latest")

    config = ZakuroPocConfig(allow_shell=True, allow_latest_images=True)
    violations = validate_security_policy(plan, config)

    assert not violations


def test_working_dir_must_be_relative_under_workspace():
    absolute = ExecutionPlan(job_name="absolute", command=["ls"], working_dir="/tmp")
    traversal = ExecutionPlan(job_name="traversal", command=["ls"], working_dir="../outside")

    assert any("working_dir" in v for v in validate_security_policy(absolute))
    assert any("working_dir" in v for v in validate_security_policy(traversal))
