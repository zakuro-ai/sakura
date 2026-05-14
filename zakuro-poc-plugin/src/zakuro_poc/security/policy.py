from pathlib import PurePosixPath

from zakuro_poc.config import ZakuroPocConfig
from zakuro_poc.models import ExecutionPlan


def _is_safe_relative_container_path(path: str) -> bool:
    parsed_path = PurePosixPath(path)
    return not parsed_path.is_absolute() and ".." not in parsed_path.parts


def validate_security_policy(
    plan: ExecutionPlan, config: ZakuroPocConfig | None = None
) -> list[str]:
    effective_config = config or ZakuroPocConfig()
    violations = []
    if not plan.command:
        violations.append("empty command")
    else:
        forbidden_shells = {"sh", "bash", "zsh", "fish", "cmd.exe", "powershell"}
        if plan.command[0] in forbidden_shells and not effective_config.allow_shell:
            violations.append(f"shell interpreters are not allowed: {plan.command[0]}")

    if plan.repo_url and not plan.repo_url.startswith("https://"):
        violations.append("non-HTTPS repo_url")

    if plan.artifact_dir and ".." in plan.artifact_dir:
        violations.append("path traversal in artifact path")

    for key in plan.env:
        upper_key = key.upper()
        if any(s in upper_key for s in ["TOKEN", "SECRET", "PASSWORD", "KEY"]):
            violations.append(f"environment variable name suggests secret: {key}")

    if (plan.image.endswith(":latest") or plan.image == "latest") and not (
        effective_config.allow_latest_images
    ):
        violations.append("image tag 'latest' is not allowed")

    if plan.network_enabled and not effective_config.allow_network:
        violations.append("network requested but config does not allow network access")

    if plan.network_enabled and not plan.repo_url:
        violations.append("network enabled without repo_url")

    if plan.resource_limits.timeout_seconds > effective_config.max_timeout_seconds:
        violations.append(
            "timeout exceeds configured maximum: "
            f"{plan.resource_limits.timeout_seconds} > {effective_config.max_timeout_seconds}"
        )

    if plan.resource_limits.memory_mb > effective_config.max_memory_mb:
        violations.append(
            "memory exceeds configured maximum: "
            f"{plan.resource_limits.memory_mb} > {effective_config.max_memory_mb}"
        )

    if plan.resource_limits.cpu_count > effective_config.max_cpu_count:
        violations.append(
            "CPU count exceeds configured maximum: "
            f"{plan.resource_limits.cpu_count} > {effective_config.max_cpu_count}"
        )

    if plan.working_dir and not _is_safe_relative_container_path(plan.working_dir):
        violations.append("working_dir must be a safe relative path under /workspace")

    return violations
