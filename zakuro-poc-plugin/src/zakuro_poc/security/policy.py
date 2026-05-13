from zakuro_poc.models import ExecutionPlan


def validate_security_policy(plan: ExecutionPlan) -> list[str]:
    violations = []
    if not plan.command:
        violations.append("empty command")
    else:
        forbidden_shells = {"sh", "bash", "zsh", "fish", "cmd.exe", "powershell"}
        if plan.command[0] in forbidden_shells:
            violations.append(f"shell interpreters are not allowed: {plan.command[0]}")

    if plan.repo_url and not plan.repo_url.startswith("https://"):
        violations.append("non-HTTPS repo_url")

    if plan.artifact_dir and ".." in plan.artifact_dir:
        violations.append("path traversal in artifact path")

    for key in plan.env:
        upper_key = key.upper()
        if any(s in upper_key for s in ["TOKEN", "SECRET", "PASSWORD", "KEY"]):
            violations.append(f"environment variable name suggests secret: {key}")

    if plan.image.endswith(":latest") or plan.image == "latest":
        violations.append("image tag 'latest' is not allowed")

    if plan.network_enabled and not plan.repo_url:
        violations.append("network enabled without repo_url")

    return violations
