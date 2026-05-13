from zakuro_poc.models import ExecutionPlan
from zakuro_poc.security.policy import validate_security_policy


def validate_plan_or_raise(plan: ExecutionPlan) -> None:
    violations = validate_security_policy(plan)
    if violations:
        raise ValueError("Security policy violations: " + ", ".join(violations))
