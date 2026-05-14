from zakuro_poc.config import ZakuroPocConfig
from zakuro_poc.models import ExecutionPlan
from zakuro_poc.security.policy import validate_security_policy


def validate_plan(plan: ExecutionPlan, config: ZakuroPocConfig | None = None) -> list[str]:
    return validate_security_policy(plan, config)


def validate_plan_or_raise(plan: ExecutionPlan, config: ZakuroPocConfig | None = None) -> None:
    violations = validate_plan(plan, config)
    if violations:
        raise ValueError("Security policy violations: " + ", ".join(violations))
