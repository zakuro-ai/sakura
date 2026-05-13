from pathlib import Path

from zakuro_poc.backends.base import ExecutionBackend
from zakuro_poc.backends.docker_backend import DockerBackend
from zakuro_poc.backends.noop_backend import NoopBackend
from zakuro_poc.config import ZakuroPocConfig
from zakuro_poc.execution.artifacts import create_artifact_dir, write_text_artifact
from zakuro_poc.execution.ids import new_job_id
from zakuro_poc.models import ExecutionPlan, ExecutionResult
from zakuro_poc.validation import validate_plan_or_raise


def execute_plan(
    plan: ExecutionPlan,
    config: ZakuroPocConfig,
) -> ExecutionResult:
    # 1. create job ID
    job_id = new_job_id()

    # 2. create artifact directory
    root_path = Path(config.artifact_root)
    artifact_dir = create_artifact_dir(root_path, job_id)

    # 3. write plan
    write_text_artifact(artifact_dir, "plan.json", plan.model_dump_json(indent=2))

    # 4. validate plan
    validate_plan_or_raise(plan)

    # 5. select backend
    if plan.backend == "noop":
        backend: ExecutionBackend = NoopBackend()
    elif plan.backend == "docker":
        backend = DockerBackend()
    else:
        raise ValueError(f"Unknown backend: {plan.backend}")

    # 6. execute backend
    result = backend.run(plan, artifact_dir, config)

    # 7. write result
    write_text_artifact(artifact_dir, "result.json", result.model_dump_json(indent=2))

    # 8. return result
    return result
