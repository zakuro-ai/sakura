from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

from zakuro_poc.backends.base import ExecutionBackend
from zakuro_poc.backends.docker_backend import DockerBackend
from zakuro_poc.backends.noop_backend import NoopBackend
from zakuro_poc.backends.zakuro_backend import ZakuroBackend
from zakuro_poc.config import ZakuroPocConfig
from zakuro_poc.execution.artifacts import (
    create_artifact_dir,
    write_execution_artifacts,
    write_text_artifact,
)
from zakuro_poc.execution.ids import new_job_id
from zakuro_poc.models import ExecutionPlan, ExecutionResult
from zakuro_poc.validation import validate_plan_or_raise

BackendFactory = Callable[[str], ExecutionBackend | None]


def _new_artifact_dir(config: ZakuroPocConfig) -> Path:
    return create_artifact_dir(Path(config.artifact_root), new_job_id())


def _build_rejected_result(
    plan: ExecutionPlan,
    artifact_dir: Path,
    started_at: datetime,
    error_message: str,
) -> ExecutionResult:
    finished_at = datetime.now(UTC)
    return ExecutionResult(
        job_id=artifact_dir.name,
        job_name=plan.job_name,
        backend=plan.backend,
        status="rejected",
        stdout="",
        stderr=error_message,
        exit_code=None,
        duration_ms=int((finished_at - started_at).total_seconds() * 1000),
        artifact_dir=str(artifact_dir),
        started_at=started_at,
        finished_at=finished_at,
        error_message=error_message,
    )


def default_backend_factory(backend_name: str) -> ExecutionBackend | None:
    if backend_name == "noop":
        return NoopBackend()
    if backend_name == "docker":
        return DockerBackend()
    if backend_name == "zakuro":
        return ZakuroBackend()
    return None


def execute_plan(
    plan: ExecutionPlan,
    config: ZakuroPocConfig,
    backend_factory: BackendFactory = default_backend_factory,
) -> ExecutionResult:
    started_at = datetime.now(UTC)

    try:
        validate_plan_or_raise(plan, config)
    except ValueError as e:
        artifact_dir = _new_artifact_dir(config)
        result = _build_rejected_result(plan, artifact_dir, started_at, str(e))
        write_execution_artifacts(artifact_dir, plan, result)
        return result

    artifact_dir = _new_artifact_dir(config)
    backend = backend_factory(plan.backend)
    if backend is None:
        result = _build_rejected_result(
            plan, artifact_dir, started_at, f"Unknown backend: {plan.backend}"
        )
        write_execution_artifacts(artifact_dir, plan, result)
        return result

    write_text_artifact(artifact_dir, "plan.json", plan.model_dump_json(indent=2))
    result = backend.run(plan, artifact_dir, config)
    write_execution_artifacts(artifact_dir, plan, result)
    return result
