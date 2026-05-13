import json
from datetime import UTC, datetime
from pathlib import Path

from zakuro_poc.backends.base import ExecutionBackend
from zakuro_poc.config import ZakuroPocConfig
from zakuro_poc.execution.artifacts import write_text_artifact
from zakuro_poc.models import ExecutionPlan, ExecutionResult


class NoopBackend(ExecutionBackend):
    def run(
        self,
        plan: ExecutionPlan,
        artifact_dir: Path,
        config: ZakuroPocConfig,  # noqa: ARG002
    ) -> ExecutionResult:
        started_at = datetime.now(UTC)

        stdout = f"NoopBackend: Executing plan {plan.job_name}\nCommand: {plan.command}"
        write_text_artifact(artifact_dir, "stdout.txt", stdout)
        write_text_artifact(artifact_dir, "stderr.txt", "")

        finished_at = datetime.now(UTC)
        duration_ms = int((finished_at - started_at).total_seconds() * 1000)

        result = ExecutionResult(
            job_id=artifact_dir.name,
            job_name=plan.job_name,
            backend="noop",
            status="succeeded",
            stdout=stdout,
            stderr="",
            exit_code=0,
            duration_ms=duration_ms,
            artifact_dir=str(artifact_dir),
            started_at=started_at,
            finished_at=finished_at,
        )

        write_text_artifact(artifact_dir, "result.json", result.model_dump_json(indent=2))

        metadata = {
            "job_id": result.job_id,
            "job_name": result.job_name,
            "backend": result.backend,
            "image": plan.image,
            "command": plan.command,
            "resource_limits": plan.resource_limits.model_dump(),
            "started_at": result.started_at.isoformat(),
            "finished_at": result.finished_at.isoformat(),
            "duration_ms": result.duration_ms,
            "status": result.status,
        }
        write_text_artifact(artifact_dir, "metadata.json", json.dumps(metadata, indent=2))

        return result
