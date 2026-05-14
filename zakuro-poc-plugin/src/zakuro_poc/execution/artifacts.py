import json
import re
from pathlib import Path

from zakuro_poc.models import ExecutionPlan, ExecutionResult


def _is_safe_job_id(job_id: str) -> bool:
    return bool(re.match(r"^[A-Za-z0-9_-]+$", job_id))


def create_artifact_dir(root: Path, job_id: str) -> Path:
    if not _is_safe_job_id(job_id):
        raise ValueError("Unsafe job ID")

    root.mkdir(parents=True, exist_ok=True)

    artifact_dir = root / job_id
    if not artifact_dir.resolve().is_relative_to(root.resolve()):
        raise ValueError("Path traversal detected")

    artifact_dir.mkdir(parents=True, exist_ok=True)
    (artifact_dir / "workspace").mkdir(parents=True, exist_ok=True)
    return artifact_dir


def write_text_artifact(path: Path, name: str, content: str) -> Path:
    if ".." in name or "/" in name or "\\" in name:
        raise ValueError("Invalid artifact name")

    file_path = path / name
    if not file_path.resolve().is_relative_to(path.resolve()):
        raise ValueError("Path traversal detected in artifact name")

    file_path.write_text(content, encoding="utf-8")
    return file_path


def write_execution_artifacts(
    artifact_dir: Path, plan: ExecutionPlan, result: ExecutionResult
) -> None:
    write_text_artifact(artifact_dir, "plan.json", plan.model_dump_json(indent=2))
    write_text_artifact(artifact_dir, "stdout.txt", result.stdout)
    write_text_artifact(artifact_dir, "stderr.txt", result.stderr)
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
        "error_message": result.error_message,
    }
    write_text_artifact(artifact_dir, "metadata.json", json.dumps(metadata, indent=2))
