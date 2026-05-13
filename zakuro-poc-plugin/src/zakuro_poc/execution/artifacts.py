import re
from pathlib import Path


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
    return artifact_dir


def write_text_artifact(path: Path, name: str, content: str) -> Path:
    if ".." in name or "/" in name or "\\" in name:
        raise ValueError("Invalid artifact name")

    file_path = path / name
    if not file_path.resolve().is_relative_to(path.resolve()):
        raise ValueError("Path traversal detected in artifact name")

    file_path.write_text(content, encoding="utf-8")
    return file_path
