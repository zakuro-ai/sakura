import re
from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field, field_validator

BackendType = Literal["noop", "docker"]
JobStatus = Literal["succeeded", "failed", "timed_out", "rejected"]


class ArtifactInfo(BaseModel):
    path: str


class ResourceLimits(BaseModel):
    cpu_count: float = Field(default=1.0, gt=0.0)
    memory_mb: int = Field(default=512, ge=128)
    gpu_count: int = Field(default=0, ge=0)
    timeout_seconds: int = Field(default=60, ge=1, le=3600)


class ExecutionPlan(BaseModel):
    job_name: str = Field(min_length=1)
    backend: BackendType = "docker"
    image: str = Field(default="python:3.11-slim", min_length=1)
    command: list[str] = Field(min_length=1)
    working_dir: str | None = None
    repo_url: str | None = None
    env: dict[str, str] = Field(default_factory=dict)
    resource_limits: ResourceLimits = Field(default_factory=ResourceLimits)
    artifact_dir: str | None = None
    network_enabled: bool = False
    created_by: str = "claude-code"

    @field_validator("repo_url")
    @classmethod
    def validate_repo_url(cls, v: str | None) -> str | None:
        if v is not None and not v.startswith("https://"):
            raise ValueError("repo_url must be HTTPS")
        return v

    @field_validator("env")
    @classmethod
    def validate_env_keys(cls, v: dict[str, str]) -> dict[str, str]:
        for key in v:
            if not re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", key):
                raise ValueError(f"invalid environment variable name: {key}")
        return v


class ExecutionResult(BaseModel):
    job_id: str
    job_name: str
    backend: str
    status: JobStatus
    stdout: str
    stderr: str
    exit_code: int | None
    duration_ms: int
    artifact_dir: str
    started_at: datetime
    finished_at: datetime
    error_message: str | None = None
