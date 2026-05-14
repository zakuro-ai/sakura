import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field


class DockerConfig(BaseModel):
    remove_container: bool = True
    read_only_root: bool = False
    network_mode: Literal["none", "bridge"] = "none"
    pids_limit: int = Field(default=256, ge=1)
    user: str = "65532:65532"


class ZakuroBackendConfig(BaseModel):
    executable: str = Field(default="zc", min_length=1)
    execute_args: list[str] = Field(default_factory=lambda: ["execute"])
    plan_arg: str = "--plan"
    json_arg: str = "--json"


class ZakuroPocConfig(BaseModel):
    artifact_root: str = Field(default="./zakuro-artifacts", min_length=1)
    default_backend: str = "docker"
    default_image: str = "python:3.11-slim"
    allow_network: bool = False
    allow_shell: bool = False
    allow_latest_images: bool = False
    max_timeout_seconds: int = Field(default=3600, ge=1)
    max_memory_mb: int = Field(default=4096, ge=128)
    max_cpu_count: float = Field(default=4.0, gt=0.0)
    docker: DockerConfig = Field(default_factory=DockerConfig)
    zakuro: ZakuroBackendConfig = Field(default_factory=ZakuroBackendConfig)


def load_config(path: str | None = None) -> ZakuroPocConfig:
    search_paths = []
    if path:
        search_paths.append(Path(path))
    else:
        search_paths.extend(
            [
                Path("./zakuro-poc.json"),
                Path("~/.claude/zakuro-poc.json").expanduser(),
                Path(__file__).parent.parent.parent / "config" / "zakuro-poc.example.json",
            ]
        )

    for p in search_paths:
        if p.is_file():
            try:
                with open(p, encoding="utf-8") as f:
                    data = json.load(f)
                return ZakuroPocConfig(**data)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON in config {p}: {e}") from e

    # Fallback to built-in defaults
    return ZakuroPocConfig()
