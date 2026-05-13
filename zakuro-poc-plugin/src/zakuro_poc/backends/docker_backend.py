import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path

from zakuro_poc.backends.base import ExecutionBackend
from zakuro_poc.config import ZakuroPocConfig
from zakuro_poc.execution.artifacts import write_text_artifact
from zakuro_poc.models import ExecutionPlan, ExecutionResult


def docker_available() -> bool:
    try:
        subprocess.run(["docker", "version"], capture_output=True, timeout=5, check=True)
        return True
    except (subprocess.SubprocessError, FileNotFoundError, OSError):
        return False


def build_docker_command(
    plan: ExecutionPlan,
    artifact_dir: Path,
    config: ZakuroPocConfig,
) -> list[str]:
    cmd = ["docker", "run"]

    if config.docker.remove_container:
        cmd.append("--rm")

    cmd.extend(
        [
            "--name",
            f"zakuro-{artifact_dir.name}",
            "--cpus",
            str(plan.resource_limits.cpu_count),
            "--memory",
            f"{plan.resource_limits.memory_mb}m",
        ]
    )

    if plan.network_enabled and config.allow_network:
        cmd.extend(["--network", "bridge"])
    else:
        cmd.extend(["--network", "none"])

    cmd.extend(
        [
            "-v",
            f"{artifact_dir.absolute()}:/zakuro-artifacts",
            "-v",
            f"{artifact_dir.absolute()}/workspace:/workspace",
            "-w",
            "/workspace",
        ]
    )

    cmd.append(plan.image)
    cmd.extend(plan.command)

    return cmd


class DockerBackend(ExecutionBackend):
    def run(
        self, plan: ExecutionPlan, artifact_dir: Path, config: ZakuroPocConfig
    ) -> ExecutionResult:
        started_at = datetime.now(UTC)

        if not docker_available():
            finished_at = datetime.now(UTC)
            duration_ms = int((finished_at - started_at).total_seconds() * 1000)
            result = ExecutionResult(
                job_id=artifact_dir.name,
                job_name=plan.job_name,
                backend="docker",
                status="failed",
                stdout="",
                stderr="Docker is not available",
                exit_code=None,
                duration_ms=duration_ms,
                artifact_dir=str(artifact_dir),
                started_at=started_at,
                finished_at=finished_at,
                error_message="Docker CLI is not available or daemon is unreachable",
            )
            write_text_artifact(artifact_dir, "stdout.txt", "")
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
            }
            write_text_artifact(artifact_dir, "metadata.json", json.dumps(metadata, indent=2))
            return result

        workspace_dir = artifact_dir / "workspace"
        workspace_dir.mkdir(parents=True, exist_ok=True)

        write_text_artifact(artifact_dir, "plan.json", plan.model_dump_json(indent=2))

        docker_cmd = build_docker_command(plan, artifact_dir, config)

        stdout_data = ""
        stderr_data = ""
        exit_code = None
        status = "failed"

        try:
            proc = subprocess.run(
                docker_cmd,
                capture_output=True,
                timeout=plan.resource_limits.timeout_seconds,
                text=True,
            )
            stdout_data = proc.stdout
            stderr_data = proc.stderr
            exit_code = proc.returncode
            status = "succeeded" if exit_code == 0 else "failed"
        except subprocess.TimeoutExpired as e:
            status = "timed_out"
            stdout_data = (
                e.stdout.decode("utf-8", errors="replace")
                if isinstance(e.stdout, bytes)
                else (e.stdout or "")
            )
            stderr_data = (
                e.stderr.decode("utf-8", errors="replace")
                if isinstance(e.stderr, bytes)
                else (e.stderr or "")
            )
            stderr_data += f"\nJob timed out after {plan.resource_limits.timeout_seconds} seconds"
        except Exception as e:
            status = "failed"
            stderr_data = f"Failed to execute docker command: {e}"

        finished_at = datetime.now(UTC)
        duration_ms = int((finished_at - started_at).total_seconds() * 1000)

        result = ExecutionResult(
            job_id=artifact_dir.name,
            job_name=plan.job_name,
            backend="docker",
            status=status,  # type: ignore
            stdout=stdout_data,
            stderr=stderr_data,
            exit_code=exit_code,
            duration_ms=duration_ms,
            artifact_dir=str(artifact_dir),
            started_at=started_at,
            finished_at=finished_at,
        )

        write_text_artifact(artifact_dir, "stdout.txt", stdout_data)
        write_text_artifact(artifact_dir, "stderr.txt", stderr_data)
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
