import subprocess
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Literal, cast

from zakuro_poc.backends.base import ExecutionBackend
from zakuro_poc.config import ZakuroPocConfig
from zakuro_poc.execution.artifacts import write_execution_artifacts
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
    container_name = docker_container_name(artifact_dir)
    cmd = ["docker", "run"]

    if config.docker.remove_container:
        cmd.append("--rm")

    cmd.extend(
        [
            "--name",
            container_name,
            "--cpus",
            str(plan.resource_limits.cpu_count),
            "--memory",
            f"{plan.resource_limits.memory_mb}m",
            "--user",
            config.docker.user,
            "--pids-limit",
            str(config.docker.pids_limit),
            "--security-opt",
            "no-new-privileges",
            "--cap-drop",
            "ALL",
        ]
    )

    if config.docker.read_only_root:
        cmd.extend(["--read-only", "--tmpfs", "/tmp:rw,noexec,nosuid,size=64m"])

    if plan.network_enabled and config.allow_network:
        cmd.extend(["--network", config.docker.network_mode])
    else:
        cmd.extend(["--network", "none"])

    for key, value in sorted(plan.env.items()):
        cmd.extend(["--env", f"{key}={value}"])

    container_workdir = "/workspace"
    if plan.working_dir:
        container_workdir = str(PurePosixPath("/workspace") / PurePosixPath(plan.working_dir))

    cmd.extend(
        [
            "-v",
            f"{artifact_dir.absolute()}:/zakuro-artifacts",
            "-v",
            f"{artifact_dir.absolute()}/workspace:/workspace",
            "-w",
            container_workdir,
        ]
    )

    cmd.append(plan.image)
    cmd.extend(plan.command)

    return cmd


def docker_container_name(artifact_dir: Path) -> str:
    return f"zakuro-{artifact_dir.name}"


def force_remove_container(container_name: str) -> None:
    subprocess.run(
        ["docker", "rm", "-f", container_name],
        capture_output=True,
        timeout=10,
        check=False,
        text=True,
    )


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
            write_execution_artifacts(artifact_dir, plan, result)
            return result

        docker_cmd = build_docker_command(plan, artifact_dir, config)
        container_name = docker_container_name(artifact_dir)

        stdout_data = ""
        stderr_data = ""
        exit_code = None
        status: Literal["succeeded", "failed", "timed_out"] = "failed"
        error_message = None

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
            if config.docker.remove_container:
                force_remove_container(container_name)
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
            error_message = f"Job timed out after {plan.resource_limits.timeout_seconds} seconds"
        except Exception as e:
            status = "failed"
            stderr_data = f"Failed to execute docker command: {e}"
            error_message = str(e)

        finished_at = datetime.now(UTC)
        duration_ms = int((finished_at - started_at).total_seconds() * 1000)

        result = ExecutionResult(
            job_id=artifact_dir.name,
            job_name=plan.job_name,
            backend="docker",
            status=cast(Literal["succeeded", "failed", "timed_out", "rejected"], status),
            stdout=stdout_data,
            stderr=stderr_data,
            exit_code=exit_code,
            duration_ms=duration_ms,
            artifact_dir=str(artifact_dir),
            started_at=started_at,
            finished_at=finished_at,
            error_message=error_message,
        )

        write_execution_artifacts(artifact_dir, plan, result)
        return result
