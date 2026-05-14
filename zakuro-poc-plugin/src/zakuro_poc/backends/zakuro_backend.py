import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal, cast

from zakuro_poc.backends.base import ExecutionBackend
from zakuro_poc.config import ZakuroPocConfig
from zakuro_poc.models import ExecutionPlan, ExecutionResult


def build_zakuro_command(
    plan_file: Path,
    config: ZakuroPocConfig,
) -> list[str]:
    return [
        config.zakuro.executable,
        *config.zakuro.execute_args,
        config.zakuro.plan_arg,
        str(plan_file),
        config.zakuro.json_arg,
    ]


class ZakuroBackend(ExecutionBackend):
    def run(
        self, plan: ExecutionPlan, artifact_dir: Path, config: ZakuroPocConfig
    ) -> ExecutionResult:
        started_at = datetime.now(UTC)
        plan_file = artifact_dir / "plan.json"
        command = build_zakuro_command(plan_file, config)

        stdout_data = ""
        stderr_data = ""
        exit_code = None
        status: Literal["succeeded", "failed", "timed_out"] = "failed"
        error_message = None

        try:
            proc = subprocess.run(
                command,
                capture_output=True,
                timeout=plan.resource_limits.timeout_seconds,
                text=True,
            )
            stdout_data = proc.stdout
            stderr_data = proc.stderr
            exit_code = proc.returncode
            status = "succeeded" if exit_code == 0 else "failed"
            if proc.returncode != 0:
                error_message = "zc execute returned a non-zero exit code"
        except FileNotFoundError:
            stderr_data = f"Zakuro executable not found: {config.zakuro.executable}"
            error_message = stderr_data
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
            stderr_data += (
                f"\nZakuro execution timed out after {plan.resource_limits.timeout_seconds} seconds"
            )
            error_message = (
                f"Zakuro execution timed out after {plan.resource_limits.timeout_seconds} seconds"
            )
        except Exception as e:
            stderr_data = f"Failed to execute Zakuro command: {e}"
            error_message = str(e)

        finished_at = datetime.now(UTC)
        return ExecutionResult(
            job_id=artifact_dir.name,
            job_name=plan.job_name,
            backend="zakuro",
            status=cast(Literal["succeeded", "failed", "timed_out", "rejected"], status),
            stdout=stdout_data,
            stderr=stderr_data,
            exit_code=exit_code,
            duration_ms=int((finished_at - started_at).total_seconds() * 1000),
            artifact_dir=str(artifact_dir),
            started_at=started_at,
            finished_at=finished_at,
            error_message=error_message,
        )
