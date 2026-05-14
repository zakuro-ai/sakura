from datetime import UTC, datetime
from pathlib import Path

from zakuro_poc.backends.base import ExecutionBackend
from zakuro_poc.config import ZakuroPocConfig
from zakuro_poc.execution.runner import execute_plan
from zakuro_poc.models import ExecutionPlan, ExecutionResult


class FakeZakuroBackend(ExecutionBackend):
    def __init__(self, *, fail: bool = False) -> None:
        self.fail = fail
        self.was_called = False
        self.plan_file_existed_before_run = False

    def run(
        self,
        plan: ExecutionPlan,
        artifact_dir: Path,
        config: ZakuroPocConfig,  # noqa: ARG002
    ) -> ExecutionResult:
        self.was_called = True
        self.plan_file_existed_before_run = (artifact_dir / "plan.json").exists()
        started_at = datetime.now(UTC)
        finished_at = datetime.now(UTC)
        if self.fail:
            return ExecutionResult(
                job_id=artifact_dir.name,
                job_name=plan.job_name,
                backend="zakuro",
                status="failed",
                stdout="",
                stderr="fake zakuro failure",
                exit_code=17,
                duration_ms=0,
                artifact_dir=str(artifact_dir),
                started_at=started_at,
                finished_at=finished_at,
                error_message="fake zakuro failure",
            )

        return ExecutionResult(
            job_id=artifact_dir.name,
            job_name=plan.job_name,
            backend="zakuro",
            status="succeeded",
            stdout="fake zakuro success",
            stderr="",
            exit_code=0,
            duration_ms=0,
            artifact_dir=str(artifact_dir),
            started_at=started_at,
            finished_at=finished_at,
        )


def test_fake_zakuro_backend_preserves_runner_contract(tmp_path):
    backend = FakeZakuroBackend()
    config = ZakuroPocConfig(artifact_root=str(tmp_path))
    plan = ExecutionPlan(job_name="zakuro-contract", backend="zakuro", command=["echo", "hello"])

    result = execute_plan(
        plan, config, backend_factory=lambda name: backend if name == "zakuro" else None
    )

    artifact_dir = tmp_path / result.job_id
    assert backend.was_called
    assert backend.plan_file_existed_before_run
    assert result.status == "succeeded"
    assert result.backend == "zakuro"
    assert result.stdout == "fake zakuro success"
    assert (artifact_dir / "plan.json").exists()
    assert (artifact_dir / "result.json").exists()
    assert (artifact_dir / "stdout.txt").read_text(encoding="utf-8") == "fake zakuro success"
    assert (artifact_dir / "stderr.txt").read_text(encoding="utf-8") == ""
    assert (artifact_dir / "metadata.json").exists()
    assert (artifact_dir / "workspace").is_dir()


def test_fake_zakuro_backend_failure_is_normalised(tmp_path):
    backend = FakeZakuroBackend(fail=True)
    config = ZakuroPocConfig(artifact_root=str(tmp_path))
    plan = ExecutionPlan(
        job_name="zakuro-contract-fail", backend="zakuro", command=["echo", "hello"]
    )

    result = execute_plan(
        plan, config, backend_factory=lambda name: backend if name == "zakuro" else None
    )

    artifact_dir = tmp_path / result.job_id
    assert backend.was_called
    assert result.status == "failed"
    assert result.exit_code == 17
    assert result.error_message == "fake zakuro failure"
    assert (artifact_dir / "stderr.txt").read_text(encoding="utf-8") == "fake zakuro failure"


def test_invalid_zakuro_plan_is_rejected_before_backend_call(tmp_path):
    backend = FakeZakuroBackend()
    config = ZakuroPocConfig(artifact_root=str(tmp_path))
    plan = ExecutionPlan(
        job_name="zakuro-contract-reject", backend="zakuro", command=["bash", "-c", "ls"]
    )

    result = execute_plan(
        plan, config, backend_factory=lambda name: backend if name == "zakuro" else None
    )

    assert not backend.was_called
    assert result.status == "rejected"
    assert "Security policy violations" in (result.error_message or "")
