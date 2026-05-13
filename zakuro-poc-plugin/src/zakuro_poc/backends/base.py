from abc import ABC, abstractmethod
from pathlib import Path

from zakuro_poc.config import ZakuroPocConfig
from zakuro_poc.models import ExecutionPlan, ExecutionResult


class ExecutionBackend(ABC):
    @abstractmethod
    def run(
        self, plan: ExecutionPlan, artifact_dir: Path, config: ZakuroPocConfig
    ) -> ExecutionResult:
        pass
