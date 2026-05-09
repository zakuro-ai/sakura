"""Benchmark harness — Workload, RunReport, BaselineRunner, SakuraRunner.

Plan 5 Task 4 ships the dataclass types; T5 fills in the runners; T6+ add
concrete workloads.
"""
from __future__ import annotations

import json
import platform
import subprocess
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Literal, Optional


@dataclass
class Workload:
    """A benchmarkable training workload — model + data + eval fn.

    `make_model`, `make_train_loader`, `make_val_loader`, `eval_fn` are
    callables so each runner can rebuild fresh state without sharing.
    """
    name: str
    tier: Literal["smoke", "ci", "perf"]
    make_model: Callable[[], Any]
    make_train_loader: Callable[[], Any]
    make_val_loader: Callable[[], Any]
    eval_fn: Callable[[Any, Any], dict]
    epochs: int = 1
    metric_target: Optional[tuple[str, float]] = None  # ("val_acc", 0.85)


@dataclass
class RunReport:
    """The output of a single Workload + Runner execution."""
    workload: str
    framework: Literal["pytorch-ddp", "lightning", "hf-trainer", "tf"]
    sakura_services: Optional[list[str]] = None
    elapsed_secs: float = 0.0
    samples_per_sec: float = 0.0
    peak_gpu_mem_mb: float = 0.0
    final_metrics: dict = field(default_factory=dict)
    per_stage_secs: dict[str, float] = field(default_factory=dict)
    git_sha: str = ""
    hardware: dict = field(default_factory=dict)

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)

    @classmethod
    def from_json(cls, s: str) -> "RunReport":
        d = json.loads(s)
        return cls(**d)


def detect_hardware() -> dict:
    """Capture hardware info for the report (CPU, GPU, RAM, OS, python)."""
    import torch

    info = {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
    }
    if torch.cuda.is_available():
        info["gpu_count"] = torch.cuda.device_count()
        try:
            info["gpu_name"] = torch.cuda.get_device_name(0)
        except Exception:
            pass
    return info


def detect_git_sha() -> str:
    try:
        out = subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL)
        return out.decode().strip()
    except Exception:
        return ""


class _DeviceLoader:
    """Wraps any iterable of batches and moves tensors to `device` on the fly.

    Used by runners so that `eval_fn` always receives device-local batches,
    even when a Workload builds plain lists of CPU tensors.
    """

    def __init__(self, loader, device: str):
        self._loader = loader
        self._device = device

    def __iter__(self):
        for batch in self._loader:
            yield _move_batch(batch, self._device)

    def __len__(self):
        return len(self._loader)


def _move_batch(batch, device: str):
    """Recursively move tensors in a batch (tuple/list/dict/tensor) to `device`."""
    if hasattr(batch, "to"):
        return batch.to(device)
    if isinstance(batch, dict):
        return {k: _move_batch(v, device) for k, v in batch.items()}
    if isinstance(batch, (tuple, list)):
        moved = [_move_batch(b, device) for b in batch]
        return type(batch)(moved)
    return batch


class BaselineRunner:
    """Run a Workload with a vanilla framework (no Sakura services)."""

    def __init__(self, framework: Literal["pytorch-ddp", "lightning", "hf-trainer"] = "pytorch-ddp"):
        self.framework = framework

    def run(self, workload: Workload) -> RunReport:
        if self.framework == "pytorch-ddp":
            return self._run_raw_pytorch(workload)
        if self.framework == "lightning":
            return self._run_lightning(workload)
        if self.framework == "hf-trainer":
            return self._run_hf(workload)
        raise NotImplementedError(f"baseline framework {self.framework!r} not yet wired")

    def _run_raw_pytorch(self, workload: Workload) -> RunReport:
        import torch

        model = workload.make_model()
        train_loader = workload.make_train_loader()
        val_loader = workload.make_val_loader()
        opt = self._make_optimizer(model)

        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = model.to(device)

        t0 = time.perf_counter()
        n_samples = 0
        for _epoch in range(workload.epochs):
            model.train()
            for batch in train_loader:
                x, y = self._batch_to_device(batch, device)
                opt.zero_grad()
                logits = model(x)
                loss = self._compute_loss(logits, y)
                loss.backward()
                opt.step()
                n_samples += y.size(0) if hasattr(y, "size") else len(y)
        elapsed = time.perf_counter() - t0

        model.eval()
        # Wrap val_loader so eval_fn receives device-placed batches even when
        # the workload builds plain lists of CPU tensors.
        dev_val_loader = _DeviceLoader(val_loader, device)
        final_metrics = workload.eval_fn(model, dev_val_loader)

        return RunReport(
            workload=workload.name,
            framework=self.framework,
            elapsed_secs=elapsed,
            samples_per_sec=n_samples / max(elapsed, 1e-9),
            peak_gpu_mem_mb=self._peak_gpu_mem_mb(),
            final_metrics=final_metrics,
            git_sha=detect_git_sha(),
            hardware=detect_hardware(),
        )

    def _run_lightning(self, workload: Workload) -> RunReport:
        # Minimal lightning runner — Plan 4 framework adapters give us a path,
        # but for the bench harness we need a self-contained loop. The simplest
        # bridge: wrap the user's model in LightningModule on the fly.
        raise NotImplementedError(
            "Lightning baseline runner is Plan 5.x — for now use 'pytorch-ddp' "
            "with a workload that returns plain torch.nn.Module."
        )

    def _run_hf(self, workload: Workload) -> RunReport:
        raise NotImplementedError(
            "HF baseline runner is Plan 5.x — for now use 'pytorch-ddp'."
        )

    @staticmethod
    def _make_optimizer(model):
        import torch
        return torch.optim.SGD(model.parameters(), lr=0.01)

    @staticmethod
    def _batch_to_device(batch, device):
        if isinstance(batch, (tuple, list)) and len(batch) == 2:
            x, y = batch
            x = _move_batch(x, device)
            y = y.to(device) if hasattr(y, "to") else y
            return x, y
        return batch

    @staticmethod
    def _compute_loss(logits, y):
        import torch
        return torch.nn.functional.cross_entropy(logits, y)

    @staticmethod
    def _peak_gpu_mem_mb() -> float:
        import torch
        if torch.cuda.is_available():
            return torch.cuda.max_memory_allocated() / (1024 * 1024)
        return 0.0


class SakuraRunner(BaselineRunner):
    """Run a Workload with Sakura services installed on a SakuraRuntime."""

    def __init__(
        self,
        framework: Literal["pytorch-ddp", "lightning", "hf-trainer"] = "pytorch-ddp",
        services: Optional[list] = None,
        compute: Optional[Any] = None,
    ):
        super().__init__(framework=framework)
        self._services = services or []
        self._compute = compute

    def run(self, workload: Workload) -> RunReport:
        from sakura.runtime import SakuraRuntime

        rt = SakuraRuntime(compute=self._compute)
        for svc in self._services:
            rt.install(svc)

        # Run with the runtime active; the services observe via emitted events.
        # For Plan 5, the bench harness uses raw pytorch + DDPAdapter for event emission.
        if self.framework == "pytorch-ddp":
            with rt:
                report = self._run_raw_pytorch_with_adapter(workload, rt)
        else:
            raise NotImplementedError(
                f"SakuraRunner framework={self.framework!r} not yet wired (Plan 5.x)"
            )
        report.sakura_services = [s.name for s in self._services]
        return report

    def _run_raw_pytorch_with_adapter(self, workload: Workload, rt) -> RunReport:
        import torch
        from sakura.adapters.ddp import DDPAdapter

        model = workload.make_model()
        train_loader = workload.make_train_loader()
        val_loader = workload.make_val_loader()
        opt = self._make_optimizer(model)

        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = model.to(device)

        adapter = DDPAdapter(rt, rank=0, world_size=1)
        adapter.on_train_begin(model, opt, train_loader, val_loader=val_loader)

        t0 = time.perf_counter()
        n_samples = 0
        metrics: dict = {}
        for epoch in range(workload.epochs):
            adapter.on_epoch_begin(epoch)
            model.train()
            for step, batch in enumerate(train_loader):
                adapter.on_train_step_begin(model, batch, step)
                x, y = self._batch_to_device(batch, device)
                opt.zero_grad()
                logits = model(x)
                loss = self._compute_loss(logits, y)
                loss.backward()
                adapter.on_optimizer_step(opt)
                opt.step()
                n_samples += y.size(0) if hasattr(y, "size") else len(y)
            model.eval()
            dev_val_loader = _DeviceLoader(val_loader, device)
            metrics = workload.eval_fn(model, dev_val_loader)
            adapter.on_epoch_end(epoch, model, opt, metrics=metrics)

        elapsed = time.perf_counter() - t0
        adapter.on_train_end(model)

        return RunReport(
            workload=workload.name,
            framework=self.framework,
            elapsed_secs=elapsed,
            samples_per_sec=n_samples / max(elapsed, 1e-9),
            peak_gpu_mem_mb=self._peak_gpu_mem_mb(),
            final_metrics=metrics,
            git_sha=detect_git_sha(),
            hardware=detect_hardware(),
        )


__all__ = ["Workload", "RunReport", "detect_hardware", "detect_git_sha", "BaselineRunner", "SakuraRunner"]
