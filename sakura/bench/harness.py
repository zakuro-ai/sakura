"""Benchmark harness — Workload, RunReport, BaselineRunner, SakuraRunner.

Plan 5 Task 4 ships the dataclass types; T5 fills in the runners; T6+ add
concrete workloads.
"""
from __future__ import annotations

import json
import platform
import subprocess
import tempfile
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Literal, Optional


_CUDA_AVAILABLE: Optional[bool] = None


def _cuda_available() -> bool:
    """Cached torch.cuda.is_available() — first call probes NVML (~5ms);
    subsequent calls are O(1). The harness queries this 4-7 times per
    RunReport so caching matters at smoke scale."""
    global _CUDA_AVAILABLE
    if _CUDA_AVAILABLE is None:
        try:
            import torch
            _CUDA_AVAILABLE = bool(torch.cuda.is_available())
        except Exception:
            _CUDA_AVAILABLE = False
    return _CUDA_AVAILABLE


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
        "cuda_available": bool(_cuda_available()),
    }
    if _cuda_available():
        info["gpu_count"] = torch.cuda.device_count()
        try:
            info["gpu_name"] = torch.cuda.get_device_name(0)
        except Exception:
            pass  # best-effort: GPU info unavailable
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

        device = "cuda" if _cuda_available() else "cpu"
        model = model.to(device)
        # Wrap val_loader so eval_fn receives device-placed batches even when
        # the workload builds plain lists of CPU tensors.
        dev_val_loader = _DeviceLoader(val_loader, device)

        t0 = time.perf_counter()
        n_samples = 0
        final_metrics: dict = {}
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
            # Per-epoch synchronous eval — matches real-world training loops
            # (early stopping, monitoring) and gives the sakura+AsyncEval
            # comparison something legitimate to overlap against. The single
            # end-of-training eval would be unfair to sakura.
            model.eval()
            final_metrics = workload.eval_fn(model, dev_val_loader)
        elapsed = time.perf_counter() - t0

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
        """Wrap the workload's nn.Module in a LightningModule and call trainer.fit.

        Works for any Workload whose make_model() returns a torch.nn.Module
        and whose train_loader yields (x, y) tuples — the auto-wrapper uses
        cross_entropy as the loss.
        """
        return self._run_lightning_impl(workload, adapter=None)

    def _run_hf(self, workload: Workload) -> RunReport:
        """HuggingFace Trainer baseline.

        Contract: workload.make_model() returns a transformers.PreTrainedModel
        whose forward(**batch) returns ModelOutput with `.loss` and `.logits`,
        and the loaders yield single-dict batches containing a `labels` key.
        See sakura/bench/workloads/distilbert_hf.py for a reference workload.
        """
        return self._run_hf_impl(workload, callbacks=None)

    def _run_hf_impl(self, workload: Workload, callbacks=None) -> RunReport:
        """Shared body of the HF Trainer baseline + sakura paths.

        When `callbacks` is None: pure baseline.
        When `callbacks` is a list (typically [HFAdapter(rt)]): the adapter
        translates Trainer hooks into runtime events so installed services
        observe the lifecycle.
        """
        from transformers import Trainer, TrainingArguments

        model = workload.make_model()
        train_loader = workload.make_train_loader()
        val_loader = workload.make_val_loader()

        # Trainer wants a Dataset + collator; pull them off the loader.
        train_dataset = getattr(train_loader, "dataset", None)
        collate_fn = getattr(train_loader, "collate_fn", None)
        batch_size = getattr(train_loader, "batch_size", 8) or 8
        if train_dataset is None:
            raise ValueError(
                "HF Trainer baseline requires workload.make_train_loader() to "
                "return a torch.utils.data.DataLoader (we read .dataset and "
                ".collate_fn off it)."
            )

        with tempfile.TemporaryDirectory(prefix="sakura-hf-") as out_dir:
            args = TrainingArguments(
                output_dir=out_dir,
                num_train_epochs=workload.epochs,
                per_device_train_batch_size=batch_size,
                logging_strategy="no",
                save_strategy="no",
                eval_strategy="no",
                report_to=[],
                disable_tqdm=True,
                use_cpu=not _cuda_available(),
                dataloader_num_workers=0,
            )
            trainer = Trainer(
                model=model,
                args=args,
                data_collator=collate_fn,
                train_dataset=train_dataset,
                callbacks=list(callbacks) if callbacks else None,
            )

            t0 = time.perf_counter()
            trainer.train()
            elapsed = time.perf_counter() - t0

        try:
            n_samples = len(train_dataset) * workload.epochs
        except (AttributeError, TypeError):
            n_samples = 0

        # Eval on the trained model. The Trainer may have moved it to GPU.
        eval_model = trainer.model
        eval_model.eval()
        device = next(eval_model.parameters()).device
        device_str = device.type if hasattr(device, "type") else str(device)
        dev_val_loader = _DeviceLoader(val_loader, device_str)
        final_metrics = workload.eval_fn(eval_model, dev_val_loader)

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

    def _run_lightning_impl(self, workload: Workload, adapter=None) -> RunReport:
        """Shared body of the Lightning baseline + sakura paths.

        When `adapter` is None: pure baseline (no Sakura).
        When `adapter` is a LightningAdapter: it's installed as a Trainer callback
        so services on the adapter's runtime observe the lifecycle events.
        """
        import lightning as L
        import torch

        base_model = workload.make_model()
        train_loader = workload.make_train_loader()
        val_loader = workload.make_val_loader()

        # Auto-wrap in a LightningModule. The wrapper assumes (x, y) batches
        # + cross-entropy loss — same contract as _run_raw_pytorch.
        class _AutoLM(L.LightningModule):
            def __init__(self, m):
                super().__init__()
                self.model = m

            def forward(self, x):
                return self.model(x)

            def training_step(self, batch, batch_idx):
                x, y = batch
                logits = self.model(x)
                return torch.nn.functional.cross_entropy(logits, y)

            def configure_optimizers(self):
                return torch.optim.SGD(self.parameters(), lr=0.01)

        lm = _AutoLM(base_model)

        callbacks = [adapter] if adapter is not None else []
        trainer = L.Trainer(
            max_epochs=workload.epochs,
            accelerator="auto",
            devices=1,
            enable_progress_bar=False,
            enable_model_summary=False,
            logger=False,
            enable_checkpointing=False,
            callbacks=callbacks,
        )

        t0 = time.perf_counter()
        trainer.fit(lm, train_loader)
        elapsed = time.perf_counter() - t0

        # Approximate sample count (dataset size * epochs).
        try:
            n_samples = len(train_loader.dataset) * workload.epochs
        except (AttributeError, TypeError):
            n_samples = 0
            for _ in train_loader:
                n_samples += 1
            n_samples *= workload.epochs

        # Eval against the wrapped LightningModule's underlying model, since
        # Lightning may have moved it to GPU.
        eval_model = lm.model
        eval_model.eval()
        device = next(eval_model.parameters()).device
        device_str = device.type if hasattr(device, "type") else str(device)
        dev_val_loader = _DeviceLoader(val_loader, device_str)
        final_metrics = workload.eval_fn(eval_model, dev_val_loader)

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
        if _cuda_available():
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
        if self.framework == "pytorch-ddp":
            with rt:
                report = self._run_raw_pytorch_with_adapter(workload, rt)
        elif self.framework == "lightning":
            with rt:
                report = self._run_lightning_with_adapter(workload, rt)
        elif self.framework == "hf-trainer":
            with rt:
                report = self._run_hf_with_adapter(workload, rt)
        else:
            raise NotImplementedError(
                f"SakuraRunner framework={self.framework!r} not yet wired"
            )
        report.sakura_services = [s.name for s in self._services]
        return report

    def _run_lightning_with_adapter(self, workload: Workload, rt) -> RunReport:
        """Sakura + Lightning: install LightningAdapter on the runtime + Trainer callback.

        Reuses BaselineRunner._run_lightning_impl with adapter=LightningAdapter(rt).
        """
        from sakura.adapters.lightning import LightningAdapter
        adapter = LightningAdapter(rt, rank=0, world_size=1)
        return self._run_lightning_impl(workload, adapter=adapter)

    def _run_hf_with_adapter(self, workload: Workload, rt) -> RunReport:
        """Sakura + HF Trainer: install HFAdapter as a Trainer callback.

        Reuses BaselineRunner._run_hf_impl with callbacks=[HFAdapter(rt)] so
        runtime services observe Trainer's lifecycle hooks.
        """
        from sakura.adapters.huggingface import HFAdapter
        adapter = HFAdapter(rt, rank=0, world_size=1)
        return self._run_hf_impl(workload, callbacks=[adapter])

    def _run_raw_pytorch_with_adapter(self, workload: Workload, rt) -> RunReport:
        import torch
        from sakura.adapters.ddp import DDPAdapter

        model = workload.make_model()
        train_loader = workload.make_train_loader()
        val_loader = workload.make_val_loader()
        opt = self._make_optimizer(model)

        device = "cuda" if _cuda_available() else "cpu"
        model = model.to(device)

        # Bench-harness bridges for the async services. Each service the CLI
        # factory built carries a `_bench_snapshot` dict; the harness writes
        # the current model state_dict into all of them after each epoch so
        # AsyncEval / AsyncCheckpoint can run their work on a different
        # execution context (thread / subprocess) without coordinating with
        # the training loop further. AsyncEval also reads val_loader from
        # its snapshot — pinned to CPU so eval doesn't contend with the
        # training device for compute.
        bridge_snapshots: list[dict] = []
        for svc_name in ("async_eval", "async_checkpoint"):
            svc = rt.find(svc_name)
            snap = getattr(svc, "_bench_snapshot", None) if svc is not None else None
            if snap is not None:
                bridge_snapshots.append(snap)
        async_eval_svc = rt.find("async_eval")
        async_eval_bridge = getattr(async_eval_svc, "_bench_snapshot", None) \
            if async_eval_svc is not None else None
        async_eval_bridge_active = async_eval_bridge is not None
        if async_eval_bridge is not None:
            async_eval_bridge["val_loader"] = _DeviceLoader(val_loader, "cpu")

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
                # Services may scale the loss before backward (fp16 GradScaler).
                loss = rt.scale_loss(loss)
                loss.backward()
                adapter.on_optimizer_step(opt)
                # Services may drive the step (e.g. fp16 GradScaler.step+update).
                # If none claims it, the loop steps as usual.
                if not rt.optimizer_step(opt):
                    opt.step()
                n_samples += y.size(0) if hasattr(y, "size") else len(y)
            if bridge_snapshots:
                # Snapshot state_dict once, share across all bridged services.
                sd = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                for snap in bridge_snapshots:
                    snap["state_dict"] = sd
            if async_eval_bridge_active:
                # AsyncEval handles eval; final metrics pulled from its history at end.
                metrics = {}
            else:
                model.eval()
                dev_val_loader = _DeviceLoader(val_loader, device)
                metrics = workload.eval_fn(model, dev_val_loader)
            adapter.on_epoch_end(epoch, model, opt, metrics=metrics)

        # AsyncEval drains pending evals during on_train_end; that drain time
        # is part of the wallclock the user pays for. Stop the timer AFTER it.
        adapter.on_train_end(model)
        elapsed = time.perf_counter() - t0
        if async_eval_bridge_active and async_eval_svc.history:
            metrics = {k: v for k, v in async_eval_svc.history[-1].items()
                       if k not in ("epoch", "skipped", "reason")}

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
