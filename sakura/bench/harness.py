"""Benchmark harness — Workload, RunReport, BaselineRunner, SakuraRunner.

Plan 5 Task 4 ships the dataclass types; T5 fills in the runners; T6+ add
concrete workloads.
"""
from __future__ import annotations

import json
import os
import platform
import subprocess
import tempfile
import threading
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Iterator, Literal, Optional, cast


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


def _cuda_reset_peak() -> None:
    """Reset the CUDA allocator's peak-memory counter so a subsequent
    ``torch.cuda.max_memory_allocated()`` reflects only the current run.

    The peak counter is process-cumulative, so when a baseline and a sakura
    run share one process the second run would otherwise inherit the first
    run's peak. No-op when CUDA is unavailable (the bench runs on CPU in CI).
    """
    if _cuda_available():
        import torch
        torch.cuda.reset_peak_memory_stats()


def _cuda_sync() -> None:
    """Block until all queued CUDA kernels complete.

    CUDA kernels dispatch asynchronously, so a ``time.perf_counter()`` reading
    taken right after the Python training loop would stop the clock before the
    GPU finishes — undercounting wall-clock and inflating tokens/sec. Call
    this immediately before every ``elapsed=`` computation. No-op without CUDA.
    """
    if _cuda_available():
        import torch
        torch.cuda.synchronize()


def _parse_smi_line(line: str) -> Optional[tuple[float, float]]:
    """Parse one line of `nvidia-smi --query-gpu=utilization.gpu,memory.used
    --format=csv,noheader,nounits` into (util_pct, mem_used_mb).

    Example: '37, 1234' -> (37.0, 1234.0). Returns None for blank/garbage
    lines (the driver can interleave '[N/A]' or warnings on stdout)."""
    parts = [p.strip() for p in line.split(",")]
    if len(parts) < 2:
        return None
    try:
        return float(parts[0]), float(parts[1])
    except ValueError:
        return None


class _GpuUtilSampler:
    """Sample GPU utilization (%) + device-used memory (MB) on a background
    thread for the duration of a `with` block.

    Backend preference:
      1. pynvml (the `nvidia-ml-py` package) in-process — accurate, cheap.
      2. a long-lived `nvidia-smi ... -lms <ms>` subprocess whose stdout is
         drained by a reader thread (zero extra dep, coarser).
    No-op (all zeros) when CUDA is unavailable; sampling never raises.

    Results are valid after the `with` block exits:
      .mean_pct  .max_pct  .n_samples  .peak_mem_mb
    """

    def __init__(self, device: int = 0, interval_s: float = 0.1):
        self.device = device
        self.interval_s = interval_s
        self._utils: list[float] = []
        self._mems: list[float] = []
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._proc: Optional[subprocess.Popen[str]] = None
        self._backend: Optional[str] = None
        self._nvml: Any = None
        self._handle: Any = None

    @property
    def mean_pct(self) -> float:
        return sum(self._utils) / len(self._utils) if self._utils else 0.0

    @property
    def max_pct(self) -> float:
        return max(self._utils) if self._utils else 0.0

    @property
    def n_samples(self) -> int:
        return len(self._utils)

    @property
    def peak_mem_mb(self) -> float:
        return max(self._mems) if self._mems else 0.0

    def __enter__(self) -> "_GpuUtilSampler":
        if not _cuda_available():
            return self
        if self._start_pynvml():
            self._backend = "pynvml"
            self._thread = threading.Thread(target=self._poll_pynvml, daemon=True)
            self._thread.start()
        elif self._start_smi():
            self._backend = "smi"
            self._thread = threading.Thread(target=self._read_smi, daemon=True)
            self._thread.start()
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self._stop.set()
        # Terminate the subprocess FIRST: the smi reader thread blocks in
        # `for line in proc.stdout:` and _stop alone cannot unblock it — only
        # closing stdout (via terminate) does. Joining before terminate would
        # always burn the full 2s timeout on the smi fallback path.
        if self._proc is not None:
            try:
                self._proc.terminate()
                self._proc.wait(timeout=2.0)
            except Exception:
                try:
                    self._proc.kill()
                except Exception:
                    pass
                try:
                    self._proc.wait(timeout=2.0)
                except Exception:
                    pass
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        if self._backend == "pynvml" and self._nvml is not None:
            try:
                self._nvml.nvmlShutdown()
            except Exception:
                pass

    def _start_pynvml(self) -> bool:
        try:
            import pynvml
        except Exception:
            return False
        try:
            pynvml.nvmlInit()
            self._handle = pynvml.nvmlDeviceGetHandleByIndex(self.device)
            self._nvml = pynvml
            return True
        except Exception:
            return False

    def _poll_pynvml(self) -> None:
        nvml = self._nvml
        handle = self._handle
        while not self._stop.is_set():
            try:
                util = nvml.nvmlDeviceGetUtilizationRates(handle)
                mem = nvml.nvmlDeviceGetMemoryInfo(handle)
                self._utils.append(float(util.gpu))
                self._mems.append(float(mem.used) / (1024 * 1024))
            except Exception:
                pass
            self._stop.wait(self.interval_s)

    def _start_smi(self) -> bool:
        import shutil
        if shutil.which("nvidia-smi") is None:
            return False
        interval_ms = max(int(self.interval_s * 1000), 1)
        try:
            self._proc = subprocess.Popen(
                [
                    "nvidia-smi",
                    f"--id={self.device}",
                    "--query-gpu=utilization.gpu,memory.used",
                    "--format=csv,noheader,nounits",
                    "-lms", str(interval_ms),
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
            )
            return True
        except Exception:
            self._proc = None
            return False

    def _read_smi(self) -> None:
        proc = self._proc
        if proc is None or proc.stdout is None:
            return
        for line in proc.stdout:
            if self._stop.is_set():
                break
            parsed = _parse_smi_line(line)
            if parsed is not None:
                util, mem = parsed
                self._utils.append(util)
                self._mems.append(mem)


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
    eval_fn: Callable[[Any, Any], dict[str, Any]]
    # Optional custom training step: `loss_fn(model, batch, device) -> (loss, n)`.
    # Lets a workload express a non-classification objective (e.g. CTC for ASR)
    # whose batches aren't (x, y) tuples. None → default cross-entropy path.
    loss_fn: Optional[Callable[[Any, Any, str], Any]] = None
    epochs: int = 1
    metric_target: Optional[tuple[str, float]] = None  # ("val_acc", 0.85)
    tokens_per_sample: Optional[int] = None  # for GPT-2 = seq_len; enables tokens/sec
    block_types: tuple[type, ...] = ()  # transformer block classes for activation_checkpoint


@dataclass
class RunReport:
    """The output of a single Workload + Runner execution."""
    workload: str
    framework: Literal["pytorch-ddp", "lightning", "hf-trainer", "tf"]
    sakura_services: Optional[list[str]] = None
    elapsed_secs: float = 0.0
    samples_per_sec: float = 0.0
    peak_gpu_mem_mb: float = 0.0
    final_metrics: dict[str, Any] = field(default_factory=dict)
    per_stage_secs: dict[str, float] = field(default_factory=dict)
    git_sha: str = ""
    hardware: dict[str, Any] = field(default_factory=dict)
    # --- P0 measurement instrumentation (all defaulted: old JSON still loads) ---
    gpu_util_mean_pct: float = 0.0
    gpu_util_max_pct: float = 0.0
    gpu_util_samples: int = 0
    gpu_mem_used_peak_mb: float = 0.0  # NVML/nvidia-smi device-used peak (complements peak_gpu_mem_mb allocator peak)
    tokens_per_sec: float = 0.0
    total_tokens: int = 0
    reached_target: Optional[bool] = None
    epochs_to_target: Optional[int] = None

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)

    @classmethod
    def from_json(cls, s: str) -> "RunReport":
        d = json.loads(s)
        return cls(**d)


def detect_hardware() -> dict[str, Any]:
    """Capture hardware info for the report (CPU, GPU, RAM, OS, python)."""
    import torch

    info: dict[str, Any] = {
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

    def __init__(self, loader: Any, device: str) -> None:
        self._loader = loader
        self._device = device

    def __iter__(self) -> Iterator[Any]:
        for batch in self._loader:
            yield _move_batch(batch, self._device)

    def __len__(self) -> int:
        return len(self._loader)


def _move_batch(batch: Any, device: str) -> Any:
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

    def __init__(
        self,
        framework: Literal["pytorch-ddp", "lightning", "hf-trainer"] = "pytorch-ddp",
        *,
        mode: Literal["fixed", "time-to-target"] = "fixed",
        max_epochs: Optional[int] = None,
        warmup_steps: Optional[int] = None,
    ):
        self.framework = framework
        self.mode = mode
        self.max_epochs = max_epochs
        if warmup_steps is None:
            env_val = os.environ.get("SAKURA_BENCH_WARMUP_STEPS")
            warmup_steps = int(env_val) if env_val is not None else 2
        self.warmup_steps = warmup_steps

    def run(self, workload: Workload) -> RunReport:
        self._validate_mode()
        if self.framework == "pytorch-ddp":
            return self._run_raw_pytorch(workload)
        if self.framework == "lightning":
            return self._run_lightning(workload)
        if self.framework == "hf-trainer":
            return self._run_hf(workload)
        raise NotImplementedError(f"baseline framework {self.framework!r} not yet wired")

    def _run_raw_pytorch(self, workload: Workload) -> RunReport:

        model = workload.make_model()
        train_loader = workload.make_train_loader()
        val_loader = workload.make_val_loader()
        opt = self._make_optimizer(model)

        device = "cuda" if _cuda_available() else "cpu"
        # Reset the allocator peak so peak_gpu_mem_mb reflects only THIS run
        # (the counter is process-cumulative). Reset before moving the model
        # to the device so the weight-residency allocation is part of the peak.
        _cuda_reset_peak()
        model = model.to(device)
        # Wrap val_loader so eval_fn receives device-placed batches even when
        # the workload builds plain lists of CPU tensors.
        dev_val_loader = _DeviceLoader(val_loader, device)

        # Warmup: run warmup_steps training steps before the timed region so
        # that torch.compile's autotuning transients are excluded from peak-
        # memory and wall-clock measurements.  Steps run on the first batch
        # and do NOT count toward n_samples / total_tokens.  A fresh
        # _cuda_sync() + _cuda_reset_peak() after warmup flushes autotune
        # allocations so that the timed region's peak is clean.
        if self.warmup_steps > 0:
            try:
                _first_batch = next(iter(train_loader))
            except StopIteration:
                _first_batch = None
            if _first_batch is not None:
                for _ in range(self.warmup_steps):
                    opt.zero_grad()
                    _wloss, _ = self._forward_loss(workload, model, _first_batch, device)
                    _wloss.backward()
                    opt.step()
                _cuda_sync()
                _cuda_reset_peak()

        with _GpuUtilSampler() as gpu:
            t0 = time.perf_counter()
            n_samples = 0
            final_metrics: dict[str, Any] = {}
            reached_target: Optional[bool] = None
            epochs_to_target: Optional[int] = None
            for epoch in range(self._epoch_budget(workload)):
                model.train()
                for batch in train_loader:
                    opt.zero_grad()
                    loss, batch_n = self._forward_loss(workload, model, batch, device)
                    loss.backward()
                    opt.step()
                    n_samples += batch_n
                # Per-epoch synchronous eval — matches real-world training loops
                # (early stopping, monitoring) and gives the sakura+AsyncEval
                # comparison something legitimate to overlap against. The single
                # end-of-training eval would be unfair to sakura.
                model.eval()
                final_metrics = workload.eval_fn(model, dev_val_loader)
                if self._should_stop_for_target(workload, final_metrics):
                    reached_target = True
                    epochs_to_target = epoch + 1
                    break
            if self.mode == "time-to-target" and reached_target is None:
                reached_target = False
            # CUDA kernels are async; block on them before stopping the clock so we
            # time the GPU work, not just the Python dispatch loop.
            _cuda_sync()
            elapsed = time.perf_counter() - t0

        total_tokens = n_samples * (workload.tokens_per_sample or 0)
        return RunReport(
            workload=workload.name,
            framework=self.framework,
            elapsed_secs=elapsed,
            samples_per_sec=n_samples / max(elapsed, 1e-9),
            peak_gpu_mem_mb=self._peak_gpu_mem_mb(),
            tokens_per_sec=total_tokens / max(elapsed, 1e-9),
            total_tokens=total_tokens,
            final_metrics=final_metrics,
            reached_target=reached_target,
            epochs_to_target=epochs_to_target,
            git_sha=detect_git_sha(),
            hardware=detect_hardware(),
            gpu_util_mean_pct=gpu.mean_pct,
            gpu_util_max_pct=gpu.max_pct,
            gpu_util_samples=gpu.n_samples,
            gpu_mem_used_peak_mb=gpu.peak_mem_mb,
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

    def _run_hf_impl(self, workload: Workload, callbacks: Optional[list[Any]] = None) -> RunReport:
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

    def _run_lightning_impl(self, workload: Workload, adapter: Optional[Any] = None) -> RunReport:
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
            def __init__(self, m: Any) -> None:
                super().__init__()
                self.model = m

            def forward(self, x: Any) -> Any:
                return self.model(x)

            def training_step(self, batch: Any, batch_idx: int) -> Any:
                x, y = batch
                logits = self.model(x)
                return torch.nn.functional.cross_entropy(logits, y)

            def configure_optimizers(self) -> Any:
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

    def _validate_mode(self) -> None:
        """Reject unsupported (mode, framework) combinations.

        Time-to-target early stopping needs a synchronous per-epoch eval and
        a plain Python training loop; only the pytorch-ddp runner has that.
        Lightning and HF Trainer own their own loops and always run fixed
        epochs.
        """
        if self.mode not in ("fixed", "time-to-target"):
            raise ValueError(
                f"unknown mode {self.mode!r}; expected 'fixed' or 'time-to-target'"
            )
        if self.mode == "time-to-target" and self.framework != "pytorch-ddp":
            raise ValueError(
                "time-to-target mode is only supported for framework='pytorch-ddp'; "
                "lightning and hf-trainer always run fixed epochs."
            )

    def _epoch_budget(self, workload: Workload) -> int:
        """How many epochs the training loop may run.

        Fixed mode runs exactly workload.epochs. Time-to-target runs until the
        target is met or max_epochs epochs elapse (falling back to
        workload.epochs as the cap when max_epochs is unset).
        """
        if self.mode == "time-to-target":
            return self.max_epochs if self.max_epochs is not None else workload.epochs
        return workload.epochs

    def _should_stop_for_target(self, workload: Workload, metrics: dict[str, Any]) -> bool:
        """True when time-to-target is active and the latest eval metrics meet
        workload.metric_target. No-op (False) in fixed mode, when no target is
        set, or when the target metric is absent from `metrics`."""
        if self.mode != "time-to-target" or workload.metric_target is None:
            return False
        name, value = workload.metric_target
        observed = metrics.get(name)
        if observed is None:
            return False
        return self._target_reached(name, float(observed), float(value))

    @staticmethod
    def _target_reached(metric_name: str, observed: float, target: float) -> bool:
        """Direction-aware target check, inferred from the metric name.

        Higher-is-better metrics (accuracy/f1/auc/bleu/score) are reached when
        observed >= target. Everything else — loss/perplexity/error/nll and any
        unrecognized name — is treated as lower-is-better and reached when
        observed <= target (the safe default for LM workloads whose headline
        metric is validation loss).
        """
        name = metric_name.lower()
        higher_is_better = any(
            k in name for k in ("acc", "f1", "auc", "bleu", "score")
        )
        lower_is_better = any(
            k in name for k in ("loss", "perplex", "ppl", "error", "nll")
        )
        if higher_is_better and not lower_is_better:
            return observed >= target
        return observed <= target

    @staticmethod
    def _make_optimizer(model: Any) -> Any:
        import torch
        return torch.optim.SGD(model.parameters(), lr=0.01)

    @staticmethod
    def _batch_to_device(batch: Any, device: str) -> Any:
        if isinstance(batch, (tuple, list)) and len(batch) == 2:
            x, y = batch
            x = _move_batch(x, device)
            y = y.to(device) if hasattr(y, "to") else y
            return x, y
        return batch

    @staticmethod
    def _compute_loss(logits: Any, y: Any) -> Any:
        import torch
        return torch.nn.functional.cross_entropy(logits, y)

    def _forward_loss(
        self, workload: Workload, model: Any, batch: Any, device: str
    ) -> tuple[Any, int]:
        """Return ``(loss, batch_n_samples)`` for one training batch.

        Default path is supervised classification: ``model(x)`` ->
        ``cross_entropy(logits, y)`` with ``n = len(y)``. A workload may set
        ``loss_fn(model, batch, device) -> (loss, n)`` to express a different
        objective (e.g. CTC for ASR) whose batches aren't ``(x, y)`` tuples;
        ``n`` is the per-batch sample count used for samples/sec accounting.
        """
        if workload.loss_fn is not None:
            return cast("tuple[Any, int]", workload.loss_fn(model, batch, device))
        x, y = self._batch_to_device(batch, device)
        logits = model(x)
        loss = self._compute_loss(logits, y)
        n = y.size(0) if hasattr(y, "size") else len(y)
        return loss, n

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
        services: Optional[list[Any]] = None,
        compute: Optional[Any] = None,
        *,
        mode: Literal["fixed", "time-to-target"] = "fixed",
        max_epochs: Optional[int] = None,
        warmup_steps: Optional[int] = None,
    ):
        super().__init__(framework=framework, mode=mode, max_epochs=max_epochs,
                         warmup_steps=warmup_steps)
        self._services = services or []
        self._compute = compute

    def run(self, workload: Workload) -> RunReport:
        from sakura.runtime import SakuraRuntime

        self._validate_mode()
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

    def _run_lightning_with_adapter(self, workload: Workload, rt: Any) -> RunReport:
        """Sakura + Lightning: install LightningAdapter on the runtime + Trainer callback.

        Reuses BaselineRunner._run_lightning_impl with adapter=LightningAdapter(rt).
        """
        from sakura.adapters.lightning import LightningAdapter
        adapter = LightningAdapter(rt, rank=0, world_size=1)
        return self._run_lightning_impl(workload, adapter=adapter)

    def _run_hf_with_adapter(self, workload: Workload, rt: Any) -> RunReport:
        """Sakura + HF Trainer: install HFAdapter as a Trainer callback.

        Reuses BaselineRunner._run_hf_impl with callbacks=[HFAdapter(rt)] so
        runtime services observe Trainer's lifecycle hooks.
        """
        from sakura.adapters.huggingface import HFAdapter
        adapter = HFAdapter(rt, rank=0, world_size=1)
        return self._run_hf_impl(workload, callbacks=[adapter])

    def _run_raw_pytorch_with_adapter(self, workload: Workload, rt: Any) -> RunReport:
        from sakura.adapters.ddp import DDPAdapter

        model = workload.make_model()
        train_loader = workload.make_train_loader()
        val_loader = workload.make_val_loader()
        opt = self._make_optimizer(model)

        device = "cuda" if _cuda_available() else "cpu"
        # Reset allocator peak so this run's peak_gpu_mem_mb isn't contaminated
        # by a baseline run earlier in the same process.
        _cuda_reset_peak()
        model = model.to(device)

        # Bench-harness bridges for the async services. Each service the CLI
        # factory built carries a `_bench_snapshot` dict; the harness writes
        # the current model state_dict into all of them after each epoch so
        # AsyncEval / AsyncCheckpoint can run their work on a different
        # execution context (thread / subprocess) without coordinating with
        # the training loop further. AsyncEval also reads val_loader from
        # its snapshot — pinned to CPU so eval doesn't contend with the
        # training device for compute.
        bridge_snapshots: list[dict[str, Any]] = []
        for svc_name in ("async_eval", "async_checkpoint"):
            svc = rt.find(svc_name)
            snap = getattr(svc, "_bench_snapshot", None) if svc is not None else None
            if snap is not None:
                bridge_snapshots.append(snap)
        async_eval_svc = rt.find("async_eval")
        async_eval_bridge: Any = getattr(async_eval_svc, "_bench_snapshot", None) \
            if async_eval_svc is not None else None
        async_eval_bridge_active = async_eval_bridge is not None
        if self.mode == "time-to-target" and async_eval_bridge_active:
            raise ValueError(
                "time-to-target mode is incompatible with the async_eval service: "
                "per-epoch metrics are produced asynchronously, so the early-stop "
                "check has no synchronous metric to test. Run time-to-target "
                "without async_eval (synchronous per-epoch eval)."
            )
        if async_eval_bridge_active:
            async_eval_bridge["val_loader"] = _DeviceLoader(val_loader, "cpu")

        adapter = DDPAdapter(rt, rank=0, world_size=1)
        adapter.on_train_begin(model, opt, train_loader, val_loader=val_loader)

        # Warmup: run warmup_steps training steps before the timed region so
        # that torch.compile's autotuning transients are excluded from peak-
        # memory and wall-clock measurements.  No adapter events are emitted
        # during warmup; async services are unaffected.  Steps do NOT count
        # toward n_samples / total_tokens.
        if self.warmup_steps > 0:
            try:
                _first_batch = next(iter(train_loader))
            except StopIteration:
                _first_batch = None
            if _first_batch is not None:
                for _ in range(self.warmup_steps):
                    opt.zero_grad()
                    _wloss, _ = self._forward_loss(workload, model, _first_batch, device)
                    _wloss = rt.scale_loss(_wloss)
                    _wloss.backward()
                    if not rt.optimizer_step(opt):
                        opt.step()
                _cuda_sync()
                _cuda_reset_peak()

        with _GpuUtilSampler() as gpu:
            t0 = time.perf_counter()
            n_samples = 0
            metrics: dict[str, Any] = {}
            reached_target: Optional[bool] = None
            epochs_to_target: Optional[int] = None
            for epoch in range(self._epoch_budget(workload)):
                adapter.on_epoch_begin(epoch)
                model.train()
                for step, batch in enumerate(train_loader):
                    adapter.on_train_step_begin(model, batch, step)
                    opt.zero_grad()
                    loss, batch_n = self._forward_loss(workload, model, batch, device)
                    # Services may scale the loss before backward (fp16 GradScaler).
                    loss = rt.scale_loss(loss)
                    loss.backward()
                    adapter.on_optimizer_step(opt)
                    # Services may drive the step (e.g. fp16 GradScaler.step+update).
                    # If none claims it, the loop steps as usual.
                    if not rt.optimizer_step(opt):
                        opt.step()
                    n_samples += batch_n
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
                if self._should_stop_for_target(workload, metrics):
                    reached_target = True
                    epochs_to_target = epoch + 1
                    break
            if self.mode == "time-to-target" and reached_target is None:
                reached_target = False

            # AsyncEval drains pending evals during on_train_end; that drain time
            # is part of the wallclock the user pays for. Stop the timer AFTER it.
            adapter.on_train_end(model)
            # CUDA kernels are async; block on them before stopping the clock.
            _cuda_sync()
            elapsed = time.perf_counter() - t0
        if async_eval_bridge_active and async_eval_svc.history:
            metrics = {k: v for k, v in async_eval_svc.history[-1].items()
                       if k not in ("epoch", "skipped", "reason")}

        total_tokens = n_samples * (workload.tokens_per_sample or 0)
        return RunReport(
            workload=workload.name,
            framework=self.framework,
            elapsed_secs=elapsed,
            samples_per_sec=n_samples / max(elapsed, 1e-9),
            peak_gpu_mem_mb=self._peak_gpu_mem_mb(),
            tokens_per_sec=total_tokens / max(elapsed, 1e-9),
            total_tokens=total_tokens,
            final_metrics=metrics,
            reached_target=reached_target,
            epochs_to_target=epochs_to_target,
            git_sha=detect_git_sha(),
            hardware=detect_hardware(),
            gpu_util_mean_pct=gpu.mean_pct,
            gpu_util_max_pct=gpu.max_pct,
            gpu_util_samples=gpu.n_samples,
            gpu_mem_used_peak_mb=gpu.peak_mem_mb,
        )


__all__ = ["Workload", "RunReport", "detect_hardware", "detect_git_sha", "BaselineRunner", "SakuraRunner"]
