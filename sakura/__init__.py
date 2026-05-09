"""Sakura — SOTA training services for PyTorch DDP / Lightning / HuggingFace Trainer.

Plan 1 added the sakura-wire transport (Rust). Plan 2 the Python orchestration
surface (SakuraRuntime, Service ABC, Dispatcher). Plan 3 the seven v1 services.
Plan 4 the framework adapters (Lightning/HF/DDP) and removed v0.1.x.
Plan 5 (future) the benchmark harness + multi-rank ZeRO1 + maturin packaging.

Users on v0.1.x should pin `sakura-ml<1.0` if they're not migrating to the
new SakuraRuntime + Adapter + Service surface.
"""

__version__ = "1.0.0a0"
__build__ = "2026-05-08T00:00:00Z"

from sakura.adapters import Adapter, DDPAdapter, HFAdapter, LightningAdapter
from sakura.dispatch import Compute, Dispatcher, Future, Result
from sakura.dispatch.zakuro import ZakuroDispatcher
from sakura.events import (
    Event,
    OnEpochBegin,
    OnEpochEnd,
    OnError,
    OnOptimizerStep,
    OnSave,
    OnTrainBegin,
    OnTrainEnd,
    OnTrainStepBegin,
)
from sakura.runtime import SakuraRuntime
from sakura.service import BaseService, Service
from sakura.services import (
    ActivationCheckpoint,
    AsyncCheckpoint,
    AsyncEval,
    Compile,
    MixedPrecision,
    Telemetry,
    ZeRO1,
)
from sakura.zero import ShardedOptimizer

__all__ = [
    "ActivationCheckpoint",
    "Adapter",
    "ShardedOptimizer",
    "AsyncCheckpoint",
    "AsyncEval",
    "BaseService",
    "Compile",
    "Compute",
    "DDPAdapter",
    "Dispatcher",
    "Event",
    "Future",
    "HFAdapter",
    "LightningAdapter",
    "MixedPrecision",
    "OnEpochBegin",
    "OnEpochEnd",
    "OnError",
    "OnOptimizerStep",
    "OnSave",
    "OnTrainBegin",
    "OnTrainEnd",
    "OnTrainStepBegin",
    "Result",
    "SakuraRuntime",
    "Service",
    "Telemetry",
    "ZakuroDispatcher",
    "ZeRO1",
    "__build__",
    "__version__",
]
