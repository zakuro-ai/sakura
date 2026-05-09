"""Sakura — SOTA training services for PyTorch DDP / Lightning / HuggingFace Trainer.

Plan 1: Rust sakura-wire transport (codec + QUIC + worker subprocess).
Plan 2: Python orchestration surface (SakuraRuntime, Service ABC, Dispatcher).
Plan 3: seven v1 services (Telemetry, MixedPrecision, Compile, ZeRO1, ...).
Plan 4: framework adapters (Lightning/HF/DDP) + v0.1.x removal.
Plan 5: benchmark harness, multi-rank ZeRO1, maturin packaging, codec
        zero-copy producer path, README/migration docs.

Users on v0.1.x should pin `sakura-ml<1.0` if they're not migrating to the
new SakuraRuntime + Adapter + Service surface.
"""

__version__ = "1.0.0a1"
__build__ = "2026-05-09T00:00:00Z"

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
