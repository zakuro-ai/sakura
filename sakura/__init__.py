"""Sakura — SOTA training services for PyTorch DDP / Lightning / HuggingFace Trainer.

Plan 1 of the v1.0 redesign added the sakura-wire transport (Rust). Plan 2
adds the Python orchestration surface: SakuraRuntime, Service ABC, event
types, and the Dispatcher abstraction. Plans 3-5 add concrete services,
framework adapters, and the benchmark harness.

Existing v0.1.x submodules (sakura.lightning, sakura.huggingface,
sakura.tensorflow, sakura.ddp, sakura.ml) continue to import. They will be
removed in Plan 4 once the migration path is validated.
"""

__version__ = "1.0.0a0"
__build__ = "2026-05-08T00:00:00Z"

from sakura.dispatch import Compute, Dispatcher, Future, Result
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

__all__ = [
    "ActivationCheckpoint",
    "AsyncCheckpoint",
    "AsyncEval",
    "BaseService",
    "Compile",
    "Compute",
    "Dispatcher",
    "Event",
    "Future",
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
    "ZeRO1",
    "__build__",
    "__version__",
]
