"""L'Atelier's training engine: one light, replayable YAML in, a trained and
binarised model out, on any of the phase-1 segments.

``python -m sakura.atelier run SPEC.yaml --out OUT`` is the whole interface a
runner needs. Nothing here imports a training stack; each backend imports its
own, inside its own venv. Design and contracts: ``docs/atelier/``.
"""

from sakura.atelier.spec import AtelierSpec, dump_spec, load_spec, spec_sha256
from sakura.atelier.types import (
    Artifact,
    Backend,
    Checkpoint,
    MaterialisedData,
    MetricsSink,
    ResolvedSpec,
    RunDir,
    RuntimeEval,
    TrainResult,
    read_metrics,
)

__all__ = [
    "Artifact", "AtelierSpec", "Backend", "Checkpoint", "MaterialisedData",
    "MetricsSink", "ResolvedSpec", "RunDir", "RuntimeEval", "TrainResult",
    "dump_spec", "load_spec", "read_metrics", "spec_sha256",
]
