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

#: Optional engine capabilities a runner can probe without running anything (CONTRACTS §2.1):
#: ``wrap_up`` = every training backend honours the ``OUT/wrap-up-requested`` signal; ``limits`` = the
#: spec's ``limits`` block is enforced; ``synthetic_data`` = ``synthetic://`` data sources (SAK-0019).
FEATURES = frozenset({"wrap_up", "limits", "synthetic_data"})
#: The MVP capability matrix (SAK-0019, CONTRACTS §1.5, §1.6, §2.3): ``init_random`` = ``init: random``,
#: ``distillation`` = ``teacher``, ``dpo`` = ``objective: dpo``, ``onnx_int8`` = that export format,
#: ``curation`` = the spec's ``curation`` block.
FEATURES |= {"init_random", "distillation", "dpo", "onnx_int8", "curation"}
#: ``lr_schedule`` = the peft text-generation schedule, clipping, evaluation cadence, early stop and best
#: checkpoint keys; ``text_metrics`` = its metrics, generation length, baseline and runtime scoring
#: (CONTRACTS §2.2, 2026-11-15).
FEATURES |= {"lr_schedule", "text_metrics"}

__all__ = [
    "Artifact", "AtelierSpec", "Backend", "Checkpoint", "FEATURES", "MaterialisedData",
    "MetricsSink", "ResolvedSpec", "RunDir", "RuntimeEval", "TrainResult",
    "dump_spec", "load_spec", "read_metrics", "spec_sha256",
]
