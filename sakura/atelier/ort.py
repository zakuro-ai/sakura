"""The one way the pipeline opens an onnxruntime session.

Every `InferenceSession` in Atelier (the backends' `runtime_eval`, the serving
`predict` paths, and through them the notebook's Evaluation cell) is built
here with explicit thread counts. The reason is an onnxruntime behaviour, not
a preference: with the default `SessionOptions` ORT sizes its intra-op pool
to the *host's* core count and pins each pool thread to one core. On a runner
whose cgroup cpuset allows only a subset of those cores every pin outside the
set fails and ORT prints, at [E] severity, one line per thread:

    [E:onnxruntime:Default, env.cc:298 ThreadMain] pthread_setaffinity_np
    failed for thread: 222, index: 8, mask: {20, }, error code: 22 error msg:
    Invalid argument. Specify the number of threads explicitly so the
    affinity is not set.

That is ~30 "errors" per session on a job that is fine (staging job
cbd7b717, 56 of 77 log lines), twice per run (engine eval + notebook check).
As the message says, an explicit `intra_op_num_threads` turns the pinning
off. The count is what the process may actually use: the scheduler's allowed
CPU set, further capped by a cgroup v2 CPU quota when there is one, with
`SAKURA_ATELIER_ORT_THREADS` as an operator override.
"""

from __future__ import annotations

import math
import os
from collections.abc import Iterable
from pathlib import Path
from typing import Any

ENV_THREADS = "SAKURA_ATELIER_ORT_THREADS"
_CPU_MAX = Path("/sys/fs/cgroup/cpu.max")


def _allowed_cpus() -> int:
    """CPUs this process may be scheduled on (the cpuset), not the host's."""
    getaffinity = getattr(os, "sched_getaffinity", None)  # Linux only
    if getaffinity is not None:
        try:
            return len(getaffinity(0))
        except OSError:
            pass
    return os.cpu_count() or 1


def _cgroup_cpu_quota() -> float | None:
    """cgroup v2 ``cpu.max`` as a number of CPUs, or None when unlimited / unreadable."""
    try:
        quota, period = _CPU_MAX.read_text(encoding="utf-8").split()[:2]
    except (OSError, ValueError):
        return None
    if quota == "max":
        return None
    try:
        q, p = int(quota), int(period)
    except ValueError:
        return None
    return q / p if q > 0 and p > 0 else None


def allowed_threads() -> int:
    """CPUs this process may actually use: the cpuset, capped by the cgroup CPU quota,
    never 0. The number any CPU-bound tool should be told to use (ORT intra-op threads,
    ``whisper-cli -t``), as opposed to ``os.cpu_count()``, which is the host's."""
    n = _allowed_cpus()
    quota = _cgroup_cpu_quota()
    if quota is not None:
        n = min(n, math.ceil(quota))
    return max(1, n)


def thread_count() -> int:
    """Intra-op threads for a session: ``SAKURA_ATELIER_ORT_THREADS`` if set to a
    positive integer, else the allowed CPU set capped by the cgroup CPU quota.
    Never 0 -- 0 hands the decision back to ORT and re-enables the pinning."""
    raw = os.environ.get(ENV_THREADS, "").strip()
    if raw:
        try:
            n = int(raw)
        except ValueError:
            n = 0
        if n > 0:
            return n
    return allowed_threads()


def session_options(ort: Any = None) -> Any:
    """`onnxruntime.SessionOptions` with both pool sizes set explicitly.
    ``ort`` is the onnxruntime module (imported here when not given)."""
    if ort is None:
        import onnxruntime  # optional dependency, imported on use

        ort = onnxruntime
    so = ort.SessionOptions()
    so.intra_op_num_threads = thread_count()
    so.inter_op_num_threads = 1  # sequential execution mode never uses more
    return so


def inference_session(path: str | os.PathLike[str], providers: Iterable[str] = ("CPUExecutionProvider",)) -> Any:
    """`onnxruntime.InferenceSession(path)` built with `session_options()`."""
    import onnxruntime as ort

    return ort.InferenceSession(str(path), sess_options=session_options(ort), providers=list(providers))


#: The operators whose constant weights ``quantize_int8`` turns into int8. A transformer's weights sit in
#: its linear layers, which export as ``MatMul``; convolutions are left in float because onnxruntime's CPU
#: provider does not run every ``ConvInteger`` the quantiser can emit.
INT8_OP_TYPES = ("MatMul",)


def quantize_int8(src: str | os.PathLike[str], dst: str | os.PathLike[str],
                  op_types: Iterable[str] = INT8_OP_TYPES) -> dict[str, Any]:
    """Write ``dst``: the float ONNX graph ``src`` with the weights of its ``op_types`` nodes quantised to
    int8 by ``onnxruntime.quantization.quantize_dynamic`` (dynamic quantisation: no calibration data, the
    activations are quantised at run time). Returns what the report says about the file:
    ``{quantization, quantized_tensors, float_bytes, size_ratio}``. Raises ``ValueError`` when the graph has
    no weight to quantise, because a file named int8 that holds only float weights would mislead."""
    import onnx
    from onnxruntime.quantization import QuantType, quantize_dynamic

    src, dst = Path(src), Path(dst)
    quantize_dynamic(str(src), str(dst), weight_type=QuantType.QInt8, op_types_to_quantize=list(op_types))
    int_types = (onnx.TensorProto.INT8, onnx.TensorProto.UINT8)
    quantized = sum(1 for init in onnx.load(str(dst), load_external_data=False).graph.initializer
                    if init.data_type in int_types and len(init.dims) >= 2)
    if not quantized:
        dst.unlink(missing_ok=True)
        raise ValueError(f"{src.name} has no {'/'.join(op_types)} weight to quantise to int8")
    float_bytes, int8_bytes = src.stat().st_size, dst.stat().st_size
    return {"quantization": "dynamic_int8", "quantized_tensors": quantized, "float_bytes": float_bytes,
            "size_ratio": round(int8_bytes / float_bytes, 4) if float_bytes else None}
