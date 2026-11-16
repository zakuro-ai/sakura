"""`sakura.atelier.ort`: every onnxruntime session the pipeline opens goes
through one helper that sets the thread counts explicitly.

Why: with the default `SessionOptions` onnxruntime sizes its intra-op pool to
the host's core count and pins each thread to a core. Inside a cgroup cpuset
(the Atelier runners) cores outside the set exist but cannot be pinned, so
every session start prints ~30 `pthread_setaffinity_np failed ... error code:
22` lines at [E] severity, once in the engine's runtime_eval and once more in
the notebook check. The message itself says the cure: "Specify the number of
threads explicitly so the affinity is not set."
"""

from __future__ import annotations

import sys
import types

import pytest

from sakura.atelier import ort as ort_helper


class _FakeSessionOptions:
    def __init__(self) -> None:
        self.intra_op_num_threads = 0
        self.inter_op_num_threads = 0


class _FakeSession:
    def __init__(self, path, sess_options=None, providers=None):
        self.path = path
        self.sess_options = sess_options
        self.providers = providers


@pytest.fixture
def fake_ort(monkeypatch):
    mod = types.ModuleType("onnxruntime")
    mod.SessionOptions = _FakeSessionOptions
    mod.InferenceSession = _FakeSession
    monkeypatch.setitem(sys.modules, "onnxruntime", mod)
    return mod


def test_session_options_always_set_both_thread_counts(fake_ort, monkeypatch):
    monkeypatch.delenv("SAKURA_ATELIER_ORT_THREADS", raising=False)
    so = ort_helper.session_options()
    assert so.intra_op_num_threads >= 1, "0 means 'let ORT decide', which re-enables affinity pinning"
    assert so.intra_op_num_threads == ort_helper.thread_count()
    assert so.inter_op_num_threads == 1


def test_thread_count_follows_the_process_cpu_set(monkeypatch):
    monkeypatch.delenv("SAKURA_ATELIER_ORT_THREADS", raising=False)
    monkeypatch.setattr(ort_helper, "_allowed_cpus", lambda: 6)
    monkeypatch.setattr(ort_helper, "_cgroup_cpu_quota", lambda: None)
    assert ort_helper.thread_count() == 6


def test_thread_count_is_capped_by_the_cgroup_cpu_quota(monkeypatch):
    monkeypatch.delenv("SAKURA_ATELIER_ORT_THREADS", raising=False)
    monkeypatch.setattr(ort_helper, "_allowed_cpus", lambda: 32)
    monkeypatch.setattr(ort_helper, "_cgroup_cpu_quota", lambda: 3.5)
    assert ort_helper.thread_count() == 4  # ceil(3.5), never 0


def test_thread_count_env_override_wins(monkeypatch):
    monkeypatch.setenv("SAKURA_ATELIER_ORT_THREADS", "2")
    monkeypatch.setattr(ort_helper, "_allowed_cpus", lambda: 32)
    assert ort_helper.thread_count() == 2
    monkeypatch.setenv("SAKURA_ATELIER_ORT_THREADS", "0")  # nonsense -> ignored, never 0
    assert ort_helper.thread_count() >= 1


def test_cgroup_cpu_quota_parses_cgroup_v2_cpu_max(tmp_path, monkeypatch):
    f = tmp_path / "cpu.max"
    monkeypatch.setattr(ort_helper, "_CPU_MAX", f)
    f.write_text("max 100000\n")
    assert ort_helper._cgroup_cpu_quota() is None
    f.write_text("250000 100000\n")
    assert ort_helper._cgroup_cpu_quota() == 2.5
    f.unlink()
    assert ort_helper._cgroup_cpu_quota() is None


def test_inference_session_passes_options_and_providers(fake_ort, tmp_path):
    sess = ort_helper.inference_session(tmp_path / "model.onnx", providers=("CPUExecutionProvider",))
    assert sess.path == str(tmp_path / "model.onnx")
    assert sess.providers == ["CPUExecutionProvider"]
    assert sess.sess_options.intra_op_num_threads >= 1
    assert sess.sess_options.inter_op_num_threads == 1


def test_no_backend_opens_a_session_without_the_helper():
    """The point of the helper is that nothing bypasses it (the serving path
    included): a grep guard so the next call site cannot regress silently."""
    import pathlib

    import sakura.atelier as pkg

    root = pathlib.Path(pkg.__file__).parent
    offenders = []
    for py in root.rglob("*.py"):
        if py.name == "ort.py":
            continue
        text = py.read_text(encoding="utf-8")
        if "InferenceSession(" in text:
            offenders.append(str(py.relative_to(root)))
    assert offenders == []


def test_real_onnxruntime_honours_the_options(tmp_path):
    """With the real libraries: a session built by the helper reports the
    thread counts we asked for and still runs."""
    onnx = pytest.importorskip("onnx")
    pytest.importorskip("onnxruntime")
    import numpy as np
    from onnx import TensorProto, helper

    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 2])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 2])
    graph = helper.make_graph([helper.make_node("Identity", ["x"], ["y"])], "id", [x], [y])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8
    path = tmp_path / "id.onnx"
    onnx.save(model, str(path))

    sess = ort_helper.inference_session(path, providers=["CPUExecutionProvider"])
    out = sess.run(["y"], {"x": np.ones((1, 2), dtype=np.float32)})[0]
    assert out.tolist() == [[1.0, 1.0]]
    assert sess.get_session_options().intra_op_num_threads == ort_helper.thread_count()
    assert sess.get_session_options().inter_op_num_threads == 1
