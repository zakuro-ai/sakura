"""The ludwig backend's TorchScript export traces the same wrapped forward the
ONNX export uses. Ludwig 0.17 has no `LudwigModel.to_torchscript`, which the
export once called -- every job asking for torchscript failed after training
(hub job f591f66b, 2026-10-12). Needs torch only, not Ludwig: the trained ECD
module is stood in for by a small nn.Module with Ludwig's two entry points
(`get_model_inputs()` and a dict-in/dict-out forward)."""
from __future__ import annotations

import types

import pytest

torch = pytest.importorskip("torch")

from sakura.atelier.backends import ludwig as backend  # noqa: E402


class _Ecd(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.lin = torch.nn.Linear(4, 3)

    def get_model_inputs(self):
        return {"img": torch.ones(2, 4)}

    def forward(self, inputs):
        logits = self.lin(inputs["img"])
        return {"label::logits": logits, "label::probabilities": logits.softmax(-1),
                "other::logits": logits[:, :1]}


class _Model:
    def __init__(self) -> None:
        self.model = _Ecd().eval()


def test_torchscript_export_is_the_traced_wrapper(tmp_path):
    model = _Model()
    path = tmp_path / "model.torchscript.pt"

    backend._export_torchscript(model, "label", path)

    traced = torch.jit.load(str(path))
    x = torch.randn(5, 4)
    with torch.no_grad():
        got = traced(x)
        want = model.model({"img": x})
    # The output feature's tensors only, in sorted key order (logits, probabilities), batch size free.
    assert len(got) == 2
    assert torch.allclose(got[0], want["label::logits"])
    assert torch.allclose(got[1], want["label::probabilities"])


def test_onnx_and_torchscript_share_one_wrapper(tmp_path):
    model = _Model()
    wrapper, example_args, sample_inputs, input_names, output_keys = backend._traceable(model, "label")
    assert input_names == ["img"] and output_keys == ["label::logits", "label::probabilities"]
    assert tuple(a.shape for a in example_args) == ((2, 4),)
    with torch.no_grad():
        assert len(wrapper(*example_args)) == 2


def test_export_dispatches_torchscript_to_the_traced_path(tmp_path, monkeypatch):
    """`export()` no longer reaches for `LudwigModel.to_torchscript`."""
    calls = []
    monkeypatch.setattr(backend, "_export_torchscript", lambda m, f, p: calls.append((f, p)) or p.write_bytes(b"x"))
    monkeypatch.setattr(backend, "_sha256_file", lambda p: "sha")

    class _Cfg:
        output_features = [type("F", (), {"name": "label"})()]

    class _Loaded:
        config_obj = _Cfg()

    import types
    monkeypatch.setitem(__import__("sys").modules, "ludwig.api", types.SimpleNamespace(LudwigModel=type("LM", (), {"load": staticmethod(lambda *a, **k: _Loaded())})))
    monkeypatch.setitem(__import__("sys").modules, "ludwig", types.SimpleNamespace(api=__import__("sys").modules["ludwig.api"]))
    from sakura.atelier.types import RunDir, TrainResult

    out = RunDir(tmp_path / "run")
    result = TrainResult(model=_Loaded(), metrics={}, train_seconds=0.0)  # no checkpoint: exported as-is
    art = backend.export(result, "torchscript", out)
    assert calls and calls[0][0] == "label" and art.format == "torchscript" and art.path.name == "model.torchscript.pt"



def test_torchscript_export_writes_its_io_order(tmp_path):
    import json

    path = tmp_path / "model.torchscript.pt"
    backend._export_torchscript(_Model(), "label", path)
    io = json.loads(path.with_suffix(".json").read_text())
    assert io["input_order"] == ["img"]
    assert io["output_order"] == ["label__logits", "label__probabilities"]
    assert io["input_dtypes"] == {"img": "torch.float32"}

class _Batcher:
    """Ludwig's batcher over a preprocessed split, cut to what the scorer reads."""

    def __init__(self, batches):
        self._batches, self._i = batches, 0

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def next_batch(self):
        self._i += 1
        return self._batches[self._i - 1]

    def last_batch(self):
        return self._i >= len(self._batches)


def test_torchscript_export_is_rescored_on_the_validation_split(tmp_path, monkeypatch):
    """BE-9: the TorchScript file gets its own accuracy and row count, through
    the same scoring loop as ONNX, instead of shipping untested."""
    import numpy as np
    import types

    from sakura.atelier.types import Artifact

    model = _Model()
    path = tmp_path / "artifacts" / "model.torchscript.pt"
    path.parent.mkdir()
    backend._export_torchscript(model, "label", path)
    assert path.with_suffix(".json").exists()

    x = np.random.default_rng(0).normal(size=(6, 4)).astype(np.float32)
    with torch.no_grad():
        target = model.model({"img": torch.from_numpy(x)})["label::logits"].argmax(-1).numpy()
    target[0] = (target[0] + 1) % 3  # one wrong row: 5 of 6
    batches = [{"img_proc": x[:4], "label_proc": target[:4]}, {"img_proc": x[4:], "label_proc": target[4:]}]
    preprocessed = types.SimpleNamespace(training_set=types.SimpleNamespace(
        initialize_batcher=lambda **kw: _Batcher(batches)))
    out_cfg = types.SimpleNamespace(name="label", type="category")
    fake = types.SimpleNamespace(config_obj=types.SimpleNamespace(output_features=[out_cfg]))
    monkeypatch.setattr(backend, "_preprocessed_validation", lambda a, d: (fake, preprocessed))
    monkeypatch.setattr(backend, "_proc_columns", lambda m: {"img": "img_proc", "label": "label_proc"})

    art = Artifact(path=path, format="torchscript", sha256="s", bytes=path.stat().st_size)
    ev = backend.runtime_eval(art, None)
    assert ev.format == "torchscript" and ev.runtime == "torch.jit" and ev.metric == "accuracy"
    assert ev.n == 6 and abs(ev.value - 5 / 6) < 1e-9 and ev.reason is None


def test_torchscript_without_its_sidecar_says_why_it_is_untested(tmp_path):
    from sakura.atelier.types import Artifact

    path = tmp_path / "model.torchscript.pt"
    path.write_bytes(b"x")
    ev = backend.runtime_eval(Artifact(path=path, format="torchscript", sha256="s", bytes=1), None)
    assert ev.value is None and ev.n == 0 and "model.torchscript.json missing" in ev.reason
