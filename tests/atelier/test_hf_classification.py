"""Generic HF image/audio/text classification backends (plan Task 3): any
AutoModelFor*Classification repo, pinned, fine-tuned, exported to ONNX and
re-scored. Tiny random repos keep this to seconds on a CPU."""

import io

import numpy as np
import pandas as pd
import pytest

transformers = pytest.importorskip("transformers")
pytest.importorskip("onnxruntime")

from sakura.atelier.backends import backend_name, load_backend  # noqa: E402
from sakura.atelier.registry import resolve  # noqa: E402
from sakura.atelier.spec import load_spec  # noqa: E402
from sakura.atelier.types import MaterialisedData, MetricsSink, ResolvedSpec, RunDir  # noqa: E402

VIT = "hf-internal-testing/tiny-random-ViTForImageClassification@5b3b44a3ed492070c273e481e30ecf4deddc5ec3"
W2V = "hf-internal-testing/tiny-random-Wav2Vec2ForSequenceClassification@2000b2022abcc37100241485f5872126b70164c9"
BERT = "hf-internal-testing/tiny-random-BertForSequenceClassification@4d5a3af95422fc0ebf732796c685c586b6d7420b"


def _hf_reachable():
    import urllib.request
    try:
        urllib.request.urlopen("https://huggingface.co/api/models/" + VIT.split("@")[0], timeout=5)
        return True
    except Exception:  # noqa: BLE001
        return False


pytestmark = pytest.mark.skipif(not _hf_reachable(), reason="needs huggingface.co")

SPEC = """
atelier: 1
name: t
task: {task}
data: {{uri: "file:///tmp/none", format: {fmt}, split: {{validation_fraction: 0.25}}}}
features:
  inputs: [{{name: x, type: {itype}}}]
  output: {{name: y, type: category}}
model: "hf:{ref}"
hp: {hp}
export: [onnx]
overrides: {{epochs: 2, batch_size: 4}}
"""


def _resolved(task, fmt, itype, ref, hp):
    return resolve(load_spec(SPEC.format(task=task, fmt=fmt, itype=itype, ref=ref, hp=hp)))


def _run(resolved: ResolvedSpec, train, val, labels, tmp_path):
    backend = load_backend(resolved.backend)
    out = RunDir(tmp_path / "out").ensure()
    data = MaterialisedData(root=tmp_path, train=train, validation=val, manifest_sha256="x", labels=labels)
    with MetricsSink(out.metrics_path) as m:
        result = backend.train(resolved, data, out, m, None)
    art = backend.export(result, "onnx", out)
    ev = backend.runtime_eval(art, data)
    return result, art, ev


def _images(tmp_path, n=8):
    from PIL import Image
    rows = []
    for i in range(n):
        p = tmp_path / f"img{i}.png"
        Image.fromarray(np.full((32, 32, 3), 40 + 20 * (i % 2), dtype=np.uint8)).save(p)
        rows.append({"x": str(p), "y": "a" if i % 2 else "b"})
    return pd.DataFrame(rows)


def test_hf_refs_route_to_the_generic_backends():
    assert backend_name("image_classification", "hf:" + VIT) == "hf_image_classification"
    assert backend_name("audio_classification", "hf:" + W2V) == "hf_audio_classification"
    assert backend_name("text_classification", "hf:" + BERT) == "hf_text_classification"


def test_image_classification_trains_and_exports_onnx(tmp_path):
    df = _images(tmp_path)
    r = _resolved("image_classification", "image_folder", "image", VIT, "image_classification/hf-finetune@1")
    result, art, ev = _run(r, df.iloc[:6], df.iloc[6:], ["a", "b"], tmp_path)
    assert "accuracy" in result.metrics and result.provenance["base_model"] == VIT.split("@")[0]
    assert art.format == "onnx" and art.path.exists()
    assert ev.runtime == "onnxruntime" and ev.metric == "accuracy" and ev.n == 2


def test_audio_classification_trains_and_exports_onnx(tmp_path):
    import soundfile as sf
    rows = []
    for i in range(8):
        p = tmp_path / f"a{i}.wav"
        t = np.arange(16000 // 4) / 16000
        sf.write(p, np.sin(2 * np.pi * (220 if i % 2 else 440) * t).astype(np.float32), 16000)
        rows.append({"x": str(p), "y": "lo" if i % 2 else "hi"})
    df = pd.DataFrame(rows)
    r = _resolved("audio_classification", "audio_folder", "audio", W2V, "audio_classification/hf-finetune@1")
    result, art, ev = _run(r, df.iloc[:6], df.iloc[6:], ["hi", "lo"], tmp_path)
    assert art.path.exists() and ev.n == 2


def test_text_classification_trains_and_exports_onnx(tmp_path):
    df = pd.DataFrame({"x": ["good film", "bad film", "great", "awful"] * 3, "y": ["pos", "neg"] * 6})
    r = _resolved("text_classification", "csv", "text", BERT, "text_classification/hf-finetune@1")
    result, art, ev = _run(r, df.iloc[:9], df.iloc[9:], None, tmp_path)
    assert art.path.exists() and ev.n == 3


def test_remote_code_refused(tmp_path, monkeypatch):
    from sakura.atelier.backends import hf_classification
    monkeypatch.setattr(hf_classification, "_config", lambda repo, rev: {"auto_map": {"AutoModel": "m.M"}})
    df = _images(tmp_path, 4)
    r = _resolved("image_classification", "image_folder", "image", VIT, "image_classification/hf-finetune@1")
    with pytest.raises(ValueError, match="needs remote code"):
        _run(r, df.iloc[:3], df.iloc[3:], ["a", "b"], tmp_path)


def test_audio_backend_leaves_deterministic_algorithms_off():
    """torch 2.5's deterministic CUDA index_put fails on wav2vec2's span masking; text and image keep it."""
    from sakura.atelier.backends.hf_classification import AUDIO, IMAGE, TEXT, ClassificationBackend

    assert ClassificationBackend(AUDIO).deterministic_algorithms is False
    assert ClassificationBackend(IMAGE).deterministic_algorithms is True
    assert ClassificationBackend(TEXT).deterministic_algorithms is True
