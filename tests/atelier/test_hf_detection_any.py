"""hf_detection fine-tunes any AutoModelForObjectDetection repo (plan Task 4), not only RT-DETR.
A tiny random YOLOS (a plain ViT detector: no timm backbone download) keeps this to seconds."""

import numpy as np
import pytest

pytest.importorskip("transformers")
pytest.importorskip("onnxruntime")
pytest.importorskip("torchmetrics")

from sakura.atelier.backends import backend_name, load_backend  # noqa: E402
from sakura.atelier.registry import resolve  # noqa: E402
from sakura.atelier.spec import load_spec  # noqa: E402
from sakura.atelier.types import MaterialisedData, MetricsSink, RunDir  # noqa: E402

YOLOS = "hf-internal-testing/tiny-random-YolosForObjectDetection@0a4aae25bfbe8b5edd4815cb00d697a6ba7d2126"


def _hf_reachable():
    import urllib.request
    try:
        urllib.request.urlopen("https://huggingface.co/api/models/" + YOLOS.split("@")[0], timeout=5)
        return True
    except Exception:  # noqa: BLE001
        return False


pytestmark = pytest.mark.skipif(not _hf_reachable(), reason="needs huggingface.co")

SPEC = f"""
atelier: 1
name: t
task: object_detection
data: {{uri: "file:///tmp/none", format: yolo, split: {{train: train, validation: val}}}}
features:
  inputs: [{{name: image, type: image}}]
  output: {{name: boxes, type: boxes}}
model: "hf:{YOLOS}"
hp: object_detection/fast@1
export: [onnx]
overrides: {{epochs: 1, batch: 2, imgsz: 64}}
"""


def _yolo_dataset(root, n=4):
    from PIL import Image
    for split in ("train", "val"):
        (root / "images" / split).mkdir(parents=True)
        (root / "labels" / split).mkdir(parents=True)
        for i in range(n):
            arr = np.full((64, 64, 3), 30, dtype=np.uint8)
            arr[16:48, 16:48] = 220
            Image.fromarray(arr).save(root / "images" / split / f"{i}.png")
            (root / "labels" / split / f"{i}.txt").write_text("0 0.5 0.5 0.5 0.5\n")
    y = root / "data.yaml"
    y.write_text(f"path: {root}\ntrain: images/train\nval: images/val\nnames: {{0: square}}\n")
    return y


def test_hf_ref_routes_to_hf_detection():
    assert backend_name("object_detection", "hf:" + YOLOS) == "hf_detection"


def test_detection_with_tiny_yolos(tmp_path):
    r = resolve(load_spec(SPEC))
    assert r.hyperparameters["base_model"] == YOLOS.split("@")[0]
    backend = load_backend(r.backend)
    out = RunDir(tmp_path / "out").ensure()
    data = MaterialisedData(root=tmp_path, train=None, validation=None, manifest_sha256="x", labels=["square"],
                            extra={"yolo_data_yaml": str(_yolo_dataset(tmp_path / "ds"))})
    with MetricsSink(out.metrics_path) as m:
        result = backend.train(r, data, out, m, None)
    assert result.provenance["base_model"] == YOLOS.split("@")[0]
    assert result.provenance["base_revision"] == YOLOS.split("@")[1]
    art = backend.export(result, "onnx", out)
    assert art.path.exists()
    ev = backend.runtime_eval(art, data)
    assert ev.runtime == "onnxruntime" and ev.n == 4
    pred = backend.predict(art, str(tmp_path / "ds" / "images" / "val" / "0.png"))
    assert isinstance(pred, (list, dict))


def test_remote_code_detector_refused(tmp_path, monkeypatch):
    from sakura.atelier.backends import hf_detection
    monkeypatch.setattr(hf_detection, "_config", lambda repo, rev: {"auto_map": {"AutoModel": "m.M"}})
    r = resolve(load_spec(SPEC))
    out = RunDir(tmp_path / "out").ensure()
    data = MaterialisedData(root=tmp_path, train=None, validation=None, manifest_sha256="x", labels=["square"],
                            extra={"yolo_data_yaml": str(_yolo_dataset(tmp_path / "ds"))})
    with MetricsSink(out.metrics_path) as m, pytest.raises(ValueError, match="needs remote code"):
        load_backend(r.backend).train(r, data, out, m, None)
