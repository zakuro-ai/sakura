"""Hugging Face RT-DETR backend: object detection, Apache-2.0 end to end.

Implements the ``Backend`` protocol (docs/atelier/CONTRACTS.md §2.2) for task
``object_detection`` / model id ``rtdetr_r18`` (checkpoint
``PekingU/rtdetr_r18vd``, Apache-2.0 per its model card -- verified before
use, see this track's handoff report). Fine-tunes via plain
``transformers``/``torch`` (no ``peft``/``accelerate`` needed: RT-DETR is
fully fine-tuned here, not LoRA'd), no ``ultralytics`` import anywhere in
this module or its import graph -- CONTRACTS §2.4's AGPL-3.0 flag on the
previous ``ultralytics`` backend does not apply to this one.

Data convention (same as the ``ultralytics`` backend it replaces; the
engine's data layer materialises a ``yolo`` tree the same way for every
object-detection backend -- ``sakura/atelier/data.py``'s
``_materialise_yolo`` is backend-agnostic):

  ``data.extra["yolo_data_yaml"]`` -- path (str) to a YOLO-format
  ``data.yaml`` (``train``/``val`` image dirs or list files, with sibling
  ``labels/`` dirs holding ``class cx cy w h`` normalised-float lines, plus
  ``names``). ``data.labels`` mirrors its ``names``, ordered by index.

This module converts that YOLO layout into Hugging Face's own COCO-style
annotation shape (``RTDetrImageProcessor(images=.., annotations=..)``) --
the pure, dependency-light parts of that conversion (listing a split's
images, mapping an image path to its label path, normalised-xywh ->
absolute-xywh box conversion) are unit tested in
``tests/atelier/test_backend_hf_detection.py`` without any heavy import.

``runtime_eval`` protocol note (same as ``ultralytics.py`` and ``tabpfn.py``):
``Backend.runtime_eval(self, artifact, data)`` is not passed the
``TrainResult``, so ``matches_training`` is left ``None`` here -- the
engine's ``run`` loop (``sakura/atelier/__main__.py``) holds
``TrainResult.metrics`` and applies the §2.3 tolerance.

mAP methodology
---------------
Training-time validation and the ONNX ``runtime_eval`` share one scoring
path (``_score``): the Hugging Face ``RTDetrImageProcessor`` does
pre/post-processing in both cases (framework code, not re-implemented per
runtime), only the model *forward pass* differs -- plain torch during
training, ``onnxruntime.InferenceSession`` against the exported graph for
``runtime_eval``/``predict``. That is what makes the two numbers comparable
at all (same conf threshold, same resize, same NMS-free top-k decoding
RT-DETR itself does). mAP is computed with ``torchmetrics``' ``pycocotools``-
free ``MeanAveragePrecision`` (Apache-2.0), at IoU 0.5 (``map50``, matching
the existing object-detection chart key) and IoU 0.5:0.95 (``map50_95``).

Licence: PekingU/rtdetr_r18vd (RT-DETR, ResNet-18 backbone) is Apache-2.0
(https://huggingface.co/PekingU/rtdetr_r18vd -- model card "License:
apache-2.0", checked 2026-10). ``transformers`` and ``torchmetrics`` are
both Apache-2.0.
"""

from __future__ import annotations

import json
import random
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, ClassVar

from sakura.atelier.types import (
    Artifact,
    Checkpoint,
    MaterialisedData,
    MetricsSink,
    ResolvedSpec,
    RunDir,
    RuntimeEval,
    TrainResult,
)
from sakura.atelier.wrap_up import WrapUpWatch, max_steps_of

LICENSE = "Apache-2.0"

BASE_MODEL = "PekingU/rtdetr_r18vd"  # the default; `hp.base_model` (an hf:<repo>@<sha> spec) names any other


def _config(repo: str, revision: str | None) -> dict[str, Any]:
    """The repo's config.json as a dict (no weights): read before loading so a repo that needs its own
    code (``auto_map``) is refused, never executed."""
    from transformers import AutoConfig

    config: dict[str, Any] = AutoConfig.from_pretrained(repo, revision=revision, trust_remote_code=False).to_dict()
    return config


def _processor_for(artifact_dir: Path) -> Any:
    """The image processor the artifact was trained with (shipped next to it since 2026-11-09); older
    artifacts were all RT-DETR, so they fall back to BASE_MODEL's."""
    from transformers import AutoImageProcessor

    shipped = artifact_dir / "preprocessor"
    return AutoImageProcessor.from_pretrained(str(shipped) if shipped.is_dir() else BASE_MODEL, use_fast=True)

IMAGE_EXTS = frozenset({".jpg", ".jpeg", ".png", ".bmp", ".webp"})


# --------------------------------------------------------------------- pure
# (no torch/transformers import -- CPU, no network, unit tested directly)

def _device() -> str:
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


def _resolve_split_path(doc: dict[str, Any], key: str) -> Path:
    """Same resolution rule ``data.yaml``'s own ``path:`` + ``train``/``val``
    entries follow in the Ultralytics layout this engine materialises: a
    relative split entry is relative to the yaml's ``path:``."""
    raw = doc[key]
    p = Path(raw)
    if p.is_absolute():
        return p
    base = doc.get("path")
    return (Path(base) / p) if base else p


def list_split_images(split_path: Path) -> list[Path]:
    """Images for one split: a list file (one path per line) or an image
    directory. Mirrors ``backends/ultralytics.py``'s ``_count_val_images``,
    but returns the paths themselves (every image, not just a count)."""
    if split_path.is_file():
        lines = [line.strip() for line in split_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        return [Path(line) for line in lines]
    if split_path.is_dir():
        return sorted(p for p in split_path.iterdir() if p.suffix.lower() in IMAGE_EXTS)
    return []


def label_path_for(image_path: Path) -> Path:
    """``images/<split>/x.jpg`` -> ``labels/<split>/x.txt`` -- the standard
    YOLO sibling-directory convention this engine's materialised tree uses.
    An image path with no ``images`` component in it is assumed to already
    be a label-sibling layout (``x.jpg`` next to ``x.txt``)."""
    parts = list(image_path.parts)
    for i in range(len(parts) - 1, -1, -1):
        if parts[i] == "images":
            parts[i] = "labels"
            return Path(*parts).with_suffix(".txt")
    return image_path.with_suffix(".txt")


def yolo_label_lines_to_coco_boxes(lines: list[str], width: int, height: int) -> list[dict[str, Any]]:
    """YOLO ``class cx cy w h`` (normalised 0..1, box CENTER) -> COCO-style
    ``{"bbox": [x, y, w, h], "category_id": int, "area": float, "iscrowd": 0}``
    (absolute pixels, TOP-LEFT corner) -- the shape
    ``RTDetrImageProcessor(images=.., annotations=..)`` expects."""
    out = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        cls_s, cx_s, cy_s, w_s, h_s = line.split()[:5]
        cls = int(float(cls_s))
        cx, cy, w, h = (float(cx_s), float(cy_s), float(w_s), float(h_s))
        abs_w, abs_h = w * width, h * height
        abs_x, abs_y = (cx - w / 2) * width, (cy - h / 2) * height
        out.append({
            "bbox": [abs_x, abs_y, abs_w, abs_h],
            "category_id": cls,
            "area": abs_w * abs_h,
            "iscrowd": 0,
        })
    return out


def coco_boxes_to_xyxy(boxes: list[dict[str, Any]]) -> list[list[float]]:
    """``[x, y, w, h]`` (top-left + size) -> ``[x0, y0, x1, y1]`` -- what
    ``torchmetrics.detection.MeanAveragePrecision(box_format="xyxy")`` wants
    for a ground-truth target, matching what
    ``processor.post_process_object_detection`` already hands back for a
    prediction."""
    out = []
    for b in boxes:
        x, y, w, h = b["bbox"]
        out.append([x, y, x + w, y + h])
    return out


def read_yolo_item(image_path: Path, label_path: Path) -> dict[str, Any]:
    """One dataset item: the image path plus its ground-truth boxes in both
    shapes (COCO xywh, for feeding the processor at train time; xyxy, for
    scoring against torchmetrics). A label-less image (background-only, a
    legal YOLO case) yields an empty box list, not an error."""
    lines = label_path.read_text(encoding="utf-8").splitlines() if label_path.exists() else []
    return {"image_path": image_path, "label_lines": lines}


def load_id2label(labels: list[str]) -> tuple[dict[int, str], dict[str, int]]:
    id2label = {i: name for i, name in enumerate(labels)}
    label2id = {name: i for i, name in enumerate(labels)}
    return id2label, label2id


# ------------------------------------------------------------- not-so-pure
# (torch/transformers/torchmetrics below; lazily imported per CONTRACTS §2)

class _ONNXForwardWrapper:
    """Wraps a torch model so ``torch.onnx.export`` sees a plain
    ``(logits, pred_boxes)`` tuple output instead of RT-DETR's dataclass
    (which also carries encoder/decoder hidden states and aux losses that
    the exported graph has no use for)."""

    def __init__(self, model: Any) -> None:
        import torch.nn as nn

        class _Wrapped(nn.Module):
            def __init__(self, inner: Any) -> None:
                super().__init__()
                self.inner = inner

            def forward(self, pixel_values: Any) -> Any:
                out = self.inner(pixel_values=pixel_values)
                return out.logits, out.pred_boxes

        self.module = _Wrapped(model)


def _load_split(data_yaml: str, split_key: str) -> list[dict[str, Any]]:
    import yaml

    doc = yaml.safe_load(Path(data_yaml).read_text(encoding="utf-8"))
    split_path = _resolve_split_path(doc, split_key)
    items = []
    for img in list_split_images(split_path):
        items.append(read_yolo_item(img, label_path_for(img)))
    return items


def _processed_batch(processor: Any, items: list[dict[str, Any]], imgsz: int) -> list[dict[str, Any]]:
    """One item -> ``(pixel_values[1,3,H,W], labels_dict, (orig_w, orig_h))``.
    Loaded/processed one at a time (not vectorised): these runs are a few
    hundred images at most (minutes-not-hours presets), and per-image PIL
    opens keep this function simple and easy to unit test around."""
    from PIL import Image

    out = []
    for item in items:
        img = Image.open(item["image_path"]).convert("RGB")
        w, h = img.size
        boxes = yolo_label_lines_to_coco_boxes(item["label_lines"], w, h)
        annotation = {"image_id": hash(str(item["image_path"])) & 0x7FFFFFFF, "annotations": boxes}
        processed = processor(images=img, annotations=annotation, return_tensors="pt", size={"height": imgsz, "width": imgsz})
        out.append({
            "pixel_values": processed["pixel_values"][0],
            "labels": processed["labels"][0],
            "orig_size": (h, w),
            "gt_xyxy": coco_boxes_to_xyxy(boxes),
            "gt_classes": [b["category_id"] for b in boxes],
        })
    return out


def _score(preds_boxes: list[Any], preds_scores: list[Any], preds_labels: list[Any],
           targets: list[dict[str, Any]], conf: float) -> tuple[float, float, int]:
    import torch
    from torchmetrics.detection import MeanAveragePrecision

    metric = MeanAveragePrecision(box_format="xyxy", backend="faster_coco_eval")
    preds = []
    tgts = []
    for pb, ps, pl, t in zip(preds_boxes, preds_scores, preds_labels, targets):
        keep = ps >= conf
        preds.append({"boxes": pb[keep], "scores": ps[keep], "labels": pl[keep]})
        tgts.append({
            "boxes": torch.tensor(t["gt_xyxy"], dtype=torch.float32) if t["gt_xyxy"] else torch.zeros((0, 4)),
            "labels": torch.tensor(t["gt_classes"], dtype=torch.int64) if t["gt_classes"] else torch.zeros((0,), dtype=torch.int64),
        })
    metric.update(preds, tgts)
    result = metric.compute()
    map50 = float(result["map_50"])
    map50_95 = float(result["map"])
    map50 = 0.0 if map50 < 0 else map50  # torchmetrics returns -1 when there is nothing to score
    map50_95 = 0.0 if map50_95 < 0 else map50_95
    return map50, map50_95, len(targets)


def _eval_torch(model: Any, processor: Any, items: list[dict[str, Any]], imgsz: int, conf: float) -> tuple[float, float, int]:
    import torch

    model.eval()
    device = next(model.parameters()).device
    processed = _processed_batch(processor, items, imgsz)
    preds_boxes, preds_scores, preds_labels = [], [], []
    with torch.no_grad():
        for p in processed:
            out = model(pixel_values=p["pixel_values"].unsqueeze(0).to(device))
            out.logits = out.logits.cpu()
            out.pred_boxes = out.pred_boxes.cpu()
            result = processor.post_process_object_detection(
                out, threshold=0.0, target_sizes=[p["orig_size"]],
            )[0]
            preds_boxes.append(result["boxes"])
            preds_scores.append(result["scores"])
            preds_labels.append(result["labels"])
    return _score(preds_boxes, preds_scores, preds_labels, processed, conf)


def _eval_onnxruntime(session: Any, processor: Any, items: list[dict[str, Any]], imgsz: int, conf: float) -> tuple[float, float, int]:
    import torch

    processed = _processed_batch(processor, items, imgsz)
    preds_boxes, preds_scores, preds_labels = [], [], []
    for p in processed:
        pixel_values = p["pixel_values"].unsqueeze(0).numpy()
        logits, pred_boxes = session.run(["logits", "pred_boxes"], {"pixel_values": pixel_values})

        out = SimpleNamespace(logits=torch.from_numpy(logits),
                              pred_boxes=torch.from_numpy(pred_boxes))
        result = processor.post_process_object_detection(out, threshold=0.0, target_sizes=[p["orig_size"]])[0]
        preds_boxes.append(result["boxes"])
        preds_scores.append(result["scores"])
        preds_labels.append(result["labels"])
    return _score(preds_boxes, preds_scores, preds_labels, processed, conf)


def _artifact(path: Path) -> Artifact:
    import hashlib

    data = path.read_bytes()
    return Artifact(path=path, format="onnx", sha256=hashlib.sha256(data).hexdigest(), bytes=len(data))


class HFDetectionBackend:
    name = "hf_detection"
    tasks = frozenset({"object_detection"})
    licence = LICENSE

    #: Set below, once the methods exist (see NOTEBOOK_CELLS at the end of the module).
    NOTEBOOK_CELLS: ClassVar[tuple[Any, ...]]

    def train(
        self,
        spec: ResolvedSpec,
        data: MaterialisedData,
        out: RunDir,
        metrics: MetricsSink,
        resume: Checkpoint | None,
    ) -> TrainResult:
        import torch
        from transformers import AutoImageProcessor, AutoModelForObjectDetection

        out.ensure()
        hp = spec.hyperparameters
        epochs = int(hp.get("epochs", 3))
        imgsz = int(hp.get("imgsz", 640))
        batch_size = int(hp.get("batch", 4))
        lr = float(hp.get("learning_rate", 1e-4))
        conf = float(hp.get("val_conf", 0.3))

        data_yaml = data.extra["yolo_data_yaml"]
        labels = data.labels or []
        id2label, label2id = load_id2label(labels)

        base = hp.get("base_model") or BASE_MODEL
        revision = hp.get("base_revision")
        if resume is not None and resume.task != "object_detection":
            raise ValueError(f"checkpoint is for task {resume.task!r}, not object_detection")
        if resume is None and _config(base, revision).get("auto_map"):
            raise ValueError(f"{base} needs remote code to load, which L'Atelier does not run")
        weights = str(resume.path / "model") if resume is not None else base
        weights_revision = None if resume is not None else revision

        processor = AutoImageProcessor.from_pretrained(weights, revision=weights_revision, use_fast=True)
        model = AutoModelForObjectDetection.from_pretrained(
            weights, revision=weights_revision, num_labels=len(labels) or None, id2label=id2label or None,
            label2id=label2id or None, ignore_mismatched_sizes=True, trust_remote_code=False,
        )
        device = _device()
        model.to(device)
        model.train()

        train_items = _load_split(data_yaml, "train")
        val_items = _load_split(data_yaml, "val")

        optim = torch.optim.AdamW(model.parameters(), lr=lr)
        rng = random.Random(spec.spec.seed)

        t0 = time.time()
        step = 0
        watch = WrapUpWatch(out.root, max_steps_of(spec))
        for epoch in range(epochs):
            order = list(range(len(train_items)))
            rng.shuffle(order)
            for i in range(0, len(order), batch_size):
                if watch.stop(epoch + i / max(len(order), 1), step):
                    break  # wrap-up: the final CPU re-validation below covers the partial epoch
                idx = order[i:i + batch_size]
                batch = _processed_batch(processor, [train_items[j] for j in idx], imgsz)
                pixel_values = torch.stack([b["pixel_values"] for b in batch]).to(device)
                labels_batch = [{k: v.to(device) for k, v in b["labels"].items()} for b in batch]
                out_ = model(pixel_values=pixel_values, labels=labels_batch)
                out_.loss.backward()
                optim.step()
                optim.zero_grad()
                metrics.log("loss", float(out_.loss), step=step, split="train", epoch=epoch + i / max(len(order), 1))
                step += 1
            map50, map50_95, _n = _eval_torch(model, processor, val_items, imgsz, conf)
            metrics.log("map50", map50, step=step, split="validation", epoch=float(epoch + 1))
            metrics.log("map50_95", map50_95, step=step, split="validation", epoch=float(epoch + 1))
            if watch.triggered:
                break
        train_seconds = time.time() - t0
        model.eval()

        # Re-validate once more on CPU at the exact settings runtime_eval()
        # will use later (same rationale as backends/ultralytics.py's module
        # docstring: this is what makes the ONNX re-score comparable at all).
        model.to("cpu")
        map50, map50_95, n = _eval_torch(model, processor, val_items, imgsz, conf)
        final_metrics = {"map50": map50, "map50_95": map50_95, "n_validation": float(n)}

        ckpt_dir = out.checkpoint
        model_dir = ckpt_dir / "model"
        model_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(str(model_dir), safe_serialization=True)
        processor.save_pretrained(str(model_dir))
        preprocess = {"imgsz": imgsz, "conf": conf, "labels": labels}
        (ckpt_dir / "preprocess.json").write_text(json.dumps(preprocess))
        (ckpt_dir / "meta.json").write_text(json.dumps({"model": spec.spec.model, "task": "object_detection"}))

        return TrainResult(
            model=model,
            metrics=final_metrics,
            train_seconds=train_seconds,
            checkpoint=Checkpoint(path=ckpt_dir, model=spec.spec.model, task="object_detection"),
            stopped_early=watch.stopped_early,
            provenance={"base_model": base, "base_revision": revision,
                        "base_commit": getattr(model.config, "_commit_hash", None) or revision},
        )

    def export(self, result: TrainResult, fmt: str, out: RunDir) -> Artifact:
        if fmt != "onnx":
            raise ValueError(f"hf_detection backend only exports 'onnx', got {fmt!r}")
        import torch

        out.ensure()
        model = result.model
        model.eval()
        imgsz = 640
        preprocess_src = None
        if result.checkpoint is not None:
            cand = result.checkpoint.path / "preprocess.json"
            if cand.exists():
                preprocess_src = json.loads(cand.read_text())
                imgsz = int(preprocess_src.get("imgsz", imgsz))

        dummy = torch.zeros(1, 3, imgsz, imgsz)
        wrapper = _ONNXForwardWrapper(model).module
        wrapper.eval()
        dest = out.artifacts / "model.onnx"
        torch.onnx.export(
            wrapper, (dummy,), str(dest),
            input_names=["pixel_values"], output_names=["logits", "pred_boxes"],
            opset_version=17, dynamic_axes=None, do_constant_folding=True,
            dynamo=False,  # the TorchScript exporter: torch >= 2.9 defaults to dynamo, which needs onnxscript
        )
        preprocess = preprocess_src or {"imgsz": imgsz, "conf": 0.3, "labels": []}
        (out.artifacts / "preprocess.json").write_text(json.dumps(preprocess))
        if result.checkpoint is not None and (result.checkpoint.path / "model").is_dir():
            from transformers import AutoImageProcessor

            # the processor this model was trained with ships next to it (runtime_eval / predict read it)
            AutoImageProcessor.from_pretrained(str(result.checkpoint.path / "model"), use_fast=True) \
                .save_pretrained(str(out.artifacts / "preprocessor"))
        if result.checkpoint is not None:
            labels_path = out.artifacts / "labels.json"
            labels_path.write_text(json.dumps(preprocess.get("labels", [])))
        return _artifact(dest)

    def runtime_eval(self, artifact: Artifact, data: MaterialisedData) -> RuntimeEval:
        from transformers import AutoImageProcessor

        from sakura.atelier.ort import inference_session

        data_yaml = data.extra["yolo_data_yaml"]
        preprocess_path = artifact.path.parent / "preprocess.json"
        preprocess = (
            json.loads(preprocess_path.read_text()) if preprocess_path.exists()
            else {"imgsz": 640, "conf": 0.3}
        )
        processor = _processor_for(artifact.path.parent)
        session = inference_session(artifact.path, providers=["CPUExecutionProvider"])
        val_items = _load_split(data_yaml, "val")
        map50, _map50_95, n = _eval_onnxruntime(
            session, processor, val_items, int(preprocess.get("imgsz", 640)), float(preprocess.get("conf", 0.3)),
        )

        # matches_training is the engine's call (Track E1's `run` loop holds
        # TrainResult.metrics; this method's signature does not).
        return RuntimeEval(
            format="onnx", runtime="onnxruntime", metric="map50",
            value=map50, n=n, matches_training=None, reason=None,
        )

    def predict(self, artifact: Artifact, inputs: Any) -> Any:
        import torch
        from PIL import Image
        from transformers import AutoImageProcessor

        from sakura.atelier.ort import inference_session

        key = str(artifact.path)
        if key not in _LOADED:  # once per process (`serve`)
            preprocess_path = artifact.path.parent / "preprocess.json"
            preprocess = (
                json.loads(preprocess_path.read_text()) if preprocess_path.exists()
                else {"imgsz": 640, "conf": 0.3, "labels": []}
            )
            processor = _processor_for(artifact.path.parent)
            session = inference_session(key, providers=["CPUExecutionProvider"])
            _LOADED[key] = (session, processor, preprocess)
        session, processor, preprocess = _LOADED[key]

        source = inputs
        if isinstance(inputs, dict) and "file" in inputs:
            source = str(inputs["file"])
        if isinstance(inputs, (bytes, bytearray)):
            import io

            source = Image.open(io.BytesIO(inputs))
        if isinstance(source, str):
            img = Image.open(source).convert("RGB")
        else:
            img = source.convert("RGB")
        w, h = img.size
        imgsz = int(preprocess.get("imgsz", 640))
        processed = processor(images=img, return_tensors="pt", size={"height": imgsz, "width": imgsz})
        pixel_values = processed["pixel_values"].numpy()
        logits, pred_boxes = session.run(["logits", "pred_boxes"], {"pixel_values": pixel_values})

        out = SimpleNamespace(logits=torch.from_numpy(logits),
                              pred_boxes=torch.from_numpy(pred_boxes))
        conf = float(preprocess.get("conf", 0.3))
        result = processor.post_process_object_detection(out, threshold=conf, target_sizes=[(h, w)])[0]
        names = preprocess.get("labels") or []
        boxes = []
        for box, score, label in zip(result["boxes"].tolist(), result["scores"].tolist(), result["labels"].tolist()):
            boxes.append({
                "label": names[label] if 0 <= label < len(names) else str(label),
                "score": float(score),
                "xyxy": [float(x) for x in box],
            })
        return {"boxes": boxes, "width": int(w), "height": int(h)}


_LOADED: dict[str, Any] = {}

#: Functions the notebook shows by source, in pipeline order (CONTRACTS §2.5): helpers before
#: `train` are the Model section, those after `train`, `export` and `runtime_eval` follow that step.
NOTEBOOK_CELLS = (
    load_id2label,
    HFDetectionBackend.train, _load_split, _processed_batch, _eval_torch,
    HFDetectionBackend.export,
    HFDetectionBackend.runtime_eval, _eval_onnxruntime, _score,
)
HFDetectionBackend.NOTEBOOK_CELLS = NOTEBOOK_CELLS

BACKEND = HFDetectionBackend()
