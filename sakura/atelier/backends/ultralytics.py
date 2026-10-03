"""Ultralytics YOLO backend: object detection.

Implements the ``Backend`` protocol (docs/atelier/CONTRACTS.md §2.2) for
task ``object_detection`` / model id ``yolo11n``. The ultralytics/torch/
onnxruntime stack is imported lazily inside methods -- ``import
sakura.atelier`` must never pull it in
(tests/atelier/test_backends_registry.py::test_importing_the_engine_imports_no_training_stack
enforces this for the whole package, this module included).

Data convention
---------------
Track E1's data materialisation (``sakura/atelier/data.py`` or similar) does
not exist yet, so the ``MaterialisedData`` shape for object detection is this
backend's own choice until the engine's loader lands:

  ``data.extra["yolo_data_yaml"]`` -- path (str) to a YOLO-format
  ``data.yaml`` (``train``/``val`` image dirs with sibling ``labels/`` dirs,
  plus ``names``). ``data.labels`` mirrors its ``names``, ordered by index.

``runtime_eval`` protocol note
------------------------------
``Backend.runtime_eval(self, artifact, data)`` (per CONTRACTS §2.2) is not
passed the ``TrainResult``, so a backend has no direct handle on the training
metric to compare ``matches_training`` against. This module (and tabpfn.py,
peft.py) leaves ``matches_training: None`` and reports only the measured
runtime value + ``n`` (plus ``reason`` when the re-score itself fails) -- the
engine's ``run`` loop (Track E1) holds both ``TrainResult.metrics`` and this
``RuntimeEval`` and is the right place to apply the §2.3 tolerance.

Why ``TrainResult.metrics["map50"]`` is a FRESH re-validation, not the
trainer's own end-of-training number
-----------------------------------------------------------------------
ONNX export is static-shaped (``dynamic=False``): the exported graph only
ever runs ``batch=1``. ultralytics' own ``model.train()`` validates at the
training batch size with ``rect=True`` (its default), which groups
same-aspect-ratio images into a batch and resizes each batch to a shared,
batch-dependent canvas -- a real preprocessing difference from a single
image's own letterbox at ``batch=1``, confirmed empirically on this track's
coco128 holdout (identical ``best.pt`` checkpoint: map50 0.560 at
batch=8/rect=True vs 0.522 at batch=1/rect=False -- a ~0.04 gap with ZERO
export involved). Comparing the trainer's batch-advantaged number against a
batch=1 ONNX re-score was therefore always going to look like "export drift"
that wasn't: at matched batch=1/rect=False, the SAME .pt checkpoint and its
ONNX export score identically (0.5220 vs 0.5220, measured during development
report). So ``train()`` re-validates the trained checkpoint once more, at the
exact settings ``runtime_eval()`` will later use, and reports THAT as the
primary metric -- the trainer's own (batch-advantaged, not comparable) number
is kept too, under ``map50_trainer_reported``, for visibility.

Licence: ultralytics is AGPL-3.0 (CONTRACTS §2.4). ``LICENSE`` below is read
by the smoke harness / run report; this module does not write report.json
itself (that is the engine's, Track E1's).
"""

from __future__ import annotations

import csv
import json
import shutil
import time
from pathlib import Path
from typing import Any

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

LICENSE = "AGPL-3.0"


def _device() -> str:
    import torch

    return "0" if torch.cuda.is_available() else "cpu"


def _replay_csv_into_sink(csv_path: Path, metrics: MetricsSink, epochs: int) -> None:
    """ultralytics writes its own per-epoch ``results.csv`` and trains to
    completion before giving us control back, so there is no hook to call
    ``MetricsSink.log`` per-step live. Replay the CSV afterwards instead --
    the sink's append-only file still ends up with one point per epoch per
    metric, same as a backend that logs as it goes."""
    if not csv_path.exists():
        return
    with open(csv_path, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    for row in rows:
        epoch = int(float(row.get("epoch", 0)))
        step = epoch
        train_loss = row.get("train/box_loss")
        if train_loss not in (None, ""):
            metrics.log("loss", float(train_loss), step=step, split="train", epoch=float(epoch))
        map50 = row.get("metrics/mAP50(B)")
        if map50 not in (None, ""):
            metrics.log("map50", float(map50), step=step, split="validation", epoch=float(epoch))
        map50_95 = row.get("metrics/mAP50-95(B)")
        if map50_95 not in (None, ""):
            metrics.log("map50_95", float(map50_95), step=step, split="validation", epoch=float(epoch))


def _count_val_images(data_yaml: str) -> int:
    """How many images ``val()`` actually scored. ``val.seen``/``val.stats``
    on the returned ``DetMetrics`` do not carry this (an earlier version of
    this module read ``len(val.stats)``, which is the metrics dict's OWN key
    count -- 5 -- not an image count; a real bug, not a hypothetical one).
    Read it directly from the data yaml's val split instead."""
    import yaml

    with open(data_yaml, encoding="utf-8") as fh:
        y = yaml.safe_load(fh)
    val = y.get("val")
    val_path = Path(val)
    base = Path(y["path"]) if "path" in y and not val_path.is_absolute() else None
    if base is not None:
        val_path = base / val_path
    if val_path.is_file():  # a list file, one image path per line
        return sum(1 for line in val_path.read_text().splitlines() if line.strip())
    if val_path.is_dir():
        return sum(1 for _ in val_path.glob("*.jpg")) + sum(1 for _ in val_path.glob("*.png"))
    return 0


def _artifact(path: Path) -> Artifact:
    import hashlib

    data = path.read_bytes()
    return Artifact(path=path, format="onnx", sha256=hashlib.sha256(data).hexdigest(), bytes=len(data))


class UltralyticsBackend:
    name = "ultralytics"
    tasks = frozenset({"object_detection"})
    licence = LICENSE

    def train(
        self,
        spec: ResolvedSpec,
        data: MaterialisedData,
        out: RunDir,
        metrics: MetricsSink,
        resume: Checkpoint | None,
    ) -> TrainResult:
        from ultralytics import YOLO

        out.ensure()
        hp = spec.hyperparameters
        epochs = int(hp.get("epochs", 3))
        imgsz = int(hp.get("imgsz", 640))
        batch = int(hp.get("batch", 8))
        # Pinned (not ultralytics' differing predict- vs val-mode defaults)
        # so training's own validation and runtime_eval's later re-score run
        # the identical NMS confidence/IoU thresholds -- a difference here
        # would masquerade as export drift. conf=0.001/iou=0.7 are
        # ultralytics' own val-mode defaults; pinning just makes the identity
        # provable instead of assumed.
        val_conf = float(hp.get("val_conf", 0.001))
        val_iou = float(hp.get("val_iou", 0.7))
        data_yaml = data.extra["yolo_data_yaml"]

        weights: str
        if resume is not None:
            if resume.task != "object_detection":
                raise ValueError(f"checkpoint is for task {resume.task!r}, not object_detection")
            weights = str(resume.path / "last.pt")
        else:
            weights = f"{spec.spec.model}.pt"
        model = YOLO(weights)

        project = out.root / "_yolo_run"
        t0 = time.time()
        results = model.train(
            data=str(data_yaml),
            epochs=epochs,
            imgsz=imgsz,
            batch=batch,
            conf=val_conf,
            iou=val_iou,
            half=False,
            device=_device(),
            seed=spec.spec.seed,
            project=str(project),
            name="train",
            exist_ok=True,
            verbose=False,
            plots=False,
            workers=2,
        )
        train_seconds = time.time() - t0

        run_dir = project / "train"
        _replay_csv_into_sink(run_dir / "results.csv", metrics, epochs=epochs)

        rd = results.results_dict
        map50_trainer_reported = float(rd.get("metrics/mAP50(B)", 0.0))
        map50_95_trainer_reported = float(rd.get("metrics/mAP50-95(B)", 0.0))

        ckpt_dir = out.checkpoint
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        best = run_dir / "weights" / "best.pt"
        last = run_dir / "weights" / "last.pt"
        shutil.copy(best if best.exists() else last, ckpt_dir / "best.pt")
        if last.exists():
            shutil.copy(last, ckpt_dir / "last.pt")
        (ckpt_dir / "meta.json").write_text(
            json.dumps({"model": spec.spec.model, "task": "object_detection"})
        )

        # batch=1/rect=False: the exact, fixed preprocessing a static-shaped
        # ONNX export can run (see module docstring) -- re-validating here,
        # once, at the settings runtime_eval() will use later is what makes
        # the two numbers comparable at all.
        preprocess = {"imgsz": imgsz, "conf": val_conf, "iou": val_iou, "batch": 1, "rect": False}
        (ckpt_dir / "preprocess.json").write_text(json.dumps(preprocess))

        trained = YOLO(str(ckpt_dir / "best.pt"))
        trained._sakura_preprocess = preprocess
        comparable = trained.val(
            data=str(data_yaml), device="cpu", verbose=False, plots=False, split="val",
            imgsz=preprocess["imgsz"], conf=preprocess["conf"], iou=preprocess["iou"],
            half=False, batch=preprocess["batch"], rect=preprocess["rect"],
        )
        map50 = float(comparable.results_dict.get("metrics/mAP50(B)", 0.0))
        map50_95 = float(comparable.results_dict.get("metrics/mAP50-95(B)", 0.0))
        final_metrics = {
            "map50": map50, "map50_95": map50_95,
            "map50_trainer_reported": map50_trainer_reported,
            "map50_95_trainer_reported": map50_95_trainer_reported,
        }

        return TrainResult(
            model=trained,
            metrics=final_metrics,
            train_seconds=train_seconds,
            checkpoint=Checkpoint(path=ckpt_dir, model=spec.spec.model, task="object_detection"),
        )

    def export(self, result: TrainResult, fmt: str, out: RunDir) -> Artifact:
        if fmt != "onnx":
            raise ValueError(f"ultralytics backend only exports 'onnx', got {fmt!r}")
        out.ensure()
        model = result.model
        preprocess = getattr(
            model, "_sakura_preprocess",
            {"imgsz": 640, "conf": 0.001, "iou": 0.7, "batch": 1, "rect": False},
        )
        exported = model.export(format="onnx", imgsz=preprocess["imgsz"], simplify=True, opset=17, dynamic=False)
        dest = out.artifacts / "model.onnx"
        shutil.move(str(exported), str(dest))
        (out.artifacts / "preprocess.json").write_text(json.dumps(preprocess))
        return _artifact(dest)

    def runtime_eval(self, artifact: Artifact, data: MaterialisedData) -> RuntimeEval:
        from ultralytics import YOLO

        data_yaml = data.extra["yolo_data_yaml"]
        preprocess_path = artifact.path.parent / "preprocess.json"
        preprocess = (
            json.loads(preprocess_path.read_text()) if preprocess_path.exists()
            else {"imgsz": 640, "conf": 0.001, "iou": 0.7, "batch": 1, "rect": False}
        )
        onnx_model = YOLO(str(artifact.path), task="detect")
        # Same imgsz/conf/iou/batch/rect training's own (re-)validation used
        # (see module docstring): batch MUST be 1 here regardless of what
        # preprocess.json says, because a static-shaped ONNX export (the only
        # kind this backend produces) cannot run any other batch size --
        # ultralytics' val() would otherwise silently fall back to batch=1
        # per-image anyway, so asserting it is the honest version of that.
        if preprocess.get("batch", 1) != 1:
            raise ValueError(
                f"preprocess.json says batch={preprocess['batch']}, but a static ONNX export only runs batch=1"
            )
        val = onnx_model.val(
            data=str(data_yaml), device="cpu", verbose=False, plots=False, split="val",
            imgsz=preprocess["imgsz"], conf=preprocess["conf"], iou=preprocess["iou"],
            half=False, batch=1, rect=preprocess.get("rect", False),
        )
        map50 = float(val.results_dict.get("metrics/mAP50(B)", 0.0))
        n = _count_val_images(data_yaml)

        # matches_training is the engine's call (Track E1's `run` loop holds
        # TrainResult.metrics; this method's signature does not), so it is
        # left None here rather than guessed at with a side-channel.
        return RuntimeEval(
            format="onnx", runtime="onnxruntime", metric="map50",
            value=map50, n=n, matches_training=None, reason=None,
        )

    def predict(self, artifact: Artifact, inputs: Any) -> Any:
        from PIL import Image
        from ultralytics import YOLO

        key = str(artifact.path)
        if key not in _LOADED:  # once per process (`serve`)
            _LOADED[key] = YOLO(key, task="detect")
        onnx_model = _LOADED[key]
        source = inputs
        if isinstance(inputs, dict) and "file" in inputs:
            # CLI --file / the hub's multipart try-it (CONTRACTS §3.1).
            source = str(inputs["file"])
        if isinstance(inputs, (bytes, bytearray)):
            import io

            source = Image.open(io.BytesIO(inputs))
        results = onnx_model.predict(source=source, device="cpu", verbose=False)
        r = results[0]
        h, w = r.orig_shape
        boxes = []
        names = r.names
        for box in r.boxes:
            cls = int(box.cls.item())
            boxes.append({
                "label": names[cls],
                "score": float(box.conf.item()),
                "xyxy": [float(x) for x in box.xyxy[0].tolist()],
            })
        return {"boxes": boxes, "width": int(w), "height": int(h)}


_LOADED: dict[str, Any] = {}

BACKEND = UltralyticsBackend()
