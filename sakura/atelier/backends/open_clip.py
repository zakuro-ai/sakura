"""open_clip backend: image regression on a SigLIP2 image tower, Apache-2.0.

Implements the ``Backend`` protocol (docs/atelier/CONTRACTS.md §2.2) for task
``image_regression`` / model id ``siglip2_b16_256`` (open_clip
``ViT-B-16-SigLIP2-256``, ``webli`` weights). The image tower is fully
fine-tuned with a linear head on the standardised target (the recipe of the
bulkorcut body-fat study: photo-only MAE 2.36 frozen -> 2.07 fine-tuned).

Data convention (``data.py``'s ``_materialise_image_table``): ``data.train`` /
``data.validation`` are DataFrames with one row per image -- the spec's image
column holding an absolute path, its output column a float.

One preprocessing path everywhere: ``to_array`` (PIL + numpy, driven by the
model's own open_clip preprocess config, recorded in ``preprocess.json``)
feeds training-time validation, the ONNX ``runtime_eval`` and ``predict``, so
the torch and ONNX scores differ only by the forward pass. The exported graph
takes normalised ``pixel_values`` and returns the prediction in the target's
own units (the standardisation is folded into the graph).

``runtime_eval`` protocol note (same as ``hf_detection.py``): this method is
not passed the ``TrainResult``, so ``matches_training`` is left ``None`` and the
engine's ``run`` loop applies the §2.3 tolerance against ``metrics["mae"]``.

Resume: ``resume_from`` may name a checkpoint directory this backend wrote, or
a single ``.pt`` holding ``{"state", "mu", "sd"}`` with ``backbone.visual.*`` /
``head.*`` keys (the bulkorcut ``finetune.py`` format), so an already
fine-tuned tower can be the starting point.

Licence: SigLIP2 weights (google, via open_clip/timm) are Apache-2.0;
open_clip is MIT, timm Apache-2.0.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any, ClassVar, cast

from sakura.atelier.spec import features_of
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

#: model id -> (open_clip architecture, pretrained tag)
ARCHS = {"siglip2_b16_256": ("ViT-B-16-SigLIP2-256", "webli")}


# --------------------------------------------------------------------- pure
# (numpy/PIL only -- unit tested without torch/open_clip)

def to_array(img: Any, cfg: dict[str, Any]) -> Any:
    """PIL image -> float32 CHW array, exactly as the model was trained:
    open_clip's ``squash`` (resize to a square) or ``shortest`` (resize the
    short side, centre crop) mode, then ``(x/255 - mean) / std``."""
    import numpy as np
    from PIL import Image

    size = int(cfg["size"])
    resample = Image.Resampling.BICUBIC if cfg.get("interpolation", "bicubic") == "bicubic" else Image.Resampling.BILINEAR
    img = img.convert("RGB")
    if cfg.get("resize_mode", "shortest") == "squash":
        img = img.resize((size, size), resample)
    else:
        w, h = img.size
        scale = size / min(w, h)
        img = img.resize((max(size, round(w * scale)), max(size, round(h * scale))), resample)
        w, h = img.size
        left, top = (w - size) // 2, (h - size) // 2
        img = img.crop((left, top, left + size, top + size))
    x = np.asarray(img, dtype=np.float32) / 255.0
    mean, std = np.asarray(cfg["mean"], dtype=np.float32), np.asarray(cfg["std"], dtype=np.float32)
    return ((x - mean) / std).transpose(2, 0, 1)


def regression_metrics(pred: Any, target: Any) -> dict[str, float]:
    import numpy as np

    err = np.asarray(pred, dtype=np.float64) - np.asarray(target, dtype=np.float64)
    return {"mae": float(np.abs(err).mean()), "rmse": float(np.sqrt((err ** 2).mean()))}


def warmup_cosine(step: int, warm: int, total: int) -> float:
    """LR multiplier: linear warmup over ``warm`` steps, then cosine to 0 (the engine's shared schedule)."""
    from sakura.atelier.lr_schedule import lr_factor

    return lr_factor(step, total, "cosine", warm, 0.0)


def remap_state(state: dict[str, Any]) -> dict[str, Any]:
    """bulkorcut ``finetune.py`` keys (``backbone.visual.*``) -> this module's
    (``visual.*``); this module's own keys pass through."""
    return {k.replace("backbone.visual.", "visual.", 1): v for k, v in state.items()}


def _artifact(path: Path, fmt: str = "onnx") -> Artifact:
    data = path.read_bytes()
    return Artifact(path=path, format=fmt, sha256=hashlib.sha256(data).hexdigest(), bytes=len(data))


# ------------------------------------------------------------- not-so-pure
# (torch/open_clip below; lazily imported per CONTRACTS §2)

def _device() -> str:
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


def _build(model_id: str) -> tuple[Any, dict[str, Any]]:
    import open_clip
    import torch
    import torch.nn as nn

    arch, tag = ARCHS[model_id]
    clip = open_clip.create_model(arch, pretrained=tag)
    pc = open_clip.get_model_preprocess_cfg(clip)
    size = pc["size"]
    cfg = {"size": int(size[0] if isinstance(size, (list, tuple)) else size),
           "mean": list(pc["mean"]), "std": list(pc["std"]),
           "interpolation": pc.get("interpolation", "bicubic"),
           "resize_mode": pc.get("resize_mode", "shortest")}

    class Net(nn.Module):
        def __init__(self, visual: Any) -> None:
            super().__init__()
            self.visual = visual
            with torch.no_grad():
                dim = visual(torch.zeros(1, 3, cfg["size"], cfg["size"])).shape[1]
            self.head = nn.Linear(dim, 1)

        def forward(self, x: Any) -> Any:
            return self.head(self.visual(x).float()).squeeze(1)

    return Net(clip.visual), cfg


class _Images:
    """torch Dataset over (path, standardised target) rows."""

    def __init__(self, paths: list[str], targets: list[float], cfg: dict[str, Any], augment: bool) -> None:
        from torchvision import transforms as T

        self.paths, self.targets, self.cfg = paths, targets, cfg
        size = cfg["size"]
        self.aug = T.Compose([
            T.RandomResizedCrop(size, scale=(0.85, 1.0), ratio=(0.9, 1.1)),
            T.RandomHorizontalFlip(),
            T.ColorJitter(0.2, 0.2, 0.1),
        ]) if augment else None

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, i: int) -> Any:
        import torch
        from PIL import Image

        img = Image.open(self.paths[i]).convert("RGB")
        if self.aug is not None:
            img = self.aug(img)
        return torch.from_numpy(to_array(img, self.cfg)), torch.tensor(self.targets[i], dtype=torch.float32)


def _loader(paths: list[str], targets: list[float], cfg: dict[str, Any], batch: int,
            augment: bool, shuffle: bool, seed: int) -> Any:
    import torch
    from torch.utils.data import DataLoader

    g = torch.Generator().manual_seed(seed)
    return DataLoader(cast(Any, _Images(paths, targets, cfg, augment)), batch_size=batch, shuffle=shuffle,
                      drop_last=shuffle and len(paths) > batch, generator=g,
                      num_workers=min(8, os.cpu_count() or 1))


def _predict_torch(net: Any, paths: list[str], cfg: dict[str, Any], mu: float, sd: float,
                   batch: int, autocast: bool) -> list[float]:
    import torch

    net.eval()
    device = next(net.parameters()).device
    out: list[float] = []
    with torch.no_grad():
        for x, _ in _loader(paths, [0.0] * len(paths), cfg, batch, False, False, 0):
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=autocast and device.type == "cuda"):
                y = net(x.to(device)).float()
            out.extend((y * sd + mu).cpu().tolist())
    return out


def _load_resume(net: Any, resume: Checkpoint) -> tuple[float, float]:
    import torch

    path = resume.path / "model.pt" if resume.path.is_dir() else resume.path
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    if resume.path.is_dir():
        meta = json.loads((resume.path / "meta.json").read_text())
        if meta.get("task") != "image_regression":
            raise ValueError(f"checkpoint is for task {meta.get('task')!r}, not image_regression")
    missing, _unexpected = net.load_state_dict(remap_state(ckpt["state"]), strict=False)
    if missing:
        raise ValueError(f"checkpoint {path} is missing weights for {missing[:5]}...")
    return float(ckpt["mu"]), float(ckpt["sd"])


class OpenClipBackend:
    name = "open_clip"
    tasks = frozenset({"image_regression"})
    licence = LICENSE

    #: Set below, once the methods exist (see NOTEBOOK_CELLS at the end of the module).
    NOTEBOOK_CELLS: ClassVar[tuple[Any, ...]]

    def train(self, spec: ResolvedSpec, data: MaterialisedData, out: RunDir,
              metrics: MetricsSink, resume: Checkpoint | None) -> TrainResult:
        import numpy as np
        import torch
        import torch.nn as nn

        out.ensure()
        hp = spec.hyperparameters
        epochs = int(hp.get("epochs", 10))
        batch = int(hp.get("batch_size", 64))
        lr = float(hp.get("learning_rate", 2e-5))
        img_col = features_of(spec.spec).inputs[0].name
        out_col = features_of(spec.spec).output.name

        net, cfg = _build(spec.spec.model)
        train_paths = [str(p) for p in data.train[img_col]]
        train_y = np.asarray(data.train[out_col], dtype=np.float64)
        val_paths = [str(p) for p in data.validation[img_col]]
        val_y = np.asarray(data.validation[out_col], dtype=np.float64)
        if resume is not None:
            mu, sd = _load_resume(net, resume)  # the head was trained on this scale: keep it
        else:
            mu, sd = float(train_y.mean()), float(train_y.std() or 1.0)

        device = _device()
        net.to(device)
        dl = _loader(train_paths, list((train_y - mu) / sd), cfg, batch,
                     bool(hp.get("augment", True)), True, spec.spec.seed)
        head_lr = lr * float(hp.get("head_lr_mult", 50))
        opt = torch.optim.AdamW([{"params": net.visual.parameters(), "lr": lr},
                                 {"params": net.head.parameters(), "lr": head_lr}],
                                weight_decay=float(hp.get("weight_decay", 0.05)))
        total = max(epochs * len(dl), 1)
        warm = int(float(hp.get("warmup_epochs", 1)) * len(dl))
        sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: warmup_cosine(s, warm, total))
        loss_fn = nn.SmoothL1Loss() if hp.get("loss", "smooth_l1") == "smooth_l1" else nn.MSELoss()
        clip = float(hp.get("grad_clip", 1.0))
        cuda = device == "cuda"

        t0 = time.time()
        step = 0
        watch = WrapUpWatch(out.root, max_steps_of(spec))
        for epoch in range(epochs):
            net.train()
            for i, (x, y) in enumerate(dl):
                if watch.stop(epoch + i / len(dl), step):
                    break  # wrap-up: the final fp32 score below covers the partial epoch
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=cuda):
                    loss = loss_fn(net(x.to(device)).float(), y.to(device))
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), clip)
                opt.step()
                sched.step()
                metrics.log("loss", float(loss), step=step, split="train", epoch=epoch + i / len(dl))
                step += 1
            m = regression_metrics(_predict_torch(net, val_paths, cfg, mu, sd, batch, True), val_y)
            metrics.log("mae", m["mae"], step=step, split="validation", epoch=float(epoch + 1))
            metrics.log("rmse", m["rmse"], step=step, split="validation", epoch=float(epoch + 1))
            if watch.triggered:
                break
        train_seconds = time.time() - t0

        # Final score in fp32, no autocast: the number the fp32 ONNX re-score is held to (§2.3).
        final = regression_metrics(_predict_torch(net, val_paths, cfg, mu, sd, batch, False), val_y)
        final["n_validation"] = float(len(val_paths))

        ckpt_dir = out.checkpoint
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        torch.save({"state": net.state_dict(), "mu": mu, "sd": sd, "cfg": cfg,
                    "model": spec.spec.model}, ckpt_dir / "model.pt")
        (ckpt_dir / "meta.json").write_text(json.dumps({"model": spec.spec.model, "task": "image_regression"}))
        preprocess = {**cfg, "input": "pixel_values", "output": "prediction", "target": out_col,
                      "layout": "NCHW", "dtype": "float32"}
        (ckpt_dir / "preprocess.json").write_text(json.dumps(preprocess))
        net.to("cpu")
        return TrainResult(model={"net": net, "mu": mu, "sd": sd, "preprocess": preprocess},
                           metrics=final, train_seconds=train_seconds,
                           checkpoint=Checkpoint(path=ckpt_dir, model=spec.spec.model, task="image_regression"),
                           stopped_early=watch.stopped_early)

    def export(self, result: TrainResult, fmt: str, out: RunDir) -> Artifact:
        if fmt != "onnx":
            raise ValueError(f"open_clip backend only exports 'onnx', got {fmt!r}")
        import torch
        import torch.nn as nn

        out.ensure()
        net, mu, sd, pre = (result.model[k] for k in ("net", "mu", "sd", "preprocess"))
        net.eval()

        class _Unscaled(nn.Module):  # prediction in the target's own units
            def __init__(self, inner: Any) -> None:
                super().__init__()
                self.inner = inner

            def forward(self, pixel_values: Any) -> Any:
                return self.inner(pixel_values) * sd + mu

        dest = out.artifacts / "model.onnx"
        size = int(pre["size"])
        torch.onnx.export(_Unscaled(net).eval(), (torch.zeros(1, 3, size, size),), str(dest),
                          input_names=["pixel_values"], output_names=["prediction"],
                          dynamic_axes={"pixel_values": {0: "batch"}, "prediction": {0: "batch"}},
                          opset_version=17, do_constant_folding=True)
        (out.artifacts / "preprocess.json").write_text(json.dumps(pre))
        return _artifact(dest)

    def runtime_eval(self, artifact: Artifact, data: MaterialisedData) -> RuntimeEval:
        import numpy as np
        from PIL import Image

        from sakura.atelier.ort import inference_session

        pre = json.loads((artifact.path.parent / "preprocess.json").read_text())
        session = inference_session(artifact.path, providers=["CPUExecutionProvider"])
        cols = list(data.validation.columns)
        paths, target = list(data.validation[cols[0]]), data.validation[cols[-1]]
        pred: list[float] = []
        for i in range(0, len(paths), 32):
            x = np.stack([to_array(Image.open(p), pre) for p in paths[i:i + 32]])
            pred.extend(session.run(["prediction"], {"pixel_values": x})[0].tolist())
        m = regression_metrics(pred, target)
        return RuntimeEval(format="onnx", runtime="onnxruntime", metric="mae",
                           value=m["mae"], n=len(paths), matches_training=None, reason=None)

    def predict(self, artifact: Artifact, inputs: Any) -> Any:
        import io

        import numpy as np
        from PIL import Image

        from sakura.atelier.ort import inference_session

        key = str(artifact.path)
        if key not in _LOADED:  # once per process (`serve`)
            pre = json.loads((artifact.path.parent / "preprocess.json").read_text())
            _LOADED[key] = (inference_session(key, providers=["CPUExecutionProvider"]), pre)
        session, pre = _LOADED[key]
        if isinstance(inputs, dict) and "file" in inputs:
            inputs = str(inputs["file"])
        if isinstance(inputs, (bytes, bytearray)):
            img = Image.open(io.BytesIO(inputs))
        elif isinstance(inputs, str):
            img = Image.open(inputs)
        else:
            img = inputs
        y = session.run(["prediction"], {"pixel_values": to_array(img, pre)[None]})[0]
        return {"prediction": float(np.asarray(y).reshape(-1)[0])}


_LOADED: dict[str, Any] = {}

#: Functions the notebook shows by source, in pipeline order (CONTRACTS §2.5): helpers before
#: `train` are the Model section, those after `train`, `export` and `runtime_eval` follow that step.
NOTEBOOK_CELLS = (
    _build,
    OpenClipBackend.train, _loader, _predict_torch,
    OpenClipBackend.export,
    OpenClipBackend.runtime_eval, to_array, regression_metrics,
)
OpenClipBackend.NOTEBOOK_CELLS = NOTEBOOK_CELLS

BACKEND = OpenClipBackend()
