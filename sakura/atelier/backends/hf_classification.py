"""Shared loop of the generic Hugging Face classification backends
(hf_image_classification, hf_audio_classification, hf_text_classification):
fine-tune any ``AutoModelFor{Image,Audio,Sequence}Classification`` repo at a
pinned commit, export ONNX (and safetensors), re-score the ONNX on the
validation split with onnxruntime.

``trust_remote_code`` is never set: a repo whose config needs its own code
(``auto_map``) is refused before any weights are downloaded. transformers /
torch are imported lazily (CONTRACTS §2: importing the engine imports no
training stack)."""

from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from sakura.atelier.recipe import init_config_of, init_of, teacher_of, teacher_source
from sakura.atelier.types import Artifact, Checkpoint, MaterialisedData, MetricsSink, ResolvedSpec, RunDir, RuntimeEval, TrainResult
from sakura.atelier.wrap_up import WrapUpWatch, max_steps_of

LICENSE = "per-base-model"
_MAX_TEXT_TOKENS = 256
_EVAL_CAP = 256


@dataclass(frozen=True)
class Kind:
    """What differs between the three backends."""
    name: str
    task: str
    auto_model: str  # transformers class name
    processor: str  # transformers class name


IMAGE = Kind("hf_image_classification", "image_classification", "AutoModelForImageClassification", "AutoImageProcessor")
AUDIO = Kind("hf_audio_classification", "audio_classification", "AutoModelForAudioClassification", "AutoFeatureExtractor")
TEXT = Kind("hf_text_classification", "text_classification", "AutoModelForSequenceClassification", "AutoTokenizer")


def _config(repo: str, revision: str | None) -> dict[str, Any]:
    from transformers import AutoConfig
    config: dict[str, Any] = AutoConfig.from_pretrained(repo, revision=revision, trust_remote_code=False).to_dict()
    return config


def _random_model(kind: Kind, repo: str, revision: str | None, init_config: dict[str, Any], processor: Any,
                  labels: list[str]) -> Any:
    """``init: random``: the architecture of ``repo`` built from its config with fresh weights (no checkpoint
    is read). ``init_config`` sets config fields first (layers, hidden size, heads), so a small model can be
    trained from scratch; a text model's vocabulary size is the tokenizer's."""
    import transformers

    config = transformers.AutoConfig.from_pretrained(
        repo, revision=revision, trust_remote_code=False, num_labels=len(labels),
        id2label=dict(enumerate(labels)), label2id={name: i for i, name in enumerate(labels)})
    for key, value in init_config.items():
        if not hasattr(config, key):
            raise ValueError(f"init_config.{key}: {type(config).__name__} has no such field")
        setattr(config, key, value)
    if kind is TEXT:
        config.vocab_size = len(processor)
        if getattr(processor, "pad_token_id", None) is not None:
            config.pad_token_id = processor.pad_token_id
    return getattr(transformers, kind.auto_model).from_config(config)


def _load_teacher(kind: Kind, ref: str, labels: list[str], device: str) -> tuple[Any, Any, list[int], str | None]:
    """``(frozen teacher, its processor, the teacher's logit column for each student label, commit)``.

    The teacher is a classifier of the same kind, loaded without remote code; its inputs are prepared by
    its own processor, so it may have another tokenizer or image size than the student. Its label names
    (``config.id2label``) must be the student's, in any order."""
    import transformers

    from sakura.atelier.backends.distill import freeze, label_permutation

    source, revision = teacher_source(ref)
    if _config(source, revision).get("auto_map"):
        raise ValueError(f"teacher {ref!r} needs remote code to load, which L'Atelier does not run")
    processor = getattr(transformers, kind.processor).from_pretrained(source, revision=revision)
    model = getattr(transformers, kind.auto_model).from_pretrained(source, revision=revision, trust_remote_code=False)
    names = model.config.id2label or {}
    columns = label_permutation([str(names[i]) for i in sorted(names)], labels, ref)
    return freeze(model.to(device)), processor, columns, getattr(model.config, "_commit_hash", None)


def _device() -> str:
    import torch
    want = os.environ.get("SAKURA_ATELIER_DEVICE")
    if want:
        return want
    return "cuda" if torch.cuda.is_available() else "cpu"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _read_audio(path: str, rate: int) -> Any:
    import numpy as np
    import soundfile as sf
    wav, sr = sf.read(path, dtype="float32", always_2d=False)
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    if sr != rate:  # linear resample: enough for classification features
        n = int(round(len(wav) * rate / sr))
        wav = np.interp(np.linspace(0, len(wav) - 1, n), np.arange(len(wav)), wav).astype("float32")
    return wav


def _encode(kind: Kind, processor: Any, values: list[str]) -> dict[str, Any]:
    """A batch of model inputs (torch tensors) from file paths or texts."""
    if kind is IMAGE:
        from PIL import Image
        images = [Image.open(v).convert("RGB") for v in values]
        return dict(processor(images=images, return_tensors="pt"))
    if kind is AUDIO:
        rate = int(getattr(processor, "sampling_rate", 16000))
        waves = [_read_audio(v, rate) for v in values]
        return dict(processor(waves, sampling_rate=rate, padding=True, return_tensors="pt"))
    return dict(processor([str(v) for v in values], padding=True, truncation=True,
                          max_length=_MAX_TEXT_TOKENS, return_tensors="pt"))


def _columns(resolved: ResolvedSpec) -> tuple[str, str]:
    f = resolved.spec.features
    if f is None:
        raise ValueError(f"{resolved.spec.task}: the spec names no features (input and output columns)")
    return f.inputs[0].name, f.output.name


def train_classifier(kind: Kind, resolved: ResolvedSpec, data: MaterialisedData, out: RunDir,
                     metrics: MetricsSink, resume: Checkpoint | None) -> TrainResult:
    import torch
    import transformers

    hp = resolved.hyperparameters
    repo, revision = hp.get("base_model"), hp.get("base_revision")
    if not repo:
        raise ValueError(f"{kind.name}: no base_model (use model: hf:<owner>/<repo>@<sha>)")
    if _config(repo, revision).get("auto_map"):
        raise ValueError(f"{repo} needs remote code to load, which L'Atelier does not run")
    in_col, out_col = _columns(resolved)
    labels = list(data.labels or sorted({str(v) for v in data.train[out_col]}))
    label2id = {l: i for i, l in enumerate(labels)}
    source = str(resume.path) if resume is not None else repo
    rev = None if resume is not None else revision
    processor = getattr(transformers, kind.processor).from_pretrained(source, revision=rev)
    # init: random builds the architecture from its config; a resume continues from the checkpoint's weights
    random_init = init_of(hp) == "random" and resume is None
    if random_init:
        model = _random_model(kind, repo, revision, init_config_of(hp), processor, labels)
    else:
        model = getattr(transformers, kind.auto_model).from_pretrained(
            source, revision=rev, num_labels=len(labels), id2label=dict(enumerate(labels)),
            label2id=label2id, ignore_mismatched_sizes=True, trust_remote_code=False)
    device = _device()
    model.to(device)
    # teacher: the loss becomes alpha * T^2 * KL(teacher || student) + (1 - alpha) * cross-entropy
    teacher = teacher_of(hp)
    t_model: Any = None
    t_processor: Any = None
    t_commit: str | None = None
    t_columns: list[int] = []
    if teacher is not None:
        from sakura.atelier.backends.distill import blend, kd_loss
        t_model, t_processor, t_columns, t_commit = _load_teacher(kind, teacher.ref, labels, device)
    epochs = int(hp.get("epochs", 3))
    batch = int(hp.get("batch_size", 8))
    opt = torch.optim.AdamW(model.parameters(), lr=float(hp.get("learning_rate", 5e-5)),
                            weight_decay=float(hp.get("weight_decay", 0.0)))
    train_x = [str(v) for v in data.train[in_col]]
    train_y = [label2id[str(v)] for v in data.train[out_col]]
    gen = torch.Generator().manual_seed(int(resolved.spec.seed or 0))
    start = time.time()
    epoch_metrics = []
    acc = 0.0
    watch = WrapUpWatch(out.root, max_steps_of(resolved))
    per_epoch = max(1, -(-len(train_x) // batch))
    for epoch in range(epochs):
        model.train()
        order = torch.randperm(len(train_x), generator=gen).tolist()
        total = soft_total = 0.0
        for i in range(0, len(order), batch):
            if watch.stop(epoch + (i // batch) / per_epoch, epoch * per_epoch + i // batch):
                break  # wrap-up: the partial epoch is scored below
            idx = order[i:i + batch]
            raw = [train_x[j] for j in idx]
            inputs = {k: v.to(device) for k, v in _encode(kind, processor, raw).items()}
            out_ = model(**inputs, labels=torch.tensor([train_y[j] for j in idx], device=device))
            loss = out_.loss
            if teacher is not None:
                with torch.no_grad():
                    t_inputs = {k: v.to(device) for k, v in _encode(kind, t_processor, raw).items()}
                    t_logits = t_model(**t_inputs).logits[:, t_columns]
                soft = kd_loss(out_.logits, t_logits, teacher.temperature)
                loss = blend(out_.loss, soft, teacher.alpha)
                soft_total += float(soft) * len(idx)
            loss.backward()
            opt.step()
            opt.zero_grad()
            total += float(loss) * len(idx)
        train_loss = total / max(1, len(train_x))
        acc = _accuracy(kind, model, processor, data.validation, in_col, out_col, label2id, device)
        metrics.log("loss", train_loss, step=epoch + 1, split="train")
        if teacher is not None:
            metrics.log("distill_loss", soft_total / max(1, len(train_x)), step=epoch + 1, split="train")
        metrics.log("accuracy", acc, step=epoch + 1, split="validation")
        epoch_metrics.append({"epoch": epoch + 1, "train_loss": train_loss, "validation_accuracy": acc})
        if watch.triggered:
            break
    model.save_pretrained(out.checkpoint)
    processor.save_pretrained(out.checkpoint)
    (out.checkpoint / "labels.json").write_text(json.dumps(labels))
    provenance: dict[str, Any] = {"base_model": repo, "base_revision": revision,
                                  "base_commit": getattr(model.config, "_commit_hash", None) or revision}
    if init_of(hp) == "random":
        # what the report says about the weights: drawn at random, never read from the checkpoint
        provenance.update(init="random", init_config=init_config_of(hp), resumed=resume is not None,
                          parameters=sum(p.numel() for p in model.parameters()))
    extra_report = None
    if teacher is not None:
        # the student and the teacher on the same validation rows, and how often they answer alike
        xs = [str(v) for v in data.validation[in_col]][:_EVAL_CAP]
        ys = [label2id.get(str(v), -1) for v in data.validation[out_col]][:_EVAL_CAP]
        s_pred = _predicted(kind, model, processor, xs, device)
        t_pred = _predicted(kind, t_model, t_processor, xs, device, t_columns)
        n = max(1, len(xs))
        extra_report = {"distillation": {
            "teacher": teacher.ref, "teacher_commit": t_commit, "alpha": teacher.alpha,
            "temperature": teacher.temperature, "metric": "accuracy", "n": len(xs),
            "student_score": sum(int(p == y) for p, y in zip(s_pred, ys)) / n,
            "teacher_score": sum(int(p == y) for p, y in zip(t_pred, ys)) / n,
            "agreement": sum(int(a == b) for a, b in zip(s_pred, t_pred)) / n}}
        provenance["teacher"] = teacher.ref
    return TrainResult(
        model={"model": model, "processor": processor, "labels": labels, "kind": kind.name},
        metrics={"accuracy": acc}, train_seconds=time.time() - start, report=extra_report,
        checkpoint=Checkpoint(path=out.checkpoint, model=resolved.spec.model, task=resolved.spec.task),
        provenance=provenance,
        epoch_metrics=epoch_metrics, stopped_early=watch.stopped_early)


def _accuracy(kind: Kind, model: Any, processor: Any, frame: Any, in_col: str, out_col: str,
              label2id: dict[str, int], device: str) -> float:
    if frame is None or len(frame) == 0:
        return 0.0
    xs = [str(v) for v in frame[in_col]][:_EVAL_CAP]
    ys = [label2id.get(str(v), -1) for v in frame[out_col]][:_EVAL_CAP]
    return sum(int(p == y) for p, y in zip(_predicted(kind, model, processor, xs, device), ys)) / len(xs)


def _predicted(kind: Kind, model: Any, processor: Any, xs: list[str], device: str,
               columns: list[int] | None = None) -> list[int]:
    """The class index the model picks for each input. ``columns`` reorders a teacher's logits into the
    student's label order first."""
    import torch
    model.eval()
    out: list[int] = []
    with torch.no_grad():
        for i in range(0, len(xs), 16):
            inputs = {k: v.to(device) for k, v in _encode(kind, processor, xs[i:i + 16]).items()}
            logits = model(**inputs).logits
            out += (logits if columns is None else logits[:, columns]).argmax(-1).tolist()
    return out


def export_classifier(kind: Kind, result: TrainResult, fmt: str, out: RunDir, sample: str | None) -> Artifact:
    import torch
    h = result.model
    model, processor = h["model"], h["processor"]
    out.artifacts.mkdir(parents=True, exist_ok=True)
    extras: dict[str, Any] | None = None
    pre = out.artifacts / "preprocessor"
    processor.save_pretrained(pre)
    (out.artifacts / "labels.json").write_text(json.dumps(h["labels"]))
    if fmt == "safetensors":
        d = out.artifacts / "model"
        model.save_pretrained(d, safe_serialization=True)
        path = d / "model.safetensors"
    elif fmt in ("onnx", "onnx_int8"):
        if sample is None:
            raise ValueError(f"{kind.name}: no sample to trace the ONNX export with")
        path = out.artifacts / "model.onnx"
        # onnx_int8 reuses the float graph this run's onnx export wrote (never a file an earlier run left in OUT)
        if fmt == "onnx" or h.get("float_onnx") != str(path):
            model.to("cpu").eval()
            inputs = _encode(kind, processor, [sample, sample])
            names = list(inputs)
            dyn = {n: {0: "batch"} for n in names}
            if kind is not IMAGE:
                dyn = {n: {0: "batch", 1: "seq"} for n in names}
            dyn["logits"] = {0: "batch"}

            class _Logits(torch.nn.Module):
                def __init__(self, m: Any) -> None:
                    super().__init__()
                    self.m = m

                def forward(self, *args: Any) -> Any:
                    return self.m(**dict(zip(names, args))).logits

            torch.onnx.export(_Logits(model), tuple(inputs[n] for n in names), str(path), input_names=names,
                              output_names=["logits"], dynamic_axes=dyn, opset_version=17, dynamo=False)
            h["float_onnx"] = str(path)
        if fmt == "onnx_int8":
            # the same graph with its linear weights quantised to int8: its own artifact, re-scored like any other
            from sakura.atelier.ort import quantize_int8
            float_path, path = path, out.artifacts / "model.int8.onnx"
            extras = quantize_int8(float_path, path)
    else:
        raise ValueError(f"{kind.name} cannot export {fmt!r} (exports onnx, onnx_int8, safetensors)")
    return Artifact(path=path, format=fmt, sha256=_sha256(path), bytes=path.stat().st_size, extras=extras)


def runtime_eval_classifier(kind: Kind, artifact: Artifact, data: MaterialisedData, in_col: str,
                            out_col: str) -> RuntimeEval:
    if artifact.format not in ("onnx", "onnx_int8"):
        return RuntimeEval(format=artifact.format, runtime="none", metric="accuracy", value=None, n=0,
                           matches_training=None, reason=f"{artifact.format} is not re-scored")
    import transformers

    from sakura.atelier.ort import inference_session
    processor = getattr(transformers, kind.processor).from_pretrained(artifact.path.parent / "preprocessor")
    labels = json.loads((artifact.path.parent / "labels.json").read_text())
    label2id = {l: i for i, l in enumerate(labels)}
    session = inference_session(artifact.path, providers=["CPUExecutionProvider"])
    frame = data.validation
    xs = [str(v) for v in frame[in_col]][:_EVAL_CAP]
    ys = [label2id.get(str(v), -1) for v in frame[out_col]][:_EVAL_CAP]
    want = {i.name for i in session.get_inputs()}
    right = 0
    for i in range(0, len(xs), 16):
        feed = {k: v.numpy() for k, v in _encode(kind, processor, xs[i:i + 16]).items() if k in want}
        pred = session.run(["logits"], feed)[0].argmax(-1).tolist()
        right += sum(int(p == y) for p, y in zip(pred, ys[i:i + 16]))
    n = len(xs)
    return RuntimeEval(format=artifact.format, runtime="onnxruntime", metric="accuracy",
                       value=(right / n if n else None), n=n, matches_training=None, reason=None)


def predict_classifier(kind: Kind, artifact: Artifact, inputs: Any) -> Any:
    import transformers

    from sakura.atelier.ort import inference_session
    processor = getattr(transformers, kind.processor).from_pretrained(artifact.path.parent / "preprocessor")
    labels = json.loads((artifact.path.parent / "labels.json").read_text())
    session = inference_session(artifact.path, providers=["CPUExecutionProvider"])
    values = inputs if isinstance(inputs, list) else [inputs]
    want = {i.name for i in session.get_inputs()}
    feed = {k: v.numpy() for k, v in _encode(kind, processor, [str(v) for v in values]).items() if k in want}
    logits = session.run(["logits"], feed)[0]
    return [{"label": labels[int(r.argmax())], "scores": dict(zip(labels, map(float, r)))} for r in logits]


class ClassificationBackend:
    """One generic HF classification backend; `kind` picks image, audio or text."""

    def __init__(self, kind: Kind):
        self.kind = kind
        #: Audio only: wav2vec2-family models mask spans with ``hidden[mask] = embed`` (SpecAugment), and
        #: the deterministic CUDA ``index_put`` of torch 2.5 fails on that broadcast ("number of flattened
        #: indices did not match number of elements in the value tensor"), so every GPU fine-tune died on
        #: its first step. Seeds and cuDNN determinism still apply.
        self.deterministic_algorithms = kind is not AUDIO
        self.name = kind.name
        self.tasks = frozenset({kind.task})
        self.licence = LICENSE
        self._sample: str | None = None
        self._cols: tuple[str, str] | None = None

    def train(self, spec: ResolvedSpec, data: MaterialisedData, out: RunDir, metrics: MetricsSink,
              resume: Checkpoint | None) -> TrainResult:
        self._cols = _columns(spec)
        frame = data.validation if data.validation is not None and len(data.validation) else data.train
        self._sample = str(frame[self._cols[0]].iloc[0])
        return train_classifier(self.kind, spec, data, out, metrics, resume)

    def export(self, result: TrainResult, fmt: str, out: RunDir) -> Artifact:
        return export_classifier(self.kind, result, fmt, out, self._sample)

    def runtime_eval(self, artifact: Artifact, data: MaterialisedData) -> RuntimeEval:
        in_col, out_col = self._cols or (data.train.columns[0], data.train.columns[-1])
        return runtime_eval_classifier(self.kind, artifact, data, in_col, out_col)

    def predict(self, artifact: Artifact, inputs: Any) -> Any:
        return predict_classifier(self.kind, artifact, inputs)
