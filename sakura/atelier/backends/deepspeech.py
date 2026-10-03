"""DeepSpeech2 backend: speech recognition trained from scratch with CTC.

Implements the ``Backend`` protocol (CONTRACTS §2.2) for task ``speech_recognition``,
model id ``deepspeech2``. The model, data pipeline and trainer live in the ``asr-deepspeech``
package; the trainer runs on the Sakura 1.0 runtime (mixed precision, evaluation overlapped
with training, checkpoint writes in a worker process). ``asr_deepspeech`` is imported lazily
inside methods -- ``import sakura.atelier`` must work without it (backends live in their own
venv, ``docs/atelier/envs/deepspeech.txt``).

Data convention (``data.format: asr_manifest``, see ``sakura.atelier.data``):
  ``data.train`` / ``data.validation`` = DataFrames with the spec's audio and transcript
  columns (plus ``offset`` / ``size`` for tar-shard clips), sorted by duration;
  ``data.labels`` = CTC alphabet, blank (``_``) first; ``data.extra["audio_root"]`` = the
  directory clip references resolve against.

Metrics are streamed per epoch to the ``MetricsSink``: ``loss`` (train) and ``cer`` / ``wer``
(validation) as fractions in [0, 1], like every other backend's metrics. Evaluation may be
overlapped with training, so a validation point can be logged one epoch late (it carries the
epoch it measures).

Exports
-------
- ``safetensors``: ``model.safetensors`` (weights) next to ``model_config.json`` (constructor
  arguments), ``labels.json`` (CTC alphabet) and ``preprocess.json`` (everything needed to turn
  a 16 kHz mono waveform into the model's input). ``runtime_eval`` rebuilds the network from
  these four files alone and re-scores the validation split, so an export that cannot stand
  on its own fails the report instead of shipping.

Resume: ``resume_from`` points at a previous run's ``checkpoint/``; the rolling checkpoint is
restored and training continues for ``hyperparameters.epochs`` *more* epochs.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
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

LICENSE = "MIT (asr-deepspeech)"

#: Constructor arguments persisted with an export (everything DeepSpeech's __init__ needs
#: besides the audio config and the label file).
_MODEL_KEYS = ("rnn_type", "rnn_hidden_size", "rnn_hidden_layers", "bidirectional", "context")


@dataclass
class _Fitted:
    model: Any  # DeepSpeech with the best weights, on CPU
    vocab: list[str]
    model_kwargs: dict[str, Any]
    preprocess: dict[str, Any]
    best_epoch: int | None


def _device() -> str:
    return os.environ.get("SAKURA_ATELIER_DEVICE") or "auto"


def _audio_conf(pre: dict[str, Any]) -> SimpleNamespace:
    return SimpleNamespace(
        sample_rate=int(pre["sample_rate"]),
        window_size=float(pre["window_size"]),
        window_stride=float(pre["window_stride"]),
        window=pre.get("window", "hamming"),
        speed_volume_perturb=False,
        spec_augment=bool(pre.get("spec_augment", False)),
        noise_dir=None,
        noise_prob=0.4,
        noise_levels=(0.0, 0.5),
    )


def _build_model(vocab: list[str], model_kwargs: dict[str, Any], pre: dict[str, Any]) -> Any:
    """A DeepSpeech whose alphabet is exactly ``vocab`` (the CTC blank is index 0).

    The alphabet is passed as a list, never through a label CSV: a CSV cannot represent a
    space (a whitespace-only row is dropped), which silently shrinks the output layer so that
    every target containing a space indexes past it -- undefined CTC behaviour, no error."""
    from asr_deepspeech.modules.deepspeech import DeepSpeech

    model = DeepSpeech(audio_conf=_audio_conf(pre), decoder=None, labels=vocab, **model_kwargs)
    if model.num_classes != len(vocab):
        raise RuntimeError(
            f"model has {model.num_classes} output classes for a {len(vocab)}-symbol alphabet")
    return model


class _MetricsCallbacks:
    """Streams the trainer's per-epoch events into ``metrics.jsonl``."""

    def __init__(self, sink: MetricsSink) -> None:
        self._sink = sink
        self.last_loss: float | None = None

    def on_train_epoch(self, epoch: int, train_loss: float, seconds: float) -> None:
        self.last_loss = float(train_loss)
        self._sink.log("loss", train_loss, step=epoch, split="train", epoch=float(epoch))

    def on_eval(self, epoch: int, wer: float, cer: float) -> None:
        self._sink.log("cer", cer / 100.0, step=epoch, split="validation", epoch=float(epoch))
        self._sink.log("wer", wer / 100.0, step=epoch, split="validation", epoch=float(epoch))


class DeepSpeechBackend:
    name = "deepspeech"
    tasks = frozenset({"speech_recognition"})
    #: The engine's ``torch.use_deterministic_algorithms(True)`` would move CTC loss onto cuDNN,
    #: whose backward returns NaN for a batch with an unalignable sample (it ignores
    #: ``zero_infinity``): on Zambezi Voice every optimizer step was skipped and the loss froze.
    #: Seeds and cuDNN determinism still apply; CTC's native CUDA backward is not bit-exact.
    deterministic_algorithms = False

    # ------------------------------------------------------------------ train

    def train(
        self,
        spec: ResolvedSpec,
        data: MaterialisedData,
        out: RunDir,
        metrics: MetricsSink,
        resume: Checkpoint | None,
    ) -> TrainResult:
        import torch
        from torch.nn import CTCLoss
        from torch.optim.lr_scheduler import StepLR

        from asr_deepspeech.checkpoint import load_checkpoint
        from asr_deepspeech.data.dataset import ManifestDataset, manifest_loader
        from asr_deepspeech.trainers import DeepSpeechTrainer

        hp = dict(spec.hyperparameters)
        hp.update(spec.spec.backend_config)  # deep dive at will (recorded in resolved.yaml)
        if not data.labels or data.labels[0] != "_":
            raise ValueError("data.labels must be a CTC alphabet with the blank '_' first")
        vocab = list(data.labels)
        labels_map = {c: i for i, c in enumerate(vocab)}
        audio_col = spec.spec.features.inputs[0].name
        text_col = spec.spec.features.output.name
        root = data.extra["audio_root"]

        pre = {
            "sample_rate": hp["sample_rate"],
            "window_size": hp["window_size"],
            "window_stride": hp["window_stride"],
            "window": "hamming",
            "normalize": True,
            "spec_augment": bool(hp.get("spec_augment", False)),
            "mono": True,
        }
        model_kwargs = {k: hp[k] for k in _MODEL_KEYS if k in hp}
        model = _build_model(vocab, model_kwargs, pre)
        conf = _audio_conf(pre)

        def _loader(df: Any, *, shuffle: bool) -> Any:
            ds = ManifestDataset(
                df, labels_map, conf, root=root, normalize=True,
                spec_augment=conf.spec_augment and shuffle, audio_col=audio_col, text_col=text_col,
            )
            loader, _ = manifest_loader(
                ds, batch_size=int(hp["batch_size"]), num_workers=int(hp.get("num_workers", 0)),
                shuffle=shuffle,
            )
            return loader

        train_loader = _loader(data.train, shuffle=True)
        val_loader = _loader(data.validation, shuffle=False)

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=float(hp["learning_rate"]),
            betas=(float(hp.get("betas", (0.9, 0.999))[0]), float(hp.get("betas", (0.9, 0.999))[1])),
            weight_decay=float(hp.get("weight_decay", 0.0)),
        )
        scheduler = StepLR(optimizer, step_size=int(hp.get("lr_step", 10)), gamma=float(hp.get("lr_gamma", 1.0)))

        rolling = out.checkpoint / "rolling"
        epochs = int(hp["epochs"])
        if resume is not None:
            src = resume.path if resume.path.is_dir() else resume.path.parent
            if (src / "rolling").is_dir():
                shutil.copytree(src / "rolling", rolling, dirs_exist_ok=True)
            newest = sorted(rolling.glob("epoch_*.pt"))
            if not newest:
                raise ValueError(f"resume_from {resume.path} holds no rolling checkpoint")
            # `epochs` is "this many more": the trainer's own bound is a total
            epochs += int(load_checkpoint(newest[-1])["epoch"]) + 1

        callbacks = _MetricsCallbacks(metrics)
        trainer = DeepSpeechTrainer(
            model,
            CTCLoss(reduction="sum", zero_infinity=True),
            optimizer,
            scheduler,
            epochs=epochs,
            model_path=str(out.checkpoint / "best.pth"),
            output_file=str(out.root / "eval_output.txt"),
            checkpoint_path=str(rolling),
            device=_device(),
            device_test=_device(),
            mixed_precision=bool(hp.get("mixed_precision", True)),
            runtime="sakura",
            async_eval=bool(hp.get("async_eval", True)),
            dispatch=str(hp.get("dispatch", "thread")),
            rolling_checkpoints=True,
            checkpoint_every_s=hp.get("checkpoint_every_s"),
            stop_cer=hp.get("stop_cer"),
            max_grad_norm=hp.get("max_grad_norm"),
            seed=spec.spec.seed,
            callbacks=callbacks,
        )
        t0 = time.time()
        run_metrics = trainer.run(train_loader, val_loader)
        train_seconds = time.time() - t0
        if run_metrics.best_cer is None:
            # Evaluations that fail inside the trainer are logged and skipped so a flaky
            # metric cannot kill a long run; but a run that never scored anything is not done.
            raise RuntimeError(
                "training finished without a single completed validation evaluation "
                f"(skipped batches: {run_metrics.skipped_batches}); see the log for the evaluation error")

        best = out.checkpoint / "best.pth"
        if best.exists():
            ckpt = load_checkpoint(best)
            model.load_state_dict(ckpt["state_dict"])
        model = model.cpu().eval()
        final: dict[str, float] = {}
        if run_metrics.best_cer is not None:
            best_rec: dict[str, Any] = next(
                (r for r in run_metrics.history if r["epoch"] == run_metrics.best_epoch), {})
            final["cer"] = run_metrics.best_cer / 100.0
            if "wer" in best_rec:
                final["wer"] = best_rec["wer"] / 100.0
        if callbacks.last_loss is not None:
            final["loss"] = callbacks.last_loss
        return TrainResult(
            model=_Fitted(model, vocab, model_kwargs, pre, run_metrics.best_epoch),
            metrics=final,
            train_seconds=train_seconds,
            checkpoint=Checkpoint(path=out.checkpoint, model=spec.spec.model, task=spec.spec.task),
        )

    # ----------------------------------------------------------------- export

    def export(self, result: TrainResult, fmt: str, out: RunDir) -> Artifact:
        if fmt != "safetensors":
            raise ValueError(f"deepspeech2 exports only 'safetensors', not {fmt!r}")
        from safetensors.torch import save_file

        fitted: _Fitted = result.model
        out.artifacts.mkdir(parents=True, exist_ok=True)
        tensors = {k: v.detach().cpu().contiguous().clone() for k, v in fitted.model.state_dict().items()}
        path = out.artifacts / "model.safetensors"
        save_file(tensors, str(path))
        (out.artifacts / "model_config.json").write_text(
            json.dumps(fitted.model_kwargs, sort_keys=True, indent=2), encoding="utf-8")
        (out.artifacts / "labels.json").write_text(
            json.dumps(fitted.vocab, ensure_ascii=False, indent=2), encoding="utf-8")
        (out.artifacts / "preprocess.json").write_text(
            json.dumps({**fitted.preprocess, "tokenizer": "characters", "blank": "_",
                        "decoder": "greedy_ctc"}, sort_keys=True, indent=2), encoding="utf-8")
        return _artifact(path, "safetensors")

    # ----------------------------------------------------------- runtime_eval

    def runtime_eval(self, artifact: Artifact, data: MaterialisedData) -> RuntimeEval:
        """Rebuild the network from the exported files alone and re-score validation."""
        from asr_deepspeech.data.dataset import ManifestDataset, manifest_loader
        from asr_deepspeech.trainers.deepspeech_trainer import evaluate

        try:
            model, pre = _load_export(artifact.path)
            conf = _audio_conf(pre)
            vocab = json.loads((artifact.path.parent / "labels.json").read_text(encoding="utf-8"))
            labels_map = {c: i for i, c in enumerate(vocab)}
            ds = ManifestDataset(
                data.validation, labels_map, conf, root=data.extra["audio_root"], normalize=True,
                audio_col=data.extra["audio_col"], text_col=data.extra["text_col"],
            )
            loader, _ = manifest_loader(ds, batch_size=16, num_workers=0, shuffle=False)
            res = evaluate(model, loader, _resolve_device(), output_file=None)
        except Exception as exc:  # an export that cannot be re-scored is reported, never dropped
            return RuntimeEval(format="safetensors", runtime="torch+safetensors", metric="cer",
                               value=None, n=len(data.validation), matches_training=None,
                               reason=f"{type(exc).__name__}: {exc}")
        return RuntimeEval(format="safetensors", runtime="torch+safetensors", metric="cer",
                           value=res.cer / 100.0, n=len(data.validation), matches_training=None)

    # ---------------------------------------------------------------- predict

    def predict(self, artifact: Artifact, inputs: Any) -> Any:
        """``{"text": "<transcript>"}`` for a 16 kHz mono wav (path, ``{"file": path}`` or bytes)."""
        import io

        import soundfile as sf
        import torch

        from asr_deepspeech.data.parsers import SpectrogramParser

        if isinstance(inputs, dict):
            inputs = inputs.get("file") or inputs.get("audio") or inputs.get("path")
        model, pre = _load_export(artifact.path)
        src = io.BytesIO(inputs) if isinstance(inputs, (bytes, bytearray)) else str(inputs)
        wave, sr = sf.read(src, dtype="float32")
        if sr != int(pre["sample_rate"]):
            raise ValueError(f"expected {pre['sample_rate']} Hz audio, got {sr} Hz")
        if wave.ndim > 1:
            wave = wave.mean(axis=1)
        spect = SpectrogramParser(_audio_conf(pre), normalize=True).parse_waveform(wave)
        x = spect.unsqueeze(0).unsqueeze(0)
        sizes = torch.tensor([x.size(3)], dtype=torch.int32)
        with torch.no_grad():
            model.eval()
            out, out_sizes = model.forward(x, sizes)
            decoded, _ = model.decoder.decode(out, out_sizes)
        return {"text": decoded[0][0]}


def _resolve_device() -> Any:
    from asr_deepspeech.device import resolve_device

    return resolve_device(_device())


def _load_export(model_path: Path) -> tuple[Any, dict[str, Any]]:
    """Rebuild a DeepSpeech from ``model.safetensors`` + its three JSON siblings."""
    from safetensors.torch import load_file

    d = model_path.parent
    vocab = json.loads((d / "labels.json").read_text(encoding="utf-8"))
    kwargs = json.loads((d / "model_config.json").read_text(encoding="utf-8"))
    pre = json.loads((d / "preprocess.json").read_text(encoding="utf-8"))
    model = _build_model(vocab, kwargs, pre)
    model.load_state_dict(load_file(str(model_path)))
    return model.eval(), pre


def _artifact(path: Path, fmt: str) -> Artifact:
    h = hashlib.sha256()
    n = 0
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
            n += len(chunk)
    return Artifact(path=path, format=fmt, sha256=h.hexdigest(), bytes=n)


BACKEND = DeepSpeechBackend()
