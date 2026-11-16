"""Whisper backend: LoRA fine-tuning for speech recognition.

Implements the ``Backend`` protocol (CONTRACTS §2.2) for task ``speech_recognition`` and the model ids
``whisper-small`` / ``whisper-large-v3-turbo`` (``openai/whisper-*``, Apache-2.0). It is the *accuracy*
route next to ``deepspeech2`` (from scratch, tiny, fast): a pretrained multilingual model adapted with a
LoRA adapter to the user's audio. ``transformers`` / ``peft`` / ``soundfile`` are imported lazily inside
methods: ``import sakura.atelier`` must work without them (backends live in their own venv,
``docs/atelier/envs/whisper.txt``).

Data convention (``data.format: asr_manifest``): ``data.train`` / ``data.validation`` are DataFrames with
the spec's audio and transcript columns (+ ``offset`` / ``size`` / ``sha256`` for tar-shard clips);
``data.extra["audio_root"]`` is the directory clip references resolve against. Clips are read with
``sakura.atelier.audio.read_clip`` (byte span, sha256-verified).

Languages: Whisper has no Lozi or Tonga. ``hyperparameters.language`` selects the decoder's language token;
``sn`` (Shona, a related Bantu language) is the proxy the La Forge Whisper job used. ``null`` lets Whisper
detect the language.

Metrics (fractions, like every backend): ``loss`` (train, per optimizer step), ``loss`` / ``cer`` / ``wer``
(validation, per epoch; greedy decoding, text normalised the same way for reference and hypothesis).

Exports
-------
- ``safetensors``: the LoRA adapter of the best epoch (by validation CER) is merged into the base weights, so
  the artifact is a plain dense checkpoint: ``model.safetensors`` + config + tokenizer/preprocessor files +
  ``atelier_whisper.json`` (language, task, decoding settings). ``runtime_eval`` loads the export from its
  own directory and re-scores validation, so an export that is not standalone fails the report.
- ``ggml`` (added 2026-10-10, PR 2 of "llm/asr jobs on the Atelier engine"): the same merged checkpoint
  converted with whisper.cpp's ``models/convert-h5-to-ggml.py`` and quantised with ``whisper-quantize`` to
  ``artifacts/model.ggml.bin`` -- what the hub's ``asr`` jobs ship today (zak-forge ``adapters/whispercpp.py``
  + ``stages/quantize_whisper.py``). ``hp.ggml_quant`` picks the type: a whisper-quantize name (``q5_0``
  default, ``q8_0``, ``q4_0``, ``q4_1``, ``q5_1``) or a La Forge profile (``WQ5_0`` / ``WQ8_0``, mapped like
  the reference ``PROFILE_TYPES``). Tool location: ``SAKURA_WHISPER_CPP_DIR`` (a whisper.cpp checkout built
  with ``build/bin/whisper-quantize`` and ``build/bin/whisper-cli``) and ``SAKURA_OPENAI_WHISPER_DIR`` (a
  github.com/openai/whisper checkout: the converter packs its ``whisper/assets/mel_filters.npz`` into the
  file) -- the whisper.cpp twins of the peft backend's ``LLAMA_CPP_DIR``. Missing tools raise
  ``GGMLExportError``. ``runtime_eval`` re-scores CER with ``whisper-cli`` on at most
  ``ATELIER_REPLAY_EVAL_ROWS`` (default 256) validation clips, ONE process for all clips; without the
  binaries it returns ``value: None`` with the reason (never raises). The artifact's ``report.json`` entry
  gains ``ggml_quant``, ``footprint_bytes`` (the file's size) and ``export_peak_rss_bytes`` (max RSS over the
  convert + quantise children, from ``getrusage``; null where unmeasurable).

Resume: ``resume_from`` points at a previous run's ``checkpoint/``; its ``adapter/`` is loaded and trained for
``epochs`` MORE epochs (the optimizer state is not carried over).
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar

from sakura.atelier.audio import error_rates, read_clip
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

LICENSE = "Apache-2.0 (openai/whisper)"
SAMPLE_RATE = 16000
MODEL_IDS = {
    "whisper-small": "openai/whisper-small",
    "whisper-large-v3-turbo": "openai/whisper-large-v3-turbo",
}
META = "atelier_whisper.json"
GGML_NAME = "model.ggml.bin"

#: La Forge profile -> whisper-quantize type. Literal copy of ``PROFILE_TYPES`` in zak-forge
#: ``src/forge/runner/adapters/whispercpp.py``, so a hub job quantised by this engine gets the
#: same type the asr runner produced for that profile.
GGML_PROFILES: dict[str, str] = {"WQ5_0": "q5_0", "WQ8_0": "q8_0"}
#: whisper-quantize's own type names (``whisper-quantize --help``; q5_0 is the reference default).
GGML_QUANT_TYPES = ("q4_0", "q4_1", "q5_0", "q5_1", "q8_0")
DEFAULT_GGML_QUANT = "q5_0"
WHISPER_CPP_ENV = "SAKURA_WHISPER_CPP_DIR"
OPENAI_WHISPER_ENV = "SAKURA_OPENAI_WHISPER_DIR"


class GGMLExportError(RuntimeError):
    """whisper.cpp is not where ``SAKURA_WHISPER_CPP_DIR`` says, is not built, or one of its
    tools (convert, quantize, cli) failed -- carries the subprocess's own stderr."""


@dataclass
class _Fitted:
    merged_dir: Path
    settings: dict[str, Any]
    best_epoch: int | None
    ggml_quant: str | None = None  # hp.ggml_quant; None = DEFAULT_GGML_QUANT


def _ggml_quant_type(quant: str | None) -> str:
    """The whisper-quantize type for ``hp.ggml_quant``: a profile (``WQ5_0``) or a bare type
    (``q5_0``, case-insensitive); None = the default."""
    if quant is None or quant == "":
        return DEFAULT_GGML_QUANT
    if quant in GGML_PROFILES:
        return GGML_PROFILES[quant]
    lowered = str(quant).lower()
    if lowered in GGML_QUANT_TYPES:
        return lowered
    raise GGMLExportError(
        f"ggml_quant={quant!r} is neither a whisper-quantize type {GGML_QUANT_TYPES} "
        f"nor a La Forge profile {sorted(GGML_PROFILES)}")


def _whisper_cpp_dir() -> Path:
    d = os.environ.get(WHISPER_CPP_ENV)
    if not d:
        raise GGMLExportError(
            f"{WHISPER_CPP_ENV} is not set -- point it at a whisper.cpp checkout with "
            "models/convert-h5-to-ggml.py, and build/bin/whisper-quantize + whisper-cli")
    return Path(d)


def _openai_whisper_dir() -> Path:
    d = os.environ.get(OPENAI_WHISPER_ENV)
    if not d:
        raise GGMLExportError(
            f"{OPENAI_WHISPER_ENV} is not set -- point it at a checkout of github.com/openai/whisper "
            "(convert-h5-to-ggml.py packs its whisper/assets/mel_filters.npz into the ggml file)")
    return Path(d)


def _whisper_cpp_bin(name: str) -> Path:
    """``build/bin/<name>`` of the whisper.cpp checkout; GGMLExportError when it is not built."""
    path = _whisper_cpp_dir() / "build" / "bin" / name
    if not path.exists():
        raise GGMLExportError(f"{path} not found -- build whisper.cpp first (cmake -B build && cmake --build build)")
    return path


#: The tokenizer files ``convert-h5-to-ggml.py`` opens by name (it builds the ggml vocab from
#: ``vocab.json`` + ``added_tokens.json``, not from ``tokenizer.json``).
CONVERTER_TOKENIZER_FILES = ("vocab.json", "added_tokens.json")


def _write_converter_tokenizer_files(merged_dir: Path) -> list[str]:
    """Make sure ``vocab.json`` and ``added_tokens.json`` sit next to the merged weights.

    ``processor.save_pretrained`` with a fast tokenizer (transformers >= 5) writes only
    ``tokenizer.json``; whisper.cpp's converter then dies with ``FileNotFoundError: vocab.json``
    (staging job 377f97f9). Both files are derived from the saved tokenizer: the base vocabulary
    (every token that is not an added one) and the added tokens, each as ``{token: id}``.
    Returns the names written; existing files are left as they are."""
    missing = [n for n in CONVERTER_TOKENIZER_FILES if not (merged_dir / n).is_file()]
    if not missing:
        return []
    import transformers  # the whisper venv has it; the test fakes the tokenizer

    tok = transformers.AutoTokenizer.from_pretrained(str(merged_dir))
    added = dict(tok.get_added_vocab())
    vocab = {t: i for t, i in tok.get_vocab().items() if t not in added}
    for name, table in (("vocab.json", vocab), ("added_tokens.json", added)):
        if name in missing:
            (merged_dir / name).write_text(json.dumps(table, ensure_ascii=False, indent=2), encoding="utf-8")
    return missing


def _ggml_convert_argv(convert_script: Path, merged_dir: Path, openai_whisper_dir: Path, out_dir: Path) -> list[str]:
    """``convert-h5-to-ggml.py <hf model dir> <openai/whisper checkout> <out dir>``: writes
    ``<out dir>/ggml-model.bin`` in f16 (the reference runner's exact call)."""
    return [sys.executable, str(convert_script), str(merged_dir), str(openai_whisper_dir), str(out_dir)]


def _ggml_quantize_argv(quantize_bin: Path, f16: Path, dest: Path, qtype: str) -> list[str]:
    """``whisper-quantize <f16> <out> <type>``; no importance matrix exists for whisper.cpp."""
    return [str(quantize_bin), str(f16), str(dest), qtype]


def _whisper_cli_argv(cli_bin: Path, model: Path, wavs: list[Path], language: str | None, out_dir: Path,
                      threads: int = 4) -> tuple[list[str], list[Path]]:
    """One ``whisper-cli`` process for every clip (``-f``/``-of`` pairs, ``-otxt``) and the transcript
    files it writes, in clip order -- the reference ``WhisperCpp.transcribe`` call."""
    argv = [str(cli_bin), "-m", str(model), "-l", language or "auto", "-t", str(threads), "-nt", "-np", "-otxt"]
    outputs: list[Path] = []
    for i, wav in enumerate(wavs):
        prefix = out_dir / f"{i:05d}"
        argv += ["-f", str(wav), "-of", str(prefix)]
        outputs.append(out_dir / f"{i:05d}.txt")
    return argv, outputs


def _children_peak_rss_bytes() -> int | None:
    """Max RSS over the subprocesses waited for so far (``getrusage(RUSAGE_CHILDREN)``), in bytes;
    None where the platform does not report it. Cheap: no probe run, read once after the quantise step."""
    try:
        import resource
    except ImportError:  # Windows
        return None
    rss = int(resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss)
    if rss <= 0:
        return None
    return rss if sys.platform == "darwin" else rss * 1024  # macOS reports bytes, Linux kilobytes


def _device() -> str:
    env = os.environ.get("SAKURA_ATELIER_DEVICE")
    if env:
        return "cuda" if env == "auto" else env
    import torch

    if torch.cuda.is_available():
        return "cuda"
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return "mps"
    return "cpu"


class _ClipDataset:
    """Map-style dataset: a row -> (log-mel features, label ids, reference text)."""

    def __init__(self, df: Any, processor: Any, root: str, audio_col: str, text_col: str,
                 max_label_tokens: int) -> None:
        self.rows = df.to_dict("records")
        self.processor, self.root = processor, root
        self.audio_col, self.text_col = audio_col, text_col
        self.max_label_tokens = max_label_tokens

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, i: int) -> dict[str, Any]:
        row = self.rows[i]
        wave = read_clip(self.root, str(row[self.audio_col]), row.get("offset"), row.get("size"),
                         row.get("sha256") or None, SAMPLE_RATE)
        feats = self.processor.feature_extractor(
            wave, sampling_rate=SAMPLE_RATE, return_tensors="np").input_features[0]
        text = str(row[self.text_col])
        ids = self.processor.tokenizer(text).input_ids[: self.max_label_tokens]
        return {"features": feats, "labels": ids, "text": text}


def _collate(start_token: int) -> Callable[[list[dict[str, Any]]], dict[str, Any]]:
    def collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
        import numpy as np
        import torch

        feats = torch.from_numpy(np.stack([b["features"] for b in batch]))
        width = max(len(b["labels"]) for b in batch)
        labels = torch.full((len(batch), width), -100, dtype=torch.long)
        for i, b in enumerate(batch):
            labels[i, : len(b["labels"])] = torch.tensor(b["labels"])
        # the model shifts labels right and prepends the decoder start token itself
        if (labels[:, 0] == start_token).all():
            labels = labels[:, 1:]
        return {"input_features": feats, "labels": labels, "texts": [b["text"] for b in batch]}

    return collate


def _generate_kwargs(settings: dict[str, Any]) -> dict[str, Any]:
    kw: dict[str, Any] = {"task": "transcribe", "max_new_tokens": int(settings["max_new_tokens"]),
                          "num_beams": 1, "do_sample": False}
    if settings.get("language"):
        kw["language"] = settings["language"]
    return kw


def _transcribe(model: Any, processor: Any, features: Any, settings: dict[str, Any], device: str) -> list[str]:
    import torch

    with torch.no_grad():
        ids = model.generate(input_features=features.to(device), **_generate_kwargs(settings))
    return [str(t) for t in processor.batch_decode(ids, skip_special_tokens=True)]


def _score(model: Any, processor: Any, loader: Any, settings: dict[str, Any], device: str,
           max_samples: int | None = None) -> tuple[float, float, float]:
    """(loss, cer, wer) of ``model`` over ``loader`` (greedy decoding)."""
    import torch

    model.eval()
    refs: list[str] = []
    hyps: list[str] = []
    losses: list[float] = []
    for batch in loader:
        with torch.no_grad():
            out = model(input_features=batch["input_features"].to(device), labels=batch["labels"].to(device))
        losses.append(float(out.loss))
        hyps += _transcribe(model, processor, batch["input_features"], settings, device)
        refs += batch["texts"]
        if max_samples is not None and len(refs) >= max_samples:
            break
    wer, cer = error_rates(refs, hyps, bool(settings.get("normalize", True)))
    return (sum(losses) / max(len(losses), 1), cer, wer)


class WhisperBackend:
    name = "whisper"
    tasks = frozenset({"speech_recognition"})

    #: Set below, once the methods exist (see NOTEBOOK_CELLS at the end of the module).
    NOTEBOOK_CELLS: ClassVar[tuple[Any, ...]]

    # ------------------------------------------------------------------ train

    def train(self, spec: ResolvedSpec, data: MaterialisedData, out: RunDir, metrics: MetricsSink,
              resume: Checkpoint | None) -> TrainResult:
        import peft
        import torch
        import transformers
        from torch.utils.data import DataLoader

        # transformers/peft static types are version-dependent; model objects are used as Any
        whisper_cls: Any = transformers.WhisperForConditionalGeneration
        processor_cls: Any = transformers.WhisperProcessor
        out.ensure()
        hp = {**spec.hyperparameters, **spec.spec.backend_config}
        model_id = str(hp.get("base_model") or MODEL_IDS[spec.spec.model])
        device = _device()
        settings = {
            "language": hp.get("language"), "task": "transcribe", "normalize": bool(hp.get("normalize", True)),
            "max_new_tokens": int(hp.get("max_new_tokens", 128)), "sample_rate": SAMPLE_RATE,
            "base_model": model_id,
        }
        audio_col, text_col = data.extra["audio_col"], data.extra["text_col"]

        processor = processor_cls.from_pretrained(model_id)
        processor.tokenizer.set_prefix_tokens(language=settings["language"], task="transcribe",
                                              predict_timestamps=False)
        model: Any = whisper_cls.from_pretrained(model_id)
        model.config.use_cache = False
        if hp.get("gradient_checkpointing"):
            model.gradient_checkpointing_enable()
            model.enable_input_require_grads()

        adapter_dir = out.checkpoint / "adapter"
        if resume is not None:
            src = (resume.path if resume.path.is_dir() else resume.path.parent) / "adapter"
            if not src.is_dir():
                raise ValueError(f"resume_from {resume.path} holds no adapter/")
            model = peft.PeftModel.from_pretrained(model, str(src), is_trainable=True)
        else:
            model = peft.get_peft_model(model, peft.LoraConfig(
                r=int(hp.get("lora_r", 32)), lora_alpha=int(hp.get("lora_alpha", 64)),
                lora_dropout=float(hp.get("lora_dropout", 0.05)), bias="none",
                target_modules=list(hp.get("target_modules", ["q_proj", "v_proj"])),
            ))
        model.to(device)

        start_token = model.config.decoder_start_token_id
        collate = _collate(start_token)
        max_label = int(hp.get("max_label_tokens", 224))

        def loader(df: Any, shuffle: bool, bs: int) -> Any:
            ds: Any = _ClipDataset(df, processor, data.extra["audio_root"], audio_col, text_col, max_label)
            return DataLoader(ds, batch_size=bs, shuffle=shuffle, collate_fn=collate,
                              num_workers=int(hp.get("num_workers", 0)))

        bs = int(hp.get("batch_size", 8))
        accum = int(hp.get("grad_accum", 2))
        train_loader = loader(data.train, True, bs)
        val_loader = loader(data.validation, False, int(hp.get("eval_batch_size", bs)))

        epochs = int(hp.get("epochs", 3))
        steps_per_epoch = max(1, (len(train_loader) + accum - 1) // accum)
        total_steps = steps_per_epoch * epochs
        warmup = int(hp.get("warmup_steps", 50))
        params = [p for p in model.parameters() if p.requires_grad]
        optim = torch.optim.AdamW(params, lr=float(hp.get("learning_rate", 1e-4)),
                                  weight_decay=float(hp.get("weight_decay", 0.0)))
        sched = torch.optim.lr_scheduler.LambdaLR(
            optim, lambda s: min((s + 1) / max(warmup, 1), max(0.0, (total_steps - s) / max(total_steps - warmup, 1))))
        amp = bool(hp.get("fp16", True)) and device == "cuda"
        scaler = torch.amp.GradScaler("cuda", enabled=amp)
        max_norm = float(hp.get("max_grad_norm", 1.0))

        best_cer: float | None = None
        best_epoch: int | None = None
        best_wer = 0.0
        last_loss = 0.0
        step = 0
        watch = WrapUpWatch(out.root, max_steps_of(spec))
        t0 = time.time()
        for epoch in range(epochs):
            model.train()
            optim.zero_grad(set_to_none=True)
            for i, batch in enumerate(train_loader):
                if watch.stop(epoch + i / len(train_loader), step):
                    break  # wrap-up: the partial epoch is scored below, so the best adapter is picked as usual
                with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
                    loss = model(input_features=batch["input_features"].to(device),
                                 labels=batch["labels"].to(device)).loss
                scaler.scale(loss / accum).backward()
                if (i + 1) % accum == 0 or i + 1 == len(train_loader):
                    scaler.unscale_(optim)
                    torch.nn.utils.clip_grad_norm_(params, max_norm)
                    scaler.step(optim)
                    scaler.update()
                    optim.zero_grad(set_to_none=True)
                    sched.step()
                    last_loss = float(loss)
                    metrics.log("loss", last_loss, step=step, split="train", epoch=epoch + (i + 1) / len(train_loader))
                    step += 1
            vloss, cer, wer = _score(model, processor, val_loader, settings, device,
                                     hp.get("eval_max_samples"))
            metrics.log("loss", vloss, step=step, split="validation", epoch=float(epoch + 1))
            metrics.log("cer", cer, step=step, split="validation", epoch=float(epoch + 1))
            metrics.log("wer", wer, step=step, split="validation", epoch=float(epoch + 1))
            if best_cer is None or cer < best_cer:
                best_cer, best_wer, best_epoch = cer, wer, epoch
                model.save_pretrained(str(adapter_dir))  # small; the best adapter is always on disk
            if watch.triggered:
                break
        train_seconds = time.time() - t0

        last_epoch = epoch if watch.triggered else epochs - 1
        if best_epoch is not None and best_epoch != last_epoch:  # restore the best epoch's adapter
            from safetensors.torch import load_file

            peft.set_peft_model_state_dict(model, load_file(str(adapter_dir / "adapter_model.safetensors")))
        merged = model.merge_and_unload()
        merged.config.use_cache = True
        merged_dir = out.checkpoint / "merged"
        merged_dir.mkdir(parents=True, exist_ok=True)
        merged.save_pretrained(str(merged_dir), safe_serialization=True)
        processor.save_pretrained(str(merged_dir))
        (merged_dir / META).write_text(json.dumps(settings, indent=2, sort_keys=True), encoding="utf-8")

        final = {"loss": last_loss}
        if best_cer is not None:
            final.update(cer=best_cer, wer=best_wer)
        ggml_quant = hp.get("ggml_quant")
        return TrainResult(
            model=_Fitted(merged_dir, settings, best_epoch, None if ggml_quant is None else str(ggml_quant)),
            metrics=final, train_seconds=train_seconds,
            checkpoint=Checkpoint(path=out.checkpoint, model=spec.spec.model, task=spec.spec.task),
            stopped_early=watch.stopped_early,
        )

    # ----------------------------------------------------------------- export

    def export(self, result: TrainResult, fmt: str, out: RunDir) -> Artifact:
        if fmt == "ggml":
            return self._export_ggml(result.model, out)
        if fmt != "safetensors":
            raise ValueError(f"whisper exports only 'safetensors' and 'ggml', not {fmt!r}")
        fitted: _Fitted = result.model
        out.artifacts.mkdir(parents=True, exist_ok=True)
        for f in fitted.merged_dir.iterdir():  # weights + config + tokenizer/preprocessor files
            if f.is_file():
                shutil.copy(f, out.artifacts / f.name)
        path = out.artifacts / "model.safetensors"
        if not path.exists():
            raise FileNotFoundError(f"merged checkpoint has no model.safetensors (sharded?): {fitted.merged_dir}")
        return _artifact(path, "safetensors")

    def _export_ggml(self, fitted: _Fitted, out: RunDir) -> Artifact:
        """Merged checkpoint -> f16 ggml (``convert-h5-to-ggml.py``) -> ``whisper-quantize`` at
        ``hp.ggml_quant`` -> ``artifacts/model.ggml.bin``. The f16 intermediate is deleted."""
        whisper_cpp = _whisper_cpp_dir()
        convert_script = whisper_cpp / "models" / "convert-h5-to-ggml.py"
        if not convert_script.exists():
            raise GGMLExportError(f"{convert_script} not found")
        quantize_bin = _whisper_cpp_bin("whisper-quantize")
        openai_whisper = _openai_whisper_dir()
        qtype = _ggml_quant_type(fitted.ggml_quant)  # validated: one bare argv element, no shell

        out.artifacts.mkdir(parents=True, exist_ok=True)
        work = out.checkpoint / "ggml"
        work.mkdir(parents=True, exist_ok=True)
        f16 = work / "ggml-model.bin"  # the converter's fixed output name
        try:
            _write_converter_tokenizer_files(fitted.merged_dir)
        except Exception as exc:  # noqa: BLE001  (the converter's own error names the file)
            logging.getLogger(__name__).warning("could not write %s next to %s: %s",
                                                "/".join(CONVERTER_TOKENIZER_FILES), fitted.merged_dir, exc)
        proc = subprocess.run(_ggml_convert_argv(convert_script, fitted.merged_dir, openai_whisper, work),
                              capture_output=True, text=True, errors="replace", stdin=subprocess.DEVNULL, check=False)
        if proc.returncode != 0:
            raise GGMLExportError(f"convert-h5-to-ggml.py failed: {proc.stderr[-4000:]}")
        if not f16.is_file():
            raise GGMLExportError(f"convert-h5-to-ggml.py wrote no {f16}")

        dest = out.artifacts / GGML_NAME
        proc = subprocess.run(  # nosemgrep: python.lang.security.audit.dangerous-subprocess-use-tainted-env-args.dangerous-subprocess-use-tainted-env-args
            _ggml_quantize_argv(quantize_bin, f16, dest, qtype),
            capture_output=True, text=True, errors="replace", stdin=subprocess.DEVNULL, check=False)
        if proc.returncode != 0:
            raise GGMLExportError(f"whisper-quantize failed: {proc.stderr[-4000:]}")
        if not dest.is_file():
            raise GGMLExportError(f"whisper-quantize wrote no {dest}")
        peak_rss = _children_peak_rss_bytes()
        f16.unlink(missing_ok=True)
        # The decoding settings (language token) travel with the file: runtime_eval and predict read
        # them from here, so a ggml-only export is standalone too.
        meta = fitted.merged_dir / META
        if meta.is_file() and not (out.artifacts / META).exists():
            shutil.copy(meta, out.artifacts / META)
        art = _artifact(dest, "ggml")
        return Artifact(path=art.path, format=art.format, sha256=art.sha256, bytes=art.bytes, extras={
            "ggml_quant": qtype, "footprint_bytes": art.bytes, "export_peak_rss_bytes": peak_rss,
        })

    # ----------------------------------------------------------- runtime_eval

    def runtime_eval(self, artifact: Artifact, data: MaterialisedData) -> RuntimeEval:
        """Load the export from its own directory and re-score the validation split."""
        if artifact.format == "ggml":
            return _rescore_ggml(artifact, data)
        try:
            from torch.utils.data import DataLoader

            model, processor, settings, device = _load_export(artifact.path)
            ds: Any = _ClipDataset(data.validation, processor, data.extra["audio_root"],
                                   data.extra["audio_col"], data.extra["text_col"], 224)
            loader: Any = DataLoader(ds, batch_size=8, shuffle=False,
                                collate_fn=_collate(model.config.decoder_start_token_id))
            _, cer, _wer = _score(model, processor, loader, settings, device)
        except Exception as exc:  # an export that cannot be re-scored is reported, never dropped
            return RuntimeEval(format="safetensors", runtime="transformers", metric="cer", value=None,
                               n=len(data.validation), matches_training=None,
                               reason=f"{type(exc).__name__}: {exc}")
        return RuntimeEval(format="safetensors", runtime="transformers", metric="cer", value=cer,
                           n=len(data.validation), matches_training=None)

    # ---------------------------------------------------------------- predict

    def predict(self, artifact: Artifact, inputs: Any) -> Any:
        """``{"text": ...}`` for a 16 kHz wav (path, ``{"file": path}`` or bytes)."""
        import io

        import soundfile as sf

        if isinstance(inputs, dict):
            inputs = inputs.get("file") or inputs.get("audio") or inputs.get("path")
        src = io.BytesIO(inputs) if isinstance(inputs, (bytes, bytearray)) else str(inputs)
        wave, sr = sf.read(src, dtype="float32")
        if sr != SAMPLE_RATE:
            raise ValueError(f"expected {SAMPLE_RATE} Hz audio, got {sr} Hz")
        if wave.ndim > 1:
            wave = wave.mean(axis=1)
        if artifact.format == "ggml":
            settings = _ggml_settings(artifact.path)
            with tempfile.TemporaryDirectory(prefix="atelier-ggml-") as tmp:
                return {"text": _whisper_cli_transcribe(artifact.path, [wave], settings, Path(tmp))[0]}
        model, processor, settings, device = _load_export(artifact.path)
        feats = processor.feature_extractor(wave, sampling_rate=SAMPLE_RATE, return_tensors="pt").input_features
        return {"text": _transcribe(model, processor, feats, settings, device)[0]}


def _ggml_settings(model_path: Path) -> dict[str, Any]:
    """Decoding settings next to a ggml export (``atelier_whisper.json``); defaults when absent."""
    meta = model_path.parent / META
    settings: dict[str, Any] = {"language": None, "normalize": True}
    if meta.is_file():
        settings.update(json.loads(meta.read_text(encoding="utf-8")))
    return settings


def _whisper_cli_transcribe(model: Path, waves: list[Any], settings: dict[str, Any], work: Path) -> list[str]:
    """Transcripts of 16 kHz ``waves`` through ONE ``whisper-cli`` process on the ggml ``model``."""
    import soundfile as sf

    cli = _whisper_cpp_bin("whisper-cli")
    clips = work / "clips"
    clips.mkdir(parents=True, exist_ok=True)
    wavs: list[Path] = []
    for i, wave in enumerate(waves):
        p = clips / f"{i:05d}.wav"
        sf.write(str(p), wave, SAMPLE_RATE, subtype="PCM_16")
        wavs.append(p)
    threads = _whisper_cli_threads()
    argv, outputs = _whisper_cli_argv(cli, model, wavs, settings.get("language"), work / "out", threads)
    (work / "out").mkdir(parents=True, exist_ok=True)
    # One process runs every clip, so the budget grows with the batch (the reference runner's rule).
    # argv is built by _whisper_cli_argv from our own paths; the only env input is the thread
    # count, coerced through int() above, so nothing user-controlled reaches the shell.
    proc = subprocess.run(argv, capture_output=True, text=True, errors="replace", stdin=subprocess.DEVNULL,  # nosemgrep: python.lang.security.audit.dangerous-subprocess-use-tainted-env-args.dangerous-subprocess-use-tainted-env-args
                          timeout=600 + 120 * len(wavs), check=False)
    if proc.returncode != 0:
        raise GGMLExportError(f"whisper-cli failed (exit {proc.returncode}): {proc.stderr[-4000:]}")
    texts: list[str] = []
    for wav, txt in zip(wavs, outputs):
        if not txt.is_file():
            raise GGMLExportError(f"whisper-cli wrote no transcript for {wav}")
        # whisper-cli writes raw token bytes: a low-resource language can end a
        # clip mid-codepoint (job 0c56d8e1: 0xe1 with no continuation), which must
        # cost a character of CER, not the whole re-score.
        texts.append(" ".join(txt.read_text(encoding="utf-8", errors="replace").split()))
    return texts


def _whisper_cli_threads() -> int:
    """``-t`` for whisper-cli: ``SAKURA_WHISPER_CPP_THREADS`` when set, else every CPU this
    process may run on (cpuset, capped by the cgroup quota) -- the fleet runner sees 3 of a
    32-thread host, and ``min(4, os.cpu_count())`` oversubscribed it."""
    raw = os.environ.get("SAKURA_WHISPER_CPP_THREADS", "").strip()
    if raw:
        try:
            return max(1, int(raw))
        except ValueError:
            pass
    from sakura.atelier import ort

    return ort.allowed_threads()


def _replay_check_rows(n: int) -> int:
    """How many validation clips the ggml re-score runs through whisper-cli.

    The engine's own run scores every clip up to ``ATELIER_REPLAY_EVAL_ROWS``. Under the notebook
    check (``ATELIER_REPLAY=check``) the cell runs the very same re-score a second time, on the
    CPU, inside the agent's 600 s budget: whisper-small encodes a 30 s window per clip (~16 s on
    3 cores), so 12 clips alone overran it (job 0c56d8e1). The check only has to prove the cell
    runs, so it scores ``SAKURA_WHISPER_CPP_CHECK_ROWS`` (default 4) clips; ``n`` says how many."""
    if os.environ.get("ATELIER_REPLAY", "") != "check":
        return n
    try:
        cap = int(os.environ.get("SAKURA_WHISPER_CPP_CHECK_ROWS", "4"))
    except ValueError:
        cap = 4
    return max(1, min(n, cap))


def _rescore_ggml(artifact: Artifact, data: MaterialisedData) -> RuntimeEval:
    """CER of the ggml export through whisper.cpp's own CLI on at most ``ATELIER_REPLAY_EVAL_ROWS``
    validation clips. A missing or failing whisper.cpp is a row with ``value: None`` and the reason,
    never an exception (CONTRACTS §2.1); ``matches_training`` is left to the engine (§2.3, ±0.05)."""
    cap = _replay_check_rows(int(os.environ.get("ATELIER_REPLAY_EVAL_ROWS", "256")))
    rows = data.validation.iloc[:cap] if hasattr(data.validation, "iloc") else data.validation[:cap]
    n = len(rows)
    try:
        settings = _ggml_settings(artifact.path)
        audio_col, text_col = data.extra["audio_col"], data.extra["text_col"]
        refs: list[str] = []
        waves: list[Any] = []
        for row in rows.to_dict("records") if hasattr(rows, "to_dict") else rows:
            waves.append(read_clip(data.extra["audio_root"], str(row[audio_col]), row.get("offset"), row.get("size"),
                                   row.get("sha256") or None, SAMPLE_RATE))
            refs.append(str(row[text_col]))
        with tempfile.TemporaryDirectory(prefix="atelier-ggml-eval-") as tmp:
            hyps = _whisper_cli_transcribe(artifact.path, waves, settings, Path(tmp))
        _wer, cer = error_rates(refs, hyps, bool(settings.get("normalize", True)))
    except Exception as exc:  # noqa: BLE001 -- whisper.cpp absent, unbuilt or failing: reported, never dropped
        return RuntimeEval(format="ggml", runtime="whisper.cpp", metric="cer", value=None, n=0,
                           matches_training=None, reason=f"{type(exc).__name__}: {exc}")
    return RuntimeEval(format="ggml", runtime="whisper.cpp", metric="cer", value=cer, n=n, matches_training=None)


_LOADED: dict[str, Any] = {}


def _load_export(model_path: Path) -> tuple[Any, Any, dict[str, Any], str]:
    """The exported model, processor and decoding settings, rebuilt from the export directory alone."""
    import transformers

    # transformers' static types for from_pretrained are version-dependent; the loaded objects are used as Any
    model_cls: Any = transformers.WhisperForConditionalGeneration
    processor_cls: Any = transformers.WhisperProcessor
    key = str(model_path)
    if key not in _LOADED:  # once per process (`serve`)
        d = model_path.parent
        settings = json.loads((d / META).read_text(encoding="utf-8"))
        device = _device()
        model = model_cls.from_pretrained(str(d)).to(device).eval()
        _LOADED[key] = (model, processor_cls.from_pretrained(str(d)), settings, device)
    loaded: tuple[Any, Any, dict[str, Any], str] = _LOADED[key]
    return loaded


def _artifact(path: Path, fmt: str) -> Artifact:
    h = hashlib.sha256()
    n = 0
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
            n += len(chunk)
    return Artifact(path=path, format=fmt, sha256=h.hexdigest(), bytes=n)


#: Functions the notebook shows by source, in pipeline order (CONTRACTS §2.5): helpers before
#: `train` are the Model section, those after `train`, `export` and `runtime_eval` follow that step.
NOTEBOOK_CELLS = (
    WhisperBackend.train, _collate, _score, _transcribe,
    WhisperBackend.export, WhisperBackend._export_ggml, _ggml_quant_type, _ggml_convert_argv, _ggml_quantize_argv,
    WhisperBackend.runtime_eval, _load_export, _rescore_ggml, _whisper_cli_transcribe, _whisper_cli_argv,
)
WhisperBackend.NOTEBOOK_CELLS = NOTEBOOK_CELLS

BACKEND = WhisperBackend()
