"""Ludwig backend: compiles an AtelierSpec + resolved hyperparameters into a
Ludwig config, trains through ``sakura.adapters.ludwig.LudwigAdapter``, and
exports ONNX / torchscript from the trained torch module directly (Ludwig
ships no ONNX exporter -- only torchscript/triton/mlflow).

Imports ludwig lazily, inside functions: this module must be importable
(by the registry/CLI, to route a spec) even where ludwig itself is not
installed. Only ``train``/``export``/``runtime_eval``/``predict`` need it.

Contract: docs/atelier/CONTRACTS.md §2.2, §2.3.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any

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

NAME = "ludwig"
TASKS = frozenset({
    "image_classification", "audio_classification",
    "tabular_classification", "tabular_regression",
    "time_series_forecasting", "text_classification",
})

#: Timeseries encoders verified end to end (train, ONNX export, re-score, reload) on Ludwig 0.17.9. Ludwig also
#: lists patchtst and nbeats, but their models cannot be reloaded (the saved config carries max_sequence_length and
#: LudwigModel.load still raises "missing 1 required positional argument"), and stacked_cnn segfaults here.
FORECAST_ENCODERS = frozenset({"rnn", "parallel_cnn", "transformer"})

# Ludwig nests a FRESH run under <output_directory>/<experiment_name>_<model_name>/model/.
# Fixed names make that deterministic instead of depending on Ludwig's own
# defaults ("api_experiment"/"run"). A RESUMED run (model_resume_path= the
# previous run's model dir) is a different story: observed on the build host, Ludwig
# writes the continued weights one level deeper UNDER model_resume_path
# itself (.../<old model dir>/model/) and leaves the new run's own
# output_directory/checkpoint/ empty -- `this run's own checkpoint folder` is
# not where a resumed run's result actually lives. So neither
# _find_model_dir() nor train()'s own Checkpoint ever hardcode a nesting
# depth OR a single root; both search every root that could plausibly hold
# the result and pick the most deeply nested match (the latest one).
EXPERIMENT_NAME = "atelier"
MODEL_NAME = "run"


def _find_model_dir(*roots: Path) -> Path:
    """The directory LudwigModel.load() wants, found by its marker file
    rather than by a guessed nesting depth or a single guessed root (see the
    note above). Picks the most deeply nested match across all roots -- a
    resumed run's real model dir is always one level deeper than the run it
    resumed from."""
    matches = [p for root in roots if root.exists() for p in root.rglob("model_hyperparameters.json")]
    if not matches:
        raise FileNotFoundError(f"no model_hyperparameters.json found under any of {list(roots)}")
    matches.sort(key=lambda p: len(p.parts), reverse=True)
    return matches[0].parent

# bitsandbytes (a ludwig.schema.optimizers import-time dependency, used only
# for 8-bit optimizer *class references*, never instantiated by any phase-1
# preset) has no precompiled kernel for CUDA 13.0 yet and raises at import
# instead of falling back. Pointing it at the closest precompiled kernel
# (12.2) is enough to satisfy the import; see docs/atelier/envs/ludwig.txt.
os.environ.setdefault("BNB_CUDA_VERSION", "122")


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


#: Ludwig's feature lists: ``backend_config`` entries merge into the compiled
#: feature of the same ``name`` instead of replacing the whole list.
_FEATURE_LISTS = ("input_features", "output_features")


def _merge_features(base: list[Any], override: list[Any]) -> list[Any]:
    """A ``backend_config`` feature list over the compiled one, by ``name``: an
    entry naming a compiled feature deep-merges into it (its ``type`` and the
    other features survive); an entry naming none is appended as a new feature.
    Replacing the list instead sent Ludwig ``[{name, preprocessing}]`` for a
    normalized target -- no ``type``, so "ConfigValidationError: 'type' is a
    required property" -- and would have dropped every input not listed."""
    merged = [dict(f) if isinstance(f, dict) else f for f in base]
    index = {f.get("name"): i for i, f in enumerate(merged) if isinstance(f, dict)}
    for entry in override:
        name = entry.get("name") if isinstance(entry, dict) else None
        if name is not None and name in index:
            merged[index[name]] = _deep_merge(merged[index[name]], entry)
        else:
            merged.append(entry)
    return merged


def _deep_merge(base: Any, override: Any) -> Any:
    """``backend_config`` merged last over the compiled config -- the
    "deep dive at will" escape hatch (CONTRACTS §1). Mappings merge key by key,
    the feature lists by feature ``name`` (`_merge_features`); any other value
    in ``backend_config`` replaces the compiled one."""
    if isinstance(base, dict) and isinstance(override, dict):
        merged = dict(base)
        for k, v in override.items():
            if k in _FEATURE_LISTS and isinstance(base.get(k), list) and isinstance(v, list):
                merged[k] = _merge_features(base[k], v)
            else:
                merged[k] = _deep_merge(base.get(k), v) if k in base else v
        return merged
    return override


#: Recipe augmentation op -> Ludwig 0.17 image-input `augmentation` entry. Ludwig applies these to the training
#: split only (validation/test batches skip the augmentation pipeline). `random_crop` has no Ludwig op: it is
#: done by `crop_train_images` on the training frame instead.
LUDWIG_AUGMENTATION: dict[str, dict[str, Any]] = {
    "horizontal_flip": {"type": "random_horizontal_flip"},
    "vertical_flip": {"type": "random_vertical_flip"},
    "rotate": {"type": "random_rotate", "degree": 15},
    "brightness": {"type": "random_brightness", "min": 0.5, "max": 2.0},
    "contrast": {"type": "random_contrast", "min": 0.5, "max": 2.0},
    "blur": {"type": "random_blur", "kernel_size": 3},
}


def augmentation_ops(spec: Any) -> list[str]:
    """The ops the run really applies: the recipe's, when it enables augmentation on an image classifier."""
    aug = getattr(spec, "augmentation", None)
    if aug is None or not aug.enabled or spec.task != "image_classification":
        return []
    return list(aug.ops)


def augmentation_report(spec: Any) -> dict[str, Any] | None:
    """``{enabled, ops}`` for report.json / the evaluation block; None when the recipe has no augmentation block."""
    if getattr(spec, "augmentation", None) is None:
        return None
    ops = augmentation_ops(spec)
    return {"enabled": bool(ops), "ops": ops}


def crop_train_images(frame: Any, column: str, out_dir: Path, seed: int) -> Any:
    """A copy of the TRAINING frame whose ``column`` images are replaced by a random crop (reflect-padded by 1/8 of
    the side, cropped back to the original size). One draw per image, fixed across epochs; the validation frame is
    never passed here."""
    import random

    from PIL import Image, ImageOps

    out_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    paths = []
    for i, src in enumerate(frame[column]):
        with Image.open(src) as opened:
            im = opened.convert("RGB")
            w, h = im.size
            pad = max(1, min(w, h) // 8)
            padded = ImageOps.expand(im, border=pad, fill=0)
            x, y = rng.randint(0, 2 * pad), rng.randint(0, 2 * pad)
            dst = out_dir / f"{i:07d}.png"
            padded.crop((x, y, x + w, y + h)).save(dst)
        paths.append(str(dst))
    out = frame.copy()
    out[column] = paths
    return out


def compile_config(resolved: ResolvedSpec) -> dict[str, Any]:
    spec = resolved.spec
    hp = resolved.hyperparameters
    task = spec.task
    output_name = features_of(spec).output.name

    if task == "image_classification":
        in_name = features_of(spec).inputs[0].name
        size = int(hp.get("image_size", 32))
        config: dict[str, Any] = {
            "input_features": [{
                "name": in_name, "type": "image",
                "encoder": {"type": "resnet", "model_variant": 18,
                            "use_pretrained": bool(hp.get("pretrained", True)),
                            # Ludwig's torchvision encoders default this to
                            # False -- a checkpoint then holds only the
                            # task-specific decoder head, and reloading it
                            # (LudwigModel.load, used by export/runtime_eval/
                            # predict/resume) silently reverts the encoder to
                            # its PRE-fine-tuning pretrained weights. Verified
                            # on the build host: a resumed CIFAR run trained to 0.616
                            # accuracy scored 0.22-0.24 after any reload
                            # without this flag, 0.59 with it (CONTRACT
                            # DEVIATION -- see report).
                            "saved_weights_in_checkpoint": True},
                "preprocessing": {"height": size, "width": size, "num_channels": 3},
            }],
            "output_features": [{"name": output_name, "type": "category"}],
            "trainer": {"epochs": int(hp.get("epochs", 3)),
                        "batch_size": int(hp.get("batch_size", 64)),
                        "learning_rate": float(hp.get("learning_rate", 0.001))},
        }
    elif task == "audio_classification":
        in_name = features_of(spec).inputs[0].name
        config = {
            "input_features": [{
                "name": in_name, "type": "audio",
                "encoder": {"type": hp.get("encoder", "stacked_cnn")},
                "preprocessing": {"audio_file_length_limit_in_s": 5,
                                   "in_memory": True},
            }],
            "output_features": [{"name": output_name, "type": "category"}],
            "trainer": {"epochs": int(hp.get("epochs", 5)),
                        "batch_size": int(hp.get("batch_size", 32)),
                        "learning_rate": float(hp.get("learning_rate", 0.001))},
        }
    elif task in ("tabular_classification", "tabular_regression"):
        type_map = {"category": "category", "number": "number", "binary": "binary", "text": "text"}
        input_features = [{"name": f.name, "type": type_map[f.type]} for f in features_of(spec).inputs]
        out_type = "category" if task == "tabular_classification" else "number"
        # ECD only. Ludwig 0.11 has no GBM model type, and the old "gbm"
        # preset fell back to a shallow ECD under a name it was not -- every
        # number reported for it was mislabelled. Tabular GBM-class accuracy
        # comes from `tabpfn_v2` (its own backend) instead.
        config = {
            "input_features": input_features,
            "output_features": [{"name": output_name, "type": out_type}],
            "combiner": {"type": "concat", "num_fc_layers": int(hp.get("fc_layers_num", 2)),
                         "output_size": int(hp.get("fc_layers_size", 128))},
            "trainer": {"epochs": int(hp.get("epochs", 20)),
                        "batch_size": int(hp.get("batch_size", 128)),
                        "learning_rate": float(hp.get("learning_rate", 0.001))},
        }
    elif task == "time_series_forecasting":
        # One series: Ludwig builds lookback windows from the column for the input feature and forward windows for
        # the output feature, so both read the SAME column under two feature names (needs Ludwig >= 0.17).
        inputs = features_of(spec).inputs
        if len(inputs) != 1 or inputs[0].type != "timeseries" or features_of(spec).output.type != "timeseries" \
                or inputs[0].name != output_name:
            raise ValueError(
                "time_series_forecasting takes exactly one series: features.inputs = [{name: X, type: timeseries}] "
                "and features.output = {name: X, type: timeseries} (the same column)")
        ts_encoder = str(hp.get("encoder", "rnn"))
        if ts_encoder not in FORECAST_ENCODERS:
            raise ValueError(
                f"forecasting encoder {ts_encoder!r} is not supported (verified: {sorted(FORECAST_ENCODERS)}). "
                "Ludwig 0.17.9: patchtst/nbeats models train but cannot be reloaded "
                "(LudwigModel.load: 'missing max_sequence_length'), so they cannot be exported; "
                "stacked_cnn segfaults the training process")
        window = int(hp.get("window_size", 48))
        horizon = int(hp.get("horizon", 12))
        config = {
            "input_features": [{
                "name": f"{output_name}_in", "column": output_name, "type": "timeseries",
                "preprocessing": {"window_size": window}, "encoder": {"type": ts_encoder},
            }],
            "output_features": [{
                "name": f"{output_name}_out", "column": output_name, "type": "timeseries",
                "preprocessing": {"horizon": horizon}, "horizon": horizon,
            }],
            "trainer": {"epochs": int(hp.get("epochs", 30)),
                        "batch_size": int(hp.get("batch_size", 64)),
                        "learning_rate": float(hp.get("learning_rate", 0.001))},
        }
    elif task == "text_classification":
        if len(features_of(spec).inputs) != 1 or features_of(spec).inputs[0].type != "text":
            raise ValueError("text_classification takes exactly one input feature of type text")
        encoder: dict[str, Any] = {"type": hp.get("encoder", "parallel_cnn")}
        if hp.get("pretrained_model"):  # fine-tune a Hugging Face encoder instead of learning one from scratch
            encoder = {"type": "auto_transformer", "pretrained_model_name_or_path": str(hp["pretrained_model"]),
                       "trainable": bool(hp.get("trainable", True))}
        config = {
            "input_features": [{"name": features_of(spec).inputs[0].name, "type": "text", "encoder": encoder}],
            "output_features": [{"name": output_name, "type": "category"}],
            "trainer": {"epochs": int(hp.get("epochs", 10)),
                        "batch_size": int(hp.get("batch_size", 64)),
                        "learning_rate": float(hp.get("learning_rate", 0.001))},
        }
    else:
        raise ValueError(f"ludwig backend does not implement task {task!r}")

    # The preset's patience and stop metric reach Ludwig's trainer: left out, Ludwig stopped on its own
    # default (5 rounds of combined loss) while the report and the UI described the preset's (BE-19).
    for key in ("early_stop", "validation_metric"):
        if hp.get(key) is not None:
            config["trainer"][key] = int(hp[key]) if key == "early_stop" else str(hp[key])
    native = [dict(LUDWIG_AUGMENTATION[o]) for o in augmentation_ops(spec) if o in LUDWIG_AUGMENTATION]
    if native:
        config["input_features"][0]["augmentation"] = native
    merged: dict[str, Any] = _deep_merge(config, dict(spec.backend_config))
    return merged


def _final_metrics(train_stats: Any, output_feature: str) -> dict[str, float]:
    from sakura.adapters.ludwig import METRIC_ALIASES

    split = None
    for key in ("validation", "valid"):
        split = getattr(train_stats, key, None) if not isinstance(train_stats, dict) else train_stats.get(key)
        if split:
            break
    if not split:
        return {}
    feature_metrics = split.get(output_feature, {}) if isinstance(split, dict) else {}
    out: dict[str, float] = {}
    for name, values in feature_metrics.items():
        if not values:
            continue
        v = values[-1]
        v = getattr(v, "value", v[-1] if isinstance(v, (tuple, list)) else v)
        if v is None:
            continue
        out[METRIC_ALIASES.get(name, name)] = float(v)
    return out


def _int_or_none(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None
#: Ludwig metrics where a larger value is better; every other one (loss, rmse, mae, ...) is minimised.
_MAXIMISED = frozenset({"accuracy", "accuracy_micro", "roc_auc", "f1", "precision", "recall", "specificity",
                        "r2", "hits_at_k", "token_accuracy", "sequence_accuracy", "jaccard"})


def _points(series: Any, every_epochs: int | None = None) -> list[tuple[int, int | None, float]]:
    """``[(epoch, step, value)]`` of a Ludwig metric series, NaN/None dropped.

    A point is a ``TrainerMetric`` namedtuple or a bare ``(epoch, step, value)``
    tuple, or -- what Ludwig 0.17's ``model.train()`` statistics actually hold
    -- a bare number, one per evaluation. Those carry no epoch: they are
    numbered from ``every_epochs`` (``evaluation_frequency`` = every k epochs,
    see :func:`_epochs_per_evaluation`) with no step, and dropped when the
    run evaluated by step, since then no epoch can be named."""
    out: list[tuple[int, int | None, float]] = []
    for i, p in enumerate(series or []):
        epoch: Any
        step: Any
        value: Any
        if isinstance(p, (int, float)) and not isinstance(p, bool):
            if not every_epochs:
                continue
            epoch, step, value = (i + 1) * every_epochs - 1, None, p
        else:
            epoch = getattr(p, "epoch", p[0] if isinstance(p, (tuple, list)) else None)
            step = getattr(p, "step", p[1] if isinstance(p, (tuple, list)) else None)
            value = getattr(p, "value", p[-1] if isinstance(p, (tuple, list)) else p)
            if step is None:
                continue
        try:
            v = float(value)
        except (TypeError, ValueError):
            continue
        if v == v and epoch is not None:  # v == v: not NaN
            out.append((int(epoch), None if step is None else int(step), v))
    return out


def _epochs_per_evaluation(train_stats: Any) -> int | None:
    """Ludwig's ``evaluation_frequency`` as "every k epochs", or None when it
    evaluated by step (or does not say)."""
    freq = (train_stats.get("evaluation_frequency") if isinstance(train_stats, dict)
            else getattr(train_stats, "evaluation_frequency", None))
    period = freq.get("period") if isinstance(freq, dict) else getattr(freq, "period", None)
    k = freq.get("frequency") if isinstance(freq, dict) else getattr(freq, "frequency", None)
    if str(period).lower().endswith("epoch") and isinstance(k, (int, float)) and k > 0:
        return int(k)
    return None


def _split_stats(train_stats: Any, *names: str) -> dict[str, Any]:
    for key in names:
        split = train_stats.get(key) if isinstance(train_stats, dict) else getattr(train_stats, key, None)
        if split:
            return dict(split)
    return {}


def epoch_metrics(train_stats: Any, output_feature: str) -> list[dict[str, Any]]:
    """One row per epoch, ``[{"epoch": 1, "step": 7, "train": {...}, "validation": {...}}]``,
    from Ludwig's training statistics: the curves a reader labels in epochs,
    not in optimizer steps (BE-19). Epochs are Ludwig's, counted from 1 (so a
    run evaluated every 2 epochs has rows 2, 4, ...); an epoch evaluated more
    than once keeps its last point, and ``step`` is present only when the
    statistics carry one. Metric names use the report's vocabulary (``rmse``, not Ludwig's
    spelling), and the combined loss of a multi-output model is not repeated."""
    from sakura.adapters.ludwig import METRIC_ALIASES

    rows: dict[int, dict[str, Any]] = {}
    every = _epochs_per_evaluation(train_stats)
    for split_name, keys in (("train", ("training", "train")), ("validation", ("validation", "valid")),
                             ("test", ("test",))):
        feature = _split_stats(train_stats, *keys).get(output_feature) or {}
        for name, series in feature.items():
            for epoch, step, value in _points(series, every):
                row = rows.setdefault(epoch, {})
                if step is not None:
                    row["step"] = max(int(row.get("step", step)), step)
                row.setdefault(split_name, {})[METRIC_ALIASES.get(name, name)] = round(value, 6)
    return [{"epoch": e + 1, **rows[e]} for e in sorted(rows)]


def early_stop_summary(train_stats: Any, *, field: str, metric: str, patience: int | None,
                       max_epochs: int | None, wrap_up: bool = False) -> dict[str, Any] | None:
    """``{metric, mode, patience, best_epoch, stopped_epoch, ...}``: what the
    stop criterion watched and where it landed, from the same validation
    series Ludwig's trainer compared (``field`` is its ``validation_field``:
    the output feature or ``combined``). ``best_epoch`` is the checkpoint
    Ludwig kept, ``stopped_epoch`` the last epoch trained, and ``reason`` says
    whether patience ran out, the epoch budget did, or a wrap-up request ended the run (``wrap_up``). Epochs count from 1, as
    :func:`epoch_metrics` does. None when the series is missing."""
    points = _points((_split_stats(train_stats, "validation", "valid").get(field) or {}).get(metric),
                     _epochs_per_evaluation(train_stats))
    if not points:
        return None
    by_epoch: dict[int, tuple[int | None, float]] = {}
    for epoch, step, value in points:
        by_epoch[epoch] = (step, value)
    epochs = sorted(by_epoch)
    mode = "max" if metric in _MAXIMISED else "min"
    values = [by_epoch[e][1] for e in epochs]
    # the first best, as Ludwig keeps it: a later equal value is no improvement
    best_i = (values.index(max(values)) if mode == "max" else values.index(min(values)))
    stopped = epochs[-1] + 1  # Ludwig counts epochs from 0; the report from 1
    rounds_since_best = len(epochs) - (best_i + 1)  # patience counts evaluation rounds
    enabled = patience is not None and patience > 0
    if wrap_up:
        reason = "wrap_up"
    elif max_epochs is not None and stopped >= max_epochs:
        reason = "max_epochs"
    elif patience is not None and patience > 0 and rounds_since_best >= patience:
        reason = "early_stop"
    else:
        reason = "other"  # stopped short of both, e.g. a time budget
    from sakura.adapters.ludwig import METRIC_ALIASES

    return {"metric": METRIC_ALIASES.get(metric, metric), "field": field, "mode": mode,
            "patience": patience if enabled else None, "enabled": enabled,
            "best_epoch": epochs[best_i] + 1, "best_step": by_epoch[epochs[best_i]][0],
            "best_value": round(values[best_i], 6), "stopped_epoch": stopped,
            "stopped_step": by_epoch[epochs[-1]][0], "max_epochs": max_epochs, "reason": reason}


def _stop_settings(model: Any, config: dict[str, Any], output_feature: str) -> tuple[str, str, int | None]:
    """``(validation_field, validation_metric, patience)`` the trainer really
    ran with: Ludwig's resolved config when it has one, the compiled config
    (and Ludwig's documented defaults) otherwise."""
    trainer_cfg = getattr(getattr(model, "config_obj", None), "trainer", None)
    compiled = config.get("trainer", {})
    field = getattr(trainer_cfg, "validation_field", None) or compiled.get("validation_field") or output_feature
    metric = getattr(trainer_cfg, "validation_metric", None) or compiled.get("validation_metric") or "loss"
    patience = getattr(trainer_cfg, "early_stop", None)
    if patience is None:
        patience = compiled.get("early_stop", 5)
    return str(field), str(metric), (int(patience) if patience is not None else None)


def apply_max_steps(config: dict[str, Any], max_steps: int | None, train_rows: int | None) -> bool:
    """``limits.max_steps`` as Ludwig's own ``trainer.train_steps`` (SAK-0019). Ludwig's ``train_steps``
    REPLACES ``epochs`` (``get_total_steps``; the same in 0.11.0 and 0.17.9), so it would lengthen a run
    whose epochs end sooner. It is therefore set only when it is known to shorten the run: the compiled
    ``epochs`` and ``batch_size`` are whole numbers and ``epochs * ceil(train_rows / batch_size)`` is more
    than ``max_steps``. Forecasting is left out, because its steps count windows, not table rows. In every
    other case the wrap-up watch stops the trainer at the same step (``LudwigAdapter.on_batch_end``).
    Returns whether the key was set."""
    trainer = config.setdefault("trainer", {})
    if max_steps is None or train_rows is None or config["output_features"][0].get("type") == "timeseries":
        return False
    epochs, batch = trainer.get("epochs"), trainer.get("batch_size")
    if isinstance(trainer.get("train_steps"), int):
        planned = int(trainer["train_steps"])
    elif isinstance(epochs, int) and isinstance(batch, int) and batch > 0:
        planned = epochs * -(-train_rows // batch)
    else:
        return False
    if max_steps >= planned:
        return False
    trainer["train_steps"] = max_steps
    return True


def train(spec: ResolvedSpec, data: MaterialisedData, out: RunDir,
          metrics: MetricsSink, resume: Checkpoint | None) -> TrainResult:
    from ludwig.api import LudwigModel

    from sakura.adapters.ludwig import LudwigAdapter

    config = compile_config(spec)
    max_steps = max_steps_of(spec)
    capped = apply_max_steps(config, max_steps, len(data.train) if hasattr(data.train, "__len__") else None)
    output_feature = str(config["output_features"][0]["name"])  # forecasting: "<column>_out", not the spec's name
    # With `train_steps` set, Ludwig ends the run itself and evaluates at its last step, so the watch only
    # listens for a wrap-up request. Without it the watch enforces the cap: Ludwig is stopped at that step,
    # before an evaluation round if none has run yet (as for a wrap-up, see LudwigAdapter._can_stop_now).
    watch = WrapUpWatch(out.root, None if capped else max_steps)
    callback = LudwigAdapter(metrics, output_feature=output_feature, wrap_up=watch,
                             epochs_max=_int_or_none(config.get("trainer", {}).get("epochs")))
    train_kwargs: dict[str, Any] = {}
    if resume is None:
        model = LudwigModel(config=config, callbacks=[callback])
    else:
        # Continue training = a WARM START from the delivered model, not
        # Ludwig's `model_resume_path`. That mechanism resumes an INTERRUPTED
        # run from its saved config and progress counters, and on a run that
        # had finished it did two wrong things, both measured on the build host: it trained
        # nothing (0 metric points; final metrics byte-identical to the
        # parent's) while still billing the preprocessing, and it rewrote the
        # PARENT job's model directory (its saved config came back with the new
        # run's epochs). Here the parent is only read; this run's epochs and
        # learning rate apply; and the parent's training-set metadata is reused
        # so the decoder's class order cannot shift under it.
        parent_dir = _find_model_dir(resume.path)
        model = LudwigModel.load(str(parent_dir), callbacks=[callback])
        model.config_obj.trainer.epochs = int(config["trainer"]["epochs"])
        model.config_obj.trainer.learning_rate = float(config["trainer"]["learning_rate"])
        if capped:
            model.config_obj.trainer.train_steps = int(config["trainer"]["train_steps"])
        train_kwargs["training_set_metadata"] = model.training_set_metadata
        _first_param_fingerprint(model, out, "resume_start")

    training_set = data.train
    if "random_crop" in augmentation_ops(spec.spec):
        training_set = crop_train_images(training_set, features_of(spec.spec).inputs[0].name,
                                         out.root / "augmented_train", spec.spec.seed)

    start = time.time()
    train_stats, _preprocessed, _output_directory = model.train(
        training_set=training_set,
        validation_set=data.validation,
        output_directory=str(out.checkpoint),
        experiment_name=EXPERIMENT_NAME,
        model_name=MODEL_NAME,
        skip_save_processed_input=True,
        random_seed=spec.spec.seed,
        **train_kwargs,
    )
    train_seconds = time.time() - start

    model_dir = _find_model_dir(out.checkpoint)
    field, stop_metric, patience = _stop_settings(model, config, output_feature)
    early_stop = early_stop_summary(train_stats, field=field, metric=stop_metric, patience=patience,
                                    max_epochs=int(config["trainer"]["epochs"]), wrap_up=watch.triggered)
    stopped_early = watch.stopped_early
    if early_stop and stopped_early and stopped_early["reason"] != "wrap_up":
        early_stop["reason"] = stopped_early["reason"]  # max_steps: the step cap, not a request
    if capped and stopped_early is None and max_steps is not None and (watch.last_step or 0) >= max_steps:
        # Ludwig ran to its `train_steps`: the cap ended the run (patience would have stopped it sooner).
        if early_stop:
            early_stop["reason"] = "max_steps"
        per_epoch = -(-len(data.train) // int(config["trainer"]["batch_size"]))
        stopped_early = {"reason": "max_steps", "epoch": max_steps / per_epoch, "step": max_steps}
    return TrainResult(
        model=model,
        metrics=_final_metrics(train_stats, output_feature),
        train_seconds=train_seconds,
        checkpoint=Checkpoint(path=model_dir, model=spec.spec.model, task=spec.spec.task),
        stopped_early=stopped_early,
        early_stop=early_stop,
        epoch_metrics=epoch_metrics(train_stats, output_feature) or None,
        augmentation=augmentation_report(spec.spec),
    )


def _first_param_fingerprint(model: Any, out: RunDir, label: str) -> None:
    """Record a fingerprint of the model's first parameter tensor, so a
    continued run can show it started from the parent's weights rather than
    a fresh encoder (`resume_evidence.json` beside metrics.jsonl)."""
    import json

    first = next(iter(model.model.parameters()))
    fp = round(float(first.detach().double().sum()), 6)
    path = out.root / "resume_evidence.json"
    data = json.loads(path.read_text()) if path.exists() else {}
    data[label] = fp
    path.write_text(json.dumps(data))


def _labels(model: Any, output_feature: str) -> list[str] | None:
    meta = model.training_set_metadata.get(output_feature, {})
    idx2str = meta.get("idx2str")
    return list(idx2str) if idx2str else None


def _onnx_sample_io(torch_model: Any) -> tuple[dict[str, Any], dict[str, Any]]:
    import torch
    torch_model.eval()
    sample_inputs = torch_model.get_model_inputs()
    with torch.no_grad():
        sample_outputs = torch_model(sample_inputs)
    return sample_inputs, sample_outputs


def _traceable(model: Any, output_feature: str) -> tuple[Any, tuple[Any, ...], dict[str, Any], list[str], list[str]]:
    """The trained ECD module wrapped for tracing: positional tensor inputs
    in, the output feature's tensors out. Shared by the ONNX and TorchScript
    exports so both carry exactly the same graph and input/output order.

    Returns `(wrapper, example_args, sample_inputs, input_names, output_keys)`.
    """
    import torch

    # Export on CPU, always: some torchvision-backed Ludwig encoders (e.g.
    # the resnet image encoder) compute their lazily-cached `.output_shape`
    # property by running a CPU-only dummy tensor through the model the
    # first time it is accessed, regardless of the model's actual device --
    # if the real model is still on GPU (as it is right after training),
    # that first forward crashes with a CPU/CUDA tensor mismatch. Moving the
    # whole model to CPU first makes every tensor involved agree, and export
    # never needs the GPU anyway.
    torch_model = model.model.to("cpu")  # the ECD nn.Module
    sample_inputs, sample_outputs = _onnx_sample_io(torch_model)
    input_names = list(sample_inputs.keys())
    output_keys = sorted(k for k in sample_outputs if k.startswith(f"{output_feature}::"))
    if not output_keys:  # defensive: an unexpected key scheme still exports *something*, never nothing
        output_keys = sorted(sample_outputs.keys())

    class _Wrapper(torch.nn.Module):
        def __init__(self, m: Any, in_names: list[str], out_keys: list[str]) -> None:
            super().__init__()
            self.m = m
            self.in_names = in_names
            self.out_keys = out_keys

        def forward(self, *args: Any) -> tuple[Any, ...]:
            inputs = dict(zip(self.in_names, args))
            out = self.m(inputs)
            return tuple(out[k] for k in self.out_keys)

    wrapper = _Wrapper(torch_model, input_names, output_keys).eval()
    example_args = tuple(sample_inputs[n] for n in input_names)
    return wrapper, example_args, sample_inputs, input_names, output_keys


def _export_torchscript(model: Any, output_feature: str, path: Path) -> None:
    """Trace the same wrapped forward the ONNX export uses and save it.

    Ludwig 0.17 has no `LudwigModel.to_torchscript` (the 0.x API this once
    called), so every job asking for `torchscript` -- the planner's default
    pairs it with onnx -- failed right after training with
    `AttributeError: 'LudwigModel' object has no attribute 'to_torchscript'`
    (hub job f591f66b, 2026-10-12). Tracing the wrapper is what the ONNX
    exporter does underneath anyway; the saved module takes the inputs in
    `preprocess.json`'s `input_order` and returns the `output_order` tensors.
    """
    import torch

    wrapper, example_args, sample_inputs, input_names, output_keys = _traceable(model, output_feature)
    with torch.no_grad():
        traced = torch.jit.trace(wrapper, example_args, check_trace=False)  # type: ignore[no-untyped-call]
    traced.save(str(path))
    # What runtime_eval needs to feed the module Ludwig's preprocessed tensors, as preprocess.json does for ONNX.
    sidecar = {"input_order": input_names, "input_dtypes": {n: str(t.dtype) for n, t in sample_inputs.items()},
               "output_order": [k.replace("::", "__") for k in output_keys]}
    path.with_suffix(".json").write_text(json.dumps(sidecar, indent=2), encoding="utf-8")


def _export_onnx(model: Any, output_feature: str, path: Path) -> dict[str, Any]:
    import torch

    wrapper, example_args, sample_inputs, input_names, output_keys = _traceable(model, output_feature)
    output_names = [k.replace("::", "__") for k in output_keys]
    dynamic_axes = {n: {0: "batch"} for n in input_names}
    dynamic_axes.update({n: {0: "batch"} for n in output_names})

    with torch.no_grad():
        # dynamo=False: the default dynamo-based exporter (torch>=2.x) needs
        # `onnxscript`, which is not in this backend's pinned env (docs/
        # atelier/envs/ludwig.txt) -- the legacy TorchScript-tracing exporter
        # (what `dynamic_axes` is for) has no such dependency and is plenty
        # for a plain ECD/resnet forward pass.
        torch.onnx.export(wrapper, example_args, str(path), input_names=input_names,
                           output_names=output_names, dynamic_axes=dynamic_axes,
                           opset_version=17, dynamo=False)

    preprocess: dict[str, Any] = {
        "task": model.config_obj.model_type,
        "input_order": input_names,
        "input_shapes": {n: list(t.shape[1:]) for n, t in sample_inputs.items()},
        "input_dtypes": {n: str(t.dtype) for n, t in sample_inputs.items()},
        "output_order": output_names,
        "onnx_output_to_ludwig_key": dict(zip(output_names, output_keys)),
        # Ludwig's own metadata is what a runtime needs to replicate
        # preprocessing exactly (resize/normalise params, vocab, idx2str, ...)
        # -- captured verbatim rather than re-derived, since Ludwig is the
        # thing that computed it.
        "ludwig_training_set_metadata": _json_safe(model.training_set_metadata),
    }
    return preprocess


def _json_safe(obj: Any) -> Any:
    try:
        json.dumps(obj)
        return obj
    except TypeError:
        if isinstance(obj, dict):
            return {str(k): _json_safe(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple, set)):
            return [_json_safe(v) for v in obj]
        try:
            import numpy as np
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, (np.integer, np.floating)):
                return obj.item()
        except ImportError:
            pass
        return str(obj)


def export(result: TrainResult, fmt: str, out: RunDir) -> Artifact:
    from ludwig.api import LudwigModel

    out.artifacts.mkdir(parents=True, exist_ok=True)

    # Reload from the checkpoint on disk rather than exporting result.model
    # (the live in-memory object) directly: observed on the build host, a resumed run's
    # final in-memory model can hold different weights than what training
    # actually persisted to disk (Ludwig's own best-checkpoint bookkeeping),
    # which showed up as an ONNX export that scored far off its own training
    # metric purely for a continued run. Reloading from disk is exactly what
    # runtime_eval/predict already do (_reload_for_preprocessing) -- this
    # makes export consistent with them and with whatever a later `predict`
    # or `resume_from` sees, instead of a third, sometimes-different, state.
    model = LudwigModel.load(str(result.checkpoint.path), callbacks=[]) if result.checkpoint else result.model
    output_feature: str = model.config_obj.output_features[0].name

    if result.checkpoint is not None:
        # runtime_eval/predict need to reload this SAME checkpoint dir later,
        # possibly in a separate process where report.json (which would
        # otherwise say `resumed_from`) does not exist yet -- e.g. right here,
        # inside this same `run`, before report.json is written. Recording it
        # directly means they never have to re-derive or guess the path (see
        # _checkpoint_dir's fallback search, kept for runs exported before
        # this file existed).
        (out.root / "_model_dir.txt").write_text(str(result.checkpoint.path), encoding="utf-8")

    if fmt == "onnx":
        path = out.artifacts / "model.onnx"
        preprocess = _export_onnx(model, output_feature, path)
        (out.artifacts / "preprocess.json").write_text(json.dumps(preprocess, indent=2), encoding="utf-8")
        labels = _labels(model, output_feature)
        (out.artifacts / "labels.json").write_text(json.dumps(labels), encoding="utf-8")
    elif fmt == "torchscript":
        path = out.artifacts / "model.torchscript.pt"
        _export_torchscript(model, output_feature, path)
    else:
        raise ValueError(f"ludwig backend does not support export format {fmt!r}")

    return Artifact(path=path, format=fmt, sha256=_sha256_file(path), bytes=path.stat().st_size)


def _np_dtype(torch_dtype: str | None) -> Any:
    """numpy dtype for an ONNX graph input, from the torch dtype recorded at export ("torch.int64" -> int64);
    float32 when unknown. Text features feed integer token ids, every other input here is float."""
    import numpy as np

    name = (torch_dtype or "torch.float32").split(".")[-1]
    return {"int64": np.int64, "int32": np.int32, "int16": np.int16, "int8": np.int8, "uint8": np.uint8,
            "bool": np.bool_, "float64": np.float64, "float16": np.float16}.get(name, np.float32)


def _checkpoint_dir(artifact_path: Path) -> Path:
    # RunDir's fixed layout (types.py): <run>/artifacts/<file>, <run>/checkpoint/.
    run_root = artifact_path.parent.parent

    # export() records exactly which dir it reloaded -- authoritative, and
    # works even DURING the same run (report.json does not exist yet there).
    marker = run_root / "_model_dir.txt"
    if marker.exists():
        return Path(marker.read_text(encoding="utf-8").strip())

    # Fallback for a run exported before _model_dir.txt existed: guess from
    # report.json's `resumed_from` (a resumed run's result lives under its
    # PARENT's tree -- see the module-level note on model_resume_path).
    roots = [run_root / "checkpoint"]
    report_path = run_root / "report.json"
    if report_path.exists():
        resumed_from = json.loads(report_path.read_text(encoding="utf-8")).get("resumed_from")
        if resumed_from and resumed_from.startswith("file://"):
            roots.append(Path(resumed_from.removeprefix("file://")).parent)
    return _find_model_dir(*roots)


def _proc_columns(model: Any) -> dict[str, str]:
    """feature name -> proc_column: the batcher's dict (and the preprocessed
    Dataset generally) is keyed by Ludwig's internal hashed proc_column, not
    the feature name ONNX input/output names use (get_model_inputs() uses the
    feature name) -- every batch read needs this translation."""
    out = {name: f.proc_column for name, f in model.model.input_features.items()}
    for name, f in model.model.output_features.items():
        out[name] = f.proc_column
    return out


def _reload_for_preprocessing(artifact: Artifact) -> Any:
    """Reload the trained LudwigModel from its sibling checkpoint/ dir --
    purely to reuse its preprocessing, so re-scoring sees the EXACT tensors
    training produced rather than a hand-rolled re-implementation that could
    silently drift from it."""
    from ludwig.api import LudwigModel

    # Loaded once per process: `serve` answers every try-it from the same
    # model instead of paying the ~10 s reload per request.
    key = str(_checkpoint_dir(artifact.path))
    if key not in _LOADED:
        _LOADED[key] = LudwigModel.load(key, callbacks=[])
    return _LOADED[key]


_LOADED: dict[str, Any] = {}


def _session_provider(session: Any) -> str | None:
    """The execution provider the session really runs on: asking for CUDA is not
    getting it (a missing libcublasLt drops the session to CPU without an error)."""
    try:
        providers = session.get_providers()
        return str(providers[0]) if providers else None
    except Exception:
        return None


def _fit_input_ranks(feed: dict[str, Any], ranks: dict[str, int]) -> dict[str, Any]:
    """Match the feed to the graph's own input ranks: a category input is exported as
    [batch, 1] while the batcher yields [batch], and onnxruntime refuses the mismatch
    ("Invalid rank")."""
    return {k: (v.reshape(-1, 1) if getattr(v, "ndim", 0) == 1 and ranks.get(k) == 2 else v)
            for k, v in feed.items()}


def runtime_eval(artifact: Artifact, data: MaterialisedData) -> RuntimeEval:
    """Re-score ``artifact`` with onnxruntime on the validation split
    (CONTRACTS §2.1/§2.3): feed Ludwig's own preprocessed validation batches
    straight into the ONNX graph (no normalisation of our own -- the ONNX
    graph starts exactly where Ludwig's preprocessing ends, which is the
    whole point of exporting "the trained torch inference module on
    preprocessed tensors"). A TorchScript export is re-scored the same way
    with `torch.jit` (`_runtime_eval_torchscript`)."""
    if artifact.format == "torchscript":
        return _runtime_eval_torchscript(artifact, data)
    try:
        import onnxruntime  # noqa: F401  -- the install check; the session comes from the helper

        from sakura.atelier.ort import inference_session
    except ImportError as exc:
        return RuntimeEval(format=artifact.format, runtime="onnxruntime", metric="accuracy",
                            value=None, n=0, matches_training=None,
                            reason=f"onnxruntime not installed: {exc}")

    if artifact.format != "onnx":
        return RuntimeEval(format=artifact.format, runtime="n/a", metric="accuracy", value=None, n=0,
                            matches_training=None, reason=f"no runtime scorer implemented for {artifact.format}")

    preprocess_path = artifact.path.parent / "preprocess.json"
    if not preprocess_path.exists():
        return RuntimeEval(format=artifact.format, runtime="onnxruntime", metric="accuracy", value=None, n=0,
                            matches_training=None, reason="preprocess.json missing next to the artifact")
    preprocess = json.loads(preprocess_path.read_text(encoding="utf-8"))
    input_order: list[str] = preprocess["input_order"]
    output_order: list[str] = preprocess["output_order"]
    input_dtypes: dict[str, str] = preprocess.get("input_dtypes", {})

    try:
        session = inference_session(artifact.path, providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
    except Exception as exc:
        return RuntimeEval(format=artifact.format, runtime="onnxruntime", metric="accuracy", value=None, n=0,
                            matches_training=None, reason=f"session load failed: {exc}")

    try:
        model, preprocessed = _preprocessed_validation(artifact, data)
    except Exception as exc:
        return RuntimeEval(format=artifact.format, runtime="onnxruntime", metric="accuracy", value=None, n=0,
                            matches_training=None, reason=f"could not reproduce preprocessing: {exc}")
    output_feature = model.config_obj.output_features[0].name
    proc_columns = _proc_columns(model)

    if model.config_obj.output_features[0].type == "timeseries":
        return _runtime_eval_forecast(model, session, preprocessed, data, input_order, output_order,
                                      output_feature, proc_columns)

    ranks = {i.name: len(i.shape) for i in session.get_inputs() if i.shape is not None}

    def _run(feed: dict[str, Any]) -> list[Any]:
        return list(session.run(output_order, _fit_input_ranks(feed, ranks)))

    n, value, is_classification = _score_validation(preprocessed, _run, input_order, input_dtypes, output_order,
                                                    output_feature, proc_columns)
    return RuntimeEval(format="onnx", runtime="onnxruntime", metric="accuracy" if is_classification else "rmse",
                       value=value, n=n, matches_training=None, provider=_session_provider(session))


def _preprocessed_validation(artifact: Artifact, data: MaterialisedData) -> tuple[Any, Any]:
    """``(model, preprocessed)``: the run's reloaded LudwigModel and the
    validation split through ITS preprocessing, the tensors every runtime scorer
    feeds the exported graph."""
    model = _reload_for_preprocessing(artifact)
    # training_set= (not dataset=, which needs no target and no further
    # split boundary): a bare `dataset=` kwarg auto-splits by proportion
    # AGAIN, re-shrinking a validation split that is already the
    # validation split (same bug as predict()'s single row). preprocess()
    # requires SOME training_set, so hand it the same frame and read it
    # back from .training_set rather than .validation_set.
    #
    # training_set_metadata=model.training_set_metadata is NOT optional:
    # left at its default (None), preprocess() infers a FRESH vocabulary
    # from whatever appears in just this one dataframe -- for a category
    # feature Ludwig ranks by frequency, so a small validation slice can
    # assign a DIFFERENT index to the same class than training did. The
    # ONNX graph's output order is frozen to the TRAINING vocabulary;
    # re-inferring it silently scrambles the label<->index mapping and
    # tanks accuracy without any actual prediction being wrong (verified
    # on the build host: a model trained to 0.29 accuracy scored 0.06 through this
    # exact bug, independent of CPU export, independent of resume).
    preprocessed = model.preprocess(training_set=data.validation,
                                     training_set_metadata=model.training_set_metadata,
                                     skip_save_processed_input=True)
    return model, preprocessed


def _score_validation(preprocessed: Any, run: Any, input_order: list[str], input_dtypes: dict[str, str],
                      output_order: list[str], output_feature: str,
                      proc_columns: dict[str, str]) -> tuple[int, float | None, bool]:
    """``(rows, value, is_classification)`` of an exported graph over the
    preprocessed validation split: accuracy for a classifier, RMSE for a
    regressor. ``run(feed)`` maps ``{input name: ndarray}`` to the graph's
    outputs in ``output_order``, so ONNX and TorchScript share this loop."""
    import numpy as np

    n = 0
    correct = 0
    sq_err = 0.0
    is_classification = True
    with preprocessed.training_set.initialize_batcher(batch_size=128, should_shuffle=False) as batcher:
        while True:
            batch = batcher.next_batch()
            feed = {name: np.asarray(batch[proc_columns[name]], dtype=_np_dtype(input_dtypes.get(name)))
                    for name in input_order}
            onnx_outputs = run(feed)
            target = np.asarray(batch[proc_columns[output_feature]])
            if np.issubdtype(target.dtype, np.floating) and target.ndim == 1 and len(output_order) == 1 \
                    and onnx_outputs[0].shape[-1] == 1:
                is_classification = False
                pred = onnx_outputs[0].reshape(-1)
                sq_err += float(((pred - target) ** 2).sum())
            else:
                logits_key = next((o for o in output_order if "logits" in o or "probabilities" in o), output_order[0])
                logits = onnx_outputs[output_order.index(logits_key)]
                pred = logits.argmax(axis=-1)
                correct += int((pred == target).sum())
            n += len(target)
            if batcher.last_batch():
                break

    if is_classification:
        return n, (correct / n if n else None), True
    return n, ((sq_err / n) ** 0.5 if n else None), False


def _runtime_eval_torchscript(artifact: Artifact, data: MaterialisedData) -> RuntimeEval:
    """Re-score the TorchScript module on the validation split, as the ONNX
    graph is: until this existed the format shipped as "delivered" without a
    single row through it (BE-9). A forecaster is not scored here; it says so."""
    def _untested(reason: str) -> RuntimeEval:
        return RuntimeEval(format="torchscript", runtime="torch.jit", metric="accuracy", value=None, n=0,
                           matches_training=None, reason=reason)

    try:
        import numpy as np
        import torch
    except ImportError as exc:
        return _untested(f"torch not installed: {exc}")
    sidecar_path = artifact.path.with_suffix(".json")
    if not sidecar_path.exists():
        return _untested(f"{sidecar_path.name} missing next to the artifact (exported before it was written)")
    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    try:
        module = torch.jit.load(str(artifact.path), map_location="cpu")  # type: ignore[no-untyped-call]
        module.eval()
    except Exception as exc:
        return _untested(f"module load failed: {exc}")
    try:
        model, preprocessed = _preprocessed_validation(artifact, data)
    except Exception as exc:
        return _untested(f"could not reproduce preprocessing: {exc}")
    if model.config_obj.output_features[0].type == "timeseries":
        return _untested("no TorchScript scorer for forecasting: the ONNX file is the tested one")

    def _run(feed: dict[str, Any]) -> list[Any]:
        with torch.no_grad():
            outs = module(*(torch.from_numpy(np.ascontiguousarray(feed[name])) for name in sidecar["input_order"]))
        return [o.numpy() for o in outs]

    try:
        n, value, is_classification = _score_validation(
            preprocessed, _run, sidecar["input_order"], sidecar.get("input_dtypes", {}), sidecar["output_order"],
            model.config_obj.output_features[0].name, _proc_columns(model))
    except Exception as exc:
        return _untested(f"scoring failed: {type(exc).__name__}: {exc}")
    return RuntimeEval(format="torchscript", runtime="torch.jit", metric="accuracy" if is_classification else "rmse",
                       value=value, n=n, matches_training=None, provider="cpu")


def _runtime_eval_forecast(model: Any, session: Any, preprocessed: Any, data: MaterialisedData,
                           input_order: list[str], output_order: list[str], output_feature: str,
                           proc_columns: dict[str, str]) -> RuntimeEval:
    """RMSE of the ONNX graph over every horizon step of the validation windows, compared with Ludwig's OWN
    evaluation of the same split: the two must agree (scale-free tolerance, so a large-valued series is not
    held to an absolute 0.05)."""
    import numpy as np

    # the graph also exposes hidden states: the forecast is the decoder output ("__logits" for a timeseries
    # feature, "__predictions" if a Ludwig version exposes it)
    pred_idx = next((i for key in ("__predictions", "__logits")
                     for i, o in enumerate(output_order) if o.endswith(key)), 0)
    sq_err, count = 0.0, 0
    with preprocessed.training_set.initialize_batcher(batch_size=128, should_shuffle=False) as batcher:
        while True:
            batch = batcher.next_batch()
            feed = {name: np.asarray(batch[proc_columns[name]], dtype=np.float32) for name in input_order}
            pred = np.asarray(session.run(output_order, feed)[pred_idx], dtype=np.float64)
            target = np.asarray(batch[proc_columns[output_feature]], dtype=np.float64).reshape(pred.shape)
            sq_err += float(((pred - target) ** 2).sum())
            count += target.size
            if batcher.last_batch():
                break
    onnx_rmse = (sq_err / count) ** 0.5 if count else None
    matches: bool | None = None
    reason: str | None = None
    try:
        stats = model.evaluate(dataset=data.validation, skip_save_eval_stats=True,
                               skip_save_predictions=True, collect_overall_stats=False)[0]
        ludwig_rmse = float(stats[output_feature]["root_mean_squared_error"])
        if onnx_rmse is not None:
            matches = bool(abs(onnx_rmse - ludwig_rmse) <= 0.02 * max(ludwig_rmse, 1e-9))
    except Exception as exc:  # the ONNX number is still reported; only the comparison is unavailable
        reason = f"could not compare with Ludwig's own evaluation: {type(exc).__name__}: {exc}"
    return RuntimeEval(format="onnx", runtime="onnxruntime", metric="rmse", value=onnx_rmse,
                       n=int(count), matches_training=matches, reason=reason,
                       provider=_session_provider(session))


def predict(artifact: Artifact, inputs: Any) -> Any:
    """CONTRACTS §3.1: raw inputs in, the task's response shape out.

    Uses the live, reloaded LudwigModel's own ``predict()`` rather than
    re-deriving preprocessing by hand a second time (as runtime_eval does for
    a whole split, where going through onnxruntime IS the point): a single
    ad-hoc row trips Ludwig's training-data validation (e.g. "a category
    column needs >=2 distinct values") when forced through the training
    preprocessing path, and Ludwig's predict path is the one actually meant
    for exactly this -- new, unlabelled rows.
    """
    model = _reload_for_preprocessing(artifact)
    input_names = [f.name for f in model.config_obj.input_features]
    output_feature = model.config_obj.output_features[0].name
    labels = _labels(model, output_feature)

    out_cfg = model.config_obj.output_features[0]
    if out_cfg.type == "timeseries":
        # {"series": [recent values, oldest first], "horizon": n} or a bare list; response {"forecast": [...]}
        import pandas as pd

        series = inputs.get("series") if isinstance(inputs, dict) else inputs
        if not isinstance(series, (list, tuple)) or not series:
            raise ValueError('forecasting predict needs {"series": [<recent values, oldest first>], "horizon": n}')
        horizon = int(inputs.get("horizon", out_cfg.horizon)) if isinstance(inputs, dict) else int(out_cfg.horizon)
        col = out_cfg.column
        frame = pd.DataFrame({col: [float(v) for v in series]})
        # Ludwig 0.17.9's forecast() puts its inputs on the default device but does not move a freshly loaded model
        # (recurrent encoders then fail: "input at cuda:0 and parameter at cpu"): place the model where it looks.
        from ludwig.utils.torch_utils import get_torch_device

        model.model.to(get_torch_device())
        fc = model.forecast(dataset=frame, horizon=horizon)
        return {"forecast": [float(v) for v in fc[col]], "horizon": horizon}

    if isinstance(inputs, dict) and "file" in inputs and len(input_names) == 1:
        # CLI --file / the HTTP multipart `file` field (CONTRACTS §3.1): one
        # raw file for a single-input task (image/audio classification).
        row = {input_names[0]: inputs["file"]}
    elif isinstance(inputs, dict):
        # {"inputs": {<feature name>: ...}} already unwrapped by the caller --
        # every input feature the model has must be present (a tabular model
        # with 5 input columns needs all 5, not just the first).
        missing = [n for n in input_names if n not in inputs]
        if missing:
            raise ValueError(f"predict inputs missing required features: {missing}")
        row = {n: inputs[n] for n in input_names}
    else:
        row = {input_names[0]: inputs}

    import pandas as pd
    df = pd.DataFrame([row])
    preds_df, _ = model.predict(dataset=df, skip_save_predictions=True, skip_save_unprocessed_output=True)
    pred_row = preds_df.iloc[0]

    if labels is not None:
        proba_col = f"{output_feature}_probabilities"
        if proba_col in pred_row:
            probs = list(pred_row[proba_col])
            order = sorted(range(len(probs)), key=lambda i: probs[i], reverse=True)[:5]
            return {"top": [{"label": labels[i], "score": float(probs[i])} for i in order]}
        pred_label = str(pred_row[f"{output_feature}_predictions"])
        return {"top": [{"label": pred_label, "score": 1.0}]}

    value = float(pred_row[f"{output_feature}_predictions"])
    proba_col = f"{output_feature}_probability"
    proba = float(pred_row[proba_col]) if proba_col in pred_row else None
    return {"prediction": value, "proba": proba}


_EVAL_SAMPLE_ROWS = 5000


def predictions(result: TrainResult, resolved: ResolvedSpec, data: MaterialisedData) -> Any:
    """Score every split with the trained model for the report's ``evaluation``
    block (CONTRACTS §2.6): category outputs -> classification, number outputs ->
    regression, anything else -> None (the generic block). A split bigger than
    ``_EVAL_SAMPLE_ROWS`` is scored on a seeded sample and says so."""
    from sakura.atelier.evaluation import ModelPredictions, SplitPredictions

    if result.checkpoint is None and result.model is None:
        return None
    from ludwig.api import LudwigModel

    # The checkpoint on disk, as export() does: the in-memory model of a resumed run can differ from it.
    key = str(result.checkpoint.path) if result.checkpoint else None
    if key is not None:
        if key not in _LOADED:
            _LOADED[key] = LudwigModel.load(key, callbacks=[])
        model = _LOADED[key]
    else:
        model = result.model
    out_cfg = model.config_obj.output_features[0]
    if out_cfg.type not in ("category", "number"):
        return None
    out_col = out_cfg.column
    in_cols = [f.column for f in model.config_obj.input_features]
    classes = _labels(model, out_cfg.name) if out_cfg.type == "category" else None
    pred = ModelPredictions(kind="classification" if classes else "regression", splits={}, classes=classes)
    if out_cfg.type == "category" and not classes:
        return None

    frames = {"train": data.train, "validation": data.validation, "test": (data.extra or {}).get("test")}
    for name, df in frames.items():
        if df is None or not hasattr(df, "iloc") or len(df) == 0 or out_col not in df.columns:
            continue
        total = len(df)
        if total > _EVAL_SAMPLE_ROWS:
            df = df.sample(n=_EVAL_SAMPLE_ROWS, random_state=resolved.spec.seed)
            pred.notes.append(f"{name} metrics are computed on a {_EVAL_SAMPLE_ROWS}-row sample of {total} rows")
        preds_df, _ = model.predict(dataset=df, skip_save_predictions=True, skip_save_unprocessed_output=True)
        inputs = [{c: row[c] for c in in_cols if c in df.columns} for row in df.to_dict("records")]
        proba: list[list[float]] | None
        y_pred: list[Any]
        y_true: list[Any]
        if classes:
            proba = [[float(v) for v in row] for row in preds_df[f"{out_cfg.name}_probabilities"]]
            y_pred = [str(v) for v in preds_df[f"{out_cfg.name}_predictions"]]
            y_true = [str(v) for v in df[out_col]]
        else:
            proba = None
            y_pred = [float(v) for v in preds_df[f"{out_cfg.name}_predictions"]]
            y_true = [float(v) for v in df[out_col]]
        pred.splits[name] = SplitPredictions(y_true=y_true, y_pred=y_pred, proba=proba, inputs=inputs,
                                             total_rows=total, row_ids=list(df.index))
    return pred


#: Functions the notebook shows by source, in pipeline order (CONTRACTS §2.5).
NOTEBOOK_CELLS = (compile_config, train, _final_metrics, export, _export_onnx, runtime_eval)


class _Backend:
    name = NAME
    tasks = TASKS
    NOTEBOOK_CELLS = NOTEBOOK_CELLS
    train = staticmethod(train)
    export = staticmethod(export)
    runtime_eval = staticmethod(runtime_eval)
    predictions = staticmethod(predictions)
    predict = staticmethod(predict)


BACKEND = _Backend()

__all__ = ["BACKEND", "compile_config"]
