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

NAME = "ludwig"
TASKS = frozenset({
    "image_classification", "audio_classification",
    "tabular_classification", "tabular_regression",
})

# Ludwig nests a FRESH run under <output_directory>/<experiment_name>_<model_name>/model/.
# Fixed names make that deterministic instead of depending on Ludwig's own
# defaults ("api_experiment"/"run"). A RESUMED run (model_resume_path= the
# previous run's model dir) is a different story: observed on a GPU host, Ludwig
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


def _deep_merge(base: Any, override: Any) -> Any:
    """``backend_config`` merged last over the compiled config -- the
    "deep dive at will" escape hatch (CONTRACTS §1)."""
    if isinstance(base, dict) and isinstance(override, dict):
        merged = dict(base)
        for k, v in override.items():
            merged[k] = _deep_merge(base.get(k), v) if k in base else v
        return merged
    return override


def compile_config(resolved: ResolvedSpec) -> dict[str, Any]:
    spec = resolved.spec
    hp = resolved.hyperparameters
    task = spec.task
    output_name = spec.features.output.name

    if task == "image_classification":
        in_name = spec.features.inputs[0].name
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
                            # on a GPU host: a resumed CIFAR run trained to 0.616
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
        in_name = spec.features.inputs[0].name
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
        input_features = [{"name": f.name, "type": type_map[f.type]} for f in spec.features.inputs]
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
    else:
        raise ValueError(f"ludwig backend does not implement task {task!r}")

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


def train(spec: ResolvedSpec, data: MaterialisedData, out: RunDir,
          metrics: MetricsSink, resume: Checkpoint | None) -> TrainResult:
    from ludwig.api import LudwigModel

    from sakura.adapters.ludwig import LudwigAdapter

    config = compile_config(spec)
    output_feature = spec.spec.features.output.name
    callback = LudwigAdapter(metrics, output_feature=output_feature)
    train_kwargs: dict[str, Any] = {}
    if resume is None:
        model = LudwigModel(config=config, callbacks=[callback])
    else:
        # Continue training = a WARM START from the delivered model, not
        # Ludwig's `model_resume_path`. That mechanism resumes an INTERRUPTED
        # run from its saved config and progress counters, and on a run that
        # had finished it did two wrong things, both measured on a GPU host: it trained
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
        train_kwargs["training_set_metadata"] = model.training_set_metadata
        _first_param_fingerprint(model, out, "resume_start")

    start = time.time()
    train_stats, _preprocessed, _output_directory = model.train(
        training_set=data.train,
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
    return TrainResult(
        model=model,
        metrics=_final_metrics(train_stats, output_feature),
        train_seconds=train_seconds,
        checkpoint=Checkpoint(path=model_dir, model=spec.spec.model, task=spec.spec.task),
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


def _export_onnx(model: Any, output_feature: str, path: Path) -> dict[str, Any]:
    import torch

    # Export on CPU, always: some torchvision-backed Ludwig encoders (e.g.
    # the resnet image encoder) compute their lazily-cached `.output_shape`
    # property by running a CPU-only dummy tensor through the model the
    # first time it is accessed, regardless of the model's actual device --
    # if the real model is still on GPU (as it is right after training),
    # that first forward crashes with a CPU/CUDA tensor mismatch. Moving the
    # whole model to CPU first makes every tensor involved agree, and export
    # never needs the GPU anyway.
    torch_model = model.model.to("cpu")  # the ECD nn.Module (what to_torchscript(model_only=True) wraps)
    sample_inputs, sample_outputs = _onnx_sample_io(torch_model)
    input_names = list(sample_inputs.keys())
    output_keys = sorted(k for k in sample_outputs if k.startswith(f"{output_feature}::"))
    if not output_keys:  # defensive: an unexpected key scheme still exports *something*, never nothing
        output_keys = sorted(sample_outputs.keys())
    output_names = [k.replace("::", "__") for k in output_keys]

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
    # (the live in-memory object) directly: observed on a GPU host, a resumed run's
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
        scripted = model.to_torchscript(model_only=True, device="cpu")  # see _export_onnx's CPU note
        scripted.save(str(path))
    else:
        raise ValueError(f"ludwig backend does not support export format {fmt!r}")

    return Artifact(path=path, format=fmt, sha256=_sha256_file(path), bytes=path.stat().st_size)


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


def runtime_eval(artifact: Artifact, data: MaterialisedData) -> RuntimeEval:
    """Re-score ``artifact`` with onnxruntime on the validation split
    (CONTRACTS §2.1/§2.3): feed Ludwig's own preprocessed validation batches
    straight into the ONNX graph (no normalisation of our own -- the ONNX
    graph starts exactly where Ludwig's preprocessing ends, which is the
    whole point of exporting "the trained torch inference module on
    preprocessed tensors")."""
    try:
        import numpy as np
        import onnxruntime as ort
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

    try:
        session = ort.InferenceSession(str(artifact.path),
                                        providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
    except Exception as exc:
        return RuntimeEval(format=artifact.format, runtime="onnxruntime", metric="accuracy", value=None, n=0,
                            matches_training=None, reason=f"session load failed: {exc}")

    try:
        model = _reload_for_preprocessing(artifact)
        output_feature = model.config_obj.output_features[0].name
        proc_columns = _proc_columns(model)
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
        # on a GPU host: a model trained to 0.29 accuracy scored 0.06 through this
        # exact bug, independent of CPU export, independent of resume).
        preprocessed = model.preprocess(training_set=data.validation,
                                         training_set_metadata=model.training_set_metadata,
                                         skip_save_processed_input=True)
    except Exception as exc:
        return RuntimeEval(format=artifact.format, runtime="onnxruntime", metric="accuracy", value=None, n=0,
                            matches_training=None, reason=f"could not reproduce preprocessing: {exc}")

    val_ds = preprocessed.training_set
    n = 0
    correct = 0
    sq_err = 0.0
    is_classification = True
    with val_ds.initialize_batcher(batch_size=128, should_shuffle=False) as batcher:
        while True:
            batch = batcher.next_batch()
            feed = {name: np.asarray(batch[proc_columns[name]], dtype=np.float32) for name in input_order}
            onnx_outputs = session.run(output_order, feed)
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
        return RuntimeEval(format="onnx", runtime="onnxruntime", metric="accuracy",
                            value=correct / n if n else None, n=n, matches_training=None)
    return RuntimeEval(format="onnx", runtime="onnxruntime", metric="rmse",
                        value=(sq_err / n) ** 0.5 if n else None, n=n, matches_training=None)


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


class _Backend:
    name = NAME
    tasks = TASKS
    train = staticmethod(train)
    export = staticmethod(export)
    runtime_eval = staticmethod(runtime_eval)
    predict = staticmethod(predict)


BACKEND = _Backend()

__all__ = ["BACKEND", "compile_config"]
