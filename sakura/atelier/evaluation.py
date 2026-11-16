"""The ``evaluation`` block of ``report.json`` (CONTRACTS §2.6).

Per-split metrics, class balance, a confusion matrix, per-class scores, the
misclassified validation rows and plain-language warnings -- what an ML
engineer reads to decide whether a run is trustworthy.

Kept free of any training stack: the metric helpers take plain lists, scikit-learn
and numpy are imported inside the functions that need them (the engine core must
import without them, CONTRACTS §2), and a missing scikit-learn skips the block
with a warning instead of failing the job. A backend takes part by exposing
``predictions(result, resolved, data) -> ModelPredictions | None``; any other
backend gets the generic block shaped from the metrics it already reports.
"""

from __future__ import annotations

import base64
import io
import math
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

#: A validation split smaller than this, or a class with fewer rows than
#: MIN_CLASS_ROWS in it, makes every validation number noisy.
MIN_VALIDATION_ROWS = 30
MIN_CLASS_ROWS = 10
#: The majority class holding more than this share of a split is an imbalance.
IMBALANCE_SHARE = 0.70
MAX_MISCLASSIFIED = 20
#: Longest side of a misclassified image's thumbnail, and its JPEG quality:
#: about 3 kB each, so the 20 rows add well under 100 kB to report.json.
THUMBNAIL_PX = 96
THUMBNAIL_QUALITY = 75
_IMAGE_SUFFIXES = frozenset({".png", ".jpg", ".jpeg", ".bmp", ".gif", ".webp", ".tif", ".tiff"})
#: Input cells that name a dataset file, whose relative path identifies a misclassified row.
_MEDIA_SUFFIXES = _IMAGE_SUFFIXES | frozenset({".wav", ".flac", ".mp3", ".ogg"})
#: Ludwig's own default when the spec does not set trainer.early_stop.
DEFAULT_EARLY_STOP = 5
_MAX_CELL_CHARS = 200
_SPLIT_ORDER = ("train", "validation", "test")


@dataclass
class SplitPredictions:
    """One split scored by the trained model."""

    y_true: list[Any]
    y_pred: list[Any]
    #: Classification only: one row per example, columns in ``classes`` order.
    proba: list[list[float]] | None = None
    #: The raw input columns of each example (for the misclassified table).
    inputs: list[dict[str, Any]] | None = None
    #: Rows the split really has, when ``y_true`` is a sample of it.
    total_rows: int | None = None
    #: Each example's row id in the split (the frame's index), so a
    #: misclassified row can be told apart from another with the same file
    #: name in a different class folder. Positions are used when absent.
    row_ids: list[Any] | None = None


@dataclass
class ModelPredictions:
    kind: str  # "classification" | "regression"
    splits: dict[str, SplitPredictions]
    classes: list[str] | None = None
    notes: list[str] = field(default_factory=list)


def _finite(x: Any) -> float | None:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return round(v, 6) if math.isfinite(v) else None


def _put(out: dict[str, Any], key: str, value: Any) -> None:
    v = _finite(value)
    if v is not None:
        out[key] = v


def _jsonable(value: Any) -> Any:
    """A raw input cell as JSON: numpy scalars unwrapped, long text cut."""
    if hasattr(value, "item") and not isinstance(value, (str, bytes)):
        try:
            value = value.item()
        except (ValueError, AttributeError):
            value = str(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, str) and len(value) > _MAX_CELL_CHARS:
        return value[:_MAX_CELL_CHARS] + "..."
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)[:_MAX_CELL_CHARS]


# ------------------------------------------------------------------ metrics

def label_distribution(labels: Sequence[Any], classes: Sequence[str] | None = None) -> dict[str, int]:
    counts = Counter(str(x) for x in labels)
    if classes is not None:
        return {c: counts.get(c, 0) for c in classes}
    return dict(sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])))


def majority_baseline(distribution: dict[str, int]) -> float | None:
    """Accuracy of always predicting the most frequent class."""
    total = sum(distribution.values())
    return round(max(distribution.values()) / total, 6) if total else None


def classification_metrics(y_true: Sequence[Any], y_pred: Sequence[Any],
                           proba: Sequence[Sequence[float]] | None, classes: Sequence[str]) -> dict[str, Any]:
    """n, loss (cross-entropy when probabilities exist), accuracy, macro/weighted F1,
    macro precision/recall and ROC AUC (binary, else macro one-vs-rest) of one split."""
    from sklearn.metrics import (accuracy_score, f1_score, log_loss, precision_score, recall_score,
                                 roc_auc_score)

    yt = [str(v) for v in y_true]
    yp = [str(v) for v in y_pred]
    labels = list(classes)
    out: dict[str, Any] = {"n": len(yt)}
    if not yt:
        return out
    if proba is not None:
        try:
            _put(out, "loss", log_loss(yt, proba, labels=labels))
        except ValueError:
            pass
    _put(out, "accuracy", accuracy_score(yt, yp))
    _put(out, "f1_macro", f1_score(yt, yp, labels=labels, average="macro", zero_division=0))
    _put(out, "f1_weighted", f1_score(yt, yp, labels=labels, average="weighted", zero_division=0))
    _put(out, "precision_macro", precision_score(yt, yp, labels=labels, average="macro", zero_division=0))
    _put(out, "recall_macro", recall_score(yt, yp, labels=labels, average="macro", zero_division=0))
    if proba is not None and len(set(yt)) >= 2:
        try:
            if len(labels) == 2:
                auc = roc_auc_score([1 if v == labels[1] else 0 for v in yt], [row[1] for row in proba])
            else:
                present = [i for i, c in enumerate(labels) if c in set(yt)]
                if len(present) == len(labels):
                    auc = roc_auc_score(yt, proba, multi_class="ovr", average="macro", labels=labels)
                elif len(present) == 2:  # only two classes occur here: a binary curve between them
                    auc = roc_auc_score([1 if v == labels[present[1]] else 0 for v in yt],
                                        [row[present[1]] for row in proba])
                else:  # a class absent from this split has no one-vs-rest curve: score the ones present
                    sub = [[row[i] for i in present] for row in proba]
                    total = [sum(r) or 1.0 for r in sub]
                    sub = [[v / t for v in r] for r, t in zip(sub, total)]
                    auc = roc_auc_score(yt, sub, multi_class="ovr", average="macro",
                                        labels=[labels[i] for i in present])
            _put(out, "roc_auc", auc)
        except ValueError:
            pass
    return out


def regression_metrics(y_true: Sequence[float], y_pred: Sequence[float]) -> dict[str, Any]:
    """n, loss (mean squared error, Ludwig's default), rmse and mae of one split."""
    n = len(y_true)
    out: dict[str, Any] = {"n": n}
    if not n:
        return out
    errs = [float(p) - float(t) for t, p in zip(y_true, y_pred)]
    mse = sum(e * e for e in errs) / n
    _put(out, "loss", mse)
    _put(out, "rmse", math.sqrt(mse))
    _put(out, "mae", sum(abs(e) for e in errs) / n)
    return out


def confusion(y_true: Sequence[Any], y_pred: Sequence[Any], classes: Sequence[str], split: str) -> dict[str, Any]:
    from sklearn.metrics import confusion_matrix

    labels = list(classes)
    m = confusion_matrix([str(v) for v in y_true], [str(v) for v in y_pred], labels=labels)
    return {"split": split, "labels": labels, "matrix": [[int(v) for v in row] for row in m]}


def per_class(y_true: Sequence[Any], y_pred: Sequence[Any], classes: Sequence[str]) -> list[dict[str, Any]]:
    from sklearn.metrics import precision_recall_fscore_support

    labels = list(classes)
    p, r, f, s = precision_recall_fscore_support([str(v) for v in y_true], [str(v) for v in y_pred],
                                                 labels=labels, zero_division=0)
    return [{"label": lab, "precision": round(float(p[i]), 6), "recall": round(float(r[i]), 6),
             "f1": round(float(f[i]), 6), "support": int(s[i])} for i, lab in enumerate(labels)]


def thumbnail(value: Any, size: int = THUMBNAIL_PX) -> str | None:
    """A small JPEG data URI of an image input cell, or None.

    An image row's input is a path on the machine that trained it, which no
    reader of the report can open; the picture itself is what tells them why
    the model got it wrong. Only a path to an existing image file qualifies,
    and any failure (Pillow absent, unreadable file) is simply no thumbnail.
    """
    if not isinstance(value, str) or Path(value).suffix.lower() not in _IMAGE_SUFFIXES:
        return None
    try:
        from PIL import Image
    except ImportError:
        return None
    try:
        with Image.open(value) as src:
            rgb = src.convert("RGB")
        rgb.thumbnail((size, size))
        buf = io.BytesIO()
        rgb.save(buf, format="JPEG", quality=THUMBNAIL_QUALITY)
    except (OSError, ValueError):
        return None
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def media_path(value: Any, root: str | Path | None = None) -> str | None:
    """The dataset-relative path of a media input cell (``airplane/022.png``), or None.

    File names alone repeat across class folders (``022.png`` in both
    ``airplane/`` and ``bird/``), so the path keeps the folders under the
    dataset root; without a root (or outside it) it keeps the file's parent
    folder, which for an image folder is its class."""
    if not isinstance(value, str) or Path(value).suffix.lower() not in _MEDIA_SUFFIXES:
        return None
    p = Path(value)
    if root is not None and p.is_absolute() and p.is_relative_to(root):
        return p.relative_to(root).as_posix()
    return "/".join(p.parts[-2:])


def misclassified(split: SplitPredictions, limit: int = MAX_MISCLASSIFIED, *, split_name: str = "validation",
                  media_root: str | Path | None = None) -> list[dict[str, Any]]:
    """The first ``limit`` wrong validation rows, most confident mistakes first.

    Every row has a ``sample_id`` unique in the report (``validation:17``, the
    row's id in its split, or its position when the split has no ``row_ids``); a row whose input is a dataset file also has
    its relative ``path`` (:func:`media_path`). A row whose input is an image
    file also carries ``thumbnail``; only the kept rows are opened, so a large
    split costs at most ``limit`` reads."""
    wrong: list[tuple[int, Any, Any, float | None]] = []
    for i, (t, p) in enumerate(zip(split.y_true, split.y_pred)):
        if str(t) == str(p):
            continue
        conf = None
        if split.proba is not None and split.proba[i] is not None:
            conf = _finite(max(split.proba[i]))
        wrong.append((i, t, p, conf))
    wrong.sort(key=lambda w: -(w[3] if w[3] is not None else -1.0))
    rows: list[dict[str, Any]] = []
    for i, t, p, conf in wrong[:limit]:
        raw = split.inputs[i] if split.inputs else {}
        row_id = split.row_ids[i] if split.row_ids is not None and i < len(split.row_ids) else i
        row: dict[str, Any] = {"sample_id": f"{split_name}:{_jsonable(row_id)}",
                               "inputs": {k: _jsonable(v) for k, v in raw.items()},
                               "true": str(t), "predicted": str(p), "confidence": conf}
        path = next((mp for mp in (media_path(v, media_root) for v in raw.values()) if mp), None)
        if path:
            row["path"] = path
        thumb = next((th for th in map(thumbnail, raw.values()) if th), None)
        if thumb:
            row["thumbnail"] = thumb
        rows.append(row)
    return rows


# ----------------------------------------------------------------- warnings

def training_params(resolved: Any) -> tuple[int | None, int | None]:
    """``(epochs, early_stop)`` the run effectively used: the preset's values, a
    ``backend_config.trainer`` override on top, Ludwig's default early stop otherwise."""
    hp = dict(getattr(resolved, "hyperparameters", None) or {})
    trainer = dict((getattr(getattr(resolved, "spec", None), "backend_config", None) or {}).get("trainer") or {})
    epochs = trainer.get("epochs", hp.get("epochs"))
    early = trainer.get("early_stop", hp.get("early_stop", DEFAULT_EARLY_STOP))
    try:
        epochs = int(epochs) if epochs is not None else None
        early = int(early) if early is not None else None
    except (TypeError, ValueError):
        return None, None
    return epochs, early


def evaluation_warnings(*, task: str | None, splits: dict[str, dict[str, Any]],
                        label_dist: dict[str, dict[str, int]] | None = None,
                        auto_split: bool = False, epochs: int | None = None,
                        early_stop: int | None = None, input_types: Sequence[str] = (),
                        split_strategy: str | None = None) -> list[str]:
    """The things a careful reader would flag before trusting the numbers."""
    out: list[str] = []
    n_val = (splits.get("validation") or {}).get("n")
    if isinstance(n_val, int) and n_val < MIN_VALIDATION_ROWS:
        out.append(f"validation split has only {n_val} rows (< {MIN_VALIDATION_ROWS}): "
                   "its metrics are too noisy to compare runs")
    dist = label_dist or {}
    val_dist = dist.get("validation")
    if val_dist:
        thin = [f"{c} ({n})" for c, n in val_dist.items() if n < MIN_CLASS_ROWS]
        if thin:
            shown = ", ".join(thin[:5]) + (f" and {len(thin) - 5} more" if len(thin) > 5 else "")
            out.append(f"validation has fewer than {MIN_CLASS_ROWS} rows for class {shown}: "
                       "per-class scores are unreliable")
    ref = dist.get("train") or val_dist
    if ref:
        total = sum(ref.values())
        top_label, top_n = max(ref.items(), key=lambda kv: kv[1])
        if total and top_n / total > IMBALANCE_SHARE:
            out.append(f"class imbalance: '{top_label}' is {top_n / total:.0%} of the "
                       f"{'train' if dist.get('train') else 'validation'} rows; "
                       "accuracy alone is misleading, read f1_macro and the per-class table")
    if auto_split:
        how = "drawn per class" if split_strategy == "stratified" else "random rows"
        out.append("the validation split was made automatically from validation_fraction "
                   f"({how} of the same dataset): metrics may be optimistic if rows are near-duplicates")
    if epochs is not None and early_stop is not None and early_stop > 0 and early_stop >= epochs:
        out.append(f"early_stop={early_stop} can never trigger with epochs={epochs}")
    if task == "tabular_classification" and len(input_types) == 1 and input_types[0] == "text":
        out.append("tabular_classification with a single text input: consider text_classification, "
                   "which has text-specific encoders")
    return out


# -------------------------------------------------------------------- build

def build_classification(pred: ModelPredictions, media_root: str | Path | None = None) -> dict[str, Any]:
    classes = list(pred.classes or sorted({str(v) for s in pred.splits.values() for v in s.y_true}))
    splits: dict[str, Any] = {}
    dists: dict[str, dict[str, int]] = {}
    baseline: dict[str, float] = {}
    for name in _SPLIT_ORDER:
        sp = pred.splits.get(name)
        if sp is None:
            continue
        splits[name] = classification_metrics(sp.y_true, sp.y_pred, sp.proba, classes)
        dists[name] = label_distribution(sp.y_true, classes)
        b = majority_baseline(dists[name])
        if b is not None:
            baseline[name] = b
    block: dict[str, Any] = {"splits": splits, "classes": classes, "label_distribution": dists,
                             "majority_baseline": baseline}
    val = pred.splits.get("validation")
    if val is not None and val.y_true:
        block["confusion_matrix"] = confusion(val.y_true, val.y_pred, classes, "validation")
        block["per_class"] = per_class(val.y_true, val.y_pred, classes)
        block["misclassified"] = misclassified(val, media_root=media_root)
    return block


def build_regression(pred: ModelPredictions) -> dict[str, Any]:
    return {"splits": {n: regression_metrics(pred.splits[n].y_true, pred.splits[n].y_pred)
                       for n in _SPLIT_ORDER if n in pred.splits}}


def _rows(obj: Any) -> int | None:
    try:
        return len(obj)
    except TypeError:
        return None


def generic_evaluation(metrics: dict[str, float], data: Any) -> dict[str, Any]:
    """Whatever per-split numbers a backend already reports: its final metrics
    are the validation split's; the other splits only get their size."""
    splits: dict[str, Any] = {}
    for name in _SPLIT_ORDER:
        n = _rows(getattr(data, name, None)) if name != "test" else _rows((getattr(data, "extra", None) or {}).get("test"))
        if n is not None:
            splits[name] = {"n": n}
    val = splits.setdefault("validation", {}) if metrics else splits.get("validation")
    if val is not None:
        for k, v in (metrics or {}).items():
            _put(val, k, v)
    return {"splits": splits}


def _auto_split(spec: Any) -> bool:
    split = getattr(getattr(spec, "data", None), "split", None)
    return getattr(split, "validation_fraction", None) is not None


def build_evaluation(backend: Any, result: Any, resolved: Any, data: Any) -> dict[str, Any]:
    """The report's ``evaluation`` block. Never raises: a failure becomes a warning."""
    spec = getattr(resolved, "spec", None)
    warnings: list[str] = []
    block: dict[str, Any] = {}
    own = getattr(result, "evaluation", None)
    if own:  # the backend measured its own held-out splits (the decision task): its block is the report's
        block = dict(own)
        block.setdefault("warnings", [])
        block.setdefault("splits", {})
        return block
    try:
        pred = None
        hook = getattr(backend, "predictions", None)
        if hook is not None:
            try:
                import sklearn  # noqa: F401
            except ImportError:
                warnings.append("scikit-learn is not installed: evaluation reduced to the backend's own metrics")
            else:
                pred = hook(result, resolved, data)
        if pred is None:
            block = generic_evaluation(getattr(result, "metrics", {}) or {}, data)
        elif pred.kind == "classification":
            block = build_classification(pred, media_root=(getattr(data, "extra", None) or {}).get("media_root"))
        else:
            block = build_regression(pred)
        if pred is not None:
            warnings.extend(pred.notes)
        features = getattr(spec, "features", None)
        input_types = [getattr(f, "type", "") for f in getattr(features, "inputs", [])] if features else []
        epochs, early = training_params(resolved)
        warnings = evaluation_warnings(
            task=getattr(spec, "task", None), splits=block.get("splits", {}),
            label_dist=block.get("label_distribution"), auto_split=_auto_split(spec),
            epochs=epochs, early_stop=early, input_types=input_types,
            split_strategy=((getattr(data, "extra", None) or {}).get("data_split") or {}).get("strategy")) + warnings
    except Exception as exc:
        warnings.append(f"evaluation could not be computed: {type(exc).__name__}: {exc}")
    if getattr(result, "augmentation", None):
        block["augmentation"] = result.augmentation
    block["warnings"] = warnings
    block.setdefault("splits", {})
    return block
