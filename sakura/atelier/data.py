"""Dataset materialisation: turn a spec's ``data`` block into a pandas
DataFrame per split, on local disk, plus the manifest whose hash pins the
run (CONTRACTS §1.2).

Every format converges on the same shape: ``MaterialisedData.train`` /
``.validation`` are pandas DataFrames whose columns are named exactly as
``spec.features`` names them. That is deliberate -- it is also exactly what
Ludwig's config wants (an image/audio column holding file paths, a category
column holding labels, numeric columns holding numbers), so the backend does
not need a second per-format data layer.

This module is imported by the CLI and by backends; it must stay free of any
*training* stack (ludwig/ultralytics/tabpfn/peft), per types.py's contract,
but may use torch/torchvision/pandas -- those are sakura-ml's own core deps.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

import pandas as pd

from sakura.atelier.spec import AtelierSpec, Data
from sakura.atelier.types import MaterialisedData

IMAGE_EXTS = frozenset({".png", ".jpg", ".jpeg", ".bmp", ".webp"})
AUDIO_EXTS = frozenset({".wav", ".flac", ".mp3", ".ogg"})


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _canonical_manifest_json(entries: list[dict[str, Any]]) -> str:
    # Sorted by path: the manifest -- and so its hash -- must not depend on
    # filesystem iteration order (CONTRACTS §1.2).
    ordered = sorted(entries, key=lambda e: str(e["path"]))
    return json.dumps(ordered, sort_keys=True, separators=(",", ":"))


def write_manifest(data_dir: Path, entries: list[dict[str, Any]]) -> str:
    text = _canonical_manifest_json(entries)
    (data_dir / "manifest.json").write_text(text, encoding="utf-8")
    return _sha256_bytes(text.encode("utf-8"))


def _entries_for_tree(root: Path, *, exclude: frozenset[str] = frozenset({"manifest.json"})) -> list[dict[str, Any]]:
    entries = []
    for p in sorted(root.rglob("*")):
        if not p.is_file() or p.name in exclude:
            continue
        entries.append({
            "path": str(p.relative_to(root)),
            "bytes": p.stat().st_size,
            "sha256": _sha256_file(p),
        })
    return entries


def _fetch_local(uri: str, dest_dir: Path) -> Path:
    """Resolve a non-hf:// data uri to a local path, downloading if remote.
    Supports file:// and plain local paths (already on disk, e.g. an NFS/LXC
    mount) and https://. s3:// is not implemented -- flagged in the report."""
    parsed = urllib.parse.urlparse(uri)
    if parsed.scheme in ("", "file"):
        return Path(parsed.path if parsed.scheme else uri)
    if parsed.scheme in ("http", "https"):
        dest_dir.mkdir(parents=True, exist_ok=True)
        local = dest_dir / Path(parsed.path).name
        if not local.exists():
            urllib.request.urlretrieve(uri, local)  # noqa: S310 -- data uri is operator-supplied, not user input
        return local
    raise ValueError(f"data.uri scheme {parsed.scheme!r} is not supported yet (uri={uri!r})")


def _select_columns(df: pd.DataFrame, spec: AtelierSpec) -> pd.DataFrame:
    """Rename/select columns to the exact names spec.features uses. Exact
    name matches are required -- renaming by position would silently feed
    the wrong column into the model on a dataset whose schema drifted."""
    wanted = [f.name for f in spec.features.inputs] + [spec.features.output.name]
    missing = [w for w in wanted if w not in df.columns]
    if missing:
        raise ValueError(
            f"dataset columns {list(df.columns)} are missing {missing} "
            f"required by spec.features (inputs + output)"
        )
    return df[wanted]


def _labels_from_column(df: pd.DataFrame, column: str) -> list[str] | None:
    if column not in df.columns:
        return None
    col = df[column]
    # pandas 3 reads text as the `str` dtype, not `object`: test by kind, not
    # by dtype name, or string targets silently get no label list.
    if (pd.api.types.is_string_dtype(col) or pd.api.types.is_object_dtype(col)
            or isinstance(col.dtype, pd.CategoricalDtype) or pd.api.types.is_bool_dtype(col)):
        return sorted(str(v) for v in col.dropna().unique())
    return None


# --------------------------------------------------------------- hf://

def _materialise_hf(data: Data, spec: AtelierSpec, data_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, list[str] | None]:
    try:
        import datasets
    except ImportError as exc:  # pragma: no cover -- exercised only where `datasets` is installed
        raise ImportError("materialising a hf:// source needs the `datasets` package") from exc

    repo = data.uri.removeprefix("hf://")

    def _load(split_expr: str) -> Any:
        return datasets.load_dataset(repo, split=split_expr)

    if data.split.validation_fraction is not None:
        full = _load(data.split.train or "train")
        parts = full.train_test_split(test_size=data.split.validation_fraction, seed=spec.seed)
        train_ds, val_ds = parts["train"], parts["test"]
    else:
        assert data.split.train is not None and data.split.validation is not None
        train_ds, val_ds = _load(data.split.train), _load(data.split.validation)

    image_cols = {n for n, f in train_ds.features.items() if isinstance(f, datasets.Image)}
    audio_cols = {n for n, f in train_ds.features.items() if isinstance(f, datasets.Audio)}
    class_label_cols = {n: f for n, f in train_ds.features.items() if isinstance(f, datasets.ClassLabel)}

    def _to_frame(ds: Any, split_name: str) -> pd.DataFrame:
        df = ds.to_pandas()
        for col in image_cols:
            out_dir = data_dir / "images" / split_name
            out_dir.mkdir(parents=True, exist_ok=True)
            paths = []
            for i, rec in enumerate(df[col]):
                img = rec["bytes"] if isinstance(rec, dict) else rec
                p = out_dir / f"{i}.png"
                if isinstance(img, (bytes, bytearray)):
                    p.write_bytes(img)
                else:  # PIL.Image
                    img.convert("RGB").save(p)
                # Absolute, not relative to data_dir: Ludwig resolves a
                # dataframe's image/audio path column against its OWN idea of
                # cwd, not ours, so a relative path here is a coin flip (CIFAR
                # happened to still read most files; esc50 did not resolve
                # audio paths at all). The manifest still hashes file
                # CONTENT by data_dir-relative path -- this is just what the
                # training dataframe cell holds.
                paths.append(str(p.resolve()))
            df[col] = paths
        for col in audio_cols:
            out_dir = data_dir / "audio" / split_name
            out_dir.mkdir(parents=True, exist_ok=True)
            paths = []
            for i, rec in enumerate(df[col]):
                p = out_dir / f"{i}.wav"
                if hasattr(rec, "get_all_samples"):
                    # An Audio column accessed through the Dataset object (not
                    # .to_pandas()) decodes via torchcodec's AudioDecoder.
                    import soundfile as sf  # lazy: only hit by audio datasets
                    samples = rec.get_all_samples()
                    sf.write(p, samples.data.numpy().T, samples.sample_rate)
                elif isinstance(rec, dict) and rec.get("bytes") is not None:
                    # `.to_pandas()` does NOT decode Audio columns -- it hands
                    # back the raw encoded file bytes (already a valid .wav/
                    # .flac/... container) under "bytes", nothing to decode.
                    p.write_bytes(rec["bytes"])
                elif isinstance(rec, dict) and rec.get("path"):
                    shutil.copyfile(rec["path"], p)
                else:
                    import soundfile as sf  # lazy: only hit by audio datasets
                    sf.write(p, rec["array"], rec["sampling_rate"])
                paths.append(str(p.resolve()))
            df[col] = paths
        for col, feat in class_label_cols.items():
            df[col] = [feat.names[i] for i in df[col]]
        return df

    train_df = _to_frame(train_ds, "train")
    val_df = _to_frame(val_ds, "validation")
    labels = None
    out_col = spec.features.output.name
    if out_col in class_label_cols:
        labels = list(class_label_cols[out_col].names)
    return _select_columns(train_df, spec), _select_columns(val_df, spec), labels


# --------------------------------------------------------- csv / parquet

def _materialise_tabular(data: Data, spec: AtelierSpec, data_dir: Path, fmt: str) -> tuple[pd.DataFrame, pd.DataFrame, list[str] | None]:
    local = _fetch_local(data.uri, data_dir / "_download")
    if fmt == "csv":
        df = pd.read_csv(local)
    elif fmt == "parquet":
        df = pd.read_parquet(local)
    else:
        df = pd.read_json(local, lines=True)  # jsonl -- one record per line
    # Keep a copy of the source bytes in data_dir so the manifest pins the
    # exact table read, independent of where data.uri later points.
    pinned = data_dir / local.name
    if local.resolve() != pinned.resolve():
        shutil.copyfile(local, pinned)

    if data.split.validation_fraction is not None:
        val_df = df.sample(frac=data.split.validation_fraction, random_state=spec.seed)
        train_df = df.drop(val_df.index)
    else:
        # Explicit splits for a single-file table select by a row-range
        # expression "start:end" (e.g. "0:1000"), the simplest reading of
        # "split expressions" that a flat CSV/parquet can support.
        train_df = _row_slice(df, data.split.train)
        val_df = _row_slice(df, data.split.validation)

    labels = _labels_from_column(df, spec.features.output.name)
    return _select_columns(train_df, spec), _select_columns(val_df, spec), labels


def _row_slice(df: pd.DataFrame, expr: str | None) -> pd.DataFrame:
    if expr is None:
        return df
    start_s, _, end_s = expr.partition(":")
    start = int(start_s) if start_s else 0
    end = int(end_s) if end_s else len(df)
    return df.iloc[start:end]


# --------------------------------------------------- image_folder / audio_folder

_SPLIT_DIR_NAMES = frozenset({"train", "training", "validation", "val", "valid", "dev", "test"})

def _materialise_labelled_folder(data: Data, spec: AtelierSpec, data_dir: Path, exts: frozenset[str]) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """``<root>/<split>/<class>/<file>`` when splits are explicit (the
    directory name under root IS the split expression), or a single
    ``<root>/<class>/<file>`` tree split by ``validation_fraction``."""
    root = _fetch_local(data.uri, data_dir)
    in_col = spec.features.inputs[0].name
    out_col = spec.features.output.name

    def _scan(split_dir: Path) -> pd.DataFrame:
        rows = []
        for class_dir in sorted(p for p in split_dir.iterdir() if p.is_dir()):
            for f in sorted(class_dir.iterdir()):
                if f.suffix.lower() in exts:
                    rows.append({in_col: str(f.resolve()), out_col: class_dir.name})
        return pd.DataFrame(rows, columns=[in_col, out_col])

    if data.split.validation_fraction is not None:
        tops = {d.name.lower() for d in root.iterdir() if d.is_dir()}
        if tops and tops <= _SPLIT_DIR_NAMES:
            # Re-splitting a train/ + validation/ tree by fraction would train
            # a classifier whose "classes" are the split names.
            raise ValueError(
                f"{root} is already split into {sorted(tops)}/; set data.split "
                "{train: <dir>, validation: <dir>} instead of validation_fraction")
        full = _scan(root)
        val_df = full.sample(frac=data.split.validation_fraction, random_state=spec.seed)
        train_df = full.drop(val_df.index)
    else:
        assert data.split.train is not None and data.split.validation is not None
        train_df = _scan(root / data.split.train)
        val_df = _scan(root / data.split.validation)

    labels = sorted(set(train_df[out_col]) | set(val_df[out_col]))
    return train_df, val_df, labels


# ------------------------------------------------------------------ public

def materialise(spec: AtelierSpec, data_dir: Path) -> MaterialisedData:
    """Fetch/convert ``spec.data`` into ``data_dir``, write ``manifest.json``,
    and verify it against ``spec.data.sha256`` when the spec pins one.

    The returned shape is per-task, not a single universal frame (per-task shapes): image/audio/tabular-via-ludwig get pandas DataFrames
    (what Ludwig's config wants); object_detection gets a YOLO ``data.yaml``
    in ``.extra``; tabular-via-tabpfn gets ``(X, y)`` pairs; text_generation
    gets a list of ``{"prompt", "completion"}`` dicts -- each backend takes
    exactly the shape it needs with no second per-backend data layer.
    """
    data_dir.mkdir(parents=True, exist_ok=True)
    fmt = spec.data.format
    extra: dict[str, Any] = {}
    train: Any
    validation: Any
    labels: list[str] | None

    if fmt == "yolo":
        train, validation, labels, extra = _materialise_yolo(spec.data, spec, data_dir)
    elif fmt == "hf":
        train, validation, labels = _materialise_hf(spec.data, spec, data_dir)
    elif fmt == "jsonl" and spec.task == "text_generation":
        train, validation, labels = _materialise_text_generation(spec.data, spec, data_dir)
    elif fmt in ("csv", "parquet", "jsonl"):
        train, validation, labels = _materialise_tabular(spec.data, spec, data_dir, fmt)
    elif fmt == "image_folder":
        train, validation, labels = _materialise_labelled_folder(spec.data, spec, data_dir, IMAGE_EXTS)
    elif fmt == "audio_folder":
        train, validation, labels = _materialise_labelled_folder(spec.data, spec, data_dir, AUDIO_EXTS)
    else:
        raise ValueError(f"data.format {fmt!r} is not supported by sakura.atelier.data yet")

    entries = _entries_for_tree(data_dir)
    manifest_sha256 = write_manifest(data_dir, entries)
    if spec.data.sha256 and spec.data.sha256 != manifest_sha256:
        raise ValueError(
            f"data.sha256 mismatch: spec pins {spec.data.sha256}, materialised "
            f"manifest hashes to {manifest_sha256} -- a replay on different "
            "bytes is not a replay (CONTRACTS §1.2)"
        )

    if fmt in ("csv", "parquet") and spec.task in ("tabular_classification", "tabular_regression"):
        train, validation = _maybe_tabpfn_shape(spec, train, validation)

    return MaterialisedData(root=data_dir, train=train, validation=validation,
                             manifest_sha256=manifest_sha256, labels=labels, extra=extra)


def _maybe_tabpfn_shape(spec: AtelierSpec, train: pd.DataFrame, validation: pd.DataFrame) -> tuple[Any, Any]:
    """tabpfn's backend wants ``(X, y)`` pairs, not a single frame with the
    target column still inside it ."""
    from sakura.atelier.backends import backend_name
    if backend_name(spec.task, spec.model) != "tabpfn":
        return train, validation
    out_col = spec.features.output.name
    in_cols = [f.name for f in spec.features.inputs]
    return ((train[in_cols], train[out_col]), (validation[in_cols], validation[out_col]))


# ------------------------------------------------------------------ yolo

def _materialise_yolo(data: Data, spec: AtelierSpec, data_dir: Path) -> tuple[Any, Any, list[str], dict[str, Any]]:
    """The source (``data.uri``) is a directory already in Ultralytics' own
    layout: an ``images/{train,val}`` + sibling ``labels/{train,val}`` tree
    and its own ``data.yaml`` (``names`` + split dirs) -- quick-and-dirty per
    CONTRACTS ("models and datasets"): copy it into ``data_dir`` verbatim so
    the manifest pins it, and hand the backend the data.yaml path directly
    rather than re-deriving Ultralytics' own format."""
    import yaml

    src = _fetch_local(data.uri, data_dir)
    src_yaml = src / "data.yaml"
    if not src_yaml.exists():
        raise ValueError(f"data.format 'yolo' needs a data.yaml at {src_yaml} (Ultralytics layout)")

    dest = data_dir / "yolo"
    if dest.resolve() != src.resolve():
        shutil.copytree(src, dest, dirs_exist_ok=True)
    names_doc = yaml.safe_load((dest / "data.yaml").read_text(encoding="utf-8"))
    # `path:` is wherever the uploader's copy lived (or relative, which
    # Ultralytics resolves against ITS datasets dir, not this file): point it
    # at this copy so train/val resolve here, on any node.
    if names_doc.get("path") != str(dest.resolve()):
        names_doc["path"] = str(dest.resolve())
        (dest / "data.yaml").write_text(yaml.safe_dump(names_doc, sort_keys=False), encoding="utf-8")
    names = names_doc.get("names")
    labels = [names[i] for i in sorted(names, key=int)] if isinstance(names, dict) else list(names)

    # train/validation are unused by the ultralytics backend (it reads the
    # data.yaml itself) but keep a placeholder DataFrame so any shared
    # reporting code (e.g. len(data.train)) does not special-case this task.
    placeholder = pd.DataFrame({"note": ["see extra['yolo_data_yaml']"]})
    return placeholder, placeholder, labels, {"yolo_data_yaml": str(dest / "data.yaml")}


# ------------------------------------------------------------ text_generation

def _materialise_text_generation(data: Data, spec: AtelierSpec, data_dir: Path) -> tuple[list[dict[str, str]], list[dict[str, str]], None]:
    local = _fetch_local(data.uri, data_dir / "_download")
    df = pd.read_json(local, lines=True)
    pinned = data_dir / local.name
    if local.resolve() != pinned.resolve():
        shutil.copyfile(local, pinned)

    prompt_col = spec.features.inputs[0].name
    completion_col = spec.features.output.name

    if data.split.validation_fraction is not None:
        val_df = df.sample(frac=data.split.validation_fraction, random_state=spec.seed)
        train_df = df.drop(val_df.index)
    else:
        train_df = _row_slice(df, data.split.train)
        val_df = _row_slice(df, data.split.validation)

    def _records(frame: pd.DataFrame) -> list[dict[str, str]]:
        return [{"prompt": str(r[prompt_col]), "completion": str(r[completion_col])}
                for _, r in frame.iterrows()]

    return _records(train_df), _records(val_df), None


__all__ = ["materialise", "write_manifest"]
