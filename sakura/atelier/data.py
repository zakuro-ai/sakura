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
import math
import re
import shutil
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

import pandas as pd

from sakura.atelier.spec import AtelierSpec, Data, _per_class_split, features_of, split_of, split_strategy
from sakura.atelier.splitting import describe_split, split_by_column, split_grouped, split_stratified
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
            # Scheme is restricted to http/https just above (no file:// or
            # custom openers), and the uri comes from the operator's spec.
            # nosemgrep: python.lang.security.audit.dynamic-urllib-use-detected.dynamic-urllib-use-detected
            urllib.request.urlretrieve(uri, local)  # noqa: S310 -- data uri is operator-supplied, not user input
        return local
    raise ValueError(f"data.uri scheme {parsed.scheme!r} is not supported yet (uri={uri!r})")


def _select_columns(df: pd.DataFrame, spec: AtelierSpec) -> pd.DataFrame:
    """Rename/select columns to the exact names spec.features uses. Exact
    name matches are required -- renaming by position would silently feed
    the wrong column into the model on a dataset whose schema drifted."""
    # each name once: a forecasting spec names its series once as input and once as output (same column)
    wanted = _feature_names(spec)
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


# ------------------------------------------------------------ limits (row caps)

def _cap_positions(total: int, n: int, labels: list[Any] | None = None, tail: bool = False) -> list[int]:
    """Which ``n`` of ``total`` rows a row cap keeps, as sorted positions.

    Plain: the first ``n`` (the last ``n`` with ``tail``). With ``labels`` (one per row, a classification
    output): the first rows of every class, each class keeping about its share and never fewer than one
    row while ``n`` is at least the number of classes, so a cap cannot drop a class. Classes are visited in
    sorted order and nothing is drawn at random: the same rows and ``n`` always give the same answer."""
    if n >= total:
        return list(range(total))
    if tail:
        return list(range(total - n, total))
    by_class: dict[str, list[int]] = {}
    for i, label in enumerate(labels or []):
        by_class.setdefault(str(label), []).append(i)
    if not by_class or n < len(by_class):
        return list(range(n))
    keys = sorted(by_class)
    # one row each, then the rest in proportion to what every class has left (largest remainder)
    spare = [len(by_class[k]) - 1 for k in keys]
    left = n - len(keys)
    share = [left * s / max(sum(spare), 1) for s in spare]
    take = [1 + int(x) for x in share]
    by_remainder = sorted(range(len(keys)), key=lambda j: (-(share[j] - int(share[j])), j))
    for j in by_remainder[:n - sum(take)]:
        take[j] += 1
    return sorted(i for k, t in zip(keys, take) for i in by_class[k][:t])


def _take(rows: Any, positions: list[int]) -> Any:
    if isinstance(rows, pd.DataFrame):
        return rows.iloc[positions]
    if isinstance(rows, list):
        return [rows[i] for i in positions]
    return rows.select(positions)  # a datasets.Dataset


def _apply_row_limits(spec: AtelierSpec, train: Any, validation: Any, notes: dict[str, Any], *,
                      label: str | None = None, chronological: bool = False,
                      rows_before: dict[str, int | None] | None = None) -> tuple[Any, Any]:
    """Cap the two sides at ``limits.max_train_rows`` / ``limits.max_eval_rows`` (SAK-0019). Called once
    per dataset, after its split and before its pin is computed, so only kept files are hashed.

    ``label`` names the class column when the output is one a holdout is stratified by: the cap then
    keeps every class (:func:`_cap_positions`). ``chronological`` (forecasting) keeps the rows on both
    sides of the train/validation boundary: the LAST training rows and the FIRST validation rows, so the
    capped series is still one contiguous stretch in time order. ``notes["limits_applied"]`` records the
    caps that removed rows and the counts before and after; nothing is recorded when nothing changed.
    ``rows_before`` overrides the counts when the caller never saw the whole split (``hf://`` streaming)."""
    limits = spec.limits
    if limits is None or (limits.max_train_rows is None and limits.max_eval_rows is None):
        return train, validation
    sides = {"train": (train, limits.max_train_rows, "max_train_rows"),
             "validation": (validation, limits.max_eval_rows, "max_eval_rows")}
    applied: dict[str, Any] = {}
    before: dict[str, int | None] = {}
    after: dict[str, int] = {}
    out: dict[str, Any] = {}
    for side, (rows, cap, key) in sides.items():
        total = len(rows)
        before[side] = total
        if cap is not None and cap < total:
            labels = None
            if label is not None and not chronological:
                labels = list(rows[label]) if not isinstance(rows, list) else None
            rows = _take(rows, _cap_positions(total, cap, labels, tail=chronological and side == "train"))
            applied[key] = cap
        out[side], after[side] = rows, len(rows)
    for side, n in (rows_before or {}).items():
        before[side] = n
        if sides[side][1] is not None:  # the stream was cut short at this side's cap
            applied[sides[side][2]] = sides[side][1]
    if applied:
        notes["limits_applied"] = {**applied, "rows_before": before, "rows_after": after}
    return out["train"], out["validation"]


def _curate(spec: AtelierSpec, notes: dict[str, Any], frame: pd.DataFrame, **files: Any) -> pd.DataFrame:
    """Apply the spec's ``curation`` block (SAK-0019) to the whole dataset, before its split and before
    ``limits``. A table when called with the frame alone, a folder dataset when ``files`` gives its root
    and columns. ``notes["curation"]`` records the report block and the curated manifest."""
    if spec.curation is None:
        return frame
    from sakura.atelier.curation import curate_files, curate_table

    kept, curated = curate_files(frame, spec, **files) if files else curate_table(frame, spec)
    if curated is not None:
        if len(kept) == 0:
            raise ValueError(f"curation removed every row ({curated.report['rules']}): nothing is left to train on")
        notes["curation"] = {"report": curated.report, "manifest": curated.manifest}
    return kept


def _class_column(spec: AtelierSpec) -> str | None:
    """The output column a row cap must keep every class of; None for any other output."""
    return features_of(spec).output.name if _per_class_split(spec) else None


def _split_table(df: pd.DataFrame, data: Data, spec: AtelierSpec, notes: dict[str, Any],
                 legacy: Any) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split a table's rows. A split column or ``stratify_by`` (CONTRACTS §1.3)
    is handled here; everything else runs ``legacy()``, the caller's original
    code, untouched -- so a spec without the new fields splits exactly as it
    always did. Either way ``notes["data_split"]`` records what happened."""
    s = split_of(data)
    out = features_of(spec).output
    excluded, warnings = 0, None
    strategy = split_strategy(spec)
    stratify_by = s.stratify_by or ([out.name] if strategy == "stratified" else None)
    if s.column is not None:
        train_df, val_df, excluded = split_by_column(df, s)
    elif s.group_by is not None:
        assert s.validation_fraction is not None
        train_df, val_df, warnings = split_grouped(df, s.validation_fraction, s.group_by, spec.seed, stratify_by)
    elif stratify_by:
        assert s.validation_fraction is not None
        train_df, val_df, warnings = split_stratified(df, s.validation_fraction, stratify_by, spec.seed)
    else:
        train_df, val_df = legacy()
    # limits: capped here, once, so `data_split` below describes the rows the run really used
    label = out.name if _class_column(spec) and out.name in df.columns else None
    train_df, val_df = _apply_row_limits(spec, train_df, val_df, notes, label=label,
                                         chronological=spec.task == "time_series_forecasting")
    notes["data_split"] = describe_split(
        train_df, val_df, split=s, output=out.name,
        categorical=out.type in ("category", "binary"), excluded=excluded, warnings=warnings,
        strategy=strategy, stratify_by=stratify_by)
    return train_df, val_df


# --------------------------------------------------------------- hf://


def _feature_names(spec: AtelierSpec) -> list[str]:
    """The columns ``spec.features`` names (inputs, then the output), each once."""
    features = features_of(spec)
    return list(dict.fromkeys([f.name for f in features.inputs] + [features.output.name]))


_UNSAFE_DIR_CHARS = re.compile(r"[^A-Za-z0-9_.-]")


def _media_dirs(columns: list[str]) -> dict[str, str]:
    """The sub-directory each media column's files are written under, inside ``<kind>/<split>/``.

    One column keeps the flat ``<kind>/<split>/<i>.<ext>`` layout, so the manifest -- and the
    ``data.sha256`` a resolved spec pins -- of every single-image or single-clip dataset is unchanged.
    Several columns (two camera views) each get their own directory: a shared one wrote every column
    to the same ``<i>.png``, so all of them pointed at the last column's files."""
    if len(columns) <= 1:
        return dict.fromkeys(columns, "")
    dirs: dict[str, str] = {}
    for col in columns:
        name = _UNSAFE_DIR_CHARS.sub("_", col).strip("._") or "column"
        while name in dirs.values():  # two names that sanitise alike (`a/b`, `a_b`) stay apart
            name += "_"
        dirs[col] = name
    return dirs


_HF_SPLIT_EXPR = re.compile(r"^([A-Za-z0-9_.\-]+)(?:\[(\d*):(\d*)\])?$")


def _hf_stream_head(datasets: Any, repo: str, load_kwargs: dict[str, Any], split_expr: str, cap: int) -> tuple[Any, bool]:
    """The first ``cap`` rows of ``split_expr``, read with ``streaming=True`` so the split is not
    downloaded (SAK-0019). Returns ``(dataset, cut)``: an in-memory ``datasets.Dataset`` and whether the
    stream held more rows than were taken. A slice in the expression must be absolute (``train[:1000]``,
    ``train[100:200]``): a stream cannot be sliced by percentage. Image and audio columns are kept as
    their encoded bytes, which is what the non-streaming path's ``to_pandas`` hands on too."""
    m = _HF_SPLIT_EXPR.match(split_expr.strip())
    if m is None:
        raise ValueError(
            f"hf:// split expression {split_expr!r} cannot be streamed: with limits.max_train_rows / "
            "max_eval_rows give a split name or an absolute slice (e.g. train, train[:1000], train[100:200])")
    name, start_s, stop_s = m.group(1), m.group(2), m.group(3)
    start = int(start_s) if start_s else 0
    want = cap if not stop_s else min(cap, max(int(stop_s) - start, 0))
    stream = datasets.load_dataset(repo, split=name, streaming=True, **load_kwargs)
    features = getattr(stream, "features", None)
    for col, feat in (features or {}).items():
        if isinstance(feat, (datasets.Image, datasets.Audio)):
            stream = stream.cast_column(col, type(feat)(decode=False))
    features = getattr(stream, "features", None)
    rows: list[Any] = []
    cut = bool(stop_s) and want < int(stop_s) - start
    for i, row in enumerate(stream):
        if i < start:
            continue
        if len(rows) == want:
            cut = cut or not stop_s  # a row past the cap was there: the split is larger than what was taken
            break
        rows.append(row)
    return datasets.Dataset.from_list(rows, features=features), cut


def _hf_holdout_source(datasets: Any, repo: str, load_kwargs: dict[str, Any]) -> str:
    """The split a ``validation_fraction`` holdout is drawn from: ``train``, else the dataset's only
    split (a repo that publishes one ``preview`` or ``test`` split and nothing else).

    A spec cannot name it: ``split`` takes explicit ``train``/``validation`` expressions OR a
    ``validation_fraction``, never both. Several splits with no ``train`` among them is ambiguous, so
    the error lists them and says how to name the rows. When the split names cannot be read, this
    keeps ``train``, which is what was loaded before."""
    try:
        names = [str(n) for n in datasets.get_dataset_split_names(repo, **load_kwargs)]
    except Exception:  # noqa: BLE001 -- no names (offline, no infos, old datasets): keep train
        return "train"
    if not names or "train" in names:
        return "train"
    if len(names) == 1:
        return names[0]
    raise ValueError(
        f"hf://{repo} has no train split to hold a validation_fraction out of; its splits are "
        f"{names}. Name the rows instead: data.split {{train: \"{names[0]}[:80%]\", "
        f"validation: \"{names[0]}[80%:]\"}} (or one split for each side)")


def _materialise_hf(data: Data, spec: AtelierSpec, data_dir: Path,
                    notes: dict[str, Any] | None = None) -> tuple[pd.DataFrame, pd.DataFrame, list[str] | None]:
    split = split_of(data)
    try:
        import datasets
    except ImportError as exc:  # pragma: no cover -- exercised only where `datasets` is installed
        raise ImportError("materialising a hf:// source needs the `datasets` package") from exc

    # hf://<repo>[@<rev>]: the revision (a branch, tag or commit of the dataset repo) is passed through
    # only when the uri names one, so an unpinned uri loads exactly as it always did.
    repo, _, revision = data.uri.removeprefix("hf://").partition("@")
    load_kwargs: dict[str, Any] = {"revision": revision} if revision else {}
    limits = spec.limits
    max_train = limits.max_train_rows if limits else None
    max_eval = limits.max_eval_rows if limits else None
    streamed: dict[str, int | None] = {}  # sides read as a capped stream: their full size is not known

    def _load(split_expr: str, side: str | None = None, cap: int | None = None) -> Any:
        if cap is None:
            return datasets.load_dataset(repo, split=split_expr, **load_kwargs)
        ds, cut = _hf_stream_head(datasets, repo, load_kwargs, split_expr, cap)
        if cut and side is not None:
            streamed[side] = None
        return ds

    if split.validation_fraction is not None:
        # One stream feeds both sides: enough rows that the holdout below leaves `max_train_rows` for
        # training and `max_eval_rows` for validation (whichever are set), plus one for rounding.
        fraction = split.validation_fraction
        need = [math.ceil(n / share) + 1 for n, share in ((max_train, 1 - fraction), (max_eval, fraction)) if n]
        source = split.train_expr or _hf_holdout_source(datasets, repo, load_kwargs)
        full = _load(source, "train", max(need) if need else None)
        if "train" in streamed:
            streamed["validation"] = None
        out_name = features_of(spec).output.name
        # datasets can only stratify a ClassLabel column; any other label is drawn at random
        stratify = (split_strategy(spec) == "stratified"
                    and isinstance(full.features.get(out_name), datasets.ClassLabel))
        parts = full.train_test_split(test_size=split.validation_fraction, seed=spec.seed,
                                      stratify_by_column=out_name if stratify else None)
        train_ds, val_ds = parts["train"], parts["test"]
    else:
        assert split.train_expr is not None and split.validation_expr is not None
        train_ds = _load(split.train_expr, "train", max_train)
        val_ds = _load(split.validation_expr, "validation", max_eval)
    out_name = features_of(spec).output.name
    label = out_name if _class_column(spec) and out_name in train_ds.column_names else None
    train_ds, val_ds = _apply_row_limits(spec, train_ds, val_ds, {} if notes is None else notes,
                                         label=label, rows_before=streamed)

    # Only the media columns the spec reads are written out: every other column is dropped by
    # `_select_columns` anyway, and writing it cost disk and, below, could land on a used column's files.
    used = set(_feature_names(spec))
    image_cols = [n for n, f in train_ds.features.items() if isinstance(f, datasets.Image) and n in used]
    audio_cols = [n for n, f in train_ds.features.items() if isinstance(f, datasets.Audio) and n in used]
    class_label_cols = {n: f for n, f in train_ds.features.items() if isinstance(f, datasets.ClassLabel)}
    image_dirs = _media_dirs(image_cols)
    audio_dirs = _media_dirs(audio_cols)

    def _to_frame(ds: Any, split_name: str) -> pd.DataFrame:
        df = ds.to_pandas()
        for col in image_cols:
            out_dir = data_dir / "images" / split_name / image_dirs[col]
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
            out_dir = data_dir / "audio" / split_name / audio_dirs[col]
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
    out_col = features_of(spec).output.name
    if out_col in class_label_cols:
        labels = list(class_label_cols[out_col].names)
    return _select_columns(train_df, spec), _select_columns(val_df, spec), labels


# --------------------------------------------------------- csv / parquet

def _materialise_tabular(data: Data, spec: AtelierSpec, data_dir: Path, fmt: str,
                         notes: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame, list[str] | None]:
    split = split_of(data)
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
    df = _curate(spec, notes, df)

    def _legacy() -> tuple[pd.DataFrame, pd.DataFrame]:
        if split.validation_fraction is not None and spec.task == "time_series_forecasting":
            # chronological: a random sample would leak the future into training and shred the windows
            n_val = max(1, int(round(len(df) * split.validation_fraction)))
            return df.iloc[:-n_val], df.iloc[-n_val:]
        if split.validation_fraction is not None:
            val = df.sample(frac=split.validation_fraction, random_state=spec.seed)
            return df.drop(val.index), val
        # Explicit splits for a single-file table select by a row-range
        # expression "start:end" (e.g. "0:1000"), the simplest reading of
        # "split expressions" that a flat CSV/parquet can support.
        return _row_slice(df, split.train_expr), _row_slice(df, split.validation_expr)

    train_df, val_df = _split_table(df, data, spec, notes, _legacy)

    labels = _labels_from_column(df, features_of(spec).output.name)
    return _select_columns(train_df, spec), _select_columns(val_df, spec), labels


def _row_slice(df: pd.DataFrame, expr: str | None) -> pd.DataFrame:
    if expr is None:
        return df
    start_s, _, end_s = expr.partition(":")
    start = int(start_s) if start_s else 0
    end = int(end_s) if end_s else len(df)
    return df.iloc[start:end]


# ------------------------------------------------------ image table (regression)

def _find_table(src: Path, fmt: str) -> Path:
    """``data.uri`` is the table itself, or a directory (an extracted hub
    archive, §1.0) holding exactly one ``.<fmt>`` table somewhere. Only
    ``data.format``'s extension counts: an archive that ships the same table
    as both parquet and csv is not ambiguous."""
    if src.is_file():
        return src
    found = sorted(p for p in src.rglob(f"*.{fmt}") if p.is_file())
    if len(found) != 1:
        raise ValueError(f"{src} must hold exactly one .{fmt} table (data.format), found {[str(p) for p in found]}")
    return found[0]


def _select_rows(df: pd.DataFrame, expr: str | None) -> pd.DataFrame:
    """A row range (``"0:1000"``), or a value of the table's own ``split``
    column (``"train"``): datasets that ship a grouped split keep it."""
    if expr is not None and "split" in df.columns and ":" not in expr:
        out = df[df["split"] == expr]
        if out.empty:
            raise ValueError(f"split expression {expr!r} selects no rows (split column holds {sorted(df['split'].unique())})")
        return out
    return _row_slice(df, expr)


def _materialise_image_table(data: Data, spec: AtelierSpec, data_dir: Path, notes: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame, None, list[dict[str, Any]]]:
    """Image regression from a table: one image column (a path, or a list of
    paths for a multi-photo record) and a numeric target. Paths are relative
    to the table's directory. Rows are split FIRST and only then exploded to
    one row per image, so one record's photos never straddle train and
    validation. Nothing is copied: the pin lists the table and every image."""
    split = split_of(data)
    table = _find_table(_fetch_local(data.uri, data_dir / "_download"), data.format)
    root = table.parent
    df = pd.read_parquet(table) if table.suffix.lower() == ".parquet" else pd.read_csv(table)
    img_col, out_col = features_of(spec).inputs[0].name, features_of(spec).output.name
    missing = [c for c in (img_col, out_col) if c not in df.columns]
    if missing:
        raise ValueError(f"{table} is missing columns {missing} required by spec.features")
    df = df.dropna(subset=[out_col])

    def _legacy() -> tuple[pd.DataFrame, pd.DataFrame]:
        if split.validation_fraction is not None:
            val = df.sample(frac=split.validation_fraction, random_state=spec.seed)
            return df.drop(val.index), val
        return _select_rows(df, split.train_expr), _select_rows(df, split.validation_expr)

    train_df, val_df = _split_table(df, data, spec, notes, _legacy)

    def _explode(frame: pd.DataFrame) -> pd.DataFrame:
        frame = frame[[img_col, out_col]].explode(img_col).dropna(subset=[img_col])
        paths = [Path(str(p)) for p in frame[img_col]]
        frame[img_col] = [str(p if p.is_absolute() else (root / p).resolve()) for p in paths]
        frame[out_col] = frame[out_col].astype(float)
        return frame.reset_index(drop=True)

    train_df, val_df = _explode(train_df), _explode(val_df)
    entries = [{"path": table.name, "bytes": table.stat().st_size, "sha256": _sha256_file(table)}]
    for p in sorted(set(train_df[img_col]) | set(val_df[img_col])):
        f = Path(p)
        if not f.is_file():
            raise ValueError(f"{table} references {f}, which does not exist")
        entries.append({"path": str(f.relative_to(root)) if f.is_relative_to(root) else str(f),
                        "bytes": f.stat().st_size, "sha256": _sha256_file(f)})
    return train_df, val_df, None, entries


# --------------------------------------------------- image_folder / audio_folder

_SPLIT_DIR_NAMES = frozenset({"train", "training", "validation", "val", "valid", "dev", "test"})

def _materialise_labelled_folder(
    data: Data, spec: AtelierSpec, data_dir: Path, exts: frozenset[str], notes: dict[str, Any]
) -> tuple[pd.DataFrame, pd.DataFrame, list[str], list[dict[str, Any]]]:
    """``<root>/<split>/<class>/<file>`` when splits are explicit (the
    directory name under root IS the split expression), or a single
    ``<root>/<class>/<file>`` tree split by ``validation_fraction``.

    The media is NOT copied under ``data_dir`` (it stays wherever the runner
    put it), so the returned pin entries list every selected file -- path
    relative to the root, size, sha256 -- exactly like image tables and ASR
    manifests. Pinning ``data_dir`` instead hashed whatever the backend
    later scattered there (Ludwig caches), which made the pin both blind to
    the dataset bytes and unstable between the run and its notebook replay
    (staging job 200fb6fa, 2026-10-10)."""
    split = split_of(data)
    root = _fetch_local(data.uri, data_dir)
    in_col = features_of(spec).inputs[0].name
    out_col = features_of(spec).output.name

    def _scan(split_dir: Path) -> pd.DataFrame:
        rows = []
        for class_dir in sorted(p for p in split_dir.iterdir() if p.is_dir()):
            for f in sorted(class_dir.iterdir()):
                if f.suffix.lower() in exts:
                    rows.append({in_col: str(f.resolve()), out_col: class_dir.name})
        return pd.DataFrame(rows, columns=[in_col, out_col])

    if split.validation_fraction is not None:
        tops = {d.name.lower() for d in root.iterdir() if d.is_dir()}
        if tops and tops <= _SPLIT_DIR_NAMES:
            # Re-splitting a train/ + validation/ tree by fraction would train
            # a classifier whose "classes" are the split names.
            raise ValueError(
                f"{root} is already split into {sorted(tops)}/; set data.split "
                "{train: <dir>, validation: <dir>} instead of validation_fraction")
        full = _curate(spec, notes, _scan(root), root=root, in_col=in_col, out_col=out_col)
        strategy = split_strategy(spec)
        warnings = None
        if strategy == "stratified":
            # per class: a 20% holdout of 50-image classes is 10 of each, not 4 to 17 (BE-18)
            train_df, val_df, warnings = split_stratified(full, split.validation_fraction, [out_col], spec.seed)
        else:
            val_df = full.sample(frac=split.validation_fraction, random_state=spec.seed)
            train_df = full.drop(val_df.index)
        train_df, val_df = _apply_row_limits(spec, train_df, val_df, notes, label=out_col)
        notes["data_split"] = describe_split(
            train_df, val_df, split=split, output=out_col, categorical=True, warnings=warnings,
            strategy=strategy, stratify_by=[out_col] if strategy == "stratified" else None)
        val_root = root
    else:
        assert split.train_expr is not None and split.validation_expr is not None
        train_df = _scan(root / split.train_expr)
        val_df = _scan(root / split.validation_expr)
        if spec.curation is not None:
            # both sides are curated as one dataset (train first), so a file repeated across them is kept once
            both = _curate(spec, notes, pd.concat([train_df, val_df], ignore_index=True), root=root,
                           in_col=in_col, out_col=out_col, prefer=root / split.validation_expr)
            n_train = len(train_df)
            train_df, val_df = both[both.index < n_train], both[both.index >= n_train]
        train_df, val_df = _apply_row_limits(spec, train_df, val_df, notes, label=out_col)
        val_root = root / split.validation_expr
    # what a misclassified validation file's path is shown relative to: `<class>/<file>` (BE-14)
    notes["media_root"] = str(val_root.resolve())

    labels = sorted(set(train_df[out_col]) | set(val_df[out_col]))
    entries = []
    for f in sorted(Path(p) for p in (*train_df[in_col], *val_df[in_col])):
        entries.append({"path": str(f.relative_to(root.resolve())) if f.is_relative_to(root.resolve()) else str(f),
                        "bytes": f.stat().st_size, "sha256": _sha256_file(f)})
    return train_df, val_df, labels, entries


# ------------------------------------------------------------------ asr_manifest

def _parse_split_expr(expr: str) -> tuple[str, str | None]:
    """``"train"`` -> (train, None); ``"train@loz"`` -> (train, loz)."""
    name, _, lang = expr.partition("@")
    return name.strip(), (lang.strip() or None)


def _materialise_asr_manifest(
    data: Data, spec: AtelierSpec, data_dir: Path, notes: dict[str, Any] | None = None
) -> tuple[pd.DataFrame, pd.DataFrame, list[str], dict[str, Any], list[dict[str, Any]]]:
    """Speech recognition from a ``manifest.csv`` of ``audio`` -> ``transcript`` rows.

    ``data.uri`` is the manifest file, or the directory holding ``manifest.csv`` (the
    dataset root: clip references are resolved against it). A row's ``audio`` is either a
    path (absolute, or relative to the root) or ``<shard>#<member>`` with ``offset`` / ``size``
    giving the clip's byte span inside an uncompressed tar shard (Zakuro hub packaging).

    Optional columns: ``split`` (``train``/``dev``/``test``...), ``language``, ``duration_ms``,
    ``sha256``. Split expressions pick rows by ``split`` and optionally ``language``:
    ``train@loz`` = the ``loz`` rows of split ``train``. Without a ``split`` column use
    ``validation_fraction``. Rows are returned sorted by duration so the bucketing loader
    batches similar lengths together.

    Nothing is copied: the dataset pin (CONTRACTS §1.2) lists the manifest and every selected
    clip with its sha256 (the manifest's own column when present, else hashed from disk).
    """
    split = split_of(data)
    src = Path(_fetch_local(data.uri, data_dir / "_download"))
    manifest = src / "manifest.csv" if src.is_dir() else src
    root = src if src.is_dir() else src.parent
    df = pd.read_csv(manifest)
    audio_col = features_of(spec).inputs[0].name
    text_col = features_of(spec).output.name
    missing = [c for c in (audio_col, text_col) if c not in df.columns]
    if missing:
        raise ValueError(f"{manifest} is missing columns {missing} required by spec.features")
    df = df.dropna(subset=[audio_col, text_col])
    df[text_col] = df[text_col].astype(str).str.strip()
    df = df[df[text_col] != ""]

    def _select(expr: str) -> pd.DataFrame:
        name, lang = _parse_split_expr(expr)
        if "split" not in df.columns:
            raise ValueError(f"split expression {expr!r} needs a 'split' column in {manifest}")
        out = df[df["split"] == name]
        if lang is not None:
            if "language" not in df.columns:
                raise ValueError(f"split expression {expr!r} needs a 'language' column in {manifest}")
            out = out[out["language"] == lang]
        if out.empty:
            raise ValueError(f"split expression {expr!r} selects no rows of {manifest}")
        return out

    if split.validation_fraction is not None:
        val_df = df.sample(frac=split.validation_fraction, random_state=spec.seed)
        train_df = df.drop(val_df.index)
    else:
        assert split.train_expr is not None and split.validation_expr is not None
        train_df, val_df = _select(split.train_expr), _select(split.validation_expr)
    # limits: the first rows in manifest order, taken before the duration sort and before any clip is hashed
    train_df, val_df = _apply_row_limits(spec, train_df, val_df, {} if notes is None else notes)

    sort_key = "duration_ms" if "duration_ms" in df.columns else None
    if sort_key:
        train_df = train_df.sort_values(sort_key, kind="stable")
        val_df = val_df.sort_values(sort_key, kind="stable")
    train_df, val_df = train_df.reset_index(drop=True), val_df.reset_index(drop=True)

    chars = sorted({c for t in pd.concat([train_df[text_col], val_df[text_col]]) for c in t})
    if "_" in chars:
        raise ValueError("transcripts must not contain '_' (reserved as the CTC blank)")
    labels = ["_"] + chars  # blank first: CTC blank index 0

    entries: list[dict[str, Any]] = [
        {"path": manifest.name, "bytes": manifest.stat().st_size, "sha256": _sha256_file(manifest)}
    ]
    seen: set[str] = set()
    for _, row in pd.concat([train_df, val_df]).iterrows():
        ref = str(row[audio_col])
        if ref in seen:
            continue
        seen.add(ref)
        sha = str(row["sha256"]) if "sha256" in df.columns and isinstance(row.get("sha256"), str) else ""
        size = int(row["size"]) if "size" in df.columns and pd.notna(row.get("size")) else 0
        if not sha or not size:
            shard_or_file = ref.split("#", 1)[0]
            p = Path(shard_or_file)
            p = p if p.is_absolute() else root / p
            if "#" in ref:
                raise ValueError(f"{ref}: shard references need 'offset', 'size' and 'sha256' columns")
            sha, size = _sha256_file(p), p.stat().st_size
        entries.append({"path": ref, "bytes": size, "sha256": sha})
    extra = {"audio_root": str(root), "manifest": str(manifest),
             "audio_col": audio_col, "text_col": text_col}
    return train_df, val_df, labels, extra, entries


# ------------------------------------------------------------------ risk_simulation

def _materialise_risk(spec: AtelierSpec, data_dir: Path) -> tuple[dict[str, Any], None, list[dict[str, Any]], dict[str, Any]]:
    """The three tables of a ``risk_simulation`` dataset (``losses``, ``factors``, ``exposure``; parquet or
    csv) from ``data.uri``: a directory holding them, or a .tar/.tar.gz/.zip of one. Each table is copied to
    ``data_dir/risk/`` and the manifest pins exactly those three files. ``train`` is ``{name: DataFrame}``;
    there is no validation split (nothing is fitted to hold out against)."""
    import tarfile
    import zipfile

    from sakura.atelier import risk

    if spec.data.format not in ("parquet", "csv"):
        raise ValueError(f"risk_simulation data must be parquet or csv tables (data.format), got {spec.data.format!r}")
    src = _fetch_local(spec.data.uri, data_dir / "_download")
    if src.is_file():
        unpacked = data_dir / "_unpacked"
        unpacked.mkdir(parents=True, exist_ok=True)
        if zipfile.is_zipfile(src):
            with zipfile.ZipFile(src) as zf:
                for m in zf.namelist():
                    if Path(m).is_absolute() or ".." in Path(m).parts:
                        raise ValueError(f"refusing archive member outside the dataset: {m!r}")
                zf.extractall(unpacked)
        elif tarfile.is_tarfile(src):
            with tarfile.open(src) as tf:
                tf.extractall(unpacked, filter="data")
        else:
            raise ValueError(f"risk_simulation data.uri must be a directory or an archive of one, got file {src.name!r}")
        src = unpacked
    found = risk.find_tables(src)
    pinned_dir = data_dir / "risk"
    pinned_dir.mkdir(parents=True, exist_ok=True)
    pinned: dict[str, Path] = {}
    for name, path in found.items():
        dest = pinned_dir / f"{name}{path.suffix}"
        if path.resolve() != dest.resolve():
            shutil.copyfile(path, dest)
        pinned[name] = dest
    tables = risk.load_tables(pinned)
    entries = [{"path": str(p.relative_to(data_dir)), "bytes": p.stat().st_size, "sha256": _sha256_file(p)}
               for p in pinned.values()]
    extra = {"risk_tables": {name: {"path": str(p), "sha256": e["sha256"], "rows": int(len(tables[name]))}
                             for (name, p), e in zip(pinned.items(), entries)}}
    return tables, None, entries, extra


# ------------------------------------------------------------------ decision

def _materialise_decision(spec: AtelierSpec, data_dir: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]], None, dict[str, Any]]:
    """The rows of a ``decision`` dataset, validated, deduplicated and split BY STATE (``decision.prepare_rows``).

    ``train`` and ``validation`` are lists of cleaned rows ``{"state", "questions", "answers"}``; here
    ``validation`` is the EVAL split (the rows the report is measured on). The calibration split, which
    the temperatures are fitted on and nothing else, is ``extra["calibration"]``; ``extra["decision"]`` has
    the counts and the small-data warnings for the report. A decision spec has no ``features`` and no ``split``:
    the rows carry their own typed questions and the engine does the splitting."""
    from sakura.atelier.decision import load_rows, prepare_rows

    if spec.data.format != "jsonl":
        raise ValueError(f"decision data must be jsonl (data.format), got {spec.data.format!r}")
    local = _fetch_local(spec.data.uri, data_dir / "_download")
    # Keep a copy of the source bytes in data_dir so the manifest pins the exact file read.
    pinned = data_dir / local.name
    if local.resolve() != pinned.resolve():
        shutil.copyfile(local, pinned)
    prepared = prepare_rows(load_rows(pinned), spec.seed)
    extra = {"calibration": prepared.calibration,
             "decision": {"stats": prepared.stats, "warnings": prepared.warnings}}
    # limits: train and eval rows are capped; the calibration split is left whole
    train, held_out = _apply_row_limits(spec, prepared.train, prepared.eval, extra)
    return train, held_out, None, extra


# ------------------------------------------------------------------ public

def materialise(spec: AtelierSpec, data_dir: Path) -> MaterialisedData:
    """Fetch/convert ``spec.data`` into ``data_dir``, write ``manifest.json``,
    and verify it against ``spec.data.sha256`` when the spec pins one.

    The returned shape is per-task, not a single universal frame:
    image/audio/tabular-via-ludwig get pandas DataFrames
    (what Ludwig's config wants); object_detection gets a YOLO ``data.yaml``
    in ``.extra``; tabular-via-tabpfn gets ``(X, y)`` pairs; text_generation
    gets a list of ``{"prompt", "completion"}`` dicts -- each backend takes
    exactly the shape it needs with no second per-backend data layer.
    """
    data_dir.mkdir(parents=True, exist_ok=True)
    fmt = spec.data.format
    if spec.data.uri.startswith("synthetic://"):
        # Generate the dataset, then read it like any file:// source. Only this function's copy of the
        # spec points at the generated files; resolved.yaml keeps the synthetic:// uri (SAK-0019).
        from sakura.atelier import synthetic

        local = synthetic.generate(spec.data.uri, data_dir, fmt)
        spec = spec.model_copy(update={"data": spec.data.model_copy(update={"uri": str(local)})})
    extra: dict[str, Any] = {}
    train: Any
    validation: Any
    labels: list[str] | None
    pinned_entries: list[dict[str, Any]] = []
    notes: dict[str, Any] = {}  # what the table paths report about their split

    if spec.task == "decision":
        train, validation, labels, extra = _materialise_decision(spec, data_dir)
    elif spec.task == "risk_simulation":
        train, validation, pinned_entries, extra = _materialise_risk(spec, data_dir)
        labels = None
    elif spec.task == "reinforcement_learning":
        from sakura.atelier.kg import materialise_triples

        train, validation, extra, kg_notes = materialise_triples(spec, data_dir)
        notes.update(kg_notes)
        labels = None
    elif fmt == "yolo":
        train, validation, labels, extra = _materialise_yolo(spec.data, spec, data_dir, notes)
    elif fmt == "hf":
        train, validation, labels = _materialise_hf(spec.data, spec, data_dir, notes)
    elif fmt in ("jsonl", "csv", "parquet") and spec.task == "text_generation":
        train, validation, labels = _materialise_text_generation(spec.data, spec, data_dir, notes, fmt)
    elif fmt in ("csv", "parquet") and spec.task == "image_regression":
        train, validation, labels, pinned_entries = _materialise_image_table(spec.data, spec, data_dir, notes)
    elif fmt in ("csv", "parquet", "jsonl"):
        train, validation, labels = _materialise_tabular(spec.data, spec, data_dir, fmt, notes)
    elif fmt == "image_folder":
        train, validation, labels, pinned_entries = _materialise_labelled_folder(
            spec.data, spec, data_dir, IMAGE_EXTS, notes)
    elif fmt == "audio_folder":
        train, validation, labels, pinned_entries = _materialise_labelled_folder(
            spec.data, spec, data_dir, AUDIO_EXTS, notes)
    elif fmt == "asr_manifest":
        train, validation, labels, extra, pinned_entries = _materialise_asr_manifest(
            spec.data, spec, data_dir, notes)
    else:
        raise ValueError(f"data.format {fmt!r} is not supported by sakura.atelier.data yet")

    # asr_manifest, image-table and labelled-folder datasets are not copied under data_dir (GBs
    # of media): their pin lists the manifest/table and every selected file instead.
    entries = pinned_entries or _entries_for_tree(data_dir)
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
                             manifest_sha256=manifest_sha256, labels=labels,
                            extra={**extra, **notes})


def _maybe_tabpfn_shape(spec: AtelierSpec, train: pd.DataFrame, validation: pd.DataFrame) -> tuple[Any, Any]:
    """tabpfn's backend wants ``(X, y)`` pairs, not a single frame with the
    target column still inside it."""
    from sakura.atelier.backends import backend_name
    if backend_name(spec.task, spec.model) != "tabpfn":
        return train, validation
    out_col = features_of(spec).output.name
    in_cols = [f.name for f in features_of(spec).inputs]
    return ((train[in_cols], train[out_col]), (validation[in_cols], validation[out_col]))


# ------------------------------------------------------------------ yolo

def _yolo_split_images(dest: Path, entry: Any) -> list[Path]:
    """The images one ``data.yaml`` split entry names: an image directory, or a list file of paths."""
    p = Path(str(entry))
    p = p if p.is_absolute() else dest / p
    if p.is_file():
        return [Path(line.strip()) for line in p.read_text(encoding="utf-8").splitlines() if line.strip()]
    if p.is_dir():
        return sorted(f for f in p.iterdir() if f.suffix.lower() in IMAGE_EXTS)
    return []


def _cap_yolo_splits(spec: AtelierSpec, dest: Path, doc: dict[str, Any], notes: dict[str, Any]) -> None:
    """limits on a YOLO dataset: a capped split becomes a list file naming its first N images (sorted by
    path), and ``data.yaml`` points at it. The copied tree is not trimmed; the list file is part of the pin."""
    import yaml

    limits = spec.limits
    caps = {"train": limits.max_train_rows if limits else None, "val": limits.max_eval_rows if limits else None}
    applied: dict[str, Any] = {}
    before: dict[str, int] = {}
    after: dict[str, int] = {}
    for key, side, name in (("train", "train", "max_train_rows"), ("val", "validation", "max_eval_rows")):
        cap = caps[key]
        if cap is None or key not in doc:
            continue
        images = _yolo_split_images(dest, doc[key])
        before[side], after[side] = len(images), min(len(images), cap)
        if cap < len(images):
            listing = dest / f"limits-{key}.txt"
            listing.write_text("".join(f"{p.resolve()}\n" for p in images[:cap]), encoding="utf-8")
            doc[key] = str(listing.resolve())
            applied[name] = cap
    if applied:
        (dest / "data.yaml").write_text(yaml.safe_dump(doc, sort_keys=False), encoding="utf-8")
        notes["limits_applied"] = {**applied, "rows_before": before, "rows_after": after}


def _materialise_yolo(data: Data, spec: AtelierSpec, data_dir: Path,
                      notes: dict[str, Any] | None = None) -> tuple[Any, Any, list[str], dict[str, Any]]:
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
    _cap_yolo_splits(spec, dest, names_doc, notes if notes is not None else {})

    # train/validation are unused by the ultralytics backend (it reads the
    # data.yaml itself) but keep a placeholder DataFrame so any shared
    # reporting code (e.g. len(data.train)) does not special-case this task.
    placeholder = pd.DataFrame({"note": ["see extra['yolo_data_yaml']"]})
    return placeholder, placeholder, labels, {"yolo_data_yaml": str(dest / "data.yaml")}


# ------------------------------------------------------------ text_generation

_LONE_SURROGATE = re.compile(r"[\ud800-\udfff]")


def _replace_invalid_unicode(df: pd.DataFrame, notes: dict[str, Any]) -> pd.DataFrame:
    """Replace each lone surrogate (a character no UTF-8 file can hold, for example the JSON escape
    ``\\udce1``) in the table's text with U+FFFD, so that everything after this point (curation, the
    tokeniser, the report, the notebook's data cell) reads and writes the same text. The count goes to
    ``notes["data_encoding"]``; a clean table adds nothing."""
    rows: set[int] = set()
    replaced = 0
    columns: list[str] = []
    out = df
    for col in df.columns:
        if not (df[col].dtype == object or pd.api.types.is_string_dtype(df[col])):
            continue
        values = df[col].tolist()
        hits = [(i, _LONE_SURROGATE.findall(v)) for i, v in enumerate(values) if isinstance(v, str) and _LONE_SURROGATE.search(v)]
        if not hits:
            continue
        if out is df:
            out = df.copy()
        out[col] = [_LONE_SURROGATE.sub("\ufffd", v) if isinstance(v, str) else v for v in values]
        rows.update(i for i, _ in hits)
        replaced += sum(len(found) for _, found in hits)
        columns.append(str(col))
    if rows:
        notes["data_encoding"] = {"rows_with_invalid_unicode": len(rows), "characters_replaced": replaced,
                                  "columns": columns}
    return out


def _materialise_text_generation(data: Data, spec: AtelierSpec, data_dir: Path, notes: dict[str, Any],
                                 fmt: str = "jsonl") -> tuple[list[dict[str, str]], list[dict[str, str]], None]:
    """``[{"prompt", "completion"}]`` records from a jsonl / csv / parquet table.

    The prompt is the spec's input column; with several inputs each row is
    rendered as ``"<name>: <value>"`` lines (the hub's La Forge LLM datasets
    are CSVs with one input and one output column, so this is the one-column
    case there). The output column is the completion; a ``category`` /
    ``binary`` output is a label the model learns to emit as text, exactly as
    the legacy runner trained it. Prompt templates (``### Instruction:`` ...)
    are the backend's business, applied from ``hp.prompt_template``, so the
    pinned records stay the dataset's own text.
    """
    split = split_of(data)
    local = _fetch_local(data.uri, data_dir / "_download")
    if fmt == "csv":
        df = pd.read_csv(local)
    elif fmt == "parquet":
        df = pd.read_parquet(local)
    else:
        df = pd.read_json(local, lines=True)
    pinned = data_dir / local.name
    if local.resolve() != pinned.resolve():
        shutil.copyfile(local, pinned)
    df = _replace_invalid_unicode(df, notes)
    df = _curate(spec, notes, df)

    prompt_cols = [f.name for f in features_of(spec).inputs]
    completion_col = features_of(spec).output.name
    missing = [c for c in [*prompt_cols, completion_col] if c not in df.columns]
    if missing:
        raise ValueError(
            f"dataset columns {list(df.columns)} are missing {missing} "
            f"required by spec.features (inputs + output)"
        )
    eval_cols = features_of(spec).eval
    if eval_cols is not None:
        absent = [c for c in eval_cols.columns if c not in df.columns]
        if absent:
            raise ValueError(f"features.eval: column(s) {absent} not found in the dataset (columns: {list(df.columns)})")
    # objective: dpo -- the rows are {prompt, chosen, rejected}: the output column is the chosen
    # completion and the recipe names the rejected one (SAK-0019)
    rejected_col = _dpo_rejected_column(spec)
    if rejected_col is not None and rejected_col not in df.columns:
        raise ValueError(
            f"objective: dpo needs a {rejected_col!r} column holding the rejected completion (name another "
            f"with overrides.rejected_column); the dataset columns are {list(df.columns)}")

    def _legacy() -> tuple[pd.DataFrame, pd.DataFrame]:
        if split.validation_fraction is not None:
            val = df.sample(frac=split.validation_fraction, random_state=spec.seed)
            return df.drop(val.index), val
        return _row_slice(df, split.train_expr), _row_slice(df, split.validation_expr)

    train_df, val_df = _split_table(df, data, spec, notes, _legacy)

    def _prompt(r: pd.Series) -> str:
        if len(prompt_cols) == 1:
            return str(r[prompt_cols[0]])
        return "\n".join(f"{c}: {r[c]}" for c in prompt_cols)

    def _records(frame: pd.DataFrame) -> list[dict[str, str]]:
        if rejected_col is not None:
            return [{"prompt": _prompt(r), "completion": str(r[completion_col]), "rejected": str(r[rejected_col])}
                    for _, r in frame.iterrows()]
        if eval_cols is not None:  # a missing completion is an empty one: the row has no target (evaluation-only)
            return [{"prompt": _prompt(r), "completion": "" if _plain(r[completion_col]) is None else str(r[completion_col])}
                    for _, r in frame.iterrows()]
        return [{"prompt": _prompt(r), "completion": str(r[completion_col])}
                for _, r in frame.iterrows()]

    train_records, val_records = _records(train_df), _records(val_df)
    if eval_cols is not None:
        # carried onto validation records only, untouched, in one `eval` entry the model never sees
        for record, (_, row) in zip(val_records, val_df.iterrows()):
            record["eval"] = _eval_fields(row, eval_cols)  # type: ignore[assignment]  # a dict among the strings
    return train_records, val_records, None


def _plain(value: Any) -> Any:
    """A table cell as plain JSON data: numpy scalars become Python ones, a missing value None."""
    if isinstance(value, (dict, list)):
        return value
    if hasattr(value, "item") and not isinstance(value, (str, bytes)):
        try:
            value = value.item()
        except (ValueError, TypeError):
            return str(value)
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    return value


def _is_true(value: Any) -> bool:
    """A ``final`` cell as a bool: a bool, 1 / 0, or the text true / false (a CSV reads it as text)."""
    value = _plain(value)
    if isinstance(value, str):
        return value.strip().lower() in ("true", "1", "yes")
    return bool(value) if value is not None else False


def _eval_fields(row: pd.Series, cols: Any) -> dict[str, Any]:
    """The ``eval`` entry of a validation record: ``reference`` (text; a JSON cell is carried as JSON text),
    ``final`` (bool), ``group`` (text) and ``keep`` ({column: value}), for the columns the spec names."""
    fields: dict[str, Any] = {}
    if cols.reference is not None:
        ref = _plain(row[cols.reference])
        fields["reference"] = None if ref is None else ref if isinstance(ref, str) else json.dumps(ref)
    if cols.final is not None:
        fields["final"] = _is_true(row[cols.final])
    if cols.group is not None:
        group = _plain(row[cols.group])
        fields["group"] = None if group is None else str(group)
    if cols.keep:
        fields["keep"] = {c: _plain(row[c]) for c in cols.keep}
    return fields


def _dpo_rejected_column(spec: AtelierSpec) -> str | None:
    """The column holding the rejected completion when the recipe's objective is ``dpo``; None otherwise.
    The objective is a hyperparameter, so a spec that is not already resolved is expanded here."""
    from sakura.atelier.recipe import objective_of, rejected_column_of

    hp = spec.hyperparameters
    if hp is None:
        from sakura.atelier.registry import resolve  # lazy: the registry does not import this module

        try:
            hp = resolve(spec).hyperparameters
        except ValueError:  # a spec `resolve` refuses never reaches a run; read what it says itself
            hp = dict(spec.overrides)
    return rejected_column_of(hp) if objective_of(hp) == "dpo" else None


__all__ = ["materialise", "write_manifest"]
