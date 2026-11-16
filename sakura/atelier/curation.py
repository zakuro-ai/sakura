"""Data curation (SAK-0019, issue 202): the spec's ``curation`` block, applied once when the
dataset is materialised, before the split and before ``limits``.

Rules run in a fixed order and a row is attributed to the first rule that drops it:
``exclude_samples`` (by sample id), ``drop_missing_target``, ``classes`` (allow or deny list),
``text_length``, ``exact_duplicates``, ``near_duplicates``. The first row of a group of duplicates is kept.

Two entry points, one per kind of dataset: :func:`curate_table` (csv, parquet, jsonl: a row's id is
``row:<n>``, its 0-based position in the source table) and :func:`curate_files` (image and audio folders: a
file's id is its path relative to the dataset root). Both return the kept rows and a :class:`Curated`
record with the ``report.json`` block and the curated manifest (every kept and dropped id, and the rule
that dropped it), so a curation can be audited and replayed.

Near-duplicate text is found without any dependency beyond numpy: the text is normalised (lower case,
punctuation to spaces, whitespace collapsed), cut into character 5-grams, and two rows are near duplicates
when the Jaccard similarity of their 5-gram sets is at least the threshold. MinHash signatures (64 hashes,
32 bands of 2) only pick the candidate pairs; every candidate is then checked with the exact Jaccard
similarity, so no row is dropped that is below the threshold. A pair above it can be missed when no band
agrees, which is unlikely at 0.5 and above (under 0.01 % of pairs at 0.5).

pandas and numpy only: no training stack.
"""

from __future__ import annotations

import hashlib
import json
import re
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

RULES = ("exclude_samples", "missing_target", "classes", "text_length", "exact_duplicates", "near_duplicates")
MANIFEST_NAME = "curated_manifest.json"
MANIFEST_SCHEMA = 1
SHINGLE = 5
_HASHES, _BAND = 64, 2
_PRIME = (1 << 61) - 1


@dataclass
class Curated:
    """What a curation did: ``report`` is ``report.json``'s ``curation`` block, ``manifest`` the
    curated-manifest artifact."""

    report: dict[str, Any]
    manifest: dict[str, Any]


# ---------------------------------------------------------------------------------------------- text

def normalise(text: Any) -> str:
    """Lower case, every run of characters that is not a letter or a digit to one space, trimmed."""
    return re.sub(r"[\W_]+", " ", str(text).lower()).strip()


def shingles(text: Any) -> frozenset[str]:
    """The character 5-grams of the normalised text (the whole text when it is shorter than 5)."""
    s = normalise(text)
    if len(s) <= SHINGLE:
        return frozenset({s})
    return frozenset(s[i:i + SHINGLE] for i in range(len(s) - SHINGLE + 1))


def jaccard(a: frozenset[str], b: frozenset[str]) -> float:
    union = len(a | b)
    return len(a & b) / union if union else 1.0


def _hash_params() -> tuple[Any, Any]:
    """64 fixed (a, b) pairs for ``(a * x + b) mod p``, from a plain LCG so they never depend on a library."""
    import numpy as np

    state, a, b = 0x5AC0_0019, [], []
    for _ in range(_HASHES):
        state = (state * 6364136223846793005 + 1442695040888963407) % (1 << 64)
        a.append(1 + (state >> 36) % ((1 << 28) - 1))
        state = (state * 6364136223846793005 + 1442695040888963407) % (1 << 64)
        b.append((state >> 36) % (1 << 28))
    return np.array(a, dtype=np.uint64)[:, None], np.array(b, dtype=np.uint64)[:, None]


def near_duplicates(texts: list[Any], threshold: float) -> dict[int, int]:
    """``{position: position of the earlier row it duplicates}`` for every text whose 5-gram Jaccard
    similarity to an earlier KEPT text is at least ``threshold``. Deterministic: rows are visited in order."""
    import numpy as np

    a, b = _hash_params()
    sets = [shingles(t) for t in texts]
    buckets: dict[tuple[int, bytes], list[int]] = {}
    dropped: dict[int, int] = {}
    for i, sh in enumerate(sets):
        x = np.fromiter((zlib.crc32(s.encode("utf-8")) for s in sh), dtype=np.uint64, count=len(sh))[None, :]
        sig = ((a * x + b) % np.uint64(_PRIME)).min(axis=1)
        keys = [(band, sig[band * _BAND:(band + 1) * _BAND].tobytes()) for band in range(_HASHES // _BAND)]
        candidates = sorted({j for key in keys for j in buckets.get(key, ())})
        match = next((j for j in candidates if jaccard(sh, sets[j]) >= threshold), None)
        if match is not None:
            dropped[i] = match
            continue
        for key in keys:
            buckets.setdefault(key, []).append(i)
    return dropped


# -------------------------------------------------------------------------------------------- shared

def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _missing(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and value != value:
        return True
    try:
        import pandas as pd

        if value is pd.NA or value is pd.NaT:
            return True
    except ImportError:  # pragma: no cover
        pass
    return isinstance(value, str) and not value.strip()


def _finish(block: Any, ids: list[str], hashes: list[str], dropped: dict[int, dict[str, Any]],
            not_found: list[str], what: str, extra: dict[str, Any]) -> Curated:
    kept = [i for i in range(len(ids)) if i not in dropped]
    counts = {rule: 0 for rule in _requested(block)}
    for d in dropped.values():
        counts[d["rule"]] = counts.get(d["rule"], 0) + 1
    kept_sha256 = _sha256_text("".join(f"{ids[i]}\t{hashes[i]}\n" for i in kept))
    report: dict[str, Any] = {"rows_before": len(ids), "rows_after": len(kept), "rules": counts,
                              "kept_sha256": kept_sha256, **extra}
    if not_found:
        report["exclude_samples_not_found"] = not_found
    manifest = {
        "schema": MANIFEST_SCHEMA, "sample_id": what, "curation": block.model_dump(mode="json"),
        "rows_before": len(ids), "rows_after": len(kept), "rules": counts, "kept_sha256": kept_sha256,
        "kept": [ids[i] for i in kept],
        "dropped": [{"id": ids[i], **d} for i, d in sorted(dropped.items())],
    }
    if not_found:
        manifest["exclude_samples_not_found"] = not_found
    return Curated(report=report, manifest=manifest)


def _requested(block: Any) -> list[str]:
    asked = {"exclude_samples": block.exclude_samples is not None, "missing_target": block.drop_missing_target,
             "classes": block.classes is not None, "text_length": block.text_length is not None,
             "exact_duplicates": block.exact_duplicates, "near_duplicates": block.near_duplicates is not None}
    return [r for r in RULES if asked[r]]


def _drop(dropped: dict[int, dict[str, Any]], i: int, rule: str, of: str | None = None) -> None:
    if i not in dropped:
        dropped[i] = {"rule": rule} if of is None else {"rule": rule, "of": of}


def _class_rule(block: Any) -> Callable[[Any], bool] | None:
    """``value -> dropped?`` for the ``classes`` rule; values are compared as text."""
    if block.classes is None:
        return None
    if block.classes.allow is not None:
        allow = {str(v) for v in block.classes.allow}
        return lambda v: str(v) not in allow
    deny = {str(v) for v in block.classes.deny or []}
    return lambda v: str(v) in deny


# --------------------------------------------------------------------------------------------- tables

def table_row_of(sample_id: Any) -> int | None:
    """The source row a table sample id names: ``17``, ``"17"``, ``"row:17"`` or the report's
    ``"validation:17"`` (a misclassified row's ``sample_id``, CONTRACTS §2.6). None when it names no row."""
    if isinstance(sample_id, bool):
        return None
    if isinstance(sample_id, int):
        return sample_id
    text = str(sample_id).rsplit(":", 1)[-1].strip()
    return int(text) if re.fullmatch(r"\d+", text) else None


def text_column(spec: Any, named: str | None, rule: str) -> str:
    """The column a text rule reads: the one it names, else the spec's first text input."""
    features = spec.features
    names = [f.name for f in features.inputs] + [features.output.name]
    if named is not None:
        if named not in names:
            raise ValueError(f"curation.{rule}.column {named!r} is not a column of spec.features ({names})")
        return named
    first = next((f.name for f in features.inputs if f.type == "text"), None)
    if first is None:
        raise ValueError(f"curation.{rule} needs a text input (or name the column: curation.{rule}.column)")
    return str(first)


def curate_table(df: Any, spec: Any) -> tuple[Any, Curated | None]:
    """Apply ``spec.curation`` to a table read whole. Returns the kept rows (index and order unchanged) and
    what was done; ``(df, None)`` when the spec has no curation block."""
    block = getattr(spec, "curation", None)
    if block is None:
        return df, None
    features = spec.features
    columns = list(dict.fromkeys([f.name for f in features.inputs] + [features.output.name]))
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise ValueError(f"dataset columns {list(df.columns)} are missing {missing} required by spec.features "
                         "(inputs + output)")
    target = features.output.name
    index = [int(i) for i in df.index]
    position = {row: pos for pos, row in enumerate(index)}
    ids = [f"row:{row}" for row in index]
    records = df[columns].astype(object).to_dict("list")
    rows = [[None if _missing(records[c][i]) and not isinstance(records[c][i], str) else records[c][i]
             for c in columns] for i in range(len(index))]
    hashes = [_sha256_text(json.dumps(r, sort_keys=True, default=str, ensure_ascii=False)) for r in rows]
    dropped: dict[int, dict[str, Any]] = {}
    not_found: list[str] = []
    extra: dict[str, Any] = {}

    for sample in block.exclude_samples or []:
        row = table_row_of(sample)
        if row is None or row not in position:
            not_found.append(str(sample))
        else:
            _drop(dropped, position[row], "exclude_samples")
    if block.drop_missing_target:
        for i, value in enumerate(records[target]):
            if _missing(value):
                _drop(dropped, i, "missing_target")
    refuse = _class_rule(block)
    if refuse is not None:
        for i, value in enumerate(records[target]):
            if refuse(value):
                _drop(dropped, i, "classes")
    if block.text_length is not None:
        column = text_column(spec, block.text_length.column, "text_length")
        low, high = block.text_length.min, block.text_length.max
        for i, value in enumerate(records[column]):
            n = 0 if _missing(value) else len(str(value))
            if (low is not None and n < low) or (high is not None and n > high):
                _drop(dropped, i, "text_length")
    if block.exact_duplicates:
        first: dict[str, int] = {}
        for i, h in enumerate(hashes):
            if i in dropped:
                continue
            if h in first:
                _drop(dropped, i, "exact_duplicates", ids[first[h]])
            else:
                first[h] = i
    if block.near_duplicates is not None:
        column = text_column(spec, block.near_duplicates.column, "near_duplicates")
        alive = [i for i in range(len(index)) if i not in dropped]
        texts = ["" if _missing(records[column][i]) else records[column][i] for i in alive]
        for pos, of in near_duplicates(texts, block.near_duplicates.threshold).items():
            _drop(dropped, alive[pos], "near_duplicates", ids[alive[of]])
        extra["near_duplicates"] = {"column": column, "threshold": block.near_duplicates.threshold,
                                    "method": f"char-{SHINGLE}gram jaccard (minhash candidates)"}
    curated = _finish(block, ids, hashes, dropped, not_found,
                      "row:<n> = the 0-based row of the source table", extra)
    if not dropped:
        return df, curated
    keep = [i for i in range(len(index)) if i not in dropped]
    return df.iloc[keep], curated


# ---------------------------------------------------------------------------------------------- files

def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def curate_files(frame: Any, spec: Any, root: Path, in_col: str, out_col: str,
                 prefer: Path | None = None) -> tuple[Any, Curated | None]:
    """Apply ``spec.curation`` to a folder dataset: ``frame`` has one row per file (``in_col`` an absolute
    path, ``out_col`` its class). A file's id is its path relative to ``root``. ``exclude_samples`` also takes
    a ``<class>/<file>`` path (what a report's misclassified row shows); when that names files in several
    split directories, the one under ``prefer`` (the validation directory) is meant. Exact duplicates are
    files with the same sha256. Returns the kept rows and what was done."""
    block = getattr(spec, "curation", None)
    if block is None:
        return frame, None
    root = root.resolve()
    paths = [Path(p) for p in frame[in_col]]
    ids = [p.relative_to(root).as_posix() if p.is_relative_to(root) else p.as_posix() for p in paths]
    labels = list(frame[out_col])
    by_id = {sid: i for i, sid in enumerate(ids)}
    dropped: dict[int, dict[str, Any]] = {}
    not_found: list[str] = []

    for sample in block.exclude_samples or []:
        wanted = str(sample).strip().lstrip("/")
        if wanted in by_id:
            _drop(dropped, by_id[wanted], "exclude_samples")
            continue
        matches = [i for i, sid in enumerate(ids) if sid.endswith("/" + wanted)]
        if len(matches) > 1 and prefer is not None:
            under = [i for i in matches if paths[i].is_relative_to(prefer.resolve())]
            matches = under or matches
        if not matches:
            not_found.append(str(sample))
        for i in matches:
            _drop(dropped, i, "exclude_samples")
    refuse = _class_rule(block)
    if refuse is not None:
        for i, value in enumerate(labels):
            if refuse(value):
                _drop(dropped, i, "classes")
    hashes = [_sha256_file(p) for p in paths]
    if block.exact_duplicates:
        first: dict[str, int] = {}
        for i, h in enumerate(hashes):
            if i in dropped:
                continue
            if h in first:
                _drop(dropped, i, "exact_duplicates", ids[first[h]])
            else:
                first[h] = i
    curated = _finish(block, ids, hashes, dropped, not_found,
                      "the file's path relative to the dataset root", {})
    if not dropped:
        return frame, curated
    keep = [i for i in range(len(ids)) if i not in dropped]
    return frame.iloc[keep], curated


__all__ = ["Curated", "MANIFEST_NAME", "RULES", "curate_files", "curate_table", "jaccard", "near_duplicates",
           "normalise", "shingles", "table_row_of", "text_column"]
