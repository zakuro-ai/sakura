"""Row-level train/validation splitting for table datasets (CONTRACTS §1.3).

Three ways to split a table that are new in this module, all deterministic
given ``spec.seed``:

* ``data.split.column`` -- the table already says which rows are which; the
  split honours it (values mapped to ``train`` / ``validation``, optionally
  some values ``exclude``d). No randomness at all.
* ``validation_fraction`` + ``data.stratify_by`` -- a random holdout drawn
  per stratum so every combination of the stratify columns keeps its share.
* ``validation_fraction`` + ``data.split.group_by`` -- the holdout is drawn over the distinct values of a
  column (a conversation, a patient) and every row of a group lands on one side.

A classification spec with a bare ``validation_fraction`` is stratified by its
output label by default (``split.strategy``, BE-18); ``strategy: random`` and
every non-classification spec keep the callers' original code path in
``data.py``, which splits byte-for-byte as before. ``describe_split`` is the
one thing every table path shares -- it builds the ``data_split`` block of
``report.json``.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from sakura.atelier.spec import Split

#: Stand-in for a missing value when a column is used as a split key.
NULL_KEY = "<null>"
#: How many offending values an error message lists before saying "and N more".
_MAX_LISTED = 10


def _keys(series: pd.Series) -> pd.Series:
    """The column as comparable strings: ``1`` and ``"1"`` are the same value
    (a spec written in YAML does not know whether the CSV reader made the
    column numeric), a missing value is :data:`NULL_KEY`."""
    return series.astype("object").where(series.notna(), NULL_KEY).astype(str)


def _as_values(v: Any) -> list[str]:
    return [str(x) for x in (v if isinstance(v, list) else [v])]


def _require_columns(df: pd.DataFrame, columns: list[str], what: str) -> None:
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise ValueError(f"{what}: column(s) {missing} not found in the dataset (columns: {list(df.columns)})")


def split_by_column(df: pd.DataFrame, split: Split) -> tuple[pd.DataFrame, pd.DataFrame, int]:
    """``(train, validation, excluded_rows)`` from the values of ``split.column``.

    Every row's value must be mapped to train or validation, or listed in
    ``exclude``; anything else is an error naming the value, because silently
    dropping (or training on) a ``test`` partition is the mistake this feature
    exists to prevent.
    """
    col = split.column
    assert col is not None and split.train is not None and split.validation is not None
    _require_columns(df, [col], "data.split.column")
    keys = _keys(df[col])
    train_vals, val_vals = _as_values(split.train), _as_values(split.validation)
    excl_vals = [str(x) for x in (split.exclude or [])]

    known = set(train_vals) | set(val_vals) | set(excl_vals)
    unmapped = keys[~keys.isin(known)].value_counts().sort_index()
    if len(unmapped):
        shown = ", ".join(f"{k!r} ({n} rows)" for k, n in list(unmapped.items())[:_MAX_LISTED])
        more = f" and {len(unmapped) - _MAX_LISTED} more" if len(unmapped) > _MAX_LISTED else ""
        raise ValueError(
            f"data.split.column {col!r} holds values that are not mapped to train or validation: "
            f"{shown}{more}. Map them in data.split.train / data.split.validation, or list them "
            "in data.split.exclude to drop those rows")

    is_train, is_val = keys.isin(train_vals), keys.isin(val_vals)
    held = sorted(set(keys))
    for role, mask, vals in (("train", is_train, train_vals), ("validation", is_val, val_vals)):
        if not mask.any():
            raise ValueError(
                f"data.split.{role} {vals} selects no rows: column {col!r} holds {held[:_MAX_LISTED]}")
    return df[is_train], df[is_val], int(len(df) - is_train.sum() - is_val.sum())


def split_stratified(df: pd.DataFrame, fraction: float, stratify_by: list[str],
                     seed: int) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """``(train, validation, warnings)``: a random holdout of ~``fraction`` of
    the rows, drawn separately inside every distinct combination of
    ``stratify_by`` so each keeps about the same share in validation.

    A stratum of ``n`` rows contributes ``round(n * fraction)`` rows to
    validation, but never none and never all of them: at least one stays in
    train. A stratum of ONE row cannot be on both sides; it goes to train and
    is named in the warnings (sklearn refuses the whole split instead, which
    would make a single stray label fail a long job). Validation size is
    therefore only approximately ``fraction`` of the table. Strata are visited
    in sorted key order and drawn from one seeded generator, so the same table
    and seed always give the same split.
    """
    _require_columns(df, stratify_by, "data.stratify_by")
    keycols = [_keys(df[c]) for c in stratify_by]
    strata = keycols[0] if len(keycols) == 1 else pd.Series(
        list(zip(*keycols)), index=df.index, dtype="object")
    rng = np.random.default_rng(seed)
    val_pos: list[int] = []
    singles: list[Any] = []
    positions = pd.Series(np.arange(len(df)), index=df.index)
    groups = positions.groupby(strata, sort=True, dropna=False)
    for key, members in groups:
        idx = members.to_numpy()
        n = len(idx)
        if n < 2:
            singles.append(key)
            continue
        n_val = min(max(int(round(n * fraction)), 1), n - 1)
        val_pos.extend(rng.choice(idx, size=n_val, replace=False).tolist())
    if not val_pos:
        raise ValueError(
            f"data.stratify_by {stratify_by} gives {groups.ngroups} groups for {len(df)} rows, "
            "none with two or more rows, so no validation rows could be drawn; stratify by "
            "fewer or lower-cardinality columns")
    mask = np.zeros(len(df), dtype=bool)
    mask[val_pos] = True
    warnings = []
    if singles:
        shown = ", ".join(repr(k) for k in singles[:_MAX_LISTED])
        warnings.append(
            f"{len(singles)} stratum(s) of data.stratify_by {stratify_by} have a single row "
            f"and were kept in train: {shown}")
    return df[~mask], df[mask], warnings


def split_grouped(df: pd.DataFrame, fraction: float, group_by: str, seed: int,
                  stratify_by: list[str] | None = None) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """``(train, validation, warnings)``: a random holdout of ~``fraction`` of the DISTINCT values of
    ``group_by``, with every row of a group on the same side.

    ``round(n_groups * fraction)`` groups are held out, at least one and never all of them. The groups are
    visited in sorted key order and drawn from one seeded generator, so the same table and seed always give
    the same split. With ``stratify_by`` the groups are drawn per stratum exactly as :func:`split_stratified`
    draws rows, which needs every group to sit in ONE stratum: a column that changes inside a group is refused
    (naming a group that does it), because a group cannot be on both sides to honour both rules."""
    _require_columns(df, [group_by], "data.split.group_by")
    groups = _keys(df[group_by])
    names = sorted(set(groups))
    if len(names) < 2:
        raise ValueError(
            f"data.split.group_by {group_by!r} has {len(names)} distinct value(s) for {len(df)} rows: a holdout "
            "of whole groups needs at least two distinct values, one for each side")
    held: set[str]
    warnings: list[str] = []
    if stratify_by:
        _require_columns(df, stratify_by, "data.stratify_by")
        strata = pd.DataFrame({c: _keys(df[c]) for c in stratify_by})
        strata["_group"] = groups.to_numpy()
        for column in stratify_by:
            varying = strata.groupby("_group", sort=True)[column].nunique()
            varying = varying[varying > 1]
            if len(varying):
                raise ValueError(
                    f"data.stratify_by {stratify_by} is not constant inside a group of data.split.group_by "
                    f"{group_by!r}: column {column!r} takes several values inside group {varying.index[0]!r} "
                    f"({len(varying)} group(s) in all), so that group has no single stratum. Stratify by a "
                    "column that is the same on every row of a group, or drop stratify_by")
        per_group = strata.drop_duplicates("_group").sort_values("_group").reset_index(drop=True)
        _, val_groups, warnings = split_stratified(per_group, fraction, stratify_by, seed)
        held = set(val_groups["_group"])
    else:
        n_val = min(max(int(round(len(names) * fraction)), 1), len(names) - 1)
        rng = np.random.default_rng(seed)
        held = {names[i] for i in rng.choice(len(names), size=n_val, replace=False)}
    mask = groups.isin(held).to_numpy()
    return df[~mask], df[mask], warnings


def describe_split(train: pd.DataFrame, validation: pd.DataFrame, *, split: Split,
                   output: str | None, categorical: bool, excluded: int = 0,
                   warnings: list[str] | None = None, strategy: str | None = None,
                   stratify_by: list[str] | None = None) -> dict[str, Any]:
    """The ``data_split`` block of ``report.json``: how the table was split and
    what each side ended up with. ``label_distribution`` (rows per output
    value) is there for category/binary outputs. ``strategy`` (``stratified``
    / ``random``) is how a ``validation_fraction`` holdout was drawn, and
    ``stratify_by`` the columns it was drawn per (``split.stratify_by``, or
    the output label when stratification is the default)."""
    stratify_by = stratify_by or split.stratify_by
    if split.column is not None:
        mode = "column"
    elif split.validation_fraction is not None:
        mode = "stratified_fraction" if stratify_by else "fraction"
    else:
        mode = "explicit"
    block: dict[str, Any] = {
        "mode": mode,
        "train_rows": int(len(train)),
        "validation_rows": int(len(validation)),
    }
    if split.column is not None:
        block["column"] = split.column
        block["excluded_rows"] = int(excluded)
    if strategy is not None:
        block["strategy"] = strategy
    if stratify_by:
        block["stratify_by"] = list(stratify_by)
    if split.group_by is not None:
        block["group_by"] = split.group_by
        if split.group_by in train.columns and split.group_by in validation.columns:
            block["groups"] = {"train": int(_keys(train[split.group_by]).nunique()),
                               "validation": int(_keys(validation[split.group_by]).nunique())}
    if categorical and output is not None and output in train.columns and output in validation.columns:
        block["label_distribution"] = {
            name: {k: int(v) for k, v in _keys(frame[output]).value_counts().sort_index().items()}
            for name, frame in (("train", train), ("validation", validation))
        }
    if warnings:
        block["warnings"] = list(warnings)
    return block
