"""TabPFN v2 backend: in-context tabular classification/regression.

Implements the ``Backend`` protocol (CONTRACTS §2.2) for
``tabular_classification`` / ``tabular_regression``, model id ``tabpfn_v2``,
pinned ``tabpfn==2.1.3`` (docs/atelier/envs/tabpfn.txt). 2.1.3 downloads the
ungated v2 weights straight from Hugging Face -- no Prior Labs account, no
API key. Releases >= 6 gate those weights behind an account; this module must
never be bumped to one without re-reading CONTRACTS §2.2's warning.

TabPFN is in-context: there is no gradient training loop, so "train" means
"fit the context" (prior-fitted transformer conditions on (X_train, y_train)
at inference time, no weight updates). ``train()`` logs k-fold CV and a
held-out validation score to the sink as a single post-fit measurement, not a
per-step curve -- there is no curve to report honestly.

Data convention: ``data.train`` = ``(X_train, y_train)``, ``data.validation``
= ``(X_val, y_val)`` (``sakura.atelier.data._maybe_tabpfn_shape``), ``X`` a
``pandas.DataFrame`` or 2D ``np.ndarray`` and ``y`` a 1D array.

ONNX export: a distilled student, not TabPFN itself (issue #67)
-----------------------------------------------------------------------
TabPFN's prediction is not one torch graph. Per ensemble member it runs a
fitted sklearn/numpy preprocessing pipeline (quantile / power transforms, an
SVD, feature shuffling, class permutation and a per-row SHA-256 "fingerprint"
feature), then the transformer on ``(context X, context y, query X)``, then
temperature-softmax averaging. The transformer core alone DOES export as a
two-input ONNX graph and matches ``predict_proba`` (measured: identical
held-out accuracy, max |dp| 0.008 -- docs/atelier/tabpfn-export.md), but the
preprocessing (the SHA-256 row hash above all) has no ONNX equivalent, so that
artefact still needs tabpfn + scikit-learn at inference, is one ~32 MB graph
per ensemble member, costs O(context) per query, and would ship the Prior Labs
weights inside the artefact. It is not a standalone binary.

So the deployable format is a **distilled student**: when the spec asks for
``onnx``, ``train()`` labels a transfer set (the training rows plus perturbed
copies of them) with TabPFN's own ``predict_proba`` / ``predict`` and fits a
small MLP to those soft labels; ``export()`` writes that MLP (imputation and
standardisation inside the graph) as ``model.onnx``.

Categorical columns (string / object / pandas category, issue #145) are
encoded inside the graph too: the artefact takes a second, string input
``X_cat`` and an ``ai.onnx.ml.LabelEncoder`` per column maps the training
vocabulary to codes 1..V and anything else (unseen, missing = ``""``) to an
unknown code 0, which one-hots to all zeros. Callers feed the raw strings; the
vocabulary lives in the graph, not in a side file a caller could drift from
(it is also in the metadata, for documentation). The student contains NO
TabPFN weights; whether a model trained on TabPFN's outputs may be shipped is
still the owner's licence call (tracked by the hub team) and is not decided here.
TabPFN's own validation metrics stay the run's training metrics, so the
engine's ``runtime_eval`` tolerance (CONTRACTS §2.3, ±0.02 for fp32 ONNX) is
literally "the student matches the teacher" -- a student that does not is
reported ``matches_training: false``, never hidden.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar

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
from sakura.atelier.wrap_up import max_steps_of

LICENSE = "Prior Labs License (TabPFN v2 weights, ungated for <=2.1.x)"

#: ONNX metadata key prefix and the student format id (bump on any graph or
#: metadata change: runtime_eval / predict read these to decode outputs).
META_PREFIX = "sakura."
STUDENT_FORMAT = "tabpfn-distilled-mlp/1"             # numeric features: one float32 input "X"
STUDENT_FORMAT_CATEGORICAL = "tabpfn-distilled-mlp/2"  # + one string input "X_cat" (issue #145)
STUDENT_FORMATS = (STUDENT_FORMAT, STUDENT_FORMAT_CATEGORICAL)

#: Distillation defaults. Every key can be overridden from the resolved
#: hyperparameters (preset or ``overrides``) under the same name.
DISTILL_DEFAULTS: dict[str, Any] = {
    "distill_augment": 20,      # transfer set = this many x the training rows ...
    "distill_max_rows": 10000,  # ... capped here (the teacher labels every row)
    "distill_swap": 0.3,        # per-cell probability of taking another row's value
    "distill_noise": 0.0,       # gaussian jitter in feature-stds (0: off-grid values hurt, see the doc)
    "distill_hidden": 256,
    "distill_layers": 2,
    "distill_epochs": 200,
    "distill_batch_size": 256,
    "distill_lr": 1e-3,
    "distill_weight_decay": 1e-4,
    "distill_max_categories": 64,  # per categorical column: the most frequent kept, the rest -> unknown
}


class TabPFNExportError(RuntimeError):
    """Raised when the ONNX student cannot be built or written. Carries the
    exact underlying reason -- never swallowed, never a placeholder file."""


@dataclass
class _Student:
    """A trained student MLP plus everything its ONNX graph bakes in."""

    module: Any  # torch.nn.Module: raw float32 features -> logits | prediction
    task: str
    classes: list[str] | None
    features: list[str]
    teacher_agreement: float  # validation: argmax agreement (clf) / RMSE to teacher (reg)
    transfer_rows: int
    hyperparameters: dict[str, Any]
    #: TabPFN's own validation metrics, stamped into the ONNX metadata so
    #: runtime_eval can judge a regression student scale-free (see there).
    teacher_metrics: dict[str, float] | None = None
    #: numeric / categorical split + vocabularies (None: all numeric, as /1)
    encoder: _Encoder | None = None
    #: ``limits.max_steps`` ended the distillation before its ``distill_epochs`` were done
    steps_capped: bool = False


@dataclass
class _Fitted:
    """``TrainResult.model`` payload: the fitted TabPFN estimator, the context
    it conditions on, and (when ONNX was requested) the distilled student."""

    estimator: Any
    task: str  # "tabular_classification" | "tabular_regression"
    X_train: Any
    y_train: Any
    classes: list[str] | None
    student: _Student | None = None
    seed: int = 42
    hyperparameters: dict[str, Any] | None = None
    #: ``limits.max_steps``: the student's distillation stops after this many optimizer steps (None: no cap).
    max_steps: int | None = None


def _device() -> str:
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


def _feature_names(X: Any) -> list[str]:
    cols = getattr(X, "columns", None)
    if cols is not None:
        return [str(c) for c in cols]
    import numpy as np

    return [f"f{i}" for i in range(np.asarray(X).shape[1])]


#: The ONNX string token for a missing categorical cell. Never a vocabulary
#: entry: missing, "" and any category unseen at training time all land in the
#: same unknown bucket (code 0, an all-zero one-hot).
MISSING_TOKEN = ""


def _as_float(col: Any) -> Any:
    """One column as float64, pandas NA / None -> NaN."""
    import numpy as np
    import pandas as pd

    return pd.Series(col).to_numpy(dtype=np.float64, na_value=np.nan)


def _tokens(col: Any) -> Any:
    """One categorical column as an object array of ``str``; missing -> ``MISSING_TOKEN``."""
    import numpy as np
    import pandas as pd

    s = pd.Series(np.asarray(col, dtype=object), dtype=object)
    return np.where(s.isna().to_numpy(), MISSING_TOKEN, s.astype(str).to_numpy()).astype(object)


def _is_number(v: Any) -> bool:
    import numbers

    return isinstance(v, numbers.Number) and not isinstance(v, (str, bytes))


def _column_kind(col: Any) -> str | None:
    """"num" | "cat" | None (unsupported, e.g. datetime). Categorical means
    string / object / pandas category -- what TabPFN's own OrdinalEncoder
    picks up. An object column holding only numbers (or missing) is numeric."""
    import numpy as np
    import pandas as pd
    from pandas.api.types import is_bool_dtype, is_numeric_dtype, is_object_dtype, is_string_dtype

    dtype = getattr(col, "dtype", None)
    if isinstance(dtype, pd.CategoricalDtype):
        return "cat"
    if dtype is not None and (is_numeric_dtype(dtype) or is_bool_dtype(dtype)):
        return "num"
    if dtype is not None and is_object_dtype(dtype):
        vals = [v for v in np.asarray(col, dtype=object) if not pd.isna(v)]
        return "num" if all(_is_number(v) for v in vals) else "cat"
    if dtype is not None and is_string_dtype(dtype):
        return "cat"
    return None


@dataclass
class _Encoder:
    """The student's view of the feature table: which training columns are
    numeric (one float32 ``X`` input, NaN-imputed + standardised in the graph)
    and which are categorical (one string ``X_cat`` input; per column a
    LabelEncoder maps the training vocabulary to codes 1..V, everything else --
    unseen, missing, beyond ``distill_max_categories`` -- to 0, then a one-hot
    with an all-zero row for 0). Built from the training frame only."""

    features: list[str]               # every training column, training order
    numeric: list[int]                # positions in ``features``
    categorical: list[int]            # positions in ``features``
    vocab: list[list[str]]            # per categorical column: code i + 1 = vocab[i]

    @property
    def numeric_names(self) -> list[str]:
        return [self.features[i] for i in self.numeric]

    @property
    def categorical_names(self) -> list[str]:
        return [self.features[i] for i in self.categorical]

    @classmethod
    def fit(cls, X: Any, max_categories: int) -> _Encoder:
        import collections

        import numpy as np

        if not hasattr(X, "columns") and np.asarray(X).ndim != 2:
            raise TabPFNExportError(f"expected a 2D feature matrix, got shape {np.asarray(X).shape}")
        features = _feature_names(X)
        cols = _columns(X, features)
        kinds = [_column_kind(c) for c in cols]
        bad = [f for f, k in zip(features, kinds) if k is None]
        if bad:
            raise TabPFNExportError(
                "the ONNX student takes numeric and categorical (string / object / category) "
                f"features only; unsupported columns: {bad} (TabPFN itself handles them -- "
                "export onnx is what cannot)"
            )
        numeric = [i for i, k in enumerate(kinds) if k == "num"]
        categorical = [i for i, k in enumerate(kinds) if k == "cat"]
        vocab: list[list[str]] = []
        for i in categorical:
            counts = collections.Counter(t for t in _tokens(cols[i]) if t != MISSING_TOKEN)
            top = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[:max(0, int(max_categories))]
            vocab.append(sorted(t for t, _ in top))
        width = len(numeric) + sum(len(v) for v in vocab)
        if width == 0:
            raise TabPFNExportError("the ONNX student needs at least one non-empty feature column")
        return cls(features=features, numeric=numeric, categorical=categorical, vocab=vocab)

    @classmethod
    def from_meta(cls, meta: dict[str, str]) -> _Encoder:
        """Rebuild the input contract from an artefact's ``sakura.*`` metadata
        (``/1`` artefacts are numeric-only)."""
        features = json.loads(meta["features"])
        cat = json.loads(meta.get("categorical") or "{}")
        categorical = [features.index(f) for f in cat]
        numeric = [i for i in range(len(features)) if i not in categorical]
        return cls(features=features, numeric=numeric, categorical=categorical, vocab=[list(cat[f]) for f in cat])

    def numeric_matrix(self, X: Any) -> Any:
        import numpy as np

        cols = _columns(X, self.features)
        n = _n_rows(X)
        if not self.numeric:
            return np.zeros((n, 0), dtype=np.float64)
        try:
            return np.stack([_as_float(cols[i]) for i in self.numeric], axis=1)
        except (TypeError, ValueError) as exc:
            raise TabPFNExportError(f"a numeric feature holds a non-numeric value: {exc}") from exc

    def raw_categorical(self, X: Any) -> Any:
        import numpy as np

        cols = _columns(X, self.features)
        if not self.categorical:
            return np.empty((_n_rows(X), 0), dtype=object)
        return np.stack([np.asarray(cols[i], dtype=object) for i in self.categorical], axis=1)

    def tokens(self, X: Any) -> Any:
        """The ``X_cat`` ONNX input: ``(n, n_categorical)`` object array of str."""
        import numpy as np

        return self._tokens_raw(self.raw_categorical(X))

    @staticmethod
    def _tokens_raw(raw: Any) -> Any:
        import numpy as np

        if raw.shape[1] == 0:
            return raw
        return np.stack([_tokens(raw[:, j]) for j in range(raw.shape[1])], axis=1)

    def codes(self, X: Any) -> Any:
        """What the graph's LabelEncoders compute: int64 codes, 0 = unknown/missing."""
        return self.codes_raw(self.raw_categorical(X))

    def codes_raw(self, raw: Any) -> Any:
        """``codes`` of an ``(n, n_categorical)`` array of raw categorical values."""
        import numpy as np

        tok = self._tokens_raw(raw)
        out = np.zeros(tok.shape, dtype=np.int64)
        for j, vocab in enumerate(self.vocab):
            lut = {t: i + 1 for i, t in enumerate(vocab)}
            out[:, j] = [lut.get(t, 0) for t in tok[:, j]]
        return out

    def teacher_input(self, X: Any) -> Any:
        """What the teacher is asked about ``X``: the numeric matrix when every
        column is numeric (as before categoricals existed), else the frame
        itself in training column order with its original values."""
        import pandas as pd

        if not self.categorical:
            return self.numeric_matrix(X)
        if hasattr(X, "columns") and all(f in X.columns for f in self.features):
            return X[self.features]
        cols = _columns(X, self.features)
        return pd.DataFrame({f: c for f, c in zip(self.features, cols)})


def _n_rows(X: Any) -> int:
    import numpy as np

    return int(X.shape[0]) if hasattr(X, "shape") else len(np.asarray(X, dtype=object))


def _columns(X: Any, features: list[str]) -> list[Any]:
    """``X``'s columns in ``features`` order: a DataFrame by name (by position
    when it carries other names), a 2D array by position."""
    import numpy as np

    if hasattr(X, "columns"):
        names = [str(c) for c in X.columns]
        if all(f in names for f in features):
            return [X.iloc[:, names.index(f)] for f in features]
        if len(names) == len(features):
            return [X.iloc[:, i] for i in range(len(features))]
        raise ValueError(f"expected features {features}, got columns {names}")
    arr = np.asarray(X) if getattr(X, "dtype", None) is not None else np.asarray(X, dtype=object)
    if arr.ndim != 2 or arr.shape[1] != len(features):
        raise ValueError(f"expected a 2D matrix of {len(features)} features, got shape {arr.shape}")
    return [arr[:, i] for i in range(arr.shape[1])]


def _perturb(num: Any, cats: Any, n_rows: int, swap: float, noise: float, rng: Any) -> tuple[Any, Any]:
    """Transfer-set rows: the training rows plus ``n_rows - n`` perturbed
    copies. Each copy starts from a random training row and takes each cell
    from another random row with probability ``swap`` (numeric and
    categorical alike, so every cell stays on a value its column really
    takes); numeric cells then get gaussian jitter of ``noise`` feature-stds.
    Categorical cells keep their ORIGINAL values (no encoding round trip), so
    the teacher is asked about exactly what the student is trained on."""
    import numpy as np

    n, dn = num.shape
    extra = max(0, n_rows - n)
    if extra == 0:
        return num.copy(), cats.copy()
    fill = np.nan_to_num(np.nanmean(num, axis=0)) if dn else np.zeros(0)
    Xf = np.where(np.isnan(num), fill, num)
    std = Xf.std(axis=0)
    base = rng.integers(0, n, size=extra)
    donor = rng.integers(0, n, size=extra)
    mask = rng.random((extra, dn + cats.shape[1])) < swap
    aug = np.where(mask[:, :dn], Xf[donor], Xf[base]) + rng.normal(size=(extra, dn)) * noise * std
    aug_cat = np.where(mask[:, dn:], cats[donor], cats[base])
    return np.concatenate([num, aug], axis=0), np.concatenate([cats, aug_cat], axis=0)


def _transfer_set(X: Any, n_rows: int, swap: float, noise: float, rng: Any) -> Any:
    """Numeric-only ``_perturb``: jitter is off by default because off-grid
    values cost 2.6 RMSE on diabetes (docs/atelier/tabpfn-export.md)."""
    import numpy as np

    num, _ = _perturb(X, np.empty((X.shape[0], 0), dtype=object), n_rows, swap, noise, rng)
    return num


def _transfer_frame(enc: _Encoder, X: Any, num: Any, cats: Any) -> Any:
    """The teacher's view of a perturbed transfer set: the numeric matrix when
    there is no categorical column, else a frame in training column order
    whose categorical columns carry their original dtype (pandas category /
    string stay what TabPFN was fitted on)."""
    import pandas as pd

    if not enc.categorical:
        return num
    orig = _columns(X, enc.features)
    out: dict[str, Any] = {}
    for j, i in enumerate(enc.numeric):
        out[enc.features[i]] = num[:, j]
    for j, i in enumerate(enc.categorical):
        dtype = getattr(orig[i], "dtype", None)
        col = pd.Series(cats[:, j], dtype=object)
        if dtype is not None and str(dtype) != "object":
            try:
                col = col.astype(dtype)
            except (TypeError, ValueError):
                pass
        out[enc.features[i]] = col
    return pd.DataFrame(out)[enc.features]


def _teacher_targets(teacher: Any, task: str, X: Any, chunk: int = 2000) -> Any:
    import numpy as np

    fn = teacher.predict_proba if task == "tabular_classification" else teacher.predict
    rows = X.iloc if hasattr(X, "iloc") else X
    return np.concatenate([np.asarray(fn(rows[i:i + chunk]), dtype=np.float64) for i in range(0, len(X), chunk)])


def _student_module(d: int, k: int, hidden: int, layers: int, mean: Any, std: Any, fill: Any,
                    y_mean: float | None, y_std: float | None, cat_sizes: list[int] | None = None) -> Any:
    """Raw features in; NaN -> training mean, standardisation and the one-hot
    of categorical codes happen INSIDE the graph, so the ONNX artefact needs
    nothing but onnxruntime. Inputs, in order: ``x`` float ``(n, d)`` when
    ``d > 0``, ``codes`` int64 ``(n, len(cat_sizes))`` when there are
    categorical columns (code 0 = unknown/missing -> all-zero one-hot)."""
    import torch

    sizes = list(cat_sizes or [])
    width_cat = sum(sizes)

    class Student(torch.nn.Module):
        fill: torch.Tensor
        mean: torch.Tensor
        std: torch.Tensor
        offsets: torch.Tensor
        onehot: torch.Tensor

        def __init__(self) -> None:
            super().__init__()
            self.register_buffer("fill", torch.as_tensor(fill, dtype=torch.float32))
            self.register_buffer("mean", torch.as_tensor(mean, dtype=torch.float32))
            self.register_buffer("std", torch.as_tensor(std, dtype=torch.float32))
            # column j's code c >= 1 -> row offsets[j] + c of a (1 + sum V, sum V)
            # table whose row 0 is all zeros: one gather + sum = the concatenated one-hots
            offs = [0]
            for v in sizes[:-1]:
                offs.append(offs[-1] + v)
            self.register_buffer("offsets", torch.as_tensor(offs[:len(sizes)], dtype=torch.int64))
            self.register_buffer("onehot", torch.cat([torch.zeros(1, width_cat), torch.eye(width_cat)], dim=0))
            self.d = d
            self.n_cat = len(sizes)
            blocks: list[torch.nn.Module] = []
            width = d + width_cat
            for _ in range(layers):
                blocks += [torch.nn.Linear(width, hidden), torch.nn.ReLU()]
                width = hidden
            blocks.append(torch.nn.Linear(width, k))
            self.net = torch.nn.Sequential(*blocks)
            self.regression = y_mean is not None
            self.y_mean = float(y_mean or 0.0)
            self.y_std = float(y_std or 1.0)

        def encode(self, *inputs: torch.Tensor) -> torch.Tensor:
            parts: list[torch.Tensor] = []
            if self.d:
                x = inputs[0]
                x = torch.where(torch.isnan(x), self.fill.expand_as(x), x)
                parts.append((x - self.mean) / self.std)
            if self.n_cat:
                codes = inputs[-1]
                idx = torch.where(codes > 0, codes + self.offsets, torch.zeros_like(codes))
                parts.append(torch.nn.functional.embedding(idx, self.onehot).sum(dim=1))
            return parts[0] if len(parts) == 1 else torch.cat(parts, dim=1)

        def core(self, *inputs: torch.Tensor) -> torch.Tensor:
            out: torch.Tensor = self.net(self.encode(*inputs))
            return out

        def forward(self, *inputs: torch.Tensor) -> torch.Tensor:
            out = self.core(*inputs)
            if self.regression:
                out = out[:, 0] * self.y_std + self.y_mean
            return out

    return Student()


def _student_inputs(enc: _Encoder, num: Any, codes: Any) -> tuple[Any, ...]:
    import torch

    parts: list[Any] = []
    if enc.numeric:
        parts.append(torch.as_tensor(num, dtype=torch.float32))
    if enc.categorical:
        parts.append(torch.as_tensor(codes, dtype=torch.int64))
    return tuple(parts)


def _distil(fitted: _Fitted, X_val: Any = None, y_val: Any = None) -> _Student:
    """Fit an MLP student to the fitted TabPFN teacher's outputs on a transfer
    set. Deterministic for a given seed on CPU. Raises TabPFNExportError on
    unsupported inputs (e.g. datetime columns)."""
    import numpy as np
    import torch

    hp = {**DISTILL_DEFAULTS, **{k: v for k, v in (fitted.hyperparameters or {}).items() if k in DISTILL_DEFAULTS}}
    enc = _Encoder.fit(fitted.X_train, int(hp["distill_max_categories"]))
    X = enc.numeric_matrix(fitted.X_train)
    n, d = X.shape
    rng = np.random.default_rng(fitted.seed)
    n_rows = int(min(int(hp["distill_max_rows"]), max(n, n * int(hp["distill_augment"]))))
    Xt, Ct_raw = _perturb(X, enc.raw_categorical(fitted.X_train), n_rows, float(hp["distill_swap"]),
                          float(hp["distill_noise"]), rng)
    teacher_X = _transfer_frame(enc, fitted.X_train, Xt, Ct_raw)
    targets = _teacher_targets(fitted.estimator, fitted.task, teacher_X)
    Ct = enc.codes_raw(Ct_raw)

    if d:
        fill = np.nan_to_num(np.nanmean(X, axis=0))
        Xfill = np.where(np.isnan(X), fill, X)
        mean, std = Xfill.mean(axis=0), Xfill.std(axis=0)
        std = np.where(std < 1e-6, 1.0, std)
    else:
        fill = mean = std = np.zeros(0)
    classification = fitted.task == "tabular_classification"
    if classification:
        k, y_mean, y_std = targets.shape[1], None, None
        target_t = torch.as_tensor(targets, dtype=torch.float32)
    else:
        y_mean, y_std = float(targets.mean()), float(targets.std() or 1.0)
        k = 1
        target_t = torch.as_tensor((targets - y_mean) / y_std, dtype=torch.float32)

    torch.manual_seed(fitted.seed)
    student = _student_module(d, k, int(hp["distill_hidden"]), int(hp["distill_layers"]),
                              mean, std, fill, y_mean, y_std, cat_sizes=[len(v) for v in enc.vocab])
    opt = torch.optim.AdamW(student.net.parameters(), lr=float(hp["distill_lr"]),
                            weight_decay=float(hp["distill_weight_decay"]))
    inputs = _student_inputs(enc, Xt, Ct)
    bs = int(hp["distill_batch_size"])
    gen = torch.Generator().manual_seed(fitted.seed)
    student.train()
    steps, max_steps = 0, fitted.max_steps
    for _ in range(int(hp["distill_epochs"])):
        if max_steps is not None and steps >= max_steps:
            break
        perm = torch.randperm(len(Xt), generator=gen)
        for i in range(0, len(perm), bs):
            if max_steps is not None and steps >= max_steps:
                break  # limits.max_steps: TabPFN itself has no loop, the distillation is what a step cap bounds
            steps += 1
            idx = perm[i:i + bs]
            loss: Any
            z = student.core(*(t[idx] for t in inputs))
            if classification:
                loss = -(target_t[idx] * torch.log_softmax(z, dim=-1)).sum(dim=-1).mean()
            else:
                loss = torch.nn.functional.mse_loss(z[:, 0], target_t[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
    student.eval()

    out = _Student(module=student, task=fitted.task, classes=fitted.classes,
                   features=enc.features, teacher_agreement=float("nan"),
                   transfer_rows=int(len(Xt)), hyperparameters=hp, encoder=enc)
    out.steps_capped = max_steps is not None and steps < int(hp["distill_epochs"]) * -(-len(Xt) // bs)
    if X_val is not None:
        with torch.no_grad():
            s_out = student(*_student_inputs(enc, enc.numeric_matrix(X_val), enc.codes(X_val))).numpy()
        t_out = _teacher_targets(fitted.estimator, fitted.task, enc.teacher_input(X_val))
        if classification:
            out.teacher_agreement = float((s_out.argmax(-1) == t_out.argmax(-1)).mean())
        else:
            out.teacher_agreement = float(np.sqrt(np.mean((s_out - t_out) ** 2)))
    return out


def _student_encoder(student: _Student) -> _Encoder:
    if student.encoder is not None:
        return student.encoder
    return _Encoder(features=list(student.features), numeric=list(range(len(student.features))),
                    categorical=[], vocab=[])


def _student_scores(student: _Student, X: Any, y: Any) -> float:
    """Student's own metric (accuracy | rmse) on a split, in torch."""
    import numpy as np
    import torch

    enc = _student_encoder(student)
    with torch.no_grad():
        out = student.module(*_student_inputs(enc, enc.numeric_matrix(X), enc.codes(X))).numpy()
    if student.task == "tabular_classification":
        labels = np.asarray(student.classes or [str(i) for i in range(out.shape[1])], dtype=object)
        return float((labels[out.argmax(-1)].astype(str) == np.asarray(y).astype(str)).mean())
    return float(np.sqrt(np.mean((out - np.asarray(y, dtype=np.float64)) ** 2)))


#: ONNX-ML opset of the in-graph LabelEncoder (string -> int64 since opset 2;
#: onnxruntime's default CPU build ships it).
ONNX_ML_OPSET = 2


def _string_input_graph(model: Any, enc: _Encoder) -> None:
    """Replace the torch graph's int64 ``cat_codes`` input with a string
    ``X_cat`` input: per categorical column a ``Gather`` takes the column and
    an ``ai.onnx.ml.LabelEncoder`` maps the training vocabulary to 1..V and
    everything else (unseen, missing = ``""``) to 0 -- the same codes
    ``_Encoder.codes`` trains the student on."""
    import numpy as np
    from onnx import TensorProto, helper, numpy_helper

    g = model.graph
    nodes: list[Any] = []
    codes: list[str] = []
    for j, vocab in enumerate(enc.vocab):
        idx, tok, code = f"sakura_cat_index_{j}", f"sakura_cat_token_{j}", f"sakura_cat_code_{j}"
        g.initializer.append(numpy_helper.from_array(np.array([j], dtype=np.int64), idx))
        nodes.append(helper.make_node("Gather", ["X_cat", idx], [tok], axis=1))
        keys, values = (list(vocab), list(range(1, len(vocab) + 1))) if vocab else ([MISSING_TOKEN], [0])
        nodes.append(helper.make_node("LabelEncoder", [tok], [code], domain="ai.onnx.ml",
                                      keys_strings=keys, values_int64s=values, default_int64=0))
        codes.append(code)
    nodes.append(helper.make_node("Concat", codes, ["cat_codes"], axis=1))
    inputs = [i for i in g.input if i.name != "cat_codes"]
    inputs.append(helper.make_tensor_value_info("X_cat", TensorProto.STRING, ["n_rows", len(enc.vocab)]))
    del g.input[:]
    g.input.extend(inputs)
    body = list(g.node)
    del g.node[:]
    g.node.extend(nodes + body)
    model.opset_import.append(helper.make_opsetid("ai.onnx.ml", ONNX_ML_OPSET))


def _write_student_onnx(student: _Student, dest: Path) -> None:
    """torch.onnx.export the student, then stamp the decode metadata
    (classes, feature order, provenance) into the model's metadata_props.
    All-numeric students keep the ``/1`` single-float-input graph; with
    categorical columns the graph is ``/2``: float ``X`` (numeric columns) +
    string ``X_cat`` (categorical columns), see ``_string_input_graph``."""
    import torch

    try:
        import onnx
    except ImportError as exc:
        raise TabPFNExportError(f"the 'onnx' package is required to export the student: {exc}") from exc
    enc = _student_encoder(student)
    out_name = "logits" if student.task == "tabular_classification" else "prediction"
    names: list[str] = []
    example: list[Any] = []
    dynamic: dict[str, dict[int, str]] = {out_name: {0: "n_rows"}}
    if enc.numeric:
        names.append("X")
        example.append(torch.zeros(2, len(enc.numeric), dtype=torch.float32))
    if enc.categorical:
        names.append("cat_codes")
        example.append(torch.zeros(2, len(enc.categorical), dtype=torch.int64))
    for name in names:
        dynamic[name] = {0: "n_rows"}
    kwargs: dict[str, Any] = {}
    if "dynamo" in inspect.signature(torch.onnx.export).parameters:
        kwargs["dynamo"] = False  # a 3-layer MLP: the TorchScript exporter is exact and needs no onnxscript
    try:
        torch.onnx.export(
            student.module, tuple(example), str(dest),
            input_names=names, output_names=[out_name], dynamic_axes=dynamic,
            opset_version=17, **kwargs,
        )
        model = onnx.load(str(dest))
        meta = {
            "format": STUDENT_FORMAT_CATEGORICAL if enc.categorical else STUDENT_FORMAT,
            "task": student.task,
            "classes": json.dumps(student.classes),
            "features": json.dumps(student.features),
            "teacher": "tabpfn_v2 (tabpfn==2.1.3)",
            "contains_teacher_weights": "false",
            "licence_gate": "Prior Labs licence: shipping a TabPFN-distilled model is the owner's call, tracked by the hub team",
            "distill": json.dumps(student.hyperparameters, sort_keys=True),
            "teacher_metrics": json.dumps(student.teacher_metrics or {}, sort_keys=True),
        }
        if enc.categorical:
            _string_input_graph(model, enc)
            meta["inputs"] = json.dumps({"X": enc.numeric_names, "X_cat": enc.categorical_names})
            meta["categorical"] = json.dumps(dict(zip(enc.categorical_names, enc.vocab)))
            meta["missing_token"] = MISSING_TOKEN
        for key, value in meta.items():
            entry = model.metadata_props.add()
            entry.key, entry.value = META_PREFIX + key, value
        onnx.checker.check_model(model)
        onnx.save(model, str(dest))
    except Exception as exc:  # noqa: BLE001 -- surfaced verbatim, not swallowed
        dest.unlink(missing_ok=True)
        raise TabPFNExportError(f"{type(exc).__name__}: {exc}") from exc


_SESSIONS: dict[str, tuple[Any, dict[str, str]]] = {}


def _session(path: Path) -> tuple[Any, dict[str, str]]:
    """(onnxruntime session, sakura.* metadata), memoised per file for the
    warm try-it worker (``python -m sakura.atelier serve``)."""
    key = str(path)
    if key not in _SESSIONS:
        from sakura.atelier.ort import inference_session

        sess = inference_session(key, providers=["CPUExecutionProvider"])
        custom = sess.get_modelmeta().custom_metadata_map or {}
        meta = {k[len(META_PREFIX):]: v for k, v in custom.items() if k.startswith(META_PREFIX)}
        if meta.get("format") not in STUDENT_FORMATS:
            raise ValueError(f"{path.name} is not a {' / '.join(STUDENT_FORMATS)} artefact "
                             f"(format={meta.get('format')!r})")
        _SESSIONS[key] = (sess, meta)
    return _SESSIONS[key]


def _feed(sess: Any, meta: dict[str, str], X: Any) -> dict[str, Any]:
    """The onnxruntime feed for rows ``X`` (frame, 2D array or list of rows):
    float32 ``X`` for the numeric columns and, on ``/2``, a string ``X_cat``
    for the categorical ones (missing -> ``""``, i.e. the unknown bucket)."""
    import numpy as np

    enc = _Encoder.from_meta(meta)
    names = [i.name for i in sess.get_inputs()]
    feed: dict[str, Any] = {}
    if enc.numeric:
        feed[names[0]] = enc.numeric_matrix(X).astype(np.float32)
    if enc.categorical:
        feed["X_cat"] = enc.tokens(X)
    return feed


class TabPFNBackend:
    name = "tabpfn"
    tasks = frozenset({"tabular_classification", "tabular_regression"})
    licence = LICENSE

    #: Set below, once the methods exist (see NOTEBOOK_CELLS at the end of the module).
    NOTEBOOK_CELLS: ClassVar[tuple[Any, ...]]

    def train(
        self,
        spec: ResolvedSpec,
        data: MaterialisedData,
        out: RunDir,
        metrics: MetricsSink,
        resume: Checkpoint | None,
    ) -> TrainResult:
        if resume is not None:
            raise ValueError("tabpfn is in-context: there is no checkpoint to resume from")
        out.ensure()
        task = spec.spec.task
        X_train, y_train = data.train
        X_val, y_val = data.validation

        import numpy as np
        from sklearn.model_selection import cross_val_score

        t0 = time.time()
        if task == "tabular_classification":
            from tabpfn import TabPFNClassifier

            est = TabPFNClassifier(device=_device())
            cv_scores = cross_val_score(est, X_train, y_train, cv=min(5, max(2, len(X_train) // 20)))
            est.fit(X_train, y_train)

            val_pred = est.predict(X_val)
            val_proba = est.predict_proba(X_val)
            accuracy = float((np.asarray(val_pred) == np.asarray(y_val)).mean())
            try:
                from sklearn.metrics import roc_auc_score

                if val_proba.shape[1] == 2:
                    auc = float(roc_auc_score(y_val, val_proba[:, 1]))
                else:
                    auc = float(roc_auc_score(y_val, val_proba, multi_class="ovr"))
            except ValueError:
                auc = float("nan")

            metrics.log("cv_accuracy", float(np.mean(cv_scores)), step=0, split="train", epoch=0.0)
            metrics.log("accuracy", accuracy, step=0, split="validation", epoch=1.0)
            if auc == auc:  # not NaN
                metrics.log("auc", auc, step=0, split="validation", epoch=1.0)

            result_metrics = {"accuracy": accuracy, "cv_accuracy": float(np.mean(cv_scores))}
            classes = [str(c) for c in getattr(est, "classes_", sorted(set(y_train)))]
        else:
            from tabpfn import TabPFNRegressor

            est = TabPFNRegressor(device=_device())
            cv_scores = cross_val_score(est, X_train, y_train, cv=min(5, max(2, len(X_train) // 20)), scoring="r2")
            est.fit(X_train, y_train)

            val_pred = est.predict(X_val)
            rmse = float(np.sqrt(np.mean((np.asarray(val_pred) - np.asarray(y_val)) ** 2)))
            metrics.log("cv_r2", float(np.mean(cv_scores)), step=0, split="train", epoch=0.0)
            metrics.log("rmse", rmse, step=0, split="validation", epoch=1.0)
            result_metrics = {"rmse": rmse, "cv_r2": float(np.mean(cv_scores))}
            classes = None

        fitted = _Fitted(estimator=est, task=task, X_train=X_train, y_train=y_train, classes=classes,
                         seed=spec.spec.seed, hyperparameters=dict(spec.hyperparameters or {}),
                         max_steps=max_steps_of(spec))
        if "onnx" in spec.spec.export:
            # The deployable binary is a student distilled from this teacher
            # (module docstring); its fit is part of the billable train time.
            fitted.student = _distil(fitted, X_val, y_val)
            student_metric = "student_accuracy" if classes is not None else "student_rmse"
            student_value = _student_scores(fitted.student, X_val, y_val)
            agreement_metric = "teacher_agreement" if classes is not None else "teacher_rmse"
            metrics.log(student_metric, student_value, step=0, split="validation", epoch=1.0)
            metrics.log(agreement_metric, fitted.student.teacher_agreement, step=0, split="validation", epoch=1.0)
            result_metrics[student_metric] = student_value
            result_metrics[agreement_metric] = fitted.student.teacher_agreement
            fitted.student.teacher_metrics = {k: v for k, v in result_metrics.items() if k in ("accuracy", "rmse")}
        train_seconds = time.time() - t0

        # No gradient checkpoint to save; record enough to refuse a resume
        # attempt loudly instead of silently ignoring resume_from.
        ckpt_dir = out.checkpoint
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        (ckpt_dir / "meta.json").write_text(json.dumps({
            "model": spec.spec.model, "task": task,
            "note": "tabpfn is in-context: nothing to resume, this file only records provenance",
        }))

        capped = fitted.student is not None and fitted.student.steps_capped
        return TrainResult(
            model=fitted, metrics=result_metrics, train_seconds=train_seconds,
            checkpoint=Checkpoint(path=ckpt_dir, model=spec.spec.model, task=task),
            limits_applied={"max_steps": fitted.max_steps} if capped else None,
        )

    def export(self, result: TrainResult, fmt: str, out: RunDir) -> Artifact:
        if fmt != "onnx":
            raise ValueError(f"tabpfn backend only exports 'onnx', got {fmt!r}")
        out.ensure()
        fitted: _Fitted = result.model
        if fitted.student is None:  # train() ran without onnx in spec.export
            fitted.student = _distil(fitted)
        dest = out.artifacts / "model.onnx"
        _write_student_onnx(fitted.student, dest)  # raises TabPFNExportError, never a fake file
        data = dest.read_bytes()
        return Artifact(path=dest, format="onnx", sha256=hashlib.sha256(data).hexdigest(), bytes=len(data))

    def runtime_eval(self, artifact: Artifact, data: MaterialisedData) -> RuntimeEval:
        X_val, y_val = data.validation
        metric = "accuracy"
        try:
            import numpy as np

            sess, meta = _session(artifact.path)
            classification = meta.get("task") == "tabular_classification"
            metric = "accuracy" if classification else "rmse"
            out = sess.run(None, _feed(sess, meta, X_val))[0]
            if classification:
                labels = np.asarray(json.loads(meta["classes"]), dtype=object)
                value = float((labels[np.argmax(out, axis=-1)].astype(str) == np.asarray(y_val).astype(str)).mean())
            else:
                value = float(np.sqrt(np.mean((np.asarray(out, dtype=np.float64) - np.asarray(y_val, dtype=np.float64)) ** 2)))
            # Classification: matches_training is the engine's call -- it holds
            # TrainResult.metrics (TabPFN's own accuracy) and applies CONTRACTS
            # §2.3's ±0.02. Regression: an absolute ±0.02 on an RMSE is
            # meaningless off unit scale, so (as backends/ludwig.py does) judge
            # the student against the teacher's RMSE stamped in the graph, ±2%.
            matches: bool | None = None
            teacher_rmse = json.loads(meta.get("teacher_metrics") or "{}").get("rmse")
            if not classification and teacher_rmse is not None:
                matches = bool(abs(value - teacher_rmse) <= 0.02 * max(teacher_rmse, 1e-9))
            return RuntimeEval(format="onnx", runtime="onnxruntime", metric=metric,
                               value=value, n=len(X_val), matches_training=matches, reason=None)
        except Exception as exc:  # noqa: BLE001
            return RuntimeEval(
                format="onnx", runtime="onnxruntime", metric=metric,
                value=None, n=len(X_val), matches_training=None,
                reason=f"onnxruntime could not re-score model.onnx: {type(exc).__name__}: {exc}",
            )

    def predict(self, artifact: Artifact, inputs: Any) -> Any:
        """CONTRACTS §3.1 tabular shape: ``{"prediction": <label or number>,
        "proba": {"<label>": p} | null}``. ``inputs`` is ``{<feature>: value}``
        (every feature required), one row as a list, or a list of rows (then
        ``{"predictions": [<one object per row>]}``)."""
        import numpy as np

        sess, meta = _session(artifact.path)
        features = json.loads(meta["features"])
        if isinstance(inputs, dict):
            missing = [f for f in features if f not in inputs]
            if missing:
                raise ValueError(f"predict inputs missing required features: {missing}")
            X = np.asarray([[inputs[f] for f in features]], dtype=object)
        else:
            X = np.asarray(inputs, dtype=object)
        single = X.ndim == 1 or len(X) == 1
        X = X.reshape(-1, len(features))
        out = sess.run(None, _feed(sess, meta, X))[0]
        if meta.get("task") == "tabular_classification":
            labels = json.loads(meta["classes"])
            proba = _softmax(out)
            rows: list[dict[str, Any]] = [
                {"prediction": labels[int(np.argmax(p))], "proba": {lab: float(v) for lab, v in zip(labels, p)}}
                for p in proba
            ]
        else:
            rows = [{"prediction": float(v), "proba": None} for v in np.asarray(out).reshape(-1)]
        return rows[0] if single else {"predictions": rows}


def _softmax(x: Any) -> Any:
    import numpy as np

    e = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


#: Functions the notebook shows by source, in pipeline order (CONTRACTS §2.5): helpers before
#: `train` are the Model section, those after `train`, `export` and `runtime_eval` follow that step.
NOTEBOOK_CELLS = (
    _student_module,
    TabPFNBackend.train, _distil, _student_scores,
    TabPFNBackend.export, _write_student_onnx,
    TabPFNBackend.runtime_eval, _session, _feed,
)
TabPFNBackend.NOTEBOOK_CELLS = NOTEBOOK_CELLS

BACKEND = TabPFNBackend()
