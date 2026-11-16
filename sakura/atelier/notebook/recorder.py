"""Cell outputs from the run's own values. Nothing here executes the notebook:
the engine already has every number, so the first copy of the file shows what
the runner saw. Odd inputs become a note in the cell, never an exception."""

from __future__ import annotations

import html
import io
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from sakura.atelier.notebook.cells import html_output, png_output, stream_output
from sakura.atelier.types import MaterialisedData, ResolvedSpec, RunDir

#: A categorical column with more distinct values than this is a free-text
#: column: report its cardinality, not a balance table.
_BALANCE_MAX_DISTINCT = 50
_BALANCE_TOP = 10
_MANIFEST_MAX_CHARS = 4000


def _table(rows: list[dict[str, Any]], columns: list[str]) -> dict[str, Any]:
    head = "".join(f"<th>{html.escape(c)}</th>" for c in columns)
    body = "".join(
        "<tr>" + "".join(f"<td>{html.escape(str(r.get(c, '')))}</td>" for c in columns) + "</tr>"
        for r in rows
    )
    text = "\n".join(" | ".join(str(r.get(c, "")) for c in columns) for r in rows)
    return html_output(f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>", text)


def _fmt(v: Any) -> Any:
    return round(v, 4) if isinstance(v, float) else v


def _dataframe_summary(df: Any, name: str) -> str:
    import pandas as pd

    lines = [f"{name}: {len(df)} rows x {df.shape[1]} columns",
             "dtypes: " + ", ".join(f"{c}={t}" for c, t in df.dtypes.astype(str).items()),
             "nulls: " + ", ".join(f"{c}={int(n)}" for c, n in df.isna().sum().items()),
             f"duplicate rows: {int(df.duplicated().sum())}"]
    for c in df.columns:
        s = df[c]
        if pd.api.types.is_bool_dtype(s):
            vc = s.value_counts(dropna=True)
            lines.append(f"{c} balance: " + ", ".join(f"{k}={v}" for k, v in vc.items()))
        elif (pd.api.types.is_string_dtype(s) or isinstance(s.dtype, pd.CategoricalDtype)
              or s.dtype == object):
            distinct = int(s.nunique(dropna=True))
            if distinct <= _BALANCE_MAX_DISTINCT:
                vc = s.value_counts(dropna=True).head(_BALANCE_TOP)
                lines.append(f"{c} balance: " + ", ".join(f"{k}={v}" for k, v in vc.items()))
            else:
                lines.append(f"{c}: {distinct} distinct values")
            lens = s.dropna().astype(str).str.len()
            if len(lens):
                lines.append(f"{c} length: min {lens.min()} median {int(lens.median())} max {lens.max()}")
        elif pd.api.types.is_numeric_dtype(s):
            d = s.describe()
            lines.append(f"{c}: mean {d['mean']:.4g} std {d['std']:.4g} "
                         f"min {d['min']:.4g} max {d['max']:.4g}")
    lines.append("head:\n" + df.head(5).to_string())
    return "\n".join(lines)


def frame_summary(obj: Any, name: str) -> str:
    """A text EDA of one split: a DataFrame, a list of examples (text
    generation), a mapping of paths (YOLO layouts, ASR manifests) or None."""
    if obj is None:
        return f"{name}: no {name} split"
    try:
        try:
            import pandas as pd
        except ImportError:
            pd = None
        if pd is not None and isinstance(obj, pd.DataFrame):
            return _dataframe_summary(obj, name)
        if isinstance(obj, list):
            keys = sorted({k for r in obj[:100] if isinstance(r, dict) for k in r})
            first = json.dumps(obj[0], ensure_ascii=False, default=str)[:500] if obj else ""
            return f"{name}: {len(obj)} examples; keys: {', '.join(map(str, keys)) or 'n/a'}\nfirst: {first}"
        if isinstance(obj, dict):
            return f"{name}: mapping with keys {sorted(map(str, obj))}"
        return f"{name}: {type(obj).__name__}"
    except Exception as e:  # a summary must never break the notebook
        return f"{name}: summary unavailable ({type(e).__name__}: {e})"


def read_curves(metrics_path: Path) -> dict[str, dict[str, list[tuple[int, float]]]]:
    """``metrics.jsonl`` -> name -> split -> [(step, value)]. A last line without
    its newline is a record still being written and is skipped."""
    out: dict[str, dict[str, list[tuple[int, float]]]] = defaultdict(lambda: defaultdict(list))
    with open(metrics_path, encoding="utf-8") as fh:
        for line in fh:
            if not line.endswith("\n"):
                break
            if not line.strip():
                continue
            p = json.loads(line)
            out[p["name"]][p["split"]].append((int(p["step"]), float(p["value"])))
    return {k: dict(v) for k, v in out.items()}


class NotebookRecorder:
    def data_summary(self, data: MaterialisedData, spec: Any) -> list[dict[str, Any]]:
        text = "\n\n".join([frame_summary(data.train, "train"),
                            frame_summary(data.validation, "validation"),
                            f"labels: {data.labels}",
                            f"manifest sha256: {data.manifest_sha256}",
                            f"spec data.sha256: {getattr(getattr(spec, 'data', None), 'sha256', None)}"])
        return [stream_output(text + "\n")]

    def curves(self, metrics_path: Path) -> list[dict[str, Any]]:
        metrics_path = Path(metrics_path)
        if not metrics_path.exists():
            return [stream_output("no metrics.jsonl was written by this run\n")]
        try:
            curves = read_curves(metrics_path)
        except Exception as e:
            return [stream_output(f"metrics.jsonl unreadable ({type(e).__name__}: {e}); curves not rendered\n")]
        if not curves:
            return [stream_output("metrics.jsonl is empty: nothing to plot\n")]
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except Exception as e:
            return [stream_output(f"matplotlib unavailable ({e}); curves not rendered\n")]
        outs: list[dict[str, Any]] = []
        for name, splits in sorted(curves.items()):
            n = sum(len(v) for v in splits.values())
            if n < 2:
                outs.append(stream_output(f"{name}: no curve: {n} point\n"))
                continue
            try:
                fig, ax = plt.subplots(figsize=(6, 3.2), dpi=110)
                for split, pts in sorted(splits.items()):
                    pts = sorted(pts)
                    ax.plot([s for s, _ in pts], [v for _, v in pts],
                            marker="o" if len(pts) < 30 else None, label=split)
                ax.set_title(name)
                ax.set_xlabel("step")
                ax.set_ylabel(name)
                ax.grid(alpha=0.3)
                ax.legend()
                buf = io.BytesIO()
                fig.tight_layout()
                fig.savefig(buf, format="png")
                plt.close(fig)
            except Exception as e:
                outs.append(stream_output(f"{name}: curve not rendered ({type(e).__name__}: {e})\n"))
                continue
            outs.append(png_output(buf.getvalue(), text=f"<figure: {name}>"))
        return outs

    def hyperparameters(self, resolved: ResolvedSpec) -> list[dict[str, Any]]:
        rows = [{"name": k, "value": json.dumps(v, default=str)}
                for k, v in sorted((resolved.hyperparameters or {}).items())]
        return [_table(rows, ["name", "value"])]

    def training_summary(self, report: dict[str, Any]) -> list[dict[str, Any]]:
        rows = [{"metric": k, "value": v} for k, v in sorted((report.get("metrics") or {}).items())]
        rows.append({"metric": "train_seconds", "value": report.get("train_seconds")})
        return [_table(rows, ["metric", "value"])]

    def eval_table(self, report: dict[str, Any]) -> list[dict[str, Any]]:
        rows = report.get("runtime_eval") or []
        if not rows:
            return [stream_output("no export was re-scored\n")]
        outs = [_table(rows, ["format", "runtime", "provider", "metric", "value", "n", "matches_training", "reason"])]
        onnx_provider = report.get("onnx_provider")
        if onnx_provider is None:
            onnx_provider = next((r.get("provider") for r in rows if r.get("format") == "onnx"), None)
        outs.append(stream_output(f"onnx provider (what the session actually ran on): {onnx_provider}\n"))
        if onnx_provider == "CPUExecutionProvider":
            outs.append(stream_output("note: ran on CPU; a GPU build of onnxruntime falls back to CPU when a "
                                      "CUDA library cannot be loaded\n"))
        return outs

    def evaluation_view(self, report: dict[str, Any]) -> list[dict[str, Any]]:
        """The report's ``evaluation`` block (CONTRACTS §2.6) as outputs: split
        table, warnings, class balance, per-class scores, confusion matrix and the
        misclassified rows. Absent keys are skipped; an absent block is one line."""
        ev = report.get("evaluation")
        if not ev:
            return [stream_output("no evaluation block was recorded for this run\n")]
        outs: list[dict[str, Any]] = []
        splits = ev.get("splits") or {}
        if splits:
            cols = ["split", "n", "loss", "accuracy", "f1_macro", "f1_weighted", "precision_macro",
                    "recall_macro", "roc_auc", "rmse", "mae"]
            seen = {k for v in splits.values() for k in v}
            cols = [c for c in cols if c == "split" or c in seen] + sorted(seen - set(cols))
            rows = [{"split": name, **{k: _fmt(v) for k, v in m.items()}} for name, m in splits.items()]
            outs.append(_table(rows, cols))
        base = ev.get("majority_baseline") or {}
        if base:
            outs.append(stream_output("majority-class baseline accuracy: "
                                      + ", ".join(f"{k}={_fmt(v)}" for k, v in base.items()) + "\n"))
        dist = ev.get("label_distribution") or {}
        if dist and ev.get("classes"):
            rows = [{"split": sp, **d} for sp, d in dist.items()]
            outs.append(_table(rows, ["split", *ev["classes"]]))
        for w in ev.get("warnings") or []:
            outs.append(stream_output(f"WARNING: {w}\n"))
        if ev.get("per_class"):
            outs.append(_table([{k: _fmt(v) for k, v in r.items()} for r in ev["per_class"]],
                               ["label", "precision", "recall", "f1", "support"]))
        cm = ev.get("confusion_matrix")
        if cm and cm.get("matrix"):
            outs.extend(self._confusion_outputs(cm))
        mis = ev.get("misclassified") or []
        if mis:
            rows = [{"inputs": json.dumps(m.get("inputs"), ensure_ascii=False, default=str)[:200],
                     "true": m.get("true"), "predicted": m.get("predicted"),
                     "confidence": _fmt(m.get("confidence"))} for m in mis]
            outs.append(_table(rows, ["inputs", "true", "predicted", "confidence"]))
        return outs

    @staticmethod
    def _confusion_outputs(cm: dict[str, Any]) -> list[dict[str, Any]]:
        labels = [str(x) for x in cm.get("labels") or []]
        matrix = cm["matrix"]
        text = "confusion matrix (" + str(cm.get("split")) + ", rows = true, columns = predicted)\n" + \
            "\n".join(" ".join(f"{v:>5}" for v in row) for row in matrix) + "\n"
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            size = max(3.0, 0.6 * len(labels) + 1.5)
            fig, ax = plt.subplots(figsize=(size, size), dpi=110)
            ax.imshow(matrix, cmap="Blues")
            ax.set_xticks(range(len(labels)), labels, rotation=45, ha="right")
            ax.set_yticks(range(len(labels)), labels)
            ax.set_xlabel("predicted")
            ax.set_ylabel("true")
            top = max((max(r) for r in matrix if r), default=0) or 1
            for i, row in enumerate(matrix):
                for j, v in enumerate(row):
                    ax.text(j, i, str(v), ha="center", va="center", color="white" if v > top / 2 else "black")
            ax.set_title(f"confusion matrix ({cm.get('split')})")
            buf = io.BytesIO()
            fig.tight_layout()
            fig.savefig(buf, format="png")
            plt.close(fig)
        except Exception as e:
            return [stream_output(text + f"(figure not rendered: {type(e).__name__}: {e})\n")]
        return [stream_output(text), png_output(buf.getvalue(), text="<figure: confusion matrix>")]

    def artifact_table(self, report: dict[str, Any]) -> list[dict[str, Any]]:
        return [_table(report.get("artifacts") or [], ["path", "format", "sha256", "bytes"])]

    def manifest_view(self, run_dir: RunDir) -> list[dict[str, Any]]:
        outs: list[dict[str, Any]] = []
        for name in ("labels.json", "preprocess.json"):
            p = run_dir.artifacts / name
            if not p.exists():
                outs.append(stream_output(f"{name}: not produced\n"))
                continue
            try:
                outs.append(stream_output(f"{name}:\n{p.read_text(encoding='utf-8')[:_MANIFEST_MAX_CHARS]}\n"))
            except Exception as e:
                outs.append(stream_output(f"{name}: unreadable ({type(e).__name__}: {e})\n"))
        return outs
