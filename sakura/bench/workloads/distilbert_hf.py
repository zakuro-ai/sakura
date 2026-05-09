"""DistilBERT + SST-2 workload, HF-Trainer-shaped (CI tier).

Distinct from `distilbert.py` because the HF Trainer expects a *different*
batch shape than the pytorch-ddp / lightning loops:

  - `make_model()` returns the raw `AutoModelForSequenceClassification`
    (no wrapper). Calling `model(**batch)` returns `ModelOutput.loss`.
  - Loaders yield a single dict batch with keys
    {`input_ids`, `attention_mask`, `labels`} — no `(x, y)` tuple split.
  - `eval_fn(model, loader)` consumes the same dict batches.

This is the contract documented by `BaselineRunner._run_hf`. The shape
stays compatible with raw HF Trainer usage out-of-the-box: pass
`loader.dataset` as `train_dataset` and `loader.collate_fn` as
`data_collator`.

Smoke shape: 200 train / 600 val examples, max_length 64, batch 32, 1 epoch.
"""
from __future__ import annotations

import os
import tempfile

import torch

from sakura.bench.harness import Workload


_MODEL_NAME = "distilbert-base-uncased"


def _collate_dict(rows):
    """Collate a list of dicts into a single dict batch with a `labels` key.

    HF datasets uses `label` (singular); synthetic uses `labels`. Both get
    normalized to `labels` here so that calling `model(**batch)` provides
    the labels argument the model's forward expects to compute loss.
    """
    input_ids = torch.stack([r["input_ids"] for r in rows])
    attention_mask = torch.stack([r["attention_mask"] for r in rows])
    label_key = "label" if "label" in rows[0] else "labels"
    labels = torch.stack([r[label_key] for r in rows])
    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


def _try_real_data(batch_size: int, n_train: int, n_val: int, max_length: int):
    from transformers import AutoTokenizer
    from datasets import load_dataset

    tok = AutoTokenizer.from_pretrained(_MODEL_NAME)
    cache_dir = os.path.join(tempfile.gettempdir(), "sakura-sst2-cache")
    os.makedirs(cache_dir, exist_ok=True)
    ds = load_dataset("glue", "sst2", cache_dir=cache_dir)

    def _tokenize(b):
        return tok(b["sentence"], padding="max_length", truncation=True, max_length=max_length)

    train = ds["train"].shuffle(seed=42).select(range(min(n_train, len(ds["train"]))))
    val = ds["validation"].shuffle(seed=42).select(range(min(n_val, len(ds["validation"]))))
    train = train.map(_tokenize, batched=True)
    val = val.map(_tokenize, batched=True)
    cols = ["input_ids", "attention_mask", "label"]
    train.set_format("torch", columns=cols)
    val.set_format("torch", columns=cols)

    return (
        torch.utils.data.DataLoader(train, batch_size=batch_size, collate_fn=_collate_dict, shuffle=True),
        torch.utils.data.DataLoader(val, batch_size=batch_size, collate_fn=_collate_dict),
    )


def _make_synthetic_loaders(batch_size: int, n_train: int, n_val: int, max_length: int):
    torch.manual_seed(0)

    def _make(n):
        ids = torch.randint(0, 30522, (n, max_length))
        mask = torch.ones((n, max_length), dtype=torch.long)
        labels = torch.randint(0, 2, (n,))
        return ids, mask, labels

    train_ids, train_mask, train_lbl = _make(n_train)
    val_ids, val_mask, val_lbl = _make(n_val)

    class _BertBatch(torch.utils.data.Dataset):
        def __init__(self, ids, mask, lbl):
            self.ids, self.mask, self.lbl = ids, mask, lbl

        def __len__(self):
            return self.ids.shape[0]

        def __getitem__(self, i):
            return {"input_ids": self.ids[i], "attention_mask": self.mask[i], "labels": self.lbl[i]}

    return (
        torch.utils.data.DataLoader(
            _BertBatch(train_ids, train_mask, train_lbl),
            batch_size=batch_size, collate_fn=_collate_dict, shuffle=True,
        ),
        torch.utils.data.DataLoader(
            _BertBatch(val_ids, val_mask, val_lbl),
            batch_size=batch_size, collate_fn=_collate_dict,
        ),
    )


def _make_loaders(batch_size: int, n_train: int, n_val: int, max_length: int):
    try:
        return _try_real_data(batch_size, n_train, n_val, max_length)
    except Exception:
        return _make_synthetic_loaders(batch_size, n_train, n_val, max_length)


def _make_model() -> torch.nn.Module:
    """Raw HF DistilBERT for SST-2 (random init for benchmarks).

    Returned as-is — no wrapper. `model(**batch)` returns
    `SequenceClassifierOutput` with `.loss` (when batch contains `labels`)
    and `.logits`.
    """
    from transformers import AutoConfig, AutoModelForSequenceClassification
    config = AutoConfig.from_pretrained(_MODEL_NAME, num_labels=2)
    return AutoModelForSequenceClassification.from_config(config)


def _eval_fn(model: torch.nn.Module, loader) -> dict:
    """Eval over dict batches. Returns {val_loss, val_acc}."""
    model.eval()
    device = next(model.parameters()).device
    correct = total = 0
    loss_sum = 0.0
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) if hasattr(v, "to") else v for k, v in batch.items()}
            out = model(**batch)
            loss_sum += float(out.loss) * batch["labels"].size(0)
            correct += int((out.logits.argmax(dim=-1) == batch["labels"]).sum())
            total += int(batch["labels"].numel())
    return {
        "val_loss": loss_sum / max(total, 1),
        "val_acc": correct / max(total, 1),
    }


def make_workload(
    *, batch_size: int = 32, epochs: int = 1,
    n_train: int = 200, n_val: int = 600, max_length: int = 64,
) -> Workload:
    """Build the HF-shaped DistilBERT + SST-2 workload."""
    train_loader, val_loader = _make_loaders(
        batch_size=batch_size, n_train=n_train, n_val=n_val, max_length=max_length,
    )
    return Workload(
        name="distilbert-sst2-hf",
        tier="ci",
        make_model=_make_model,
        make_train_loader=lambda: train_loader,
        make_val_loader=lambda: val_loader,
        eval_fn=_eval_fn,
        epochs=epochs,
    )


__all__ = ["make_workload"]
