"""DistilBERT + SST-2 workload (CI tier).

Smoke variant: 200 train / 600 val examples, max_length 64, batch 32,
1 epoch. Runs in ~30s on CPU, <5min on a small CPU box.

Real SST-2 via the `datasets` library when available; synthetic
tokenized random tensors otherwise (CI / offline).

Batch format: (dict_of_input_tensors, labels_tensor) tuples so that the
BaselineRunner training loop (which expects x, y = batch) works without
modification. The wrapper model's forward() accepts a dict as x.
"""
from __future__ import annotations

import os
import tempfile

import torch

from sakura.bench.harness import Workload


_MODEL_NAME = "distilbert-base-uncased"


def _collate(rows):
    """Collate a list of dicts into (input_dict, labels) tuple."""
    input_ids = torch.stack([r["input_ids"] for r in rows])
    attention_mask = torch.stack([r["attention_mask"] for r in rows])
    # HuggingFace datasets uses "label" (singular); synthetic uses "labels".
    label_key = "label" if "label" in rows[0] else "labels"
    labels = torch.stack([r[label_key] for r in rows])
    return {"input_ids": input_ids, "attention_mask": attention_mask}, labels


def _try_real_data(batch_size: int, n_train: int, n_val: int, max_length: int):
    """Try to load the real SST-2 dataset; return (train_loader, val_loader)
    or raise if anything fails."""
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
        torch.utils.data.DataLoader(train, batch_size=batch_size, collate_fn=_collate, shuffle=True),
        torch.utils.data.DataLoader(val, batch_size=batch_size, collate_fn=_collate),
    )


def _make_synthetic_loaders(batch_size: int, n_train: int, n_val: int, max_length: int):
    """Synthetic tokenized random tensors with the right shape."""
    torch.manual_seed(0)

    def _make(n):
        ids = torch.randint(0, 30522, (n, max_length))  # distilbert vocab ~30k
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
            batch_size=batch_size, collate_fn=_collate, shuffle=True,
        ),
        torch.utils.data.DataLoader(
            _BertBatch(val_ids, val_mask, val_lbl),
            batch_size=batch_size, collate_fn=_collate,
        ),
    )


def _make_loaders(batch_size: int = 32, n_train: int = 200, n_val: int = 600, max_length: int = 64):
    try:
        return _try_real_data(batch_size, n_train, n_val, max_length)
    except Exception:
        return _make_synthetic_loaders(batch_size, n_train, n_val, max_length)


def _make_model() -> torch.nn.Module:
    """Architecture-only DistilBERT for SST-2 (random init for benchmarks)."""
    from transformers import AutoConfig, AutoModelForSequenceClassification
    config = AutoConfig.from_pretrained(_MODEL_NAME, num_labels=2)
    return AutoModelForSequenceClassification.from_config(config)


class _BertModelWrapper(torch.nn.Module):
    """Wraps a transformers model so the benchmark runner can call model(x) → logits.

    The harness training loop calls: logits = model(x) where x is a dict of
    {input_ids, attention_mask}. This wrapper unpacks x and delegates to the
    underlying DistilBERT model.
    """

    def __init__(self, base):
        super().__init__()
        self.base = base

    def forward(self, x, *args, **kwargs):
        if isinstance(x, dict):
            return self.base(**x).logits
        # Fallback: treat as raw input_ids tensor.
        return self.base(input_ids=x).logits


def _make_model_wrapped() -> torch.nn.Module:
    return _BertModelWrapper(_make_model())


def _eval_fn(model: torch.nn.Module, loader) -> dict:
    """Eval loop — loader yields (input_dict, labels) tuples."""
    model.eval()
    device = next(model.parameters()).device
    correct = total = 0
    loss_sum = 0.0
    with torch.no_grad():
        for x, y in loader:
            # x is already moved to device by _DeviceLoader/_move_batch (dict support added).
            # y is a labels tensor.
            if isinstance(x, dict):
                x = {k: v.to(device) if hasattr(v, "to") else v for k, v in x.items()}
            elif hasattr(x, "to"):
                x = x.to(device)
            if hasattr(y, "to"):
                y = y.to(device)
            logits = model(x)
            loss_sum += float(torch.nn.functional.cross_entropy(logits, y, reduction="sum"))
            correct += int((logits.argmax(dim=-1) == y).sum())
            total += int(y.numel())
    return {
        "val_loss": loss_sum / max(total, 1),
        "val_acc": correct / max(total, 1),
    }


def make_workload(
    *, batch_size: int = 32, epochs: int = 1,
    n_train: int = 200, n_val: int = 600, max_length: int = 64,
) -> Workload:
    """Build the DistilBERT + SST-2 workload. Default is fast smoke shape."""
    train_loader, val_loader = _make_loaders(
        batch_size=batch_size, n_train=n_train, n_val=n_val, max_length=max_length,
    )
    return Workload(
        name="distilbert-sst2",
        tier="ci",
        make_model=_make_model_wrapped,
        make_train_loader=lambda: train_loader,
        make_val_loader=lambda: val_loader,
        eval_fn=_eval_fn,
        epochs=epochs,
    )


__all__ = ["make_workload"]
