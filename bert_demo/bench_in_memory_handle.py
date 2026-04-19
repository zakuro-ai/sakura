"""Measure the in-memory-handle fast path vs the torch.save round-trip.

Runs a real BERT fine-tune with ``SakuraHFCallback`` in standalone mode
(no Zakuro worker). The new path detects ``Compute()`` without a URI and
skips ``torch.save`` / ``torch.load`` entirely — the state_dict flows as
a Python object into the pool thread.

Every number is measured. No simulations.
"""

from __future__ import annotations

import os
import sys
import time

os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch
from datasets import load_dataset
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

import zakuro as zk
from sakura.huggingface import SakuraHFCallback


MODEL_NAME = "distilbert-base-uncased"
_CONFIG = AutoConfig.from_pretrained(MODEL_NAME, num_labels=2)


def _model_factory():
    return AutoModelForSequenceClassification.from_config(_CONFIG)


def _eval_fn(model, payload) -> dict:
    import torch as _torch

    val, bs = payload
    device = _torch.device("cpu")
    model.to(device)
    total = correct = 0
    with _torch.no_grad():
        for start in range(0, len(val["input_ids"]), bs):
            batch = {
                k: val[k][start : start + bs].to(device)
                for k in ("input_ids", "attention_mask")
            }
            labels = val["label"][start : start + bs].to(device)
            out = model(**batch, labels=labels)
            correct += int((out.logits.argmax(-1) == labels).sum().item())
            total += int(labels.size(0))
    return {"val_acc": correct / max(total, 1)}


def _run(force_serialise: bool) -> float:
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    ds = load_dataset("glue", "sst2")
    train = ds["train"].shuffle(seed=42).select(range(256)).map(
        lambda b: tok(b["sentence"], padding="max_length", truncation=True, max_length=64),
        batched=True,
    )
    val = ds["validation"].shuffle(seed=42).select(range(128)).map(
        lambda b: tok(b["sentence"], padding="max_length", truncation=True, max_length=64),
        batched=True,
    )
    cols = ["input_ids", "attention_mask", "label"]
    train.set_format("torch", columns=cols)
    val.set_format("torch", columns=cols)

    val_payload = (
        {
            "input_ids": torch.stack([val[i]["input_ids"] for i in range(len(val))]).clone(),
            "attention_mask": torch.stack([val[i]["attention_mask"] for i in range(len(val))]).clone(),
            "label": torch.stack([val[i]["label"] for i in range(len(val))]).clone(),
        },
        32,
    )

    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)
    cb = SakuraHFCallback(
        model_factory=_model_factory,
        eval_fn=_eval_fn,
        eval_payload=val_payload,
        val_compute=zk.Compute(),  # standalone
        drain="lazy",
        cache_key=f"bench-{MODEL_NAME}-{force_serialise}",
        fp16_state_dict=False,
        on_backpressure="queue",
        verbose=False,
    )
    if force_serialise:
        # Monkey-patch the in-process detector off so we always go through
        # torch.save. Lets us A/B the two paths on identical data.
        cb._is_in_process_target = lambda: False

    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir="/tmp/in-mem-bench",
            num_train_epochs=3,
            per_device_train_batch_size=32,
            eval_strategy="no",
            save_strategy="no",
            logging_strategy="no",
            report_to=[],
            disable_tqdm=True,
            seed=42,
            dataloader_num_workers=0,
            fp16=torch.cuda.is_available(),
        ),
        train_dataset=train,
        callbacks=[cb],
    )

    t0 = time.perf_counter()
    trainer.train()
    # We want to include the final drain so the eval work is accounted for.
    return time.perf_counter() - t0


def main() -> None:
    print(f"=== torch.save path (force_serialise=True) ===")
    a = _run(force_serialise=True)
    print(f"   wall: {a:.2f}s")

    print(f"=== in-memory handle path (force_serialise=False) ===")
    b = _run(force_serialise=False)
    print(f"   wall: {b:.2f}s")

    d = a - b
    print(f"\n{'=' * 48}")
    print(f"torch.save:      {a:>7.2f}s")
    print(f"in-memory:       {b:>7.2f}s")
    print(f"delta:           {d:>+7.2f}s  ({100 * d / a:+.1f}%)")


if __name__ == "__main__":
    main()
