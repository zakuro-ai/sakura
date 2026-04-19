"""Benchmark: BERT fine-tuning — Sakura async vs serial HuggingFace Trainer.

Runs a small sequence-classification finetune on SST-2, comparing:

  - Serial baseline: plain ``transformers.Trainer`` with
    ``eval_strategy="epoch"`` (eval blocks training between epochs).

  - Sakura async:    the same training loop with ``eval_strategy="no"``
    and a ``SakuraHFCallback`` that dispatches evaluation to Zakuro after
    each epoch. The callback reaps the previous epoch's future at the
    start of the next one, so evaluation overlaps with the next training
    pass.

The validation set is deliberately sized so that eval time per epoch is
comparable to training time per epoch — otherwise there is nothing to
overlap. Tweak ``--train-size`` / ``--val-size`` if you want a different
ratio.

Run:
    uv run python bert_demo/bench_bert.py \
        --model prajjwal1/bert-tiny \
        --train-size 200 --val-size 600 --epochs 3

No Zakuro worker required — by default the callback falls back to Zakuro's
standalone (in-process) mode. Pass ``--val-worker quic://...`` to target a
real worker.
"""

from __future__ import annotations

import argparse
import os
import time
from typing import Any, Callable

# Silence TF-related warnings that leak through transformers when TF isn't installed.
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np
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


def _tokenize(batch: dict, tokenizer, max_length: int) -> dict:
    return tokenizer(
        batch["sentence"],
        padding="max_length",
        truncation=True,
        max_length=max_length,
    )


def _build_datasets(model_name: str, train_size: int, val_size: int, max_length: int):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    ds = load_dataset("glue", "sst2")
    train = ds["train"].shuffle(seed=42).select(range(train_size))
    val = ds["validation"].shuffle(seed=42).select(
        range(min(val_size, len(ds["validation"])))
    )

    train = train.map(lambda b: _tokenize(b, tokenizer, max_length), batched=True)
    val = val.map(lambda b: _tokenize(b, tokenizer, max_length), batched=True)
    cols = ["input_ids", "attention_mask", "label"]
    train.set_format("torch", columns=cols)
    val.set_format("torch", columns=cols)
    return tokenizer, train, val


def _make_model_factory(model_name: str, num_labels: int) -> Callable[[], Any]:
    # Snapshot the config locally so the worker doesn't need a network hit.
    config = AutoConfig.from_pretrained(model_name, num_labels=num_labels)

    def factory():
        # Architecture-only instantiation: SakuraHFCallback streams over the
        # current weights, so we skip the expensive `from_pretrained` disk read
        # whose weights would be immediately overwritten.
        return AutoModelForSequenceClassification.from_config(config)

    return factory


def _eval_fn(model, payload) -> dict:
    """Runs on the worker. Plain torch eval loop: accuracy + avg cross-entropy.

    Imports are kept inside the function so cloudpickle doesn't try to
    serialise module globals (some torch submodules aren't picklable).
    """
    import torch as _torch

    val_data, batch_size = payload
    if _torch.cuda.is_available():
        device = _torch.device("cuda")
    elif getattr(_torch.backends, "mps", None) and _torch.backends.mps.is_available():
        device = _torch.device("mps")
    else:
        device = _torch.device("cpu")
    model.to(device)

    correct, total, loss_sum = 0, 0, 0.0
    with _torch.no_grad():
        for start in range(0, len(val_data["input_ids"]), batch_size):
            batch = {
                k: val_data[k][start : start + batch_size].to(device)
                for k in ("input_ids", "attention_mask")
            }
            labels = val_data["label"][start : start + batch_size].to(device)
            out = model(**batch, labels=labels)
            loss_sum += float(out.loss.item()) * labels.size(0)
            preds = out.logits.argmax(-1)
            correct += int((preds == labels).sum().item())
            total += int(labels.size(0))

    return {
        "val_loss": loss_sum / max(total, 1),
        "val_acc": correct / max(total, 1),
    }


def _args(output_dir: str, epochs: int, batch_size: int, eval_strategy: str):
    return TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        eval_strategy=eval_strategy,
        save_strategy="no",
        logging_strategy="no",
        report_to=[],
        disable_tqdm=True,
        seed=42,
        dataloader_num_workers=0,  # pickleable + avoids fork issues with eval thread
    )


def run_serial(model_name, train_ds, val_ds, *, epochs, batch_size) -> dict:
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=2
    )
    trainer = Trainer(
        model=model,
        args=_args("/tmp/sakura-bench/serial", epochs, batch_size, "epoch"),
        train_dataset=train_ds,
        eval_dataset=val_ds,
    )
    t0 = time.perf_counter()
    trainer.train()
    elapsed = time.perf_counter() - t0

    final = trainer.evaluate()
    return {
        "mode": "serial",
        "elapsed_secs": elapsed,
        "val_loss": float(final.get("eval_loss", float("nan"))),
        "val_acc": None,
    }


def run_async(model_name, train_ds, val_ds, *, epochs, batch_size, val_compute) -> dict:
    # Pre-materialise the val tensors so they pickle cleanly (no Dataset internals).
    val_payload = (
        {
            "input_ids": val_ds["input_ids"],
            "attention_mask": val_ds["attention_mask"],
            "label": val_ds["label"],
        },
        batch_size,
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=2
    )
    callback = SakuraHFCallback(
        model_factory=_make_model_factory(model_name, num_labels=2),
        eval_fn=_eval_fn,
        eval_payload=val_payload,
        val_compute=val_compute,
    )
    trainer = Trainer(
        model=model,
        args=_args("/tmp/sakura-bench/async", epochs, batch_size, "no"),
        train_dataset=train_ds,
        callbacks=[callback],
    )
    t0 = time.perf_counter()
    trainer.train()
    elapsed = time.perf_counter() - t0

    last = callback.history[-1] if callback.history else {}
    return {
        "mode": "async",
        "elapsed_secs": elapsed,
        "val_loss": last.get("val_loss"),
        "val_acc": last.get("val_acc"),
        "history": callback.history,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="prajjwal1/bert-tiny")
    parser.add_argument("--train-size", type=int, default=200)
    parser.add_argument("--val-size", type=int, default=600)
    parser.add_argument("--max-length", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--mode",
        choices=("serial", "async", "both"),
        default="both",
    )
    parser.add_argument(
        "--val-worker",
        default=os.environ.get("SAKURA_VAL_WORKER"),
        help="Zakuro URI for remote eval (quic://…). Unset → standalone fallback.",
    )
    args = parser.parse_args()

    print(f"Preparing data ({args.model}, train={args.train_size}, val={args.val_size})...")
    _, train_ds, val_ds = _build_datasets(
        args.model, args.train_size, args.val_size, args.max_length
    )

    val_compute = None
    if args.val_worker:
        val_compute = zk.Compute(uri=args.val_worker)
        print(f"Async eval → {val_compute.uri}")
    else:
        print("Async eval → zakuro standalone (in-process)")

    results = {}
    if args.mode in ("serial", "both"):
        print(f"\n=== Serial baseline ({args.epochs} epochs) ===")
        results["serial"] = run_serial(
            args.model, train_ds, val_ds,
            epochs=args.epochs, batch_size=args.batch_size,
        )
        print(f"  serial:  {results['serial']['elapsed_secs']:.2f}s  "
              f"final val_loss={results['serial']['val_loss']:.4f}")

    if args.mode in ("async", "both"):
        print(f"\n=== Sakura async ({args.epochs} epochs) ===")
        results["async"] = run_async(
            args.model, train_ds, val_ds,
            epochs=args.epochs, batch_size=args.batch_size,
            val_compute=val_compute,
        )
        print(f"  async:   {results['async']['elapsed_secs']:.2f}s")
        for h in results["async"]["history"]:
            print(f"    epoch {h['epoch']:.0f}: val_loss={h.get('val_loss'):.4f} "
                  f"val_acc={h.get('val_acc'):.4f} eval_took={h.get('elapsed_secs'):.2f}s")

    if "serial" in results and "async" in results:
        s = results["serial"]["elapsed_secs"]
        a = results["async"]["elapsed_secs"]
        delta = s - a
        pct = 100 * delta / s
        print(f"\n{'='*56}")
        print(f"Serial:  {s:>7.2f}s")
        print(f"Async:   {a:>7.2f}s")
        print(f"Δ:       {delta:>+7.2f}s ({pct:+.1f}%)")


if __name__ == "__main__":
    main()
