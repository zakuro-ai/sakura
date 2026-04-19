<h1 align="center">Sakura</h1>

<p align="center">
  ML-framework integrations for <a href="https://github.com/zakuro-ai/zakuro">Zakuro</a> —
  hide evaluation, logging, and checkpointing behind training so the main loop never waits.
</p>

<p align="center">
  <a href="#quick-start">Quick Start</a> •
  <a href="#installation">Installation</a> •
  <a href="#pytorch-lightning">Lightning</a> •
  <a href="#huggingface-trainer">HuggingFace</a> •
  <a href="#tensorflow--keras">TensorFlow</a> •
  <a href="#benchmarks--notebooks">Benchmarks</a>
</p>

--------------------------------------------------------------------------------

## What is Sakura?

Sakura wraps the framework you're already using (PyTorch Lightning, HuggingFace `Trainer`, TensorFlow `Model.fit`) with a callback that **dispatches evaluation to a [Zakuro](https://github.com/zakuro-ai/zakuro) worker** instead of running it inline. Training keeps stepping while eval runs on a side pool; metrics come back through a non-blocking queue.

The old Sakura used MPI + Redis for this plumbing. The current Sakura uses Zakuro — one `@zk.fn` dispatch per epoch, shared connection, context-aware allocation across a pool of workers. No MPI, no Redis, no `SAKURA_ROLE` fork.

## Quick start

### Laptop-only (no worker setup)

```python
import lightning as L
from sakura.lightning import SakuraTrainer

trainer = SakuraTrainer(
    max_epochs=10,
    accelerator="auto",
    model_factory=MyLightningModule,     # rebuilds on the eval worker
    val_loader_factory=lambda: val_loader,
)
trainer.run(model, train_loader)          # val_compute=None → Zakuro standalone fallback
```

No `zakuro-worker` needed — the eval runs in-process via Zakuro's standalone fallback, but the async dispatch pattern still works.

### HuggingFace `Trainer` with a real worker

```python
from transformers import Trainer, TrainingArguments
from sakura.huggingface import SakuraHFCallback
import zakuro as zk

trainer = Trainer(
    model=model,
    args=TrainingArguments(..., eval_strategy="no"),   # we handle eval
    train_dataset=train_ds,
    callbacks=[
        SakuraHFCallback(
            model_factory=lambda: AutoModelForSequenceClassification.from_config(config),
            eval_fn=my_eval_fn,
            eval_payload=(val_inputs, 32),
            val_compute=zk.Compute(uri="quic://worker:4433"),
            fp16_state_dict=True,
            on_backpressure="skip",
        )
    ],
)
trainer.train()
```

`on_backpressure="skip"` makes the callback consult `AdaptiveCompute.is_backpressured()` before every dispatch — if the allocator reports saturation (the slow eval worker can't keep up), that epoch's eval is dropped rather than blocking training.

## Installation

```bash
# Core + HuggingFace integration
pip install 'sakura-ml[huggingface]'

# Everything
pip install 'sakura-ml[huggingface,tensorflow,bench]'

# From source
git clone https://github.com/zakuro-ai/sakura && cd sakura
uv pip install -e '.[huggingface]'
```

Zakuro is pulled transitively. For a worker (HTTP or QUIC) install the `[worker]` extra on the zakuro package.

## PyTorch Lightning

`sakura.lightning.SakuraTrainer` — a drop-in replacement for the async-eval case:

```python
from sakura.lightning import SakuraTrainer

trainer = SakuraTrainer(
    max_epochs=10,
    accelerator="auto",
    # how the eval worker rebuilds the model:
    model_factory=lambda: MyLightningModule(),
    # how the eval worker rebuilds the dataloader:
    val_loader_factory=lambda: DataLoader(val_ds, batch_size=256),
    # optional: where to run eval
    val_compute=zk.Compute(uri="quic://eval-worker:4433"),
    # optional: where to save the best-loss checkpoint
    model_path="checkpoints/best.pth",
)
trainer.run(model, train_loader)

print(trainer.history)         # [{epoch, val_loss, worker_name, elapsed_secs}, ...]
print(trainer.best_val_loss)
```

## HuggingFace Trainer

`sakura.huggingface.SakuraHFCallback` is a `transformers.TrainerCallback` that cloudpickles `state_dict` on `on_epoch_end`, dispatches a remote eval, and lazily reaps futures as they finish. Knobs:

| parameter | what it does |
|---|---|
| `model_factory` | how the eval worker rebuilds the architecture (weights stream in from the callback) |
| `eval_fn(model, payload)` | the eval routine itself — runs on the worker, returns a `dict` of metrics |
| `eval_payload` | anything cloudpickle can serialise — dataset, tokenizer, batch size |
| `val_compute` | `zk.Compute` or `zk.AdaptiveCompute`; `None` → standalone |
| `drain="lazy"` *(default)* / `"strict"` | whether `on_epoch_end` blocks to reap the previous future |
| `cache_key=...` | keep the validator model architecture warm on the worker |
| `fp16_state_dict=True` | halve the wire bytes |
| `async_copy=True` *(default, CUDA-only)* | GPU→CPU snapshot on a dedicated stream, ~170 → 75 ms per epoch on x399 4090 |
| `on_backpressure={"skip","queue","block"}` | policy when `AdaptiveCompute` reports saturation |
| `max_pending` | cap on in-flight evaluations |

In-memory fast path is **automatic**: when `val_compute` resolves to standalone, `torch.save`/`torch.load` are skipped entirely — measured +23.6 % wall on a 3-epoch distilbert fine-tune vs forced serialisation.

## TensorFlow / Keras

`sakura.tensorflow.SakuraKerasCallback` — a `tf.keras.callbacks.Callback` with the same pattern:

```python
from sakura.tensorflow import SakuraKerasCallback

model.fit(
    x_train, y_train,
    epochs=10,
    callbacks=[SakuraKerasCallback(
        model_factory=lambda: tf.keras.Sequential([...]),
        val_fn=lambda m, p: m.evaluate(*p, verbose=0, return_dict=True),
        val_payload=(x_val, y_val),
        val_compute=zk.Compute(uri="quic://eval-worker:4433"),
    )],
)
```

Weights are transferred as numpy arrays via `get_weights()` / `set_weights()` — clean cloudpickle, no TF graph-state serialisation.

## Generic async trainer (framework-agnostic)

`sakura.ml.async_trainer.AsyncTrainer` — for training loops that aren't Lightning / HF / Keras. Takes any object implementing `train(loader)`, `serialized_state_dict()`, `_epochs`, `_metrics`, plus a `model_factory` and `test_fn(model) -> dict`. Same dispatch mechanics.

## Benchmarks & notebooks

- **[`bert_demo/hf_async_features.ipynb`](bert_demo/hf_async_features.ipynb)** — every `SakuraHFCallback` knob exercised on distilbert / SST-2. Runs in ~1 min on a laptop. Verified via `jupyter nbconvert --execute`.
- **[`bert_demo/bench_bert.py`](bert_demo/bench_bert.py)** — serial `Trainer` vs Sakura async, configurable.
- **[`bert_demo/bench_in_memory_handle.py`](bert_demo/bench_in_memory_handle.py)** — A/B the in-memory-handle fast path against `torch.save`. +23.6 % end-to-end measured.

## Measured performance wins (distilbert-base-uncased, 268 MB state_dict)

| slice | before | after | measured on |
|---|---|---|---|
| blocking `.cpu()` → async CUDA-stream copy | 176 ms / epoch main-thread | **75 ms** / epoch | x399 4090 |
| cloudpickle → `torch.save` for state_dict | 482 ms / epoch pool | **282 ms** / epoch | x399 CPU |
| in-memory handle for standalone | 9.12 s wall (3 epochs) | **7.59 s** | Mac MPS |

See [`zakuro/PLAN.md`](https://github.com/zakuro-ai/zakuro/blob/master/PLAN.md#measured-results-so-far) for the consolidated numbers across both repos.

## Development

```bash
git clone https://github.com/zakuro-ai/sakura && cd sakura
uv pip install -e '.[bench]'
uv run pytest tests/
```

## License

BSD-3-Clause.
