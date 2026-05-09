# Migrating from sakura-ml v0.1.x to v1.0

v1.0 is a clean break from v0.1.x. Every public class moved or was replaced.
This guide gives side-by-side examples for the four integrations.

If you can't migrate yet, pin to `sakura-ml<1.0`:

```bash
pip install 'sakura-ml<1.0'
```

## Mental model shift

**v0.1.x:** thin wrappers per framework. Each integration was a single class
that bundled the eval-dispatch pattern with framework glue.

**v1.0:** explicit composition. You instantiate a `SakuraRuntime`, install
services on it, and attach a thin adapter to the framework callback chain.
Each piece is independently swappable.

## Lightning

### v0.1.x

```python
from sakura.lightning import SakuraTrainer
import zakuro as zk

trainer = SakuraTrainer(
    max_epochs=10,
    accelerator="auto",
    val_compute=zk.Compute(uri="quic://worker:4433"),
    model_factory=lambda: MyLightningModule(),
    val_loader_factory=lambda: DataLoader(val_ds, batch_size=256),
    model_path="checkpoints/best.pth",
)
trainer.run(model, train_loader)

print(trainer.history)
print(trainer.best_val_loss)
```

### v1.0

```python
import lightning as L
from sakura import SakuraRuntime
from sakura.adapters import LightningAdapter
from sakura.dispatch import RemoteDispatcher
from sakura.services import AsyncEval, AsyncCheckpoint

dispatcher = RemoteDispatcher(uri="quic://worker:4433", cert_der=cert_bytes)

def eval_fn(epoch, payload):
    # Your eval logic — runs on the worker. payload is whatever you pass to AsyncEval.
    val_loss = run_evaluation(model, val_loader)
    return {"val_loss": val_loss, "epoch": epoch}

with SakuraRuntime() as rt:
    rt.install(AsyncEval(eval_fn=eval_fn, eval_payload={}, dispatcher=dispatcher))
    rt.install(AsyncCheckpoint(
        dir="checkpoints/", every="best", metric="val_loss",
        dispatcher=dispatcher,
        state_provider=lambda: {k: v.cpu() for k, v in model.state_dict().items()},
    ))

    trainer = L.Trainer(
        max_epochs=10,
        accelerator="auto",
        callbacks=[LightningAdapter(rt)],
    )
    trainer.fit(model, train_loader)

# Access history via the runtime or per-service:
async_eval = rt.find("async_eval")
print(async_eval.history)
```

Notable changes:
- `model_factory` and `val_loader_factory` go away. The eval runs on the
  worker via `AsyncEval`, which receives a cloudpickled `eval_fn`. You decide
  inside `eval_fn` how to reconstruct/run the model.
- `model_path` becomes `AsyncCheckpoint(dir=..., every="best")`.
- `trainer.history` becomes `async_eval.history`. `trainer.best_val_loss` is
  derivable from the same.

## HuggingFace Trainer

### v0.1.x

```python
from transformers import Trainer, TrainingArguments
from sakura.huggingface import SakuraHFCallback
import zakuro as zk

trainer = Trainer(
    model=model,
    args=TrainingArguments(..., eval_strategy="no"),
    train_dataset=train_ds,
    callbacks=[
        SakuraHFCallback(
            model_factory=lambda: AutoModelForSequenceClassification.from_config(config),
            eval_fn=my_eval_fn,
            eval_payload=(val_inputs, 32),
            val_compute=zk.Compute(uri="quic://worker:4433"),
            fp16_state_dict=True,
            on_backpressure="skip",
        ),
    ],
)
trainer.train()
```

### v1.0

```python
from transformers import Trainer
from sakura import SakuraRuntime
from sakura.adapters import HFAdapter
from sakura.dispatch import RemoteDispatcher
from sakura.services import AsyncEval

dispatcher = RemoteDispatcher(uri="quic://worker:4433", cert_der=cert_bytes)

def eval_fn(epoch, payload):
    val_inputs, batch_size = payload
    return run_eval(model, val_inputs, batch_size)

with SakuraRuntime() as rt:
    rt.install(AsyncEval(
        eval_fn=eval_fn,
        eval_payload=(val_inputs, 32),
        dispatcher=dispatcher,
        on_backpressure="skip",
    ))
    trainer = Trainer(
        model=model, args=hf_args, train_dataset=train_ds,
        callbacks=[HFAdapter(rt)],
    )
    trainer.train()
```

## Raw PyTorch DDP

### v0.1.x

```python
from sakura.ddp import DDPAsyncEvalCallback
import torch.distributed as dist
import zakuro as zk

cb = DDPAsyncEvalCallback(
    model_factory=lambda: MyModel(),
    eval_fn=my_eval_fn,
    eval_payload=(val_tensors, 32),
    val_compute=zk.Compute(uri="quic://eval-worker:4433"),
    world_size=dist.get_world_size(),
    rank=dist.get_rank(),
)

for epoch in range(num_epochs):
    train_one_epoch(model, train_loader)
    dist.barrier()
    cb.on_epoch_end(epoch, model)
cb.on_train_end()
```

### v1.0

```python
import torch.distributed as dist
from sakura import SakuraRuntime
from sakura.adapters import DDPAdapter
from sakura.dispatch import RemoteDispatcher
from sakura.services import AsyncEval

dispatcher = RemoteDispatcher(uri="quic://eval-worker:4433", cert_der=cert_bytes)

def eval_fn(epoch, payload):
    val_tensors, batch_size = payload
    return run_eval(model, val_tensors, batch_size)

with SakuraRuntime() as rt:
    rt.install(AsyncEval(eval_fn=eval_fn, eval_payload=(val_tensors, 32),
                          dispatcher=dispatcher))

    adapter = DDPAdapter(rt, rank=dist.get_rank(), world_size=dist.get_world_size())
    adapter.on_train_begin(model, optimizer, train_loader)
    for epoch in range(num_epochs):
        adapter.on_epoch_begin(epoch)
        # ... your training step + per-step adapter.on_train_step_begin / on_optimizer_step ...
        adapter.on_epoch_end(epoch, model, optimizer, metrics={})
    adapter.on_train_end(model)
```

## TensorFlow / Keras

`sakura.tensorflow.SakuraKerasCallback` was **removed entirely at v1.0** — TensorFlow integration is no longer in scope.

If you need TF support, pin `sakura-ml<1.0` and stay on v0.1.x.

## Generic async trainer

`sakura.ml.async_trainer.AsyncTrainer` was **removed at v1.0**. Use
`DDPAdapter` (it's framework-free — works for any explicit training loop) or
build your own adapter by subclassing `sakura.adapters.Adapter` and emitting
the runtime events at the right points.

## API name changes

| v0.1.x | v1.0 |
|---|---|
| `sakura.lightning.SakuraTrainer` | `SakuraRuntime + LightningAdapter + AsyncEval` |
| `sakura.lightning.SakuraLightningCallback` | same as above |
| `sakura.huggingface.SakuraHFCallback` | `SakuraRuntime + HFAdapter + AsyncEval` |
| `sakura.ddp.DDPAsyncEvalCallback` | `SakuraRuntime + DDPAdapter + AsyncEval` |
| `sakura.tensorflow.SakuraKerasCallback` | **removed** |
| `sakura.ml.async_trainer.AsyncTrainer` | **removed** (use `DDPAdapter`) |
| `sakura.ml.sakura_trainer.SakuraTrainer` | **removed** |
| `sakura.functional.*` | **removed** |
| `zakuro.Compute(uri="quic://...")` | `RemoteDispatcher(uri="quic://...", cert_der=...)` or `Compute.at("quic://...")` (resolves at runtime.start()) |

## Knobs that carried over

The following `SakuraHFCallback` knobs map directly onto `AsyncEval`:

| v0.1.x knob | v1.0 location |
|---|---|
| `eval_fn`, `eval_payload` | `AsyncEval(eval_fn=..., eval_payload=...)` |
| `cache_key` | `AsyncEval(cache_key=...)` (Plan 4 worker-side caching) |
| `fp16_state_dict` | passed via the cloudpickled spec; or `MixedPrecision(dtype="fp16")` in v1.0 |
| `max_pending` | `AsyncEval(max_pending=...)` |
| `on_backpressure` | `AsyncEval(on_backpressure=...)` |
| `drain` | `AsyncEval(drain=...)` |

## What's new in v1.0 (worth migrating for)

- **Process isolation** — `LocalDispatcher` auto-spawns a worker subprocess. The GIL never contends between training and eval.
- **Service composition** — install only what you need. `MixedPrecision`, `Compile`, `ZeRO1`, `ActivationCheckpoint` are independent services that compose with `AsyncEval` / `AsyncCheckpoint`.
- **Multi-framework** — same runtime, three adapters. Switching frameworks = swapping the adapter.
- **Telemetry** — `Telemetry(output=path)` writes a JSON-line per event. Single source of truth for benchmarking.
- **Rust transport** — `sakura-wire` over QUIC. Same wire format scales from localhost to LAN/WAN.
