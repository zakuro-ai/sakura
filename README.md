<h1 align="center">Sakura</h1>

<p align="center">
  SOTA training services for PyTorch DDP / Lightning / HuggingFace Trainer.
  Async eval, async checkpoint, mixed precision, torch.compile, ZeRO-1, all
  installable on a single runtime, all driving a Rust-backed QUIC transport.
</p>

<p align="center">
  <a href="#install">Install</a> •
  <a href="#quickstart">Quickstart</a> •
  <a href="#architecture">Architecture</a> •
  <a href="#services">Services</a> •
  <a href="#adapters">Adapters</a> •
  <a href="#dispatchers">Dispatchers</a> •
  <a href="#benchmarks">Benchmarks</a> •
  <a href="#migrating-from-v01x">Migrating from v0.1.x</a>
</p>

---

## What is Sakura?

Sakura v1.0 is a standard library that sits on top of every PyTorch frontend (`torch.distributed.DistributedDataParallel`, `lightning.Trainer`, `transformers.Trainer`) and accelerates training via **a small, explicit set of installable services**:

- `Telemetry` — JSON-line event sink
- `MixedPrecision` — autocast policies + GradScaler for fp16
- `ActivationCheckpoint` — selective `torch.utils.checkpoint` wrapping
- `Compile` — `torch.compile` with on-disk cache
- `ZeRO1` — optimizer-state sharding
- `AsyncEval` — eval at epoch end, dispatched off the training thread
- `AsyncCheckpoint` — state-dict writes, dispatched off the training thread

Async services dispatch work to a **`sakura-worker` subprocess over QUIC** (loopback or LAN/WAN). The transport is a Rust crate (`sakura-wire`) exposed to Python via PyO3. Process isolation means the GIL never contends between training and eval/checkpoint work — a real constraint that thread-pool-based async patterns hit head-on.

Three framework adapters translate framework hooks into runtime events:

- `LightningAdapter` — a `lightning.Callback`
- `HFAdapter` — a `transformers.TrainerCallback`
- `DDPAdapter` — explicit hooks for raw `torch.distributed` loops

You install services on a `SakuraRuntime`, attach an adapter to your training loop, run as usual.

## Install

```bash
pip install sakura-ml
# or with framework integrations:
pip install 'sakura-ml[lightning,huggingface]'
```

From source:

```bash
git clone https://github.com/zakuro-ai/sakura && cd sakura
uv pip install maturin
maturin develop --release
```

> **Wheel packaging** is being finalized — until then, the from-source path is the recommended install.

## Quickstart

### Lightning

```python
import lightning as L
from sakura import SakuraRuntime
from sakura.adapters import LightningAdapter
from sakura.services import MixedPrecision, Compile, AsyncEval, AsyncCheckpoint
from sakura.dispatch import InThreadDispatcher  # or LocalDispatcher() to spawn a worker

with SakuraRuntime() as rt:
    rt.install(MixedPrecision(dtype="bf16"))
    rt.install(Compile(mode="reduce-overhead"))
    rt.install(AsyncEval(
        eval_fn=lambda epoch, payload: {"val_loss": evaluate(model, val_loader)},
        eval_payload={},
        dispatcher=InThreadDispatcher(),
    ))
    rt.install(AsyncCheckpoint(
        dir="ckpt/", every="best", metric="val_loss",
        dispatcher=InThreadDispatcher(),
        state_provider=lambda: {k: v.cpu() for k, v in model.state_dict().items()},
    ))

    trainer = L.Trainer(
        max_epochs=10,
        accelerator="auto",
        callbacks=[LightningAdapter(rt)],
    )
    trainer.fit(model, train_loader)
```

### HuggingFace Trainer

```python
from transformers import Trainer
from sakura import SakuraRuntime
from sakura.adapters import HFAdapter
from sakura.services import MixedPrecision, AsyncEval

with SakuraRuntime() as rt:
    rt.install(MixedPrecision(dtype="bf16"))
    rt.install(AsyncEval(eval_fn=eval_fn, eval_payload=val_payload,
                          dispatcher=InThreadDispatcher()))
    trainer = Trainer(model=model, args=hf_args, train_dataset=train_ds,
                      callbacks=[HFAdapter(rt)])
    trainer.train()
```

### Raw PyTorch DDP

```python
import torch.distributed as dist
from sakura import SakuraRuntime
from sakura.adapters import DDPAdapter
from sakura.services import ZeRO1, AsyncEval

with SakuraRuntime() as rt:
    rt.install(ZeRO1())
    rt.install(AsyncEval(eval_fn=eval_fn, eval_payload=val_payload,
                          dispatcher=InThreadDispatcher()))

    adapter = DDPAdapter(rt, rank=dist.get_rank(), world_size=dist.get_world_size())
    adapter.on_train_begin(model, optimizer, train_loader)
    for epoch in range(num_epochs):
        adapter.on_epoch_begin(epoch)
        for step, batch in enumerate(train_loader):
            adapter.on_train_step_begin(model, batch, step)
            loss = train_one_step(model, batch, optimizer)
            adapter.on_optimizer_step(optimizer)
        adapter.on_epoch_end(epoch, model, optimizer, metrics={"train_loss": loss})
    adapter.on_train_end(model)
```

### Out-of-process worker (auto-spawned)

`LocalDispatcher` auto-spawns a `sakura-worker` subprocess on first dispatch. The eval runs in a separate Python interpreter — the GIL never contends with the training loop:

```python
from sakura.dispatch import LocalDispatcher

dispatcher = LocalDispatcher()  # spawns localhost worker over QUIC
rt.install(AsyncEval(eval_fn=eval_fn, eval_payload=val_payload, dispatcher=dispatcher))
```

To target an existing worker on another host:

```python
from sakura.dispatch import RemoteDispatcher
dispatcher = RemoteDispatcher(uri="quic://eval-host:4433", cert_der=cert_bytes)
```

## Architecture

```
┌─── Training process (Python, GIL) ──────────────────────┐
│  framework loop (Lightning / HF / raw DDP)              │
│        │ hook                                           │
│        ▼                                                │
│  Adapter — translates hooks → typed events              │
│        │                                                │
│        ▼                                                │
│  SakuraRuntime — event bus + service registry           │
│        │                                                │
│        ├─→ in-process services (MixedPrecision, …)      │
│        └─→ dispatching services (AsyncEval, AsyncCkpt)  │
│                  │                                      │
│                  ▼                                      │
│            Dispatcher (Local | Remote | InThread)       │
│                  │ PyO3 → sakura-wire (Rust)            │
│                  ▼ QUIC                                 │
└──────────────────│──────────────────────────────────────┘
                   │
┌──────────────────▼──────────────────────────────────────┐
│  sakura-worker subprocess (Python, separate GIL)        │
│      QUIC server → HandlerRegistry → user callable      │
└─────────────────────────────────────────────────────────┘
```

Five execution states cover every dispatching combination: in-thread (synchronous, for tests), in-process (single Python proc), localhost subprocess (default), remote subprocess (cluster), Zakuro-backed (existing infra). A typed event bus (`OnTrainBegin`, `OnEpochEnd`, etc.) carries `rank` and `world_size` so DDP-aware services branch on `event.rank` without each adapter doing the bookkeeping.

## Services

| Service | Priority | Hooks consumed | What it does |
|---|---|---|---|
| `Telemetry` | 0 | every event | JSON record sink (callable / file / stream) |
| `MixedPrecision` | 10 | train_begin, optimizer_step | wraps forward in `torch.autocast`; `GradScaler` for fp16 |
| `ActivationCheckpoint` | 15 | train_begin | wraps matching submodules with `torch.utils.checkpoint` |
| `Compile` | 20 | train_begin | `torch.compile` with on-disk cache |
| `ZeRO1` | 30 | train_begin, optimizer_step | optimizer-state sharding (single-rank passthrough; multi-rank in progress) |
| `AsyncEval` | 80 | epoch_end | dispatch eval to worker; lazy future drain |
| `AsyncCheckpoint` | 85 | epoch_end | dispatch state-dict write; modes: epoch / N / best |

Lower priority runs earlier. Service exceptions are isolated — one service crashing emits an `OnError` event but doesn't block the others.

## Adapters

| Adapter | Type | Use case |
|---|---|---|
| `LightningAdapter` | `lightning.Callback` | Drop-in for `lightning.Trainer` |
| `HFAdapter` | `transformers.TrainerCallback` | Drop-in for `transformers.Trainer` (>=4.38) |
| `DDPAdapter` | explicit hooks | Raw PyTorch DDP loops |

## Dispatchers

| Dispatcher | URI | When |
|---|---|---|
| `InThreadDispatcher` | — | Tests / debug; runs synchronously |
| `LocalDispatcher` | auto | Default; auto-spawns localhost `sakura-worker` |
| `RemoteDispatcher` | `quic://host:port` | Existing remote worker daemon |
| `ZakuroDispatcher` | — | Wraps `zakuro.Compute` for users with existing Zakuro infra |

## Benchmarks

A reproducible benchmark harness ships with the package: `sakura-bench`. It runs a `Workload` through either a vanilla framework loop (`BaselineRunner`) or through `SakuraRuntime` with a chosen set of services (`SakuraRunner`), writes a `RunReport` JSON, and exports markdown comparisons.

```bash
# baseline (vanilla PyTorch DDP loop)
sakura-bench run --workload mnist-mlp --runner baseline --framework pytorch-ddp \
    --output reports/

# sakura with mixed-precision + telemetry
sakura-bench run --workload cifar10-resnet50 --runner sakura --framework pytorch-ddp \
    --service telemetry --service mixed_precision:bf16 --output reports/

# HuggingFace Trainer baseline (HF-shaped workload required)
sakura-bench run --workload distilbert-sst2-hf --runner baseline --framework hf-trainer \
    --output reports/

# sakura on top of HF Trainer
sakura-bench run --workload distilbert-sst2-hf --runner sakura --framework hf-trainer \
    --service telemetry --output reports/

# pair-wise comparison
sakura-bench compare reports/cifar10-resnet50-baseline-pytorch-ddp.json \
                     reports/cifar10-resnet50-sakura-pytorch-ddp.json

# markdown export across runs
sakura-bench export reports/*.json
```

Available workloads: `mnist-mlp`, `cifar10-resnet50`, `distilbert-sst2`, `distilbert-sst2-hf`, `distilbert-glue`, `llama3-1b-finetune`, `mistral-7b-lora`. Available frameworks: `pytorch-ddp`, `lightning`, `hf-trainer`. The `hf-trainer` framework requires an HF-shaped workload — one whose `make_model()` returns a `transformers.PreTrainedModel` (or any model whose `forward(**batch)` returns an output with `.loss`) and whose loaders yield single-dict batches with a `labels` key. `distilbert-sst2-hf` is the reference HF-shaped workload; `distilbert-sst2` (with the wrapper) targets the pytorch-ddp / lightning loops.

**Honest framing.** The harness is real and ships; the headline speed numbers do not yet. Microbenchmarks on tiny CPU workloads (1-epoch MNIST/MLP, 1-epoch CIFAR with 32 train samples) show per-event service-dispatch overhead exceeding the wins from `MixedPrecision` / `Compile` at that scale — `AsyncEval` and `AsyncCheckpoint` need workloads where eval/checkpoint cost is meaningful relative to one training epoch before they amortize. Validating the speed claim requires GPU runs on the larger tiers (`cifar10-resnet50` full, `distilbert-glue`, `llama3-1b-finetune`) on real hardware, which has not been done in this release. Treat this as: harness ready, numbers TBD.

### Known limitations

- **`MixedPrecision` fp16 + GradScaler integration is incomplete.** `on_optimizer_step` unscales gradients and applies grad-clip, but does **not** call `scaler.step()` / `scaler.update()` — the framework's training loop owns `opt.step()` and the service can't currently override it. fp16 inf/nan detection and dynamic scale updates therefore don't run; if a NaN appears it propagates through the loop's step. `bf16` (the default for `dtype="auto"` on Ampere+) does not need a scaler and works as advertised.
- **`ZeRO1` multi-rank** uses cyclic dealing across ranks via `gloo` for parameter all-gather; the single-rank path is a passthrough.
- The cross-framework speed comparison vs. raw HuggingFace Trainer / Lightning *without* sakura is wired (`BaselineRunner` covers `pytorch-ddp`, `lightning`, and `hf-trainer`); populating the markdown table from real hardware runs is the next milestone.

## Migrating from v0.1.x

v0.1.x submodules (`sakura.lightning.SakuraTrainer`, `sakura.huggingface.SakuraHFCallback`, `sakura.ddp.DDPAsyncEvalCallback`, `sakura.tensorflow.*`, `sakura.ml.*`) have been **removed at v1.0**.

Users on v0.1.x should pin `sakura-ml<1.0` if they're not migrating. To migrate, see [`docs/migration-from-0.1.md`](docs/migration-from-0.1.md).

## Development

```bash
git clone https://github.com/zakuro-ai/sakura && cd sakura
uv venv && source .venv/bin/activate
uv pip install maturin pytest cloudpickle numpy torch lightning transformers
maturin develop --release
pytest tests/
```

Rust workspace: `crates/sakura-wire/` — codec + protocol + QUIC transport + PyO3 bindings.

```bash
cargo test --workspace
cargo bench -p sakura-wire
```

## License

BSD-3-Clause.
