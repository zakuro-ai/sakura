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
| `MixedPrecision` | 10 | train_begin, optimizer_step, wrap_loss, optimizer_step (step) | wraps forward in `torch.autocast`; for fp16, scales loss + drives `GradScaler.step()/update()` via runtime.optimizer_step replacing the loop's default |
| `ActivationCheckpoint` | 15 | train_begin | wraps matching submodules with `torch.utils.checkpoint` |
| `Compile` | 20 | train_begin | `torch.compile` with on-disk cache |
| `ZeRO1` | 30 | train_begin, optimizer_step | optimizer-state sharding (cyclic dealing across ranks; single-rank passthrough). Multi-rank correctness verified against single-rank reference under both **gloo (CPU)** and **NCCL (2× GPU)**. |
| `AsyncEval` | 80 | epoch_end | dispatch eval to worker; lazy future drain |
| `AsyncCheckpoint` | 85 | epoch_end | dispatch state-dict write; modes: epoch / N / best |

Lower priority runs earlier. Service exceptions are isolated — one service crashing emits an `OnError` event but doesn't block the others.

In addition to event handlers, services may implement two optional runtime-coordinated methods:

- `wrap_loss(loss) -> loss` — invoked by `runtime.scale_loss(loss)` before `loss.backward()`. Threads the loss through every service in priority order. `MixedPrecision` uses this to apply `GradScaler.scale()` for fp16; bf16/fp8/auto are passthroughs.
- `optimizer_step(optimizer) -> bool` — invoked by `runtime.optimizer_step(opt)` after the `OnOptimizerStep` event. First service to return `True` claims the step, and the loop must skip its default `opt.step()`. `MixedPrecision` returns `True` for fp16 to drive `GradScaler.step()/update()`; otherwise returns `False` and the loop steps as usual.

These hooks are honored by the bench harness's raw-pytorch loop. Lightning / HF Trainer manage their own step lifecycles and don't dispatch through these methods — use the framework's native precision config there.

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
| `ThreadDispatcher` | — | In-process Python thread; real parallelism for tensor ops (torch releases the GIL) without subprocess pickle cost. Best for `AsyncEval` / `AsyncCheckpoint` when isolation isn't needed. |
| `LocalDispatcher` | auto | Auto-spawns localhost `sakura-worker` subprocess; full GIL isolation, ~50ms+ pickle overhead per round-trip |
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

# sakura overlapping per-epoch eval with next-epoch training (1.57x on CPU; see "Measured results")
sakura-bench run --workload mnist-mlp-multi --runner sakura --framework pytorch-ddp \
    --service async_eval:thread --output reports/

# both async services together — eval + checkpoint dispatched off-thread per epoch
sakura-bench run --workload mnist-mlp-multi --runner sakura --framework pytorch-ddp \
    --service async_eval:thread --service async_checkpoint:thread --output reports/

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

Available workloads: `mnist-mlp`, `mnist-mlp-multi`, `cifar10-resnet50`, `distilbert-sst2`, `distilbert-sst2-hf`, `distilbert-glue`, `llama3-1b-finetune`, `mistral-7b-lora`. Available frameworks: `pytorch-ddp`, `lightning`, `hf-trainer`. The `hf-trainer` framework requires an HF-shaped workload — one whose `make_model()` returns a `transformers.PreTrainedModel` (or any model whose `forward(**batch)` returns an output with `.loss`) and whose loaders yield single-dict batches with a `labels` key. `distilbert-sst2-hf` is the reference HF-shaped workload; `distilbert-sst2` (with the wrapper) targets the pytorch-ddp / lightning loops. `mnist-mlp-multi` is a multi-epoch synthetic workload sized so per-epoch eval ≈ per-epoch training cost — the regime where `AsyncEval` overlap pays off.

### Measured results (CPU)

After the runtime hot-path optimizations (cached CUDA detection, fast path for empty service stack, opt-out history bookkeeping), sakura's overhead vs. raw PyTorch is **within noise** on tiny workloads (median over 20 trials, MNIST/MLP smoke):

| Configuration | Median | vs. baseline |
|---|---|---|
| baseline (raw PyTorch loop) | 155 ms | — |
| sakura runtime, no services | 153 ms | −1.2% |
| sakura + Telemetry | 152 ms | −1.7% |

(All within trial-to-trial noise; the practical claim is "no overhead.")

The structural win arrives when there's something to overlap. `AsyncEval` + `ThreadDispatcher` runs the per-epoch evaluation on a background thread — torch's C++ kernels release the GIL so this is real parallelism on CPU — so eval overlaps with the next epoch's training. The win is reproducible from the CLI in two commands:

```bash
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 sakura-bench run \
    --workload mnist-mlp-multi --runner baseline --framework pytorch-ddp \
    --output reports/

OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 sakura-bench run \
    --workload mnist-mlp-multi --runner sakura --framework pytorch-ddp \
    --service async_eval:thread --output reports/

sakura-bench compare reports/mnist-mlp-multi-baseline-pytorch-ddp.json \
                     reports/mnist-mlp-multi-sakura-pytorch-ddp.json
```

| Configuration | Median (5 epochs, 5 evals × 16k samples, 256-hidden MLP) | Speedup |
|---|---|---|
| baseline (eval blocks each epoch) | 1179 ms | 1.00× |
| sakura + `--service async_eval:thread` | 750 ms | **1.57×** (−36% wallclock) |

Median over 7 trials, `OMP_NUM_THREADS=2`. Identical val_acc at convergence (eval is correct — only the timing changes). The win comes from N−1 of the N evals finishing during training time that would have happened anyway. The sweet spot is when per-epoch eval cost is comparable to per-epoch training cost; if eval is much heavier than training, AsyncEval's backpressure forces the next epoch to wait and the speedup shrinks.

### Measured results (GPU, RTX 4090)

bf16 mixed precision is the headline GPU win — Ada Lovelace tensor cores are roughly 2× the throughput of fp32 on GEMM-heavy ops. Two end-to-end benchmarks on a single RTX 4090 (compute 8.9, 24 GB VRAM):

**ResNet-50 / CIFAR-10 (224×224)** — 3 epochs, batch 128, 4096 train / 1024 val:

| Config | elapsed | samples/sec | peak GPU mem | speedup |
|---|---|---|---|---|
| baseline (fp32) | 23.55s | 522 | 10714 MB | 1.00× |
| `--service mixed_precision:bf16` | 17.85s | 689 | 5641 MB | **1.32×** (47% memory cut) |
| `--service mixed_precision:fp16` | 17.87s | 688 | 5641 MB | **1.32×** (GradScaler stable, dynamic scale 65536 → 8192) |

The `fp16` row exercises sakura's runtime-coordinated GradScaler integration end-to-end: `wrap_loss(loss)` calls `scaler.scale`, `optimizer_step(opt)` returns True so the loop skips its default `opt.step()` and the scaler drives `scaler.step` + `scaler.update` instead. The scale factor adjusts from the canonical fp16 init `2^16 = 65536` down to a stable `8192` during training — confirming the dynamic-scaling math runs (the inf/nan check + scale adjustment that the previous "unscale-only" integration was missing). val_loss tracks bf16 within ~0.5; no NaN explosion.

**DistilBERT / SST-2** — HF Trainer path, 2 epochs, batch 64, 4096 train, max_length 128:

| Config | elapsed | samples/sec | peak GPU mem | speedup |
|---|---|---|---|---|
| `hf+fp32` (baseline) | 9.14s | 896 | 3359 MB | 1.00× |
| `hf+bf16` (Trainer's native bf16) | 5.07s | 1615 | 2522 MB | **1.80×** |
| `hf+bf16` + sakura HFAdapter + Telemetry | 5.07s | 1616 | 2522 MB | 1.80× |

The DistilBERT result also confirms sakura's HF integration cost is **zero**: installing HFAdapter + Telemetry on top of HF Trainer's native bf16 produces identical wallclock to vanilla. Reproduce with `scripts/bench_gpu_resnet50.py` and `scripts/bench_gpu_distilbert.py` (single-trial; bf16 wins are large enough to dominate noise).

**What didn't work on GPU (honest negatives).** Two configs we tried that *don't* speed up the GPU runs at this scale:

- `--service compile` (torch.compile) — at 3 epochs the JIT compilation cost (~70s) doesn't amortize. Useful only when training runs many epochs over the same compiled graph.
- `--service async_eval:thread` — the bridge runs eval on CPU to overlap with GPU training, but ResNet-50 *eval* on CPU is far slower than GPU *training*, so AsyncEval blocks. AsyncEval's CPU-thread overlap is the right pattern when eval ≈ train cost (the `mnist-mlp-multi` 1.57× CPU result above), not when GPU training dwarfs CPU eval. A GPU-stream-based dispatcher would address this regime; not yet implemented.

**Net.** sakura's bf16 path is the real headline: 1.32× on ResNet-50, 1.80× on DistilBERT, no overhead on the HF Trainer integration. AsyncEval's CPU-overlap pattern is correct on CPU-bound workloads, miscast on GPU-bound ones — the failure mode is documented above.

### Known limitations

- **`MixedPrecision` fp16 in non-DDP framework loops.** Lightning's automatic-optimization and HF Trainer own `opt.step()` themselves and don't expose an interception point — for those frameworks, configure precision via `L.Trainer(precision="16-mixed")` or HF's `TrainingArguments(fp16=True)` instead of installing this service. The runtime-coordinated step replacement (`runtime.scale_loss(loss)` + `runtime.optimizer_step(opt)`) is honored by sakura's raw-DDP loop, where the full fp16 path (scale → backward → unscale → grad-clip → scaler.step → scaler.update) runs end-to-end. Validated end-to-end on RTX 4090: dynamic scale converges, no NaN explosion, wallclock matches bf16 within 0.1% (see "Measured results (GPU)" above).
- **`ZeRO1` multi-rank** uses cyclic dealing (param `i` owned by rank `i % world_size`) and per-shard `opt.step()` followed by parameter broadcast. Verified bit-equivalent to a single-rank `opt.step()` over 1- and 5-step SGD trajectories under `gloo` on CPU (`tests/zero/test_sharded_optimizer_multi_rank.py`) and over 1-step SGD under `NCCL` on two CUDA devices (`tests/zero/test_sharded_optimizer_nccl.py`). NCCL test sets `NCCL_SHM_DISABLE=1` + `NCCL_P2P_DISABLE=1` to fall back to the socket transport for sandboxed environments where the shm segment can't be sized; production users on bare metal should leave the defaults.
- **`Compile` on CPU.** `torch.compile` is GPU-optimized (Inductor's biggest wins are kernel fusion + cudagraphs). On the CPU smoke workloads in `sakura-bench`, installing `--service compile` is at-best neutral (within trial noise) and often a small regression because the JIT/cache-warmup cost outweighs the inner-loop savings. Recommend `--service compile` for GPU runs only; on CPU, leave it off.
- The cross-framework speed comparison vs. raw HuggingFace Trainer / Lightning *without* sakura is wired (`BaselineRunner` covers `pytorch-ddp`, `lightning`, and `hf-trainer`); the HF `hf+bf16+sakura` GPU row above is the first populated entry. ResNet-50 + CIFAR-10 in the perf tier shipped here; `distilbert-glue` (multi-task GLUE) and `llama3-1b-finetune` are still stubs.

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
