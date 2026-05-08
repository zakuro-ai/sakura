---
title: Sakura v1.0 redesign — plugin runtime, Rust wire codec, SOTA training services
date: 2026-05-08
status: approved (brainstorming complete; pending implementation plan)
authors: ZakuroAI (Jean Maximilien Cadic) + Claude (brainstorming partner)
supersedes: sakura-ml v0.1.x architecture
---

# Sakura v1.0 — Redesign Spec

## 1. Executive summary

Sakura v0.1.x is a Python wrapper layer (~1.5k lines of integration code) that adds a single SOTA pattern — *async eval-during-train* — on top of PyTorch Lightning, HuggingFace `Trainer`, TF/Keras, and raw PyTorch DDP, by dispatching evaluation through Zakuro (`@zk.fn` + `zk.Compute`). Real but narrow.

Sakura v1.0 is a clean rewrite as a **plugin runtime + Rust transport** standard library that integrates with every major PyTorch frontend and bundles a curated set of SOTA training services. It owns its own dispatch fabric (Rust `sakura-wire` crate over QUIC), so process isolation, GIL avoidance, and crash isolation are first-class — and the same wire format scales unchanged from localhost loopback to multi-node QUIC/RDMA.

The headline promise: **measurably faster training across PyTorch DDP / Lightning / HF Trainer with a small, explicit set of installable services**, proven by a permanent two-tier benchmark suite that compares against vanilla baselines (PyTorch DDP, Lightning, HF Trainer, TensorFlow) on standard workloads.

## 2. Goals and non-goals

### Goals (v1)

- **Speed** — Sakura demonstrably trains the same models faster than vanilla PyTorch DDP, Lightning, and HF Trainer on a representative single-node 4090/A100 workload set.
- **Integration breadth across PyTorch frontends** — three first-class adapters: raw DDP, Lightning, HF Trainer.
- **Plugin runtime architecture** — a `SakuraRuntime` event bus + `Service` abstraction lets new SOTA techniques drop in without touching adapters, and new framework adapters drop in without touching services.
- **Rust wire codec from day 1** — `sakura-wire` crate (PyO3) provides zero-copy tensor transport over QUIC; same crate scales to multi-node in v2.
- **Pluggable dispatcher** — `LocalDispatcher` (auto-spawned localhost worker), `RemoteDispatcher` (QUIC to remote `sakura-worker`), `ZakuroDispatcher` (wraps `@zk.fn` for users with existing Zakuro infra), `InThreadDispatcher` (debug only).
- **Three SOTA service clusters in v1** — async-everything (eval + checkpoint + telemetry), compute acceleration (`torch.compile` + mixed precision policies), memory efficiency (ZeRO-1 + selective activation checkpointing).
- **Two-tier benchmark suite** — CI-lightweight tier (MNIST + CIFAR-10/ResNet-50 + DistilBERT/SST-2) always runs; LLM tier (Llama-3-1B + Mistral-7B LoRA + DistilBERT/GLUE) runs when GPU resources allow.
- **Clean break at v1.0** — old `SakuraTrainer`, `SakuraHFCallback`, etc. classes removed; v0.1.x stays on PyPI under `sakura-ml<1.0` for users who haven't migrated.

### Non-goals (v1, deferred to v2 or later)

- TensorFlow/Keras integration (`sakura.tensorflow` removed; TF stays as benchmark *baseline* only).
- JAX integration (deferred to v2; functional/no-callback model needs different abstraction).
- Multi-node distributed training optimizations (gradient compression, RDMA transport, custom collectives) — architecture anticipates them; implementation lands in v2.
- ZeRO-2 / ZeRO-3 (gradient/parameter sharding) — v1 stops at stage 1.
- Tensor parallelism, pipeline parallelism — out of scope.
- A "preset" / `with_preset()` ergonomic shorthand — explicit `runtime.install(...)` calls in v1; presets considered for v1.x.
- CUDA IPC handle path for colocated workers — designed-in, but feature-gated, lands in v1.x.
- `unix-shm://` transport for same-host zero-copy — designed-in, ships in v1.x.

## 3. Resolved decisions (brainstorming inputs)

| Decision | Choice |
|---|---|
| Headline goal | Speed (beat baselines by measurable %) + Integration breadth |
| Topology | Single-node first, multi-node next; same code path |
| Rust scope | `sakura-wire` codec + transport from day 1 |
| Dispatch architecture | Pluggable `Dispatcher` ABC; Zakuro is one of N backends |
| Frameworks | PyTorch DDP + Lightning + HF Trainer; TF dropped, JAX deferred |
| v1 service clusters | Async-everything + compute-acceleration + memory-efficiency |
| Benchmark scope | Two-tier (CI-lightweight always; LLM tier when resources allow) |
| API migration | Clean break at v1.0; no compatibility shim |
| Architecture pattern | Plugin runtime: `SakuraRuntime` + `Service` + `Adapter` + `Dispatcher` |

## 4. Architecture

### 4.1 The four primitives

1. **`SakuraRuntime`** — owns the event bus, the service registry, the telemetry sink, and the `WorkerSupervisor`. Lifecycle: `runtime.start()` → adapters and services emit/consume events → `runtime.shutdown()` drains in-flight work and reaps workers. Auto-spawns a localhost `sakura-worker` subprocess on first dispatch when no `compute=` is configured.
2. **`Service`** — installable unit of behavior. Two kinds:
   - **In-process services** run entirely in the training process. PyTorch ops they wrap release the GIL during CUDA work, so no contention with the training step. v1 in-process services: `MixedPrecision`, `Compile`, `ZeRO1`, `ActivationCheckpoint`, `Telemetry`.
   - **Dispatching services** submit work to a worker process via the dispatcher. The work runs out-of-process (its own GIL); the service only orchestrates submission and `Future` collection. v1 dispatching services: `AsyncEval`, `AsyncCheckpoint`.
3. **`Adapter`** — thin per-framework bridge. Translates framework hooks to runtime events. Never runs techniques; only emits events with the model/optimizer/loaders surfaced. v1: `LightningAdapter`, `HFAdapter`, `DDPAdapter`.
4. **`Dispatcher`** — abstract `submit(handler_id, tensors, aux_payload, *, timeout_ms) -> Future`. Concrete: `LocalDispatcher` (loopback QUIC to spawned worker), `RemoteDispatcher` (QUIC to remote daemon), `ZakuroDispatcher` (wraps `@zk.fn`), `InThreadDispatcher` (debug only).

### 4.2 Two-process minimum, identical to multi-node

```
┌─────────────────────── Training process (Python, GIL) ───────────────────────┐
│  framework loop (Lightning / HF Trainer / raw DDP)                           │
│        │ hook                                                                │
│        ▼                                                                     │
│  Adapter (LightningAdapter / HFAdapter / DDPAdapter)                         │
│        │ emits events                                                        │
│        ▼                                                                     │
│  SakuraRuntime ── event bus ──┬─→ MixedPrecision     (in-process)            │
│        │                      ├─→ Compile            (in-process)            │
│        │                      ├─→ ZeRO1              (in-process, NCCL)      │
│        │                      ├─→ ActivationCkpt     (in-process)            │
│        │                      ├─→ Telemetry          (in-process, async)     │
│        │                      ├─→ AsyncEval ────┐                            │
│        │                      └─→ AsyncCkpt ────┤                            │
│        ▼                                        ▼                            │
│  Dispatcher ── PyO3 ─→ sakura-wire (Rust): Codec ─→ Protocol ─→ Transport    │
│                                                                  │           │
└──────────────────────────────────────────────────────────────────│───────────┘
                                                                   │ QUIC over UDP
                                                                   │ (loopback or LAN/WAN)
┌──────────────────────────────────────────────────────────────────▼───────────┐
│  Worker process (Python, separate GIL) — `sakura-worker` daemon              │
│      sakura-wire QUIC server ─→ Protocol decode ─→ Handler registry          │
│        │                                                                     │
│        ▼                                                                     │
│  user-supplied callable: eval_fn(model, payload) | checkpoint_fn(state, dir) │
└──────────────────────────────────────────────────────────────────────────────┘
```

**Why subprocess + QUIC + Rust over threads:**

| Concern | Threads (v0.1.x) | Subprocess + QUIC (v1) |
|---|---|---|
| GIL contention with training loop | Yes — `cloudpickle`, fp16 cast, `torch.save` all hold GIL | No — worker has its own interpreter |
| Crash isolation | No — eval bug kills the run | Yes — worker dies, training continues |
| CUDA context isolation | Shared (eval competes for stream/SMs) | Independent (worker can pin GPU 1 while training uses GPU 0) |
| Single-node ↔ multi-node code path | Different (threads vs network) | Same (URI changes; codec/protocol/transport identical) |
| Default-no-setup laptop demo | "Standalone" eats GIL | Auto-spawn localhost worker; user sees nothing |
| Multi-node v2 work | New code path needed | Same dispatcher, worker is on another host |

### 4.3 Concurrency model

- Training process: framework's main thread runs the training step normally. Services run synchronously inside event handlers (cheap operations like setting up an autocast context). Dispatching services submit RPCs and return — they do not block.
- The Rust transport runs on its own tokio runtime in a Rust-owned OS thread (the GIL is released while in Rust). Result delivery raises a Python event/condvar that the runtime polls non-blockingly at event-handler entry.
- Worker process: single-threaded by default; CUDA work parallelizes across SMs as usual. Multiple workers (via `WorkerSupervisor.scale(n)`) for parallel eval against multiple checkpoints.

### 4.4 Anti-coupling rules

- A service must not import from `sakura.lightning` / `sakura.huggingface` / `sakura.ddp`. Services are framework-agnostic; they consume model/optimizer references provided by the adapter through events.
- Adapters must not call services directly — only emit events. This keeps services swappable and makes "fewer services + minimal adapter" the testing baseline.
- The dispatcher abstraction is the only path for cross-process or off-host work. Services never spawn their own threads or processes.

## 5. Public API surface

### 5.1 Top-level type catalog

```python
# sakura/__init__.py — top-level surface
class SakuraRuntime:
    def __init__(self, *, compute: Compute | None = None,
                 logger: Callable[[dict], None] | None = None) -> None: ...
    def install(self, service: Service) -> None: ...
    def uninstall(self, name: str) -> None: ...
    def history(self) -> list[dict]: ...
    def start(self) -> None: ...
    def shutdown(self, *, timeout: float = 30.0) -> None: ...
    def __enter__(self) -> "SakuraRuntime": ...
    def __exit__(self, *exc) -> None: ...

class Service(Protocol):
    name: str
    priority: int                                # 0 = early, 100 = late
    requires: tuple[str, ...]
    def on_install(self, runtime: SakuraRuntime) -> None: ...
    def on_event(self, event: Event) -> None: ...

class Compute:
    @classmethod
    def local(cls, *, n_workers: int = 1, gpus: list[int] | None = None) -> Compute: ...
    @classmethod
    def at(cls, uri: str) -> Compute: ...
    @classmethod
    def pool(cls, uris: list[str]) -> Compute: ...
    @classmethod
    def zakuro(cls, zk_compute) -> Compute: ...
    @classmethod
    def in_thread(cls) -> Compute: ...
```

### 5.2 Event schema (adapter ↔ service contract)

| Event | Payload |
|---|---|
| `on_train_begin` | `model, optimizer, train_loader, val_loader?` |
| `on_epoch_begin` | `epoch` |
| `on_train_step_begin` | `model, batch, step` |
| `on_optimizer_step` | `optimizer` |
| `on_epoch_end` | `epoch, model, optimizer, metrics` |
| `on_save` | `path, state_dict` |
| `on_train_end` | `model, history` |
| `on_error` | `exc, context` |

For DDP, every event payload includes `rank` and `world_size`. Services that should run on rank 0 only check `event.rank == 0`.

### 5.3 User-facing examples

**Lightning quickstart (auto-spawned local worker):**

```python
import lightning as L
from sakura import SakuraRuntime
from sakura.lightning import LightningAdapter
from sakura.services import MixedPrecision, Compile, AsyncEval, AsyncCheckpoint

with SakuraRuntime() as rt:                                 # auto-spawns localhost worker
    rt.install(MixedPrecision(dtype="bf16"))
    rt.install(Compile(mode="reduce-overhead"))
    rt.install(AsyncEval(model_factory=MyModule,
                          val_loader_factory=val_loader_fn,
                          eval_fn=val_step))
    rt.install(AsyncCheckpoint(dir="ckpt/", every="best", metric="val_loss"))

    trainer = L.Trainer(max_epochs=10, accelerator="auto",
                        callbacks=[LightningAdapter(rt)])
    trainer.fit(model, train_loader)
    print(rt.history())
```

**HF Trainer with explicit pool of workers:**

```python
from sakura import SakuraRuntime, Compute
from sakura.huggingface import HFAdapter
from sakura.services import MixedPrecision, ZeRO1, AsyncEval, AsyncCheckpoint

compute = Compute.pool(["quic://gpu-eval-1.lan:4433", "quic://gpu-eval-2.lan:4433"])

with SakuraRuntime(compute=compute) as rt:
    rt.install(MixedPrecision(dtype="bf16"))
    rt.install(ZeRO1())
    rt.install(AsyncEval(model_factory=lambda: AutoModelForSequenceClassification.from_config(cfg),
                          eval_fn=eval_fn, eval_payload=val_payload,
                          on_backpressure="skip"))
    rt.install(AsyncCheckpoint(dir="ckpt/", every="best", metric="val_loss"))

    trainer = Trainer(model=model, args=hf_args, train_dataset=train_ds,
                      callbacks=[HFAdapter(rt)])
    trainer.train()
```

**Raw PyTorch DDP (explicit hooks):**

```python
import torch.distributed as dist
from sakura import SakuraRuntime, Compute
from sakura.ddp import DDPAdapter
from sakura.services import MixedPrecision, ZeRO1, AsyncEval, AsyncCheckpoint

with SakuraRuntime(compute=Compute.local(n_workers=1, gpus=[1])) as rt:
    rt.install(MixedPrecision(dtype="bf16"))
    rt.install(ZeRO1())
    rt.install(AsyncEval(model_factory=MyModel, eval_fn=eval_fn, eval_payload=val_payload))
    rt.install(AsyncCheckpoint(dir="ckpt/"))

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

**Custom service (10 lines, no internals):**

```python
from sakura import Service
from sakura.events import OnEpochEnd

class GradNormLogger(Service):
    name = "grad_norm_logger"
    priority = 90
    requires = ()

    def on_event(self, event):
        if isinstance(event, OnEpochEnd):
            total = sum(p.grad.norm().item() ** 2
                        for p in event.model.parameters() if p.grad is not None) ** 0.5
            self.runtime.telemetry.emit({"epoch": event.epoch, "grad_norm": total})
```

**One-line diff to point at a remote worker:**

```python
# Before:  SakuraRuntime()                                       → localhost worker
# After:   SakuraRuntime(compute=Compute.at("quic://eval-host:4433"))  → remote worker
```

### 5.4 `sakura-worker` daemon CLI

```bash
# Start a worker that listens on QUIC, pinned to GPU 1, with self-signed TLS for dev.
sakura-worker --listen quic://0.0.0.0:4433 --gpu 1 --tls-self-signed

# Pool of workers on one host, sharing GPUs 0-3 round-robin.
sakura-worker --listen quic://0.0.0.0:4433 --pool-size 4 --gpus 0,1,2,3
```

## 6. v1 service catalog

Service execution order = `(priority asc, install-order)`. Lower priority runs earlier.

### 6.1 `MixedPrecision` (priority 10, in-process)

Wraps forward pass in `torch.autocast`; manages `GradScaler` for fp16; bf16/fp8 paths skip the scaler.

- **Knobs**: `dtype: "fp16" | "bf16" | "fp8" | "auto"`, `loss_scale: float | "dynamic" | None`, `cache_enabled: bool`, `grad_clip: float | None`.
- **Events**: `on_train_step_begin` (enters autocast context), `on_optimizer_step` (unscale, clip, scaler.step, scaler.update), `on_train_begin` (capability check).
- **Edge cases**: must wrap *before* `Compile`; coordinates with `ZeRO1` so the sharded optimizer accepts a `GradScaler`; bad dtype-on-hardware → install-time error with clear hint.

### 6.2 `Compile` (priority 20, in-process)

`torch.compile` lazily on first step, with on-disk cache that survives process restart.

- **Knobs**: `mode: "default" | "reduce-overhead" | "max-autotune"`, `backend: "inductor" | "aot_eager" | "cudagraphs"`, `dynamic: bool | None`, `fullgraph: bool`, `cache_dir: str | None` (default `~/.cache/sakura/compile`), `apply_to: "model" | "training_step" | Callable`.
- **Events**: `on_train_begin` (wraps target with `torch.compile`), `on_train_step_begin` (telemetry: first-step compile time recorded as `compile_secs`).
- **Edge cases**: each DDP rank compiles independently; shared `cache_dir` across ranks amortizes from second run on. Compile failure → emit `on_error` and fall back to eager.

### 6.3 `ZeRO1` (priority 30, in-process; uses NCCL)

Optimizer-state sharding (stage 1 only — gradients/params replicated; optimizer state partitioned across ranks).

- **Knobs**: `process_group: ProcessGroup | None`, `bucket_size_mb: int = 16`, `cpu_offload: bool = False`, `param_groups_override: list[dict] | None`, `optimizer_class: type | None`.
- **Events**: `on_train_begin` (wraps user's optimizer in `sakura.zero.ShardedOptimizer` via cyclic param-group dealing), `on_optimizer_step` (step on local shard, all-gather updated weights), `on_save` (gather full optimizer state for checkpoint), `on_train_end` (restore unsharded optimizer).
- **Edge cases**: uneven rank/parameter ratio handled via cyclic dealing (within-1 imbalance); custom optimizers warned at install + state shape introspected via dummy step; with `MixedPrecision` the sharded optimizer respects `GradScaler.unscale_`.

### 6.4 `ActivationCheckpoint` (priority 15, in-process)

Wraps matching submodules with `torch.utils.checkpoint` to trade recompute for activation memory.

- **Knobs**: `selective: bool | "auto" | int`, `target_types: tuple[type, ...]` (default sensible HF/torchvision module names), `non_reentrant: bool = True`, `preserve_rng_state: bool = True`.
- **Events**: `on_train_begin` (one-shot walk + wrap).
- **Edge cases**: must run *before* `Compile`; non-reentrant required when stacked with `ZeRO1`.

### 6.5 `AsyncEval` (priority 80, dispatching)

Dispatches eval to a worker via the runtime's dispatcher; reaps futures lazily.

- **Knobs**: `model_factory: Callable[[], nn.Module]`, `eval_fn: Callable[[model, payload], dict]`, `eval_payload: Any | val_loader_factory: Callable`, `cache_key: str | None = "default"`, `fp16_state_dict: bool = False`, `max_pending: int = 4`, `on_backpressure: "skip" | "queue" | "block" = "skip"`, `drain: "lazy" | "strict" = "lazy"`, `every: "epoch" | int = "epoch"`.
- **Events**: `on_epoch_end` (snapshot via async-CUDA-stream when on CUDA, submit RPC), `on_train_end` (drain), `on_error` (handle worker crashes / backpressure).
- **Edge cases**: rank-0-only in DDP (checks `event.rank`); ZeRO1 interaction calls `gather_state_dict()` before snapshot; worker crash → `WorkerSupervisor` restart + epoch recorded as skipped; backpressure="skip" → record skip in history.

### 6.6 `AsyncCheckpoint` (priority 85, dispatching)

Snapshot model + (optional) optimizer state, write to disk asynchronously.

- **Knobs**: `dir: str`, `every: "epoch" | "best" | int`, `keep: int | None = 3`, `format: "torch" | "safetensors"`, `metric: str | None`, `mode: "min" | "max"`, `include_optimizer: bool = True`, `compress: "none" | "lz4" | "zstd"`.
- **Events**: `on_epoch_end` (trigger logic), `on_save` (explicit), `on_train_end` (drain).
- **Edge cases**: `every="best"` requires a metric source — `requires=("async_eval",)` at install time; remote worker filesystem reachability handled via `--ship-back` mode (off by default); `ZeRO1` interaction gathers full state before snapshot.

### 6.7 `Telemetry` (priority 0, in-process, async sink)

Captures every event's timestamps + service-emitted records.

- **Knobs**: `output: Callable[[dict], None] | str | None`, `level: "minimal" | "standard" | "verbose"`, `wall_per_service: bool = True`.
- **Events**: all (priority 0 = first; drains last on shutdown).
- **Edge cases**: sink runs in a small Python thread (just dict marshalling); crash-safe flush on shutdown; worker-side telemetry merged into the same sink via a side QUIC stream.

### 6.8 Service interaction matrix

The off-diagonal cells are explicit integration contracts the implementation plan must address.

| → consumer / ↓ producer | MixedPrec | Compile | ZeRO1 | ActCkpt | AsyncEval | AsyncCkpt | Telemetry |
|---|---|---|---|---|---|---|---|
| **MixedPrec** | — | wraps inside autocast | scaler-aware optimizer | non-reentrant required | gathered state for fp16 cast | – | timed |
| **Compile** | runs inside | — | optimizer wrapped after | wraps wrapped modules | – | – | timed |
| **ZeRO1** | scaler-aware | – | — | non-reentrant required | `gather_state_dict()` | `gather_state_dict()` | timed |
| **ActCkpt** | – | first | non-reentrant | — | – | – | timed |
| **AsyncEval** | – | – | gathers | – | — | provides `metric` | timed |
| **AsyncCkpt** | – | – | gathers | – | depends-on for "best" | — | timed |
| **Telemetry** | passthrough | passthrough | passthrough | passthrough | sink | sink | — |

## 7. Rust `sakura-wire` crate

### 7.1 Crate layout

```
crates/sakura-wire/
├── Cargo.toml                   # crate-type = ["cdylib", "rlib"], maturin build
├── src/
│   ├── lib.rs                   # re-exports + #[pymodule]
│   ├── codec/
│   │   ├── mod.rs               # public: Encoder, Decoder
│   │   ├── header.rs            # postcard structs
│   │   ├── tensor.rs            # zero-copy pack/unpack against &[u8] views
│   │   └── cast.rs              # SIMD fp32→fp16/bf16 + reverse
│   ├── protocol/
│   │   ├── mod.rs               # RPC framing + state machine
│   │   ├── handlers.rs          # well-known handler IDs
│   │   └── error.rs             # WireError → Python exceptions
│   ├── transport/
│   │   ├── mod.rs               # trait Transport + URI selectors
│   │   ├── quic.rs              # quinn-based; default
│   │   ├── shm.rs               # v1.x: memfd + Unix socket
│   │   └── rdma.rs              # v2 stub: feature = "rdma"
│   ├── runtime.rs               # tokio multi-thread runtime, owned by Rust
│   ├── supervisor.rs            # spawn/monitor worker subprocesses
│   └── pyo3_bindings.rs         # Dispatcher, Future, WorkerSupervisor, Compute
├── benches/                     # criterion benches (codec, RTT, MB/s)
└── tests/                       # cargo test + interop tests via maturin develop in CI
```

### 7.2 Wire format

Two tiers: **headers** in postcard (compact, schema-versioned), **tensor payloads** as raw zero-copy bytes.

Postcard structs:

```rust
pub enum WireVersion { V1 = 1 }

pub struct RpcRequestHeader {
    pub version: WireVersion,
    pub request_id: u64,
    pub handler_id: u32,
    pub n_tensors: u32,
    pub aux_payload_bytes: u32,
    pub deadline_ms: Option<u32>,
    pub trace_id: u128,
}

pub struct TensorDesc {
    pub shape: SmallVec<[u32; 8]>,
    pub dtype: Dtype,                       // F32 / F16 / BF16 / F8E4M3 / I64 / I32 / U8 / BOOL / ...
    pub n_bytes: u64,
    pub device_hint: Device,                // Cpu | Cuda(u8) | CudaIpc(handle)
    pub fp16_cast_on_wire: bool,
}

pub struct RpcResponseHeader {
    pub version: WireVersion,
    pub request_id: u64,
    pub status: RpcStatus,                  // Ok | Error(WireError) | Cancelled
    pub n_result_tensors: u32,
    pub aux_payload_bytes: u32,
    pub elapsed_us: u64,
}
```

**One RPC = one bidirectional QUIC stream** with frames in order:

```
client → server:
   [RpcRequestHeader (postcard)]
   [TensorDesc × n_tensors (postcard, length-prefixed)]
   [tensor bytes × n_tensors (raw, sized by TensorDesc.n_bytes)]
   [aux_payload (cloudpickle, sized by header.aux_payload_bytes)]
   <stream FIN>

server → client:
   [RpcResponseHeader]
   [TensorDesc × n_result_tensors]
   [tensor bytes × n_result_tensors]
   [aux_payload (cloudpickle metrics)]
   <stream FIN>
```

**Zero-copy in/out:**
- Producer: PyO3 grabs `&[u8]` view of `tensor.data_ptr()` via buffer protocol (no Python copy); `quinn::SendStream::write_all_chunks` writes directly.
- Consumer: pre-allocates `Vec<u8>` of exact size from descriptor; `quinn::RecvStream::read_exact` fills it; PyO3 hands back `np.ndarray` zero-copy via `PyArray::from_owned_slice`; Python wraps via `torch.from_numpy(...)` (also zero-copy).
- Same-host fast path (v1.x): `device_hint = CudaIpc(handle)` → receiver imports the IPC handle, no copy.

**Optional fp16/bf16 cast on the wire:** SIMD pass via `wide` crate (AVX2 / NEON). Producer flag halves bytes; receiver casts back at load. IEEE 754 round-to-nearest-even — bit-identical to a Python-side `.to(torch.float16)`.

### 7.3 Protocol — handler IDs

```rust
pub const HANDLER_EXEC_CLOUDPICKLED: u32 = 0x0001;
pub const HANDLER_MODEL_CACHE_GET:    u32 = 0x0002;
pub const HANDLER_HEARTBEAT:          u32 = 0x0003;
pub const HANDLER_SHUTDOWN:           u32 = 0x0004;
pub const HANDLER_SAVE_BLOB:          u32 = 0x0005;
pub const HANDLER_CUSTOM_BASE:        u32 = 0x1000;
```

Almost every real RPC is `HANDLER_EXEC_CLOUDPICKLED`. Dispatching services cloudpickle `(eval_fn, model_factory, payload)` once at install time and ship a content-hash reference per call; only the *tensor state_dict* travels per-call (zero-copy). Per-epoch wire cost is dominated by tensor bytes, not Python serialization.

### 7.4 Error propagation

```rust
pub enum WireError {
    HandlerNotFound { handler_id: u32 },
    DecodeFailed { what: String, detail: String },
    HandlerPanic { msg: String, trace: Vec<String> },
    Timeout { deadline_ms: u32 },
    WorkerCrashed,
    BackpressureSaturated,
}
```

PyO3 maps each variant to a Python exception subclass (`SakuraWireError`, `HandlerPanic`, `Timeout`, `WorkerCrashed`, `BackpressureSaturated`, …). `Future.result()` raises with worker-side traceback attached.

### 7.5 Transport selection by URI scheme

| URI | Transport | When |
|---|---|---|
| `quic://host:port` | `quic::QuicTransport` (quinn) | default; loopback or LAN/WAN |
| `quic+mtls://host:port` | same with mutual TLS | cross-host, verified certs |
| `unix-shm://path` | `shm::ShmTransport` | v1.x; same-host zero-copy via memfd |
| `rdma://host:port` | `rdma::RdmaTransport` | v2; feature-gated |
| `zakuro://...` | (handled in Python via `ZakuroDispatcher`) | wraps `@zk.fn` |

QUIC: `quinn` ≥ 0.10, `rustls` for TLS 1.3. `LocalDispatcher` uses self-signed ephemeral cert; cross-host requires CA-signed or mutual TLS. Tokio runtime owned by Rust in its own OS thread; PyO3 `submit()` returns a `Future` whose `.result()` blocks the calling thread on a Rust-owned oneshot channel via `py.allow_threads(...)`.

### 7.6 PyO3 surface

```python
# from sakura_wire import ...
class Dispatcher:
    def __init__(self, target: str, *, tls: TlsConfig | None = None) -> None: ...
    def submit(self, handler_id: int, tensors: list[TensorView], aux_payload: bytes,
               *, timeout_ms: int | None = None) -> Future: ...
    def stats(self) -> dict: ...
    def shutdown(self, timeout_s: float = 30.0) -> None: ...

class Future:
    def result(self, timeout: float | None = None) -> Result: ...
    def cancel(self) -> bool: ...
    def done(self) -> bool: ...

class Result:
    tensors: list[np.ndarray]
    aux: bytes
    elapsed_us: int

class WorkerSupervisor:
    def __init__(self, *, n_workers: int = 1, gpus: list[int] | None = None,
                 listen_addr: str = "quic://127.0.0.1:0",
                 worker_cmd: list[str] | None = None,
                 env: dict[str, str] | None = None) -> None: ...
    def start(self) -> list[str]: ...
    def shutdown(self, timeout_s: float = 30.0) -> None: ...
    def restart_failed(self) -> int: ...
    def stats(self) -> dict: ...

class TlsConfig:
    cert: bytes; key: bytes; ca: bytes | None; verify_peer: bool
```

### 7.7 Performance budget (CI-asserted)

Verified by `cargo bench` (criterion) on every Rust release, against `x399 4090` reference (loopback):

| Metric | Target |
|---|---|
| Codec encode (268 MB state_dict, fp32) | < 30 ms |
| Codec decode | < 30 ms |
| QUIC loopback RTT (heartbeat) | < 0.5 ms |
| Per-epoch eval RPC (268 MB → tiny metrics) | < 50 ms wire-only |
| Steady-state throughput on loopback | > 8 GB/s |
| fp16-cast-on-wire vs no cast | ~0.5× bytes, < 5 ms cast cost |

CI fails before the wheel ships if a regression > 5% on any target.

## 8. Dispatcher implementations (Python)

```python
class LocalDispatcher(Dispatcher):
    """Spawns a localhost sakura-worker subprocess; talks QUIC over loopback.
    Auto-spawned by SakuraRuntime when no compute= is set."""

class RemoteDispatcher(Dispatcher):
    """Connects to a sakura-worker daemon at quic://host:port.
    Default for multi-node and dedicated-eval-box deployments."""

class ZakuroDispatcher(Dispatcher):
    """Wraps zakuro's @zk.fn + zk.Compute. Wire format is Zakuro's, not sakura-wire's
    (loses codec wins, keeps the integration). Use when Zakuro infra needed."""

class InThreadDispatcher(Dispatcher):
    """Debug/test only: runs handlers synchronously in the calling thread."""
```

`Compute` resolves to a Dispatcher at `runtime.start()`:

```python
Compute.local(n_workers=2, gpus=[0,1])  → LocalDispatcher(...)
Compute.at("quic://eval-1:4433")        → RemoteDispatcher(target="quic://eval-1:4433")
Compute.pool(["quic://e1","quic://e2"]) → RemoteDispatcher(target=[...], strategy="least-loaded")
Compute.zakuro(zk_compute)              → ZakuroDispatcher(zk_compute)
Compute.in_thread()                     → InThreadDispatcher()
```

## 9. `WorkerSupervisor` lifecycle

1. **Spawn** — `subprocess.Popen([sys.executable, "-m", "sakura.worker", "--listen", "quic://127.0.0.1:0", ...])`. Reads worker's stdout to pick up the actual port (since `:0` = ephemeral). Pins GPUs via `CUDA_VISIBLE_DEVICES`.
2. **Health** — heartbeat RPC every N seconds; on failure, kill (SIGTERM → SIGKILL) and respawn. Outstanding `Future`s on the dead worker raise `WorkerCrashed`; the dispatcher resubmits queued requests to a healthy peer.
3. **Shutdown** — at `runtime.shutdown()`: send `HANDLER_SHUTDOWN`, wait up to `timeout_s`, then SIGTERM → SIGKILL. Drain in-flight `Future`s with `Cancelled`.
4. **Pool scaling** — `supervisor.scale(n)` adds/removes workers between epochs (rare).

## 10. Framework adapters

### 10.1 `LightningAdapter` (`lightning.Callback`)

| Lightning hook | Runtime event |
|---|---|
| `on_train_start` | `on_train_begin(model, optimizer=trainer.optimizers[0], train_loader=trainer.train_dataloader, val_loader=trainer.val_dataloaders)` (fires after `setup` so `trainer.optimizers` is populated) |
| `on_train_epoch_start` | `on_epoch_begin(epoch=trainer.current_epoch)` |
| `on_train_batch_start` | `on_train_step_begin(model, batch, step=batch_idx)` |
| `on_before_optimizer_step` | `on_optimizer_step(optimizer)` |
| `on_train_epoch_end` | `on_epoch_end(epoch, model, optimizer, metrics=dict(trainer.callback_metrics))` |
| `on_train_end` | `on_train_end(model, history)` |
| `on_exception` | `on_error(exc, context={"hook": "lightning"})` |

**Tricky bit:** `ZeRO1` mutates the optimizer in place at `on_train_begin` (replaces `step` / `zero_grad` / `state_dict` / `load_state_dict`); Lightning doesn't re-instantiate, so the wrap survives. Users do not subclass `LightningModule`.

### 10.2 `HFAdapter` (`transformers.TrainerCallback`)

| HF hook | Runtime event |
|---|---|
| `on_train_begin(args, state, control, **kw)` | `on_train_begin(model=kw["model"], optimizer=kw["optimizer"], train_loader=kw["train_dataloader"])` |
| `on_step_begin` | `on_train_step_begin(model, batch=kw["inputs"], step=state.global_step)` |
| `on_pre_optimizer_step` (≥ 4.38) | `on_optimizer_step(optimizer)` |
| `on_epoch_end` | `on_epoch_end(epoch=int(state.epoch), model, optimizer, metrics=dict(state.log_history[-1]) if state.log_history else {})` |
| `on_train_end` | `on_train_end(model, history)` |

For `transformers < 4.38` we hook `on_step_end` instead, with one-step optimizer-step lag — documented as a minor limitation; adapter sets `min_transformers_version = "4.38"` and warns.

### 10.3 `DDPAdapter` (no callback model — explicit hooks)

```python
class DDPAdapter:
    def __init__(self, runtime: SakuraRuntime, *, rank: int, world_size: int): ...
    def on_train_begin(self, model, optimizer, train_loader, val_loader=None): ...
    def on_epoch_begin(self, epoch: int): ...
    def on_train_step_begin(self, model, batch, step: int): ...
    def on_optimizer_step(self, optimizer): ...
    def on_epoch_end(self, epoch: int, model, optimizer, metrics: dict): ...
    def on_train_end(self, model): ...
```

Embeds `rank` / `world_size` in every event payload.

## 11. Benchmark harness (`sakura.bench`)

### 11.1 Artifacts

```python
@dataclass
class Workload:
    name: str                                   # "cifar10-resnet50"
    tier: Literal["ci", "perf"]
    make_model: Callable[[], nn.Module]
    make_train_loader: Callable[[], DataLoader]
    make_val_loader: Callable[[], DataLoader]
    eval_fn: Callable[[nn.Module, Any], dict]
    metric_target: tuple[str, float] | None     # convergence sanity gate
    epochs: int

class BaselineRunner:                            # vanilla framework, no Sakura
    def __init__(self, framework: Literal["pytorch-ddp", "lightning", "hf-trainer", "tf"]): ...
    def run(self, workload: Workload) -> RunReport: ...

class SakuraRunner:
    def __init__(self, framework: Literal["pytorch-ddp", "lightning", "hf-trainer"],
                 services: list[Service], compute: Compute = Compute.local()): ...
    def run(self, workload: Workload) -> RunReport: ...

@dataclass
class RunReport:
    workload: str
    framework: str
    sakura_services: list[str] | None
    elapsed_secs: float
    samples_per_sec: float
    peak_gpu_mem_mb: float
    final_metrics: dict
    per_stage_secs: dict[str, float]            # from TelemetryService
    git_sha: str
    hardware: dict
```

### 11.2 Two-tier suite

**Tier 1 — CI lightweight** (always runs, nightly + on perf-relevant PRs):
- `mnist-mlp` — smoke, < 2 min
- `cifar10-resnet50` — ~5–10 min
- `distilbert-sst2` — ~5 min

**Tier 2 — Perf / LLM** (when GPU resources allow; nightly on dedicated runner):
- `llama3-1b-instruct-finetune` — small instruction set
- `mistral-7b-lora-finetune`
- `distilbert-glue` (full GLUE)

Each workload exercises baselines: `pytorch-ddp`, `lightning`, `hf-trainer`, plus `tf` *where a canonical TF impl exists* (CIFAR-10/ResNet via `keras`; DistilBERT via `keras-nlp`). Sakura runs are configured via service combos.

### 11.3 CLI

```bash
sakura-bench run --tier ci --output reports/
sakura-bench run --workload cifar10-resnet50 --runner sakura
sakura-bench compare reports/baseline.json reports/sakura.json
sakura-bench export reports/ --format markdown > BENCHMARKS.md
```

### 11.4 Comparison report format

```
Workload: cifar10-resnet50  (RTX 4090, batch 256, 5 epochs)
                                wall      sps      peak_mem   final_acc
  pytorch-ddp (baseline)        47.2s     2716     1830 MB    91.4%
  lightning   (baseline)        49.1s     2611     1862 MB    91.5%
  tf          (baseline)        58.4s     2196     1740 MB    91.0%
  sakura+pytorch-ddp [bf16,compile,asyncEval,asyncCkpt]
                                33.8s     3793     1620 MB    91.4%   (1.40× vs pytorch-ddp)
  sakura+lightning  [...]       34.5s     3716     1640 MB    91.5%   (1.42× vs lightning)
```

The same JSON behind it is consumed by `sakura-bench compare` for CI perf gates.

## 12. Repo layout

```
sakura/
├── Cargo.toml                          # workspace
├── pyproject.toml                      # maturin build
├── rust-toolchain.toml                 # pinned rustc 1.78+
├── crates/
│   └── sakura-wire/                    # see §7.1
├── python/sakura/
│   ├── __init__.py                     # SakuraRuntime, Compute, Service, …
│   ├── runtime.py                      # SakuraRuntime
│   ├── service.py                      # Service ABC + registry
│   ├── events.py                       # Event types
│   ├── dispatch/{local,remote,zakuro,in_thread}.py
│   ├── services/{mixed_precision,compile,zero1,activation_checkpoint,
│   │              async_eval,async_checkpoint,telemetry}.py
│   ├── lightning/adapter.py
│   ├── huggingface/adapter.py
│   ├── ddp/adapter.py
│   ├── worker/{__init__,__main__}.py   # sakura-worker daemon
│   ├── bench/
│   │   ├── harness.py                  # Workload, BaselineRunner, SakuraRunner
│   │   ├── workloads/{mnist,cifar,distilbert,llama,mistral}.py
│   │   └── __main__.py                 # sakura-bench CLI
│   └── _internal/cuda_snapshot.py      # async-CUDA-stream trick (factored out)
├── tests/
│   ├── unit/                           # services, runtime, dispatcher (in-thread), adapters mocked
│   ├── integration/                    # services × LocalDispatcher (real subprocess), small models
│   ├── e2e/                            # Lightning + HF + DDP, CPU-only, 2 epochs
│   └── perf/                           # GPU-only, gates on perf budget
├── docs/
│   ├── superpowers/specs/              # this spec
│   ├── migration-from-0.1.md
│   └── (user docs)
├── .github/workflows/{test,publish,perf,bench}.yml
├── compose.yaml
├── docker/Dockerfile                   # CUDA + maturin + Rust toolchain
├── README.md
└── Taskfile.yml + taskfiles/
```

## 13. Packaging

- **Build**: `maturin develop` (dev), `maturin build --release` (wheel).
- **Wheel matrix**: Python 3.10–3.13 × Linux x86_64/aarch64 + macOS arm64 + Windows x86_64. Compiled Rust extension bundled in each wheel.
- **PyPI package**: `sakura-ml` (continuity).
- **Optional extras**:
  - `[lightning]` → `lightning>=2.0`
  - `[huggingface]` → `transformers>=4.38, datasets, accelerate`
  - `[zakuro]` → `zakuro-ai>=0.2.3` (enables `ZakuroDispatcher`)
  - `[bench]` → `[lightning, huggingface]` + `tensorflow>=2.15` (TF baselines), `safetensors`
- **Console scripts**: `sakura`, `sakura-worker`, `sakura-bench`.
- **Rust toolchain** pinned via `rust-toolchain.toml` for reproducible wheel builds.

## 14. Testing strategy

| Tier | Hardware | Runs | Speed | Gate |
|---|---|---|---|---|
| **Unit** | CPU only | every PR | < 60 s | services, runtime, adapters with `InThreadDispatcher` and mock model/optimizer |
| **Integration** | CPU + subprocess | every PR | < 5 min | `LocalDispatcher` round-trip with a real `sakura-worker` subprocess, tiny models |
| **E2E** | CPU only | every PR | < 15 min | Lightning + HF + DDP × small models × 2 epochs, verify `rt.history()` correctness |
| **Perf** | one A100 / 4090 runner | nightly + on-tag | < 30 min | tier-1 bench harness with perf budget assertions; gate release if regression > 5% |

**Rust side**: `cargo test` (codec / QUIC / error tests); `cargo bench` (criterion) gates on §7.7 budget. Wired into `.github/workflows/test.yml`.

**GIL-vs-subprocess regression test**: `tests/integration/test_no_gil_blocking.py` measures main-thread CPU share during `AsyncEvalService` dispatches. If main-thread % drops below 90% of pure-training baseline we have a subprocess regression.

## 15. Migration from v0.1.x

| v0.1.x | v1.0 | Status |
|---|---|---|
| `from sakura.lightning import SakuraTrainer` | `SakuraRuntime() + LightningAdapter + services.AsyncEval` | replaced |
| `from sakura.lightning import SakuraLightningCallback` | same | replaced |
| `from sakura.huggingface import SakuraHFCallback` | `SakuraRuntime() + HFAdapter + services.AsyncEval` (knobs `cache_key`, `fp16_state_dict`, `on_backpressure` carry over) | replaced |
| `from sakura.ddp import DDPAsyncEvalCallback` | `SakuraRuntime() + DDPAdapter + services.AsyncEval` | replaced |
| `from sakura.tensorflow import SakuraKerasCallback` | **removed** (TF stays as benchmark *baseline*) | removed |
| `from sakura.ml.async_trainer import AsyncTrainer` | **removed** (use `DDPAdapter` for raw loops) | removed |
| `from sakura.ml.sakura_trainer import SakuraTrainer` | **removed** (dead MPI+Redis era) | removed |
| `from sakura.functional import asr_metrics, defaultMetrics` | **removed** (unused) | removed |
| `zk.Compute(uri="quic://…")` | `Compute.at("quic://…")` (Sakura-owned URI parser; `ZakuroDispatcher` keeps `zk.Compute`) | replaced |

A `docs/migration-from-0.1.md` with side-by-side examples for each integration. v0.1.x stays available on PyPI under `sakura-ml<1.0` for users who haven't migrated.

## 16. Error handling, telemetry, security

### 16.1 Error contract

| Failure | Where it surfaces | What runs after |
|---|---|---|
| Service install fails (e.g. bf16 on pre-Ampere) | `runtime.install(...)` raises | nothing installed; caller must fix or skip |
| Service `on_event` raises | runtime catches, emits `on_error`, continues | other services keep running |
| `WorkerCrashed` mid-RPC | `Future.result()` raises; `AsyncEvalService` catches → `{epoch, skipped, reason}` | `WorkerSupervisor` restarts; next epoch dispatches normally |
| `Timeout` on `Future.result(timeout=...)` | raises | per-service: `AsyncEval` drops the epoch, `AsyncCheckpoint` retries once then drops |
| `BackpressureSaturated` | dispatcher rejects submit | service honors `on_backpressure` policy |
| Adapter raises in framework hook | propagates to framework's error handler | runtime emits `on_error` for telemetry record |

### 16.2 Telemetry record schema

Every record is one JSON-line:

```json
{"ts":1730000000.123,"event":"on_epoch_end","service":"async_eval","epoch":3,
 "elapsed_us":4523,"trace_id":"…","payload":{"val_loss":0.241,"worker":"localhost:48211"}}
```

`Telemetry` is the single source of truth — the bench harness consumes this stream for per-stage time attribution.

### 16.3 Security

- Loopback worker uses self-signed TLS pinned to the supervisor's cert.
- Cross-host `RemoteDispatcher` requires `tls=` config explicitly (no anonymous QUIC); `--tls-self-signed` flag on `sakura-worker` is permitted for dev but emits a banner.
- Cloudpickle bytes from the network are NEVER deserialized blindly — workers reject `HANDLER_EXEC_CLOUDPICKLED` from unauthenticated clients.

## 17. Open questions / explicitly out of scope

Out of scope for v1, designed-in for v2 or v1.x:

- **`unix-shm://` transport** for same-host zero-copy — module placeholder in `sakura-wire`, wired up in v1.x.
- **CUDA IPC handle path** for colocated CUDA workers — descriptor field already in wire format; consumer logic v1.x.
- **RDMA backend** (`rdma://`) — Rust feature-gated stub; v2 lights it up.
- **ZeRO-2 / ZeRO-3** — out of scope; v1 stops at stage 1.
- **Tensor / pipeline parallelism** — not on the roadmap.
- **Gradient compression** (PowerSGD, top-k, fp8 reduce-scatter) — v2 work, single-node returns are modest.
- **JAX adapter** — v2 work; functional model needs a different abstraction.
- **TF integration revival** — not planned; TF stays as benchmark baseline only.
- **Preset bundles** (`SakuraRuntime.with_preset("max-throughput")`) — v1.x ergonomic layer once the underlying services have stabilized.

## 18. Glossary

| Term | Meaning |
|---|---|
| **Adapter** | Per-framework bridge that translates framework hooks to runtime events. |
| **Dispatcher** | Submits work to a worker process; abstracts transport. |
| **Event** | Typed message on the runtime's event bus (e.g., `on_epoch_end`). |
| **Handler** | A callable registered on a worker, identified by `handler_id`. Almost all real handlers are `HANDLER_EXEC_CLOUDPICKLED`. |
| **In-process service** | A service that runs entirely in the training process (e.g., `MixedPrecision`). |
| **Dispatching service** | A service that submits RPCs to a worker process (e.g., `AsyncEval`). |
| **Runtime** | `SakuraRuntime` — the central orchestrator owning event bus, services, dispatcher, supervisor. |
| **`sakura-wire`** | The Rust crate (codec + protocol + transport) that backs the dispatcher. |
| **`sakura-worker`** | Python entry point that runs the QUIC server side; receives RPCs and runs handlers. |
| **Service** | An installable unit of behavior subscribing to runtime events. |
| **Tier (benchmark)** | Lightweight (CI) vs perf/LLM (resource-allowing). |
| **WireError** | Rust-side error variant; PyO3 maps to Python exception subclass. |
| **WorkerSupervisor** | Manages the lifecycle of `sakura-worker` subprocesses (spawn, health, shutdown, restart, scale). |
