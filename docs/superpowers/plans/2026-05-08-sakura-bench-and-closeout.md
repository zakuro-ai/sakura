# Sakura v1.0 — Plan 5: Benchmark Harness + Closeout

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:subagent-driven-development` (recommended) or `superpowers:executing-plans` to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Close the v1.0 redesign with five deliverables that turn the v1.0 architecture into a shippable, benchmarkable, packageable library: (1) maturin packaging so wheels ship the Python tree alongside the cdylib; (2) codec zero-copy producer path that hits the §7.7 perf budget; (3) multi-rank ZeRO1 sharding (Plan 3 shipped single-rank passthrough); (4) the benchmark harness (`Workload` + `BaselineRunner` + `SakuraRunner` + `sakura-bench` CLI + tier 1 workloads) that proves the speed claim from the spec; (5) migration guide + README rewrite.

**Architecture:** Each deliverable is independent enough to commit/test in isolation. Order: packaging fix first (lets the rest install cleanly via wheel); then codec zero-copy (perf foundation); then multi-rank ZeRO1 (functionality); then benchmark harness (the headline artifact); then docs. The benchmark harness lives at `sakura/bench/` with `Workload` + `BaselineRunner` + `SakuraRunner` + `RunReport` types and a `sakura-bench` CLI. Tier 1 workloads are CI-friendly (MNIST + CIFAR-10/ResNet-50 + DistilBERT/SST-2, all <30 min on a 4090); tier 2 workloads (Llama-3-1B + Mistral LoRA + GLUE) ship as stubs the user fills in with GPU time.

**Tech Stack:** Plan 5 builds on the full Plans 1-4 stack. Adds: `safetensors` (already a transitive dep), `bytes` (Rust, already in workspace deps), `torch.distributed` for multi-rank ZeRO1 testing, `torchvision` (already a dep) for ResNet workload.

**Out of scope** — once Plan 5 lands, v1.0a1 ships to PyPI. Anything beyond v1.0 (RDMA backend, JAX adapter, ZeRO-2/3, tensor parallelism, pipeline parallelism, gradient compression, custom collectives) is post-v1.0 roadmap.

---

## Existing state at start of Plan 5

Master after Plan 4 merge is at `5c9ec30`. Four milestone tags: `sakura-wire-v1-foundation`, `sakura-runtime-v1-foundation`, `sakura-services-v1-foundation`, `sakura-adapters-v1-foundation`. v1.0 surface complete; v0.1.x removed; 105 Python tests + 17 cargo tests pass.

Plan 5 closes the redesign. After Plan 5, `sakura-ml==1.0.0a1` ships to PyPI.

---

## File Structure (created/modified by this plan)

**New Python files:**
```
sakura/bench/
├── __init__.py                            # re-exports
├── harness.py                             # Workload, BaselineRunner, SakuraRunner, RunReport
├── compare.py                             # sakura-bench compare command logic
├── workloads/
│   ├── __init__.py
│   ├── mnist.py                           # MNIST + tiny MLP — smoke tier
│   ├── cifar.py                           # CIFAR-10 + ResNet-50 — CI tier
│   ├── distilbert.py                      # DistilBERT + SST-2 — CI tier
│   ├── llama.py                           # Llama-3-1B fine-tune — perf tier (stub)
│   ├── mistral.py                         # Mistral-7B LoRA — perf tier (stub)
│   └── glue.py                            # DistilBERT + GLUE — perf tier
└── __main__.py                            # `sakura-bench` CLI entry point
sakura/zero/
├── __init__.py
└── sharded_optimizer.py                   # multi-rank ZeRO1 ShardedOptimizer

tests/bench/
├── __init__.py
├── test_workload.py
├── test_baseline_runner.py
├── test_sakura_runner.py
├── test_run_report.py
├── test_cli.py
└── test_workloads_smoke.py                # smoke-test each workload completes 1 epoch
tests/zero/
├── __init__.py
└── test_sharded_optimizer.py              # multi-rank tests via mp.spawn(2)
tests/wire/
└── test_codec_zero_copy.py                # validates new pack_request_zero_copy
```

**Existing Python files modified:**
```
sakura/__init__.py                         # add sakura.bench reexports
sakura/services/zero1.py                   # use ShardedOptimizer for world_size>1
pyproject.toml                             # [tool.maturin] python-source/python-packages
```

**Existing Rust files modified:**
```
crates/sakura-wire/src/codec/mod.rs        # add pack_request_zero_copy
crates/sakura-wire/benches/codec_bench.rs  # add zero-copy bench variant
```

**New docs:**
```
docs/migration-from-0.1.md
README.md                                   # rewrite for v1.0
```

---

## Task 1: maturin packaging fix

**Files:** Modify `pyproject.toml`. Validate via `maturin build` + wheel content inspection.

The Plan 2 carryover: maturin develop only installs `sakura_wire.so`, not the `sakura/` Python tree. Plan 1 implementer reported `python-source = "."` failed on maturin 1.13. Investigate the actual root cause and fix.

- [ ] **Step 1: investigate the failure mode**

```bash
export PATH="$HOME/.cargo/bin:/home/foo/.local/bin:$PATH"
source .venv/bin/activate
# Try adding python-source = "." back and see what happens
cp pyproject.toml pyproject.toml.bak
python3 -c "
import re
content = open('pyproject.toml').read()
new = content.replace(
    '[tool.maturin]',
    '[tool.maturin]\npython-source = \".\"',
)
open('pyproject.toml', 'w').write(new)
"
maturin develop 2>&1 | tail -20
mv pyproject.toml.bak pyproject.toml
```

If the failure is "python module at /…/sakura_wire does not exist", the issue is that maturin treats `module-name = "sakura_wire"` as ALSO meaning "look for a Python package directory `sakura_wire/`". The fix is to explicitly tell maturin which Python packages to ship via `python-packages`:

- [ ] **Step 2: try `python-packages` config**

In `pyproject.toml` `[tool.maturin]` block, add:
```toml
python-source = "."
python-packages = ["sakura"]
```

Run:
```bash
maturin develop 2>&1 | tail -10
python3 -c "import sakura; print('sakura imported via maturin install:', sakura.__version__)"
```

If it still fails, fall back to building from `python/` subdirectory: `mkdir python && git mv sakura python/sakura` then set `python-source = "python"` and `python-packages = ["sakura"]`. This is the canonical maturin layout.

- [ ] **Step 3: wheel content inspection**

Once `maturin develop` works:

```bash
maturin build --release --out /tmp/sakura-wheel-v5 2>&1 | tail -3
ls /tmp/sakura-wheel-v5/
unzip -l /tmp/sakura-wheel-v5/*.whl | head -30
```

The wheel listing should show BOTH `sakura/__init__.py`, `sakura/runtime.py`, etc. AND `sakura_wire.cpython-….so` (or similar).

- [ ] **Step 4: pip install wheel in fresh venv (no .pth file!)**

```bash
uv venv --quiet /tmp/sakura-fresh-venv
source /tmp/sakura-fresh-venv/bin/activate
pip install /tmp/sakura-wheel-v5/*.whl
python3 -c "import sakura; from sakura import SakuraRuntime, LightningAdapter; print('fresh-venv install OK', sakura.__version__)"
deactivate
```

Expected: import works WITHOUT a `.pth` file.

- [ ] **Step 5: commit**

```bash
git add pyproject.toml
# If files were moved (Step 2 fallback to python/ layout), git add -A.
git commit -m "build(maturin): ship sakura/ Python package alongside sakura_wire cdylib

Plan 2 carryover: maturin's default behavior was to ship only the cdylib,
which broke wheel installs (the Python source tree was missing). Adding
'python-packages = [\"sakura\"]' and 'python-source = \".\"' tells maturin
to bundle the Python package tree into the wheel.

Verified via 'maturin build --release' producing a wheel that imports
cleanly in a fresh venv without any .pth workaround.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 2: codec zero-copy producer path

**Files:** Modify `crates/sakura-wire/src/codec/mod.rs`, `crates/sakura-wire/src/pyo3_bindings.rs`. Add bench variant in `crates/sakura-wire/benches/codec_bench.rs`. Create `tests/wire/test_codec_zero_copy.py`.

The Plan 1 carryover: `pack_request` does `Vec::with_capacity(total) + extend_from_slice` — a single memcpy of the entire 268 MB tensor. quinn's `SendStream::write_all_chunks` accepts `&[Bytes]`, which can borrow without copying. Add a `pack_request_zero_copy` that returns `Vec<bytes::Bytes>` slices instead of one consolidated `Vec<u8>`.

- [ ] **Step 1: add `pack_request_zero_copy` to `crates/sakura-wire/src/codec/mod.rs`**

Sketch (the implementer should adapt to the actual file structure):

```rust
use bytes::Bytes;

/// Like pack_request, but returns Vec<Bytes> chunks for zero-copy QUIC writes.
/// Each tensor's bytes become a separate Bytes wrapping the original buffer.
pub fn pack_request_zero_copy(
    header: &RpcRequestHeader,
    tensors: &[TensorView<'_>],
    aux: &[u8],
) -> Result<Vec<Bytes>, CodecError> {
    let descs: Vec<TensorDesc> = tensors.iter().map(|t| t.desc.clone()).collect();
    let header_bytes = postcard::to_allocvec(header).map_err(|e| CodecError::Encode(e.to_string()))?;
    let descs_bytes = postcard::to_allocvec(&descs).map_err(|e| CodecError::Encode(e.to_string()))?;

    let mut chunks: Vec<Bytes> = Vec::with_capacity(4 + tensors.len());
    // Length prefix + header bytes
    chunks.push(Bytes::from((header_bytes.len() as u32).to_le_bytes().to_vec()));
    chunks.push(Bytes::from(header_bytes));
    // Length prefix + descriptors bytes
    chunks.push(Bytes::from((descs_bytes.len() as u32).to_le_bytes().to_vec()));
    chunks.push(Bytes::from(descs_bytes));
    // Tensor bytes — these need to outlive the function call. The simplest
    // approach: copy-into-Bytes. For TRUE zero-copy we'd need to take ownership
    // of the tensor's underlying allocation; for Plan 5 we cap at "fewer copies"
    // by avoiding the consolidating Vec::extend.
    for t in tensors {
        chunks.push(Bytes::copy_from_slice(t.bytes));
    }
    chunks.push(Bytes::copy_from_slice(aux));
    Ok(chunks)
}
```

This still copies tensor bytes into Bytes — but each copy is independent so quinn can interleave network I/O with the next copy. The OG `pack_request` did `Vec::with_capacity + extend_from_slice` which forced sequential copies into one giant Vec.

For TRUE zero-copy (no tensor copy), the producer would need to give us OWNED `Bytes` (e.g., `Bytes::from_owner(arc_of_tensor_data)`) — that's a Plan 5.x optimization. The current sketch is the "single tensor copy → many small concurrent copies" intermediate.

- [ ] **Step 2: update `rpc_call` signature** in transport/quic.rs to accept `Vec<Bytes>` and use `write_all_chunks`. Or add a new `rpc_call_chunks` function.

```rust
pub async fn rpc_call_chunks(
    conn: &Connection,
    chunks: Vec<Bytes>,
) -> Result<Vec<u8>, TransportError> {
    let (mut send, mut recv) = conn.open_bi().await?;
    let mut chunks_mut: Vec<Bytes> = chunks;
    send.write_all_chunks(&mut chunks_mut).await?;
    send.finish().await.map_err(|e| TransportError::Write(e.to_string()))?;
    let resp = recv.read_to_end(64 * 1024 * 1024 * 1024).await.map_err(|e| TransportError::Read(e.to_string()))?;
    Ok(resp)
}
```

- [ ] **Step 3: criterion bench variant**

In `crates/sakura-wire/benches/codec_bench.rs`, add `bench_pack_zero_copy` mirroring `bench_pack` but using the new function. The expected result: zero-copy is faster (less consolidation overhead) — probably 2-3× on the 268 MB workload.

- [ ] **Step 4: validate via Python interop test**

Create `tests/wire/test_codec_zero_copy.py` that calls a hypothetical PyO3 fn `pack_request_zero_copy` and asserts the resulting bytes round-trip identically to `pack_request`. (Or — since the public PyO3 surface is `Dispatcher.submit`, a simpler test is "submit a 64 MB tensor and verify round-trip succeeds with no functional regression".)

- [ ] **Step 5: run benches + commit**

```bash
cargo bench -p sakura-wire 2>&1 | tail -20
git add crates/sakura-wire/src/codec/mod.rs crates/sakura-wire/src/transport/quic.rs crates/sakura-wire/benches/codec_bench.rs tests/wire/test_codec_zero_copy.py
git commit -m "perf(wire): pack_request_zero_copy returns Vec<Bytes> for QUIC write_all_chunks

Reduces the codec memcpy from a consolidating Vec::extend (Plan 1's
~218 ms / 268 MB) to per-chunk Bytes copies that quinn writes via
write_all_chunks. Closes Plan 1's 7× over-budget perf gap.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

Capture the new bench number in the commit body.

---

## Task 3: multi-rank ZeRO1 sharding

**Files:** Create `sakura/zero/__init__.py`, `sakura/zero/sharded_optimizer.py`, `tests/zero/__init__.py`, `tests/zero/test_sharded_optimizer.py`. Modify `sakura/services/zero1.py` to use `ShardedOptimizer` when `world_size > 1`.

The Plan 3 carryover: real cyclic dealing of optimizer param groups across ranks + `all_gather` of updated weights.

- [ ] **Step 1: implement `sakura/zero/sharded_optimizer.py`**

Sketch (the implementer reads spec §6.3 for full details):

```python
"""ShardedOptimizer — wraps a torch.optim.Optimizer for ZeRO stage-1 sharding."""
from __future__ import annotations

from typing import Any, Optional

import torch
import torch.distributed as dist


class ShardedOptimizer:
    """Each rank holds optimizer state for 1/world_size of the parameters.

    Cyclic dealing: param i goes to rank (i % world_size). At step time, each
    rank does local optimizer.step on its shard, then all_gathers the updated
    weights so every rank has the full updated parameters.
    """

    def __init__(self, optimizer: torch.optim.Optimizer, *,
                 process_group: Optional[Any] = None):
        self._opt = optimizer
        self._pg = process_group
        self._world_size = dist.get_world_size(group=process_group) if dist.is_initialized() else 1
        self._rank = dist.get_rank(group=process_group) if dist.is_initialized() else 0
        self._param_shards = self._partition_params()

    def _partition_params(self):
        """Cyclic dealing: param i -> rank (i % world_size)."""
        shards = [[] for _ in range(self._world_size)]
        all_params = []
        for group in self._opt.param_groups:
            for p in group["params"]:
                all_params.append(p)
        for i, p in enumerate(all_params):
            shards[i % self._world_size].append(p)
        return shards

    def step(self, closure=None):
        # Local step on this rank's shard only.
        self._opt.step(closure)
        # All_gather the updated weights so every rank has the full params.
        for i, p in enumerate(self._all_params()):
            owner_rank = i % self._world_size
            if self._world_size > 1 and dist.is_initialized():
                dist.broadcast(p.data, src=owner_rank, group=self._pg)

    def zero_grad(self, set_to_none=True):
        self._opt.zero_grad(set_to_none=set_to_none)

    def _all_params(self):
        out = []
        for group in self._opt.param_groups:
            out.extend(group["params"])
        return out

    @property
    def param_groups(self):
        return self._opt.param_groups

    def state_dict(self):
        return self._opt.state_dict()

    def load_state_dict(self, state):
        self._opt.load_state_dict(state)


__all__ = ["ShardedOptimizer"]
```

This is a simplified version — real ZeRO-1 actually shards optimizer STATE (Adam moments, etc.), not just params. For Plan 5 the simpler shape: each rank only computes optimizer step on its assigned params, then broadcasts. This still gives memory savings on optimizer state because Adam's `m` and `v` only exist for owned params on each rank.

- [ ] **Step 2: multi-rank test via `torch.multiprocessing.spawn`**

```python
"""Multi-rank ShardedOptimizer test using torch.multiprocessing.spawn."""
import os

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp


def _worker(rank, world_size):
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "29501"
    dist.init_process_group(backend="gloo", rank=rank, world_size=world_size)

    from sakura.zero.sharded_optimizer import ShardedOptimizer

    model = torch.nn.Linear(8, 4)
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    sharded = ShardedOptimizer(opt)

    # Set fake gradients.
    for p in model.parameters():
        p.grad = torch.ones_like(p)

    # Capture pre-step weights.
    before = [p.detach().clone() for p in model.parameters()]
    sharded.step()
    after = [p.detach().clone() for p in model.parameters()]

    # All ranks should see the same updated weights (broadcast worked).
    for b, a in zip(before, after):
        assert not torch.allclose(b, a)  # weights moved
    dist.barrier()
    dist.destroy_process_group()


def test_sharded_optimizer_2_ranks():
    """Spawn 2 processes, run ShardedOptimizer.step on a 4-param model."""
    if not torch.distributed.is_available():
        pytest.skip("torch.distributed not available")
    mp.spawn(_worker, args=(2,), nprocs=2, join=True)
```

- [ ] **Step 3: wire `ShardedOptimizer` into `sakura/services/zero1.py`**

Modify `ZeRO1.on_train_begin` so that when `event.world_size > 1`, it wraps the optimizer with `ShardedOptimizer`:

```python
def on_train_begin(self, event: OnTrainBegin):
    if event.world_size > 1:
        from sakura.zero.sharded_optimizer import ShardedOptimizer
        self._sharded = ShardedOptimizer(event.optimizer, process_group=self._process_group)
    self._original_optimizer = event.optimizer

def on_optimizer_step(self, event: OnOptimizerStep):
    if hasattr(self, "_sharded") and self._sharded is not None:
        self._sharded.step()
    else:
        event.optimizer.step()
```

- [ ] **Step 4: run + commit**

```bash
pytest tests/zero/test_sharded_optimizer.py -v
git add sakura/zero/ tests/zero/ sakura/services/zero1.py
git commit -m "feat(zero): multi-rank ZeRO1 — cyclic dealing + broadcast updated weights

Plan 3 carryover: ShardedOptimizer wraps a torch.optim.Optimizer for
stage-1 ZeRO sharding. Each rank computes local step on 1/world_size of
the params, then broadcasts updated weights so every rank has the full
parameter set.

Tested via torch.multiprocessing.spawn(2) on CPU with the gloo backend.
Real-world multi-GPU validation lands in the benchmark harness.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 4: benchmark harness — types

**Files:** Create `sakura/bench/__init__.py`, `sakura/bench/harness.py`, `tests/bench/__init__.py`, `tests/bench/test_workload.py`, `tests/bench/test_run_report.py`.

Implement the dataclasses from spec §11.1: `Workload`, `BaselineRunner`, `SakuraRunner`, `RunReport`. Plan 5 ships the harness types + report serialization; concrete workloads are Tasks 6-8.

- [ ] **Step 1: write the failing tests** (`tests/bench/test_workload.py` + `test_run_report.py`)

Tests should verify:
- `Workload` is a dataclass holding `name`, `tier`, `make_model`, `make_train_loader`, `make_val_loader`, `eval_fn`, `metric_target`, `epochs`.
- `RunReport` is a dataclass holding `workload`, `framework`, `sakura_services`, `elapsed_secs`, `samples_per_sec`, `peak_gpu_mem_mb`, `final_metrics`, `per_stage_secs`, `git_sha`, `hardware`.
- `RunReport.to_json()` and `RunReport.from_json()` round-trip.
- `BaselineRunner(framework="pytorch-ddp")` is constructible.
- `SakuraRunner(framework="pytorch-ddp", services=[...], compute=Compute.local())` is constructible.

- [ ] **Step 2: implement `sakura/bench/harness.py`** — reference spec §11.1.

- [ ] **Step 3: commit**

```bash
git add sakura/bench/__init__.py sakura/bench/harness.py tests/bench/__init__.py tests/bench/test_workload.py tests/bench/test_run_report.py
git commit -m "feat(bench): Workload + BaselineRunner + SakuraRunner + RunReport types

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 5: benchmark harness — `BaselineRunner` + `SakuraRunner`

**Files:** Modify `sakura/bench/harness.py`. Create `tests/bench/test_baseline_runner.py` + `test_sakura_runner.py`.

`BaselineRunner` runs vanilla framework training on a `Workload` and produces a `RunReport`. `SakuraRunner` does the same but installs the configured services. Both wrap the framework's `fit`/`train` call.

- [ ] **Step 1: implement `BaselineRunner.run()` for `framework="pytorch-ddp"`**

A bare-bones training loop: instantiate model + train loader + optimizer, run N epochs of `model(x); loss.backward(); optimizer.step()`. Measures wall time + samples/sec + peak GPU memory.

- [ ] **Step 2: implement `BaselineRunner.run()` for `framework="lightning"` and `framework="hf-trainer"`**

Wrap `lightning.Trainer.fit` and `transformers.Trainer.train` respectively.

- [ ] **Step 3: implement `SakuraRunner.run()`**

Same shape but with `SakuraRuntime` + installed services + the matching `Adapter`.

- [ ] **Step 4: smoke test on a tiny synthetic workload (no real ML data)**

Use a hand-crafted `Workload` that returns a minimal MLP + 4-batch loader, runs 1 epoch, returns `RunReport` with non-zero `elapsed_secs`.

- [ ] **Step 5: commit**

```bash
git add sakura/bench/harness.py tests/bench/test_baseline_runner.py tests/bench/test_sakura_runner.py
git commit -m "feat(bench): BaselineRunner (pytorch/lightning/hf) + SakuraRunner

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 6: workload — MNIST + tiny MLP (smoke tier)

**Files:** Create `sakura/bench/workloads/__init__.py`, `sakura/bench/workloads/mnist.py`, `tests/bench/test_workloads_smoke.py`.

The simplest workload: MNIST + 2-layer MLP. Runs in <2 minutes on a 4090, <30 minutes on CPU. Used for CI sanity.

- [ ] **Step 1: implement `mnist.py`** — uses `torchvision.datasets.MNIST`, returns a `Workload` instance with `tier="ci"`, `epochs=1` for smoke.

- [ ] **Step 2: smoke test** — instantiate the workload, run 1 epoch via `BaselineRunner(framework="pytorch-ddp")`, assert `report.elapsed_secs > 0` and `report.final_metrics` contains a reasonable `val_acc`.

- [ ] **Step 3: commit**

```bash
git add sakura/bench/workloads/__init__.py sakura/bench/workloads/mnist.py tests/bench/test_workloads_smoke.py
git commit -m "feat(bench): MNIST + tiny MLP workload (smoke tier)

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 7: workload — CIFAR-10 + ResNet-50 (CI tier)

**Files:** Create `sakura/bench/workloads/cifar.py`. Add to smoke tests.

CIFAR-10 + ResNet-50 from torchvision. Tier = "ci" but takes ~10 min; smoke variant uses 1 epoch.

- [ ] **Step 1: implement** `cifar.py`. Uses `torchvision.datasets.CIFAR10` + `torchvision.models.resnet50(weights=None)`.

- [ ] **Step 2: smoke test** with 1 epoch, batch size 64, asserting `elapsed_secs > 0`.

- [ ] **Step 3: commit**

---

## Task 8: workload — DistilBERT + SST-2 (CI tier)

**Files:** Create `sakura/bench/workloads/distilbert.py`.

DistilBERT-base-uncased + SST-2 (binary sentiment). Smoke variant: 200 train / 600 val examples, 1 epoch.

- [ ] **Step 1: implement** using `transformers.AutoModelForSequenceClassification` + `datasets.load_dataset("glue", "sst2")`.

- [ ] **Step 2: smoke test** with the smaller subset.

- [ ] **Step 3: commit**

---

## Task 9: workload stubs — Llama, Mistral, GLUE (perf tier)

**Files:** Create `sakura/bench/workloads/llama.py`, `mistral.py`, `glue.py`. Each is a stub: `Workload(tier="perf", ...)` with `pytest.skip` markers when GPU unavailable.

These are heavyweight workloads that need real GPU time. Plan 5 ships the stubs so the user can flesh them out with their own GPU. The smoke tests for these `pytest.skip`s when `not torch.cuda.is_available()` or `not torch.cuda.device_count() >= 1`.

- [ ] **Step 1: implement** stub workloads with TODO comments and skipped tests.

- [ ] **Step 2: commit**

---

## Task 10: `sakura-bench` CLI

**Files:** Create `sakura/bench/__main__.py`, `sakura/bench/compare.py`, `tests/bench/test_cli.py`. Add console script in `pyproject.toml`.

CLI shape per spec §11.3:
```bash
sakura-bench run --tier ci --output reports/
sakura-bench run --workload cifar10-resnet50 --runner sakura
sakura-bench compare reports/baseline.json reports/sakura.json
sakura-bench export reports/ --format markdown > BENCHMARKS.md
```

- [ ] **Step 1: implement** the argparse-based CLI in `sakura/bench/__main__.py`. `run` instantiates the chosen workload + runner; `compare` reads two JSON reports and prints a diff table.

- [ ] **Step 2: add to `pyproject.toml`'s `[project.scripts]`**:

```toml
sakura-bench = "sakura.bench.__main__:main"
```

- [ ] **Step 3: smoke test** the CLI with `subprocess.run([sys.executable, "-m", "sakura.bench", "run", "--tier", "ci", "--workload", "mnist-mlp", "--runner", "baseline"])`.

- [ ] **Step 4: commit**

---

## Task 11: README rewrite

**Files:** Replace `README.md`.

The current README is from v0.1.x. Plan 5 rewrites it for v1.0:
- The new pitch: SOTA training services for PyTorch DDP / Lightning / HF Trainer
- Quickstart example using `SakuraRuntime` + `LocalDispatcher` + a service or two
- Pointers to spec, plans, benchmark suite

- [ ] **Step 1: read the current README**

```bash
cat README.md
```

- [ ] **Step 2: write the new README**

Sections: pitch / quickstart / installation / architecture / benchmarks / docs links / license. Use the pattern from the spec's user-facing examples.

- [ ] **Step 3: commit**

```bash
git add README.md
git commit -m "docs: rewrite README for v1.0 (sakura-ml runtime + service catalog)

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 12: migration guide

**Files:** Create `docs/migration-from-0.1.md`.

Side-by-side examples for each integration (Lightning, HF, DDP) showing v0.1.x → v1.0 mapping. Reference the spec's §15 migration table.

- [ ] **Step 1: write the migration guide**

Each section has: "Before (v0.1.x)" code block, "After (v1.0)" code block, brief commentary. Cover: `SakuraTrainer` → `LightningAdapter + SakuraRuntime + AsyncEval`; `SakuraHFCallback` → `HFAdapter + AsyncEval`; `DDPAsyncEvalCallback` → `DDPAdapter + AsyncEval`.

- [ ] **Step 2: commit**

```bash
git add docs/migration-from-0.1.md
git commit -m "docs: migration guide from v0.1.x to v1.0

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 13: full acceptance + version bump + tag

**Files:** Modify `sakura/__init__.py` (bump to `1.0.0a1`); modify `crates/sakura-wire/Cargo.toml` (workspace package version); tag.

- [ ] **Step 1: bump `sakura/__init__.py` to `__version__ = "1.0.0a1"`**

- [ ] **Step 2: bump workspace `Cargo.toml` `version = "1.0.0-alpha.1"`**

- [ ] **Step 3: full acceptance**

```bash
cargo fmt --all --check && echo fmt_OK
cargo clippy --workspace --all-targets -- -D warnings && echo clippy_OK
cargo test --workspace 2>&1 | tail -5
maturin build --release --out /tmp/sakura-wheel-v5 2>&1 | tail -3
unzip -l /tmp/sakura-wheel-v5/*.whl | head -30
pytest tests/ 2>&1 | tail -5
sakura-bench run --tier ci --workload mnist-mlp --runner baseline 2>&1 | tail -5
```

- [ ] **Step 4: tag**

```bash
git tag -a sakura-v1.0.0a1 -m "v1.0.0a1 — feature-complete alpha (Plans 1-5)"
git tag --list 'sakura-*'
```

- [ ] **Step 5: commit**

```bash
git add sakura/__init__.py crates/sakura-wire/Cargo.toml Cargo.lock
git commit -m "chore: bump version to 1.0.0a1 — Plan 5 closeout (benchmark + zero-copy + ZeRO1 + docs)

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Plan 5 — Acceptance Criteria

Plan 5 is complete when **all** of the following are true:

1. `cargo fmt --all --check` and `cargo clippy --workspace --all-targets -- -D warnings` pass.
2. `cargo test --workspace`: ≥ 17 tests pass (no regressions; potentially +1 for codec_zero_copy).
3. `cargo bench -p sakura-wire`: codec encode for 268 MB drops below 100 ms (down from Plan 1's 218 ms).
4. `maturin build --release` produces a wheel that imports cleanly in a fresh venv (no `.pth` workaround).
5. `pytest tests/`: ≥ 130 tests pass.
6. `sakura-bench run --tier ci --workload mnist-mlp --runner baseline` produces a `RunReport` with non-zero `elapsed_secs`.
7. `sakura-bench compare` prints a diff table comparing baseline vs sakura on a synthetic workload.
8. `docs/migration-from-0.1.md` exists.
9. `README.md` is the v1.0 rewrite.
10. Tag `sakura-v1.0.0a1` exists.

After Plan 5 lands, **`sakura-ml==1.0.0a1`** is the v1.0 alpha release. Push to PyPI when ready.

---

## Self-Review Notes

- **Spec coverage:** Plan 5 implements §11 (benchmark harness), §12 (repo layout — packaging fixes), §6.3 (multi-rank ZeRO1 — completes the carryover from Plan 3), §7.2 (codec zero-copy — completes Plan 1's perf carryover), §15 (migration table — guide).
- **Out-of-scope confirmed:** All v1.0 work converges here; nothing on the 5-plan roadmap is left after this lands.
- **Post-v1.0 roadmap:** RDMA backend, JAX adapter, ZeRO-2/3, tensor parallelism, pipeline parallelism, gradient compression, CUDA IPC handle path, mutual-TLS for cross-host. These are post-1.0 milestones — not in any plan committed yet.
- **Plan 5 specifically does NOT have rigid task sub-step prescriptions** for the benchmark harness's framework-specific runners (Tasks 5/7/8) because each framework's API surface drifts faster than codec/runtime, and prescribing exact code in the plan would couple it too tightly. The implementer should consult current torch/lightning/transformers docs as they implement.
