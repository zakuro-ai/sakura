# Sakura v1.0 — Plan 1: `sakura-wire` Rust Foundation

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:subagent-driven-development` (recommended) or `superpowers:executing-plans` to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Stand up the `sakura-wire` Rust crate with PyO3 bindings — a zero-copy tensor codec + RPC protocol over QUIC + worker subprocess supervisor — providing the transport foundation that all later plans (runtime, services, adapters) build on.

**Architecture:** Single Cargo workspace at the repo root. One published Rust crate (`sakura-wire`) with three internal layers (codec / protocol / transport) and a `pyo3_bindings` module exposing `Dispatcher`, `Future`, `WorkerSupervisor`, and `TlsConfig` to Python. `quinn` provides QUIC; `rustls` provides TLS; `postcard` serializes headers; tensor payloads ride raw over the wire (zero-copy via the buffer protocol). A minimal Python `sakura.worker` daemon and a `HANDLER_ECHO` test handler prove end-to-end round-trip.

**Tech Stack:** Rust 2021 (rustc 1.78+), `pyo3 0.21` (abi3-py310), `tokio 1.x`, `quinn 0.10`, `rustls 0.21`, `rcgen 0.12`, `postcard 1.x`, `serde`, `bytes`, `smallvec`, `half`, `thiserror`, `tracing`, `criterion` (benches). Python build via `maturin >= 1.4`. Existing `sakura/` Python package stays in place; the wheel ships both `sakura` (Python) and `sakura_wire` (compiled Rust extension).

**Out of scope for this plan** (deferred to Plan 2+): `SakuraRuntime`, `Service` ABC, framework adapters, real handlers (eval/checkpoint), `SHM` transport, `RDMA` transport, mutual TLS for cross-host, CUDA IPC handle path, removing v0.1.x code.

---

## Prerequisites

Install once on the development machine (these are not part of any task):

```bash
# Rust toolchain (rustup self-installs)
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain none

# maturin (Python build backend for PyO3)
uv pip install --system maturin>=1.4

# Verify
rustup --version && cargo --version && maturin --version
```

Hardware: any x86_64 or aarch64 Linux box. No GPU required for Plan 1 (the codec handles tensors as raw bytes; CUDA paths are exercised in later plans).

---

## File Structure (created/modified by this plan)

**New files:**
```
Cargo.toml                                       # workspace
rust-toolchain.toml                              # pinned rustc
crates/sakura-wire/
  Cargo.toml
  src/
    lib.rs                                       # crate root + #[pymodule]
    codec/
      mod.rs                                     # public Encoder/Decoder facade
      header.rs                                  # postcard structs (RpcRequestHeader, ...)
      tensor.rs                                  # TensorView, pack/unpack
      cast.rs                                    # fp32 ↔ fp16 conversion
    protocol/
      mod.rs                                     # request/response framing
      handlers.rs                                # well-known handler IDs
      error.rs                                   # WireError + Python exception bridge
    transport/
      mod.rs                                     # Transport trait, URI selectors
      quic.rs                                    # quinn-based default
    runtime.rs                                   # tokio runtime owned by Rust
    supervisor.rs                                # spawn/manage worker subprocesses
    pyo3_bindings.rs                             # Dispatcher, Future, WorkerSupervisor, TlsConfig
  tests/
    codec_roundtrip.rs                           # cross-module integration test
    quic_loopback.rs                             # transport+protocol+codec round-trip
  benches/
    codec_bench.rs                               # criterion: encode/decode MB/s
    transport_bench.rs                           # criterion: RTT, throughput
sakura/wire/__init__.py                          # thin Python wrapper around sakura_wire (re-exports + docs)
sakura/worker/__init__.py
sakura/worker/__main__.py                        # `sakura-worker` CLI: echo handler for Plan 1
tests/wire/__init__.py
tests/wire/test_e2e_echo.py                      # spawn-worker → submit → echo round-trip
```

**Existing files modified:**
```
pyproject.toml                                   # switch build-system to maturin; add scripts
.gitignore                                       # add Rust target/, *.so, .python-version, dist/
.github/workflows/test.yml                       # add Rust toolchain + cargo test + maturin develop
.github/workflows/publish.yml                    # add maturin wheel matrix
```

**Files NOT touched (deferred to later plans):**
```
sakura/**                                        # existing v0.1.x code stays — Plan 2 owns the rewrite
tests/test_*.py (existing)                       # existing v0.1.x tests stay
main.py, bert_demo/, mnist_demo/, compose.yaml, docker/, Taskfile.yml, taskfiles/
```

---

## Task 1: Cargo workspace + maturin pyproject + rust-toolchain + .gitignore

**Files:**
- Create: `Cargo.toml`, `rust-toolchain.toml`, `crates/sakura-wire/Cargo.toml`, `crates/sakura-wire/src/lib.rs`
- Modify: `pyproject.toml`, `.gitignore`

- [ ] **Step 1: Write the failing build verification**

Run:
```bash
cargo --version
```
Expected: cargo prints version. (Prerequisite check; skip if already verified.)

- [ ] **Step 2: Create the workspace `Cargo.toml`**

Write `Cargo.toml`:
```toml
[workspace]
members = ["crates/*"]
resolver = "2"

[workspace.package]
version = "1.0.0-alpha.0"
edition = "2021"
license = "BSD-3-Clause"
repository = "https://github.com/zakuro-ai/sakura"

[workspace.dependencies]
pyo3 = { version = "0.21", features = ["extension-module", "abi3-py310"] }
tokio = { version = "1", features = ["macros", "rt-multi-thread", "sync", "time", "io-util", "process"] }
quinn = "0.10"
rustls = { version = "0.21", default-features = false, features = ["tls12"] }
rustls-pemfile = "1.0"
rcgen = "0.12"
postcard = { version = "1.0", default-features = false, features = ["alloc", "use-std"] }
serde = { version = "1.0", features = ["derive"] }
bytes = "1.5"
smallvec = { version = "1.13", features = ["serde", "union"] }
half = { version = "2.4", default-features = false, features = ["std", "num-traits"] }
thiserror = "1.0"
tracing = "0.1"
tracing-subscriber = { version = "0.3", features = ["env-filter"] }
criterion = { version = "0.5", default-features = false, features = ["html_reports"] }
tempfile = "3.10"

[profile.release]
lto = "thin"
codegen-units = 1
debug = 1                                       # keep symbols for perf profiling
```

- [ ] **Step 3: Create `rust-toolchain.toml`**

Write `rust-toolchain.toml`:
```toml
[toolchain]
channel = "1.78.0"
components = ["rustfmt", "clippy"]
profile = "minimal"
```

- [ ] **Step 4: Create `crates/sakura-wire/Cargo.toml`**

Write `crates/sakura-wire/Cargo.toml`:
```toml
[package]
name = "sakura-wire"
version.workspace = true
edition.workspace = true
license.workspace = true
repository.workspace = true
description = "Sakura: zero-copy tensor wire codec + RPC over QUIC for SOTA training pipelines."

[lib]
name = "sakura_wire"
crate-type = ["cdylib", "rlib"]

[dependencies]
pyo3 = { workspace = true }
tokio = { workspace = true }
quinn = { workspace = true }
rustls = { workspace = true }
rustls-pemfile = { workspace = true }
rcgen = { workspace = true }
postcard = { workspace = true }
serde = { workspace = true }
bytes = { workspace = true }
smallvec = { workspace = true }
half = { workspace = true }
thiserror = { workspace = true }
tracing = { workspace = true }

[dev-dependencies]
tempfile = { workspace = true }
criterion = { workspace = true }
tracing-subscriber = { workspace = true }

[[bench]]
name = "codec_bench"
harness = false

[[bench]]
name = "transport_bench"
harness = false
```

- [ ] **Step 5: Create `crates/sakura-wire/src/lib.rs` (PyO3 module skeleton)**

Write `crates/sakura-wire/src/lib.rs`:
```rust
//! sakura-wire: zero-copy tensor codec + RPC over QUIC.

#![deny(unsafe_op_in_unsafe_fn)]

use pyo3::prelude::*;

#[pymodule]
fn sakura_wire(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    Ok(())
}
```

- [ ] **Step 6: Update `pyproject.toml` — switch to maturin**

Replace contents of `pyproject.toml`:
```toml
[build-system]
requires = ["maturin>=1.4,<2"]
build-backend = "maturin"

[project]
name = "sakura-ml"
version = "1.0.0a0"
description = "Sakura: SOTA training services for PyTorch DDP / Lightning / HuggingFace Trainer."
readme = "README.md"
license = "BSD-3-Clause"
requires-python = ">=3.10"
authors = [{ name = "ZakuroAI", email = "git@zakuro.ai" }]
dependencies = [
    "tqdm>=4.64.0",
    "torch",
    "torchvision",
    "numpy",
    "gnutools-python",
    "six",
    "chardet",
    "lightning",
    "ipython",
    "charset-normalizer==3.1.0",
    "zakuro-ai>=0.2.3",
]
classifiers = [
    "Intended Audience :: Developers",
    "Intended Audience :: Education",
    "Intended Audience :: Science/Research",
    "License :: OSI Approved :: BSD License",
    "Topic :: Scientific/Engineering",
    "Topic :: Scientific/Engineering :: Artificial Intelligence",
    "Programming Language :: Python :: 3",
]

[project.optional-dependencies]
huggingface = [
    "transformers>=4.40.0",
    "datasets>=2.14.0",
    "accelerate>=1.1.0",
]
tensorflow = ["tensorflow>=2.15.0"]
bench = ["sakura-ml[huggingface]"]

[dependency-groups]
dev = ["pytest", "maturin>=1.4"]

[project.scripts]
sakura = "sakura.__main__:main"
sakura-benchmark = "main:main"
sakura-worker = "sakura.worker.__main__:main"

[tool.pytest.ini_options]
testpaths = ["tests"]

[tool.maturin]
python-source = "."
manifest-path = "crates/sakura-wire/Cargo.toml"
module-name = "sakura_wire"
features = ["pyo3/extension-module"]
include = [
    { path = "sakura/config.yaml", format = "sdist" },
    { path = "sakura/config.yaml", format = "wheel" },
    { path = "main.py", format = "wheel" },
]
```

- [ ] **Step 7: Update `.gitignore` — Rust artifacts**

Append to `.gitignore`:
```
# Rust
/target/
**/*.rs.bk
*.pdb

# maturin
/dist/
*.so
*.dylib
*.pyd
.python-version
```

- [ ] **Step 8: Verify the build**

Run:
```bash
cargo check --workspace
```
Expected: compiles cleanly, exits 0.

Run:
```bash
maturin develop
```
Expected: builds the extension, installs into the active Python env, exits 0.

Run:
```bash
python -c "import sakura_wire; print(sakura_wire.__version__)"
```
Expected: prints `1.0.0-alpha.0`.

- [ ] **Step 9: Commit**

```bash
git add Cargo.toml rust-toolchain.toml crates/ pyproject.toml .gitignore
git commit -m "$(cat <<'EOF'
build: scaffold Cargo workspace + maturin pyproject for sakura-wire

Adds the Rust workspace at the repo root with a single sakura-wire crate,
pins rustc 1.78, and switches the Python build to maturin so the same
wheel ships sakura (Python) and sakura_wire (compiled Rust extension).
Existing sakura/ v0.1.x code is untouched.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 2: Codec — wire types (postcard headers)

**Files:**
- Create: `crates/sakura-wire/src/codec/mod.rs`, `crates/sakura-wire/src/codec/header.rs`

- [ ] **Step 1: Add the codec module declaration**

Edit `crates/sakura-wire/src/lib.rs` — append after the existing imports:
```rust
pub mod codec;
```

- [ ] **Step 2: Write the failing test for header round-trip**

Create `crates/sakura-wire/src/codec/header.rs` with the test stub:
```rust
//! Postcard-encoded headers for the wire format.

use serde::{Deserialize, Serialize};
use smallvec::SmallVec;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[repr(u8)]
pub enum WireVersion {
    V1 = 1,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[repr(u8)]
pub enum Dtype {
    F32 = 0,
    F16 = 1,
    BF16 = 2,
    F8E4M3 = 3,
    I64 = 10,
    I32 = 11,
    U8 = 12,
    Bool = 13,
}

impl Dtype {
    pub fn size_bytes(&self) -> usize {
        match self {
            Dtype::F32 | Dtype::I32 => 4,
            Dtype::F16 | Dtype::BF16 => 2,
            Dtype::F8E4M3 | Dtype::U8 | Dtype::Bool => 1,
            Dtype::I64 => 8,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Device {
    Cpu,
    Cuda(u8),
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TensorDesc {
    pub shape: SmallVec<[u32; 8]>,
    pub dtype: Dtype,
    pub n_bytes: u64,
    pub device_hint: Device,
    pub fp16_cast_on_wire: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum RpcStatus {
    Ok,
    Error,
    Cancelled,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RpcRequestHeader {
    pub version: WireVersion,
    pub request_id: u64,
    pub handler_id: u32,
    pub n_tensors: u32,
    pub aux_payload_bytes: u32,
    pub deadline_ms: Option<u32>,
    pub trace_id: u128,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RpcResponseHeader {
    pub version: WireVersion,
    pub request_id: u64,
    pub status: RpcStatus,
    pub n_result_tensors: u32,
    pub aux_payload_bytes: u32,
    pub elapsed_us: u64,
}

#[cfg(test)]
mod tests {
    use super::*;
    use postcard::{from_bytes, to_allocvec};
    use smallvec::smallvec;

    #[test]
    fn rpc_request_header_roundtrip() {
        let h = RpcRequestHeader {
            version: WireVersion::V1,
            request_id: 0xDEAD_BEEF,
            handler_id: 0xDEAD,
            n_tensors: 3,
            aux_payload_bytes: 1024,
            deadline_ms: Some(5000),
            trace_id: 0x0123_4567_89AB_CDEF_0000_1111_2222_3333,
        };
        let bytes = to_allocvec(&h).unwrap();
        let decoded: RpcRequestHeader = from_bytes(&bytes).unwrap();
        assert_eq!(decoded.request_id, h.request_id);
        assert_eq!(decoded.handler_id, h.handler_id);
        assert_eq!(decoded.n_tensors, h.n_tensors);
        assert_eq!(decoded.deadline_ms, h.deadline_ms);
        assert_eq!(decoded.trace_id, h.trace_id);
    }

    #[test]
    fn tensor_desc_roundtrip() {
        let t = TensorDesc {
            shape: smallvec![768, 3, 224, 224],
            dtype: Dtype::F32,
            n_bytes: 768 * 3 * 224 * 224 * 4,
            device_hint: Device::Cuda(0),
            fp16_cast_on_wire: false,
        };
        let bytes = to_allocvec(&t).unwrap();
        let decoded: TensorDesc = from_bytes(&bytes).unwrap();
        assert_eq!(decoded, t);
    }

    #[test]
    fn rpc_response_header_roundtrip() {
        let h = RpcResponseHeader {
            version: WireVersion::V1,
            request_id: 42,
            status: RpcStatus::Ok,
            n_result_tensors: 1,
            aux_payload_bytes: 64,
            elapsed_us: 4523,
        };
        let bytes = to_allocvec(&h).unwrap();
        let decoded: RpcResponseHeader = from_bytes(&bytes).unwrap();
        assert_eq!(decoded.request_id, h.request_id);
        assert_eq!(decoded.status, h.status);
        assert_eq!(decoded.elapsed_us, h.elapsed_us);
    }

    #[test]
    fn dtype_size_bytes() {
        assert_eq!(Dtype::F32.size_bytes(), 4);
        assert_eq!(Dtype::F16.size_bytes(), 2);
        assert_eq!(Dtype::BF16.size_bytes(), 2);
        assert_eq!(Dtype::F8E4M3.size_bytes(), 1);
        assert_eq!(Dtype::I64.size_bytes(), 8);
        assert_eq!(Dtype::I32.size_bytes(), 4);
        assert_eq!(Dtype::U8.size_bytes(), 1);
        assert_eq!(Dtype::Bool.size_bytes(), 1);
    }
}
```

- [ ] **Step 3: Create `crates/sakura-wire/src/codec/mod.rs`**

Write `crates/sakura-wire/src/codec/mod.rs`:
```rust
//! Wire-format codec: postcard headers + raw zero-copy tensor payloads.

pub mod header;

pub use header::{
    Device, Dtype, RpcRequestHeader, RpcResponseHeader, RpcStatus, TensorDesc, WireVersion,
};
```

- [ ] **Step 4: Run the tests — verify they pass (no impl change needed; types ARE the impl)**

Run:
```bash
cargo test -p sakura-wire codec::header::tests
```
Expected:
```
running 4 tests
test codec::header::tests::dtype_size_bytes ... ok
test codec::header::tests::rpc_request_header_roundtrip ... ok
test codec::header::tests::rpc_response_header_roundtrip ... ok
test codec::header::tests::tensor_desc_roundtrip ... ok
```

- [ ] **Step 5: Commit**

```bash
git add crates/sakura-wire/src/codec/ crates/sakura-wire/src/lib.rs
git commit -m "feat(wire): postcard headers — RpcRequestHeader, TensorDesc, RpcResponseHeader

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 3: Codec — fp32 ↔ fp16 cast (`half` crate)

**Files:**
- Create: `crates/sakura-wire/src/codec/cast.rs`
- Modify: `crates/sakura-wire/src/codec/mod.rs`

- [ ] **Step 1: Write the failing test**

Create `crates/sakura-wire/src/codec/cast.rs`:
```rust
//! fp32 ↔ fp16 / bf16 conversion. Bit-identical to torch's `.to(torch.float16)`
//! (IEEE 754 round-to-nearest-even via the `half` crate).

use half::{bf16, f16};

/// Cast a slice of f32 bytes into a Vec<u8> of fp16 bytes (length halves).
pub fn cast_f32_to_f16(input: &[u8]) -> Vec<u8> {
    debug_assert!(input.len() % 4 == 0, "input must be 4-byte aligned (f32)");
    let n = input.len() / 4;
    let mut out = Vec::with_capacity(n * 2);
    for chunk in input.chunks_exact(4) {
        let v = f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
        let h = f16::from_f32(v);
        out.extend_from_slice(&h.to_le_bytes());
    }
    out
}

/// Cast a slice of fp16 bytes back to f32 bytes (length doubles).
pub fn cast_f16_to_f32(input: &[u8]) -> Vec<u8> {
    debug_assert!(input.len() % 2 == 0, "input must be 2-byte aligned (f16)");
    let n = input.len() / 2;
    let mut out = Vec::with_capacity(n * 4);
    for chunk in input.chunks_exact(2) {
        let h = f16::from_le_bytes([chunk[0], chunk[1]]);
        let v = h.to_f32();
        out.extend_from_slice(&v.to_le_bytes());
    }
    out
}

/// Cast a slice of f32 bytes into a Vec<u8> of bf16 bytes (length halves).
pub fn cast_f32_to_bf16(input: &[u8]) -> Vec<u8> {
    debug_assert!(input.len() % 4 == 0, "input must be 4-byte aligned (f32)");
    let n = input.len() / 4;
    let mut out = Vec::with_capacity(n * 2);
    for chunk in input.chunks_exact(4) {
        let v = f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
        let h = bf16::from_f32(v);
        out.extend_from_slice(&h.to_le_bytes());
    }
    out
}

/// Cast a slice of bf16 bytes back to f32 bytes (length doubles).
pub fn cast_bf16_to_f32(input: &[u8]) -> Vec<u8> {
    debug_assert!(input.len() % 2 == 0, "input must be 2-byte aligned (bf16)");
    let n = input.len() / 2;
    let mut out = Vec::with_capacity(n * 4);
    for chunk in input.chunks_exact(2) {
        let h = bf16::from_le_bytes([chunk[0], chunk[1]]);
        let v = h.to_f32();
        out.extend_from_slice(&v.to_le_bytes());
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn f32_bytes(values: &[f32]) -> Vec<u8> {
        let mut out = Vec::with_capacity(values.len() * 4);
        for &v in values {
            out.extend_from_slice(&v.to_le_bytes());
        }
        out
    }

    fn bytes_to_f32(input: &[u8]) -> Vec<f32> {
        input
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect()
    }

    #[test]
    fn fp16_roundtrip_preserves_representable_values() {
        let original = [0.0_f32, 1.0, -1.0, 0.5, -0.5, 256.0, 1e-4];
        let bytes = f32_bytes(&original);
        let half = cast_f32_to_f16(&bytes);
        assert_eq!(half.len(), bytes.len() / 2);
        let back = cast_f16_to_f32(&half);
        let recovered = bytes_to_f32(&back);
        for (orig, got) in original.iter().zip(recovered.iter()) {
            assert!(
                (orig - got).abs() < 1e-3,
                "fp16 round-trip drift: {} vs {}",
                orig,
                got
            );
        }
    }

    #[test]
    fn bf16_roundtrip_preserves_dynamic_range() {
        let original = [0.0_f32, 1.0, -1.0, 1e10, -1e10, 1e-30];
        let bytes = f32_bytes(&original);
        let bf = cast_f32_to_bf16(&bytes);
        assert_eq!(bf.len(), bytes.len() / 2);
        let back = cast_bf16_to_f32(&bf);
        let recovered = bytes_to_f32(&back);
        for (orig, got) in original.iter().zip(recovered.iter()) {
            // bf16 has 7 mantissa bits — relative precision ~1%.
            let rel = if orig.abs() > 1e-6 { (orig - got).abs() / orig.abs() } else { (orig - got).abs() };
            assert!(rel < 1e-2, "bf16 round-trip drift: {} vs {}", orig, got);
        }
    }

    #[test]
    fn empty_slice_yields_empty_output() {
        assert!(cast_f32_to_f16(&[]).is_empty());
        assert!(cast_f16_to_f32(&[]).is_empty());
        assert!(cast_f32_to_bf16(&[]).is_empty());
        assert!(cast_bf16_to_f32(&[]).is_empty());
    }
}
```

- [ ] **Step 2: Run the test — should fail at compile (cast module not in mod.rs)**

Run:
```bash
cargo test -p sakura-wire codec::cast 2>&1 | head -20
```
Expected: error — `module 'cast' not found in 'codec'` or similar.

- [ ] **Step 3: Add `pub mod cast;` to codec/mod.rs**

Edit `crates/sakura-wire/src/codec/mod.rs` to add:
```rust
pub mod cast;
```

The full file is now:
```rust
//! Wire-format codec: postcard headers + raw zero-copy tensor payloads.

pub mod cast;
pub mod header;

pub use header::{
    Device, Dtype, RpcRequestHeader, RpcResponseHeader, RpcStatus, TensorDesc, WireVersion,
};
```

- [ ] **Step 4: Run the test — should now pass**

Run:
```bash
cargo test -p sakura-wire codec::cast::tests
```
Expected:
```
running 3 tests
test codec::cast::tests::bf16_roundtrip_preserves_dynamic_range ... ok
test codec::cast::tests::empty_slice_yields_empty_output ... ok
test codec::cast::tests::fp16_roundtrip_preserves_representable_values ... ok
```

- [ ] **Step 5: Commit**

```bash
git add crates/sakura-wire/src/codec/cast.rs crates/sakura-wire/src/codec/mod.rs
git commit -m "feat(wire): fp32↔fp16 and fp32↔bf16 cast via half crate

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 4: Codec — `TensorView` and zero-copy pack/unpack

**Files:**
- Create: `crates/sakura-wire/src/codec/tensor.rs`
- Modify: `crates/sakura-wire/src/codec/mod.rs`

- [ ] **Step 1: Write the failing test**

Create `crates/sakura-wire/src/codec/tensor.rs`:
```rust
//! Owning + borrowing tensor types used by the codec.
//!
//! `TensorView` borrows external bytes (zero-copy from the buffer protocol);
//! `OwnedTensor` owns its bytes (used on the receiver side after assembling
//! a buffer from the wire).

use crate::codec::header::{Device, Dtype, TensorDesc};
use smallvec::SmallVec;

/// A borrowed view of tensor bytes, paired with a descriptor.
/// Producers create these from PyO3 buffer protocol slices — no copy.
pub struct TensorView<'a> {
    pub desc: TensorDesc,
    pub bytes: &'a [u8],
}

impl<'a> TensorView<'a> {
    pub fn new(
        shape: impl Into<SmallVec<[u32; 8]>>,
        dtype: Dtype,
        device_hint: Device,
        bytes: &'a [u8],
    ) -> Self {
        let desc = TensorDesc {
            shape: shape.into(),
            dtype,
            n_bytes: bytes.len() as u64,
            device_hint,
            fp16_cast_on_wire: false,
        };
        Self { desc, bytes }
    }

    pub fn with_fp16_cast(mut self) -> Self {
        self.desc.fp16_cast_on_wire = true;
        self
    }

    pub fn n_elements(&self) -> usize {
        self.desc.shape.iter().product::<u32>() as usize
    }
}

/// A heap-allocated tensor blob used after assembling from the wire.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OwnedTensor {
    pub desc: TensorDesc,
    pub bytes: Vec<u8>,
}

impl OwnedTensor {
    pub fn new(desc: TensorDesc, bytes: Vec<u8>) -> Self {
        debug_assert_eq!(bytes.len() as u64, desc.n_bytes);
        Self { desc, bytes }
    }

    pub fn from_view(view: &TensorView<'_>) -> Self {
        Self::new(view.desc.clone(), view.bytes.to_vec())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use smallvec::smallvec;

    #[test]
    fn view_descriptor_matches_byte_length() {
        let bytes = vec![0u8; 4 * 12];
        let v = TensorView::new(smallvec![3u32, 4], Dtype::F32, Device::Cpu, &bytes);
        assert_eq!(v.desc.shape.as_slice(), &[3, 4]);
        assert_eq!(v.desc.dtype, Dtype::F32);
        assert_eq!(v.desc.n_bytes, 48);
        assert_eq!(v.n_elements(), 12);
        assert!(!v.desc.fp16_cast_on_wire);
    }

    #[test]
    fn view_fp16_cast_flag_flips() {
        let bytes = vec![0u8; 16];
        let v = TensorView::new(smallvec![4u32], Dtype::F32, Device::Cpu, &bytes).with_fp16_cast();
        assert!(v.desc.fp16_cast_on_wire);
    }

    #[test]
    fn owned_tensor_roundtrips_from_view() {
        let bytes: Vec<u8> = (0..16).collect();
        let v = TensorView::new(smallvec![4u32], Dtype::F32, Device::Cpu, &bytes);
        let owned = OwnedTensor::from_view(&v);
        assert_eq!(owned.bytes, bytes);
        assert_eq!(owned.desc.dtype, Dtype::F32);
    }
}
```

- [ ] **Step 2: Run — fails at compile (module not declared)**

Run:
```bash
cargo test -p sakura-wire codec::tensor 2>&1 | head -20
```
Expected: `module 'tensor' not found`.

- [ ] **Step 3: Add `pub mod tensor;` and re-exports to codec/mod.rs**

Replace `crates/sakura-wire/src/codec/mod.rs` with:
```rust
//! Wire-format codec: postcard headers + raw zero-copy tensor payloads.

pub mod cast;
pub mod header;
pub mod tensor;

pub use header::{
    Device, Dtype, RpcRequestHeader, RpcResponseHeader, RpcStatus, TensorDesc, WireVersion,
};
pub use tensor::{OwnedTensor, TensorView};
```

- [ ] **Step 4: Run — should pass**

Run:
```bash
cargo test -p sakura-wire codec
```
Expected: all codec tests pass (header + cast + tensor).

- [ ] **Step 5: Commit**

```bash
git add crates/sakura-wire/src/codec/tensor.rs crates/sakura-wire/src/codec/mod.rs
git commit -m "feat(wire): TensorView (borrowing) + OwnedTensor (owned) types

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 5: Codec — full pack/unpack integration test

**Files:**
- Create: `crates/sakura-wire/tests/codec_roundtrip.rs`
- Modify: `crates/sakura-wire/src/codec/mod.rs` (add `pack_request` / `unpack_request` helpers)

- [ ] **Step 1: Write the failing integration test**

Create `crates/sakura-wire/tests/codec_roundtrip.rs`:
```rust
//! Integration test: pack a request (header + descriptors + payloads + aux)
//! into a Vec<u8> and unpack it back. Verifies the public codec facade.

use sakura_wire::codec::{
    pack_request, unpack_request, Device, Dtype, RpcRequestHeader, TensorView, WireVersion,
};
use smallvec::smallvec;

#[test]
fn pack_then_unpack_preserves_request() {
    let h = RpcRequestHeader {
        version: WireVersion::V1,
        request_id: 7,
        handler_id: 0xDEAD,
        n_tensors: 2,
        aux_payload_bytes: 5,
        deadline_ms: None,
        trace_id: 0,
    };

    // Two tensors with distinguishable byte patterns.
    let t1_bytes: Vec<u8> = (0u8..16).collect();
    let t2_bytes: Vec<u8> = (32u8..40).collect();
    let t1 = TensorView::new(smallvec![4u32], Dtype::F32, Device::Cpu, &t1_bytes);
    let t2 = TensorView::new(smallvec![2u32], Dtype::F32, Device::Cpu, &t2_bytes);
    let aux = b"hello";

    let packed = pack_request(&h, &[t1, t2], aux).expect("pack");
    let (got_header, got_descs, got_tensor_bytes, got_aux) =
        unpack_request(&packed).expect("unpack");

    assert_eq!(got_header.request_id, h.request_id);
    assert_eq!(got_header.handler_id, h.handler_id);
    assert_eq!(got_descs.len(), 2);
    assert_eq!(got_tensor_bytes.len(), 2);
    assert_eq!(got_tensor_bytes[0], t1_bytes);
    assert_eq!(got_tensor_bytes[1], t2_bytes);
    assert_eq!(got_aux, aux);
}

#[test]
fn pack_request_with_zero_tensors_works() {
    let h = RpcRequestHeader {
        version: WireVersion::V1,
        request_id: 1,
        handler_id: 0x0001,
        n_tensors: 0,
        aux_payload_bytes: 4,
        deadline_ms: Some(1000),
        trace_id: 42,
    };
    let aux = b"NOOP";
    let packed = pack_request(&h, &[], aux).unwrap();
    let (got_h, descs, tensors, got_aux) = unpack_request(&packed).unwrap();
    assert_eq!(got_h.handler_id, 0x0001);
    assert!(descs.is_empty());
    assert!(tensors.is_empty());
    assert_eq!(got_aux, aux);
}
```

- [ ] **Step 2: Run — fails because pack_request / unpack_request don't exist**

Run:
```bash
cargo test -p sakura-wire --test codec_roundtrip 2>&1 | head -30
```
Expected: `unresolved imports`, `pack_request`, `unpack_request`.

- [ ] **Step 3: Implement `pack_request` and `unpack_request` in `codec/mod.rs`**

Replace `crates/sakura-wire/src/codec/mod.rs` with:
```rust
//! Wire-format codec: postcard headers + raw zero-copy tensor payloads.

pub mod cast;
pub mod header;
pub mod tensor;

pub use header::{
    Device, Dtype, RpcRequestHeader, RpcResponseHeader, RpcStatus, TensorDesc, WireVersion,
};
pub use tensor::{OwnedTensor, TensorView};

use thiserror::Error;

#[derive(Debug, Error)]
pub enum CodecError {
    #[error("postcard encode failed: {0}")]
    Encode(String),
    #[error("postcard decode failed: {0}")]
    Decode(String),
    #[error("truncated payload: expected {expected} bytes, got {got}")]
    Truncated { expected: usize, got: usize },
    #[error("descriptor count mismatch: header.n_tensors = {header}, descriptors = {got}")]
    DescriptorCountMismatch { header: u32, got: usize },
}

impl From<postcard::Error> for CodecError {
    fn from(e: postcard::Error) -> Self {
        CodecError::Decode(e.to_string())
    }
}

/// Pack a request into a single contiguous buffer. Used by the client side.
///
/// Layout: [postcard(header)] [postcard(Vec<TensorDesc>)] [tensor bytes...] [aux bytes].
pub fn pack_request(
    header: &RpcRequestHeader,
    tensors: &[TensorView<'_>],
    aux: &[u8],
) -> Result<Vec<u8>, CodecError> {
    let descs: Vec<TensorDesc> = tensors.iter().map(|t| t.desc.clone()).collect();
    let header_bytes =
        postcard::to_allocvec(header).map_err(|e| CodecError::Encode(e.to_string()))?;
    let descs_bytes =
        postcard::to_allocvec(&descs).map_err(|e| CodecError::Encode(e.to_string()))?;

    let total: usize = header_bytes.len()
        + descs_bytes.len()
        + tensors.iter().map(|t| t.bytes.len()).sum::<usize>()
        + aux.len();
    let mut out = Vec::with_capacity(total);

    // Length-prefix each postcard chunk so the unpacker knows its size.
    out.extend_from_slice(&(header_bytes.len() as u32).to_le_bytes());
    out.extend_from_slice(&header_bytes);
    out.extend_from_slice(&(descs_bytes.len() as u32).to_le_bytes());
    out.extend_from_slice(&descs_bytes);
    for t in tensors {
        out.extend_from_slice(t.bytes);
    }
    out.extend_from_slice(aux);
    Ok(out)
}

/// Unpack a request buffer into header + descriptors + per-tensor byte slices + aux bytes.
/// Returns owned bytes for tensors so the caller can reuse the input buffer.
pub fn unpack_request(
    buf: &[u8],
) -> Result<(RpcRequestHeader, Vec<TensorDesc>, Vec<Vec<u8>>, Vec<u8>), CodecError> {
    let mut cursor = 0usize;
    let read_u32 = |cur: &mut usize, buf: &[u8]| -> Result<u32, CodecError> {
        if buf.len() < *cur + 4 {
            return Err(CodecError::Truncated {
                expected: *cur + 4,
                got: buf.len(),
            });
        }
        let v = u32::from_le_bytes([buf[*cur], buf[*cur + 1], buf[*cur + 2], buf[*cur + 3]]);
        *cur += 4;
        Ok(v)
    };

    let header_len = read_u32(&mut cursor, buf)? as usize;
    if buf.len() < cursor + header_len {
        return Err(CodecError::Truncated {
            expected: cursor + header_len,
            got: buf.len(),
        });
    }
    let header: RpcRequestHeader = postcard::from_bytes(&buf[cursor..cursor + header_len])?;
    cursor += header_len;

    let descs_len = read_u32(&mut cursor, buf)? as usize;
    if buf.len() < cursor + descs_len {
        return Err(CodecError::Truncated {
            expected: cursor + descs_len,
            got: buf.len(),
        });
    }
    let descs: Vec<TensorDesc> = postcard::from_bytes(&buf[cursor..cursor + descs_len])?;
    cursor += descs_len;

    if descs.len() != header.n_tensors as usize {
        return Err(CodecError::DescriptorCountMismatch {
            header: header.n_tensors,
            got: descs.len(),
        });
    }

    let mut tensors = Vec::with_capacity(descs.len());
    for d in &descs {
        let n = d.n_bytes as usize;
        if buf.len() < cursor + n {
            return Err(CodecError::Truncated {
                expected: cursor + n,
                got: buf.len(),
            });
        }
        tensors.push(buf[cursor..cursor + n].to_vec());
        cursor += n;
    }

    let aux_len = header.aux_payload_bytes as usize;
    if buf.len() < cursor + aux_len {
        return Err(CodecError::Truncated {
            expected: cursor + aux_len,
            got: buf.len(),
        });
    }
    let aux = buf[cursor..cursor + aux_len].to_vec();
    Ok((header, descs, tensors, aux))
}
```

- [ ] **Step 4: Run — should pass**

Run:
```bash
cargo test -p sakura-wire --test codec_roundtrip
```
Expected:
```
running 2 tests
test pack_request_with_zero_tensors_works ... ok
test pack_then_unpack_preserves_request ... ok
```

- [ ] **Step 5: Commit**

```bash
git add crates/sakura-wire/src/codec/mod.rs crates/sakura-wire/tests/codec_roundtrip.rs
git commit -m "feat(wire): pack_request / unpack_request — full codec round-trip

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 6: Protocol — handler IDs + WireError

**Files:**
- Create: `crates/sakura-wire/src/protocol/mod.rs`, `crates/sakura-wire/src/protocol/handlers.rs`, `crates/sakura-wire/src/protocol/error.rs`
- Modify: `crates/sakura-wire/src/lib.rs`

- [ ] **Step 1: Write `protocol/handlers.rs`**

Create `crates/sakura-wire/src/protocol/handlers.rs`:
```rust
//! Well-known handler IDs.

pub const HANDLER_EXEC_CLOUDPICKLED: u32 = 0x0001;
pub const HANDLER_MODEL_CACHE_GET:   u32 = 0x0002;
pub const HANDLER_HEARTBEAT:         u32 = 0x0003;
pub const HANDLER_SHUTDOWN:          u32 = 0x0004;
pub const HANDLER_SAVE_BLOB:         u32 = 0x0005;

/// First handler ID available for user-registered handlers.
pub const HANDLER_CUSTOM_BASE: u32 = 0x1000;

/// Echo handler used in tests: the worker bounces all input tensors back
/// (with byte-identical content) and an empty aux payload.
pub const HANDLER_ECHO: u32 = 0xDEAD;
```

- [ ] **Step 2: Write `protocol/error.rs` with the failing test**

Create `crates/sakura-wire/src/protocol/error.rs`:
```rust
//! Wire-level error types. Maps to Python exceptions in pyo3_bindings.

use serde::{Deserialize, Serialize};
use thiserror::Error;

#[derive(Debug, Clone, Error, Serialize, Deserialize)]
pub enum WireError {
    #[error("handler {handler_id:#x} not found")]
    HandlerNotFound { handler_id: u32 },

    #[error("decode failed in {what}: {detail}")]
    DecodeFailed { what: String, detail: String },

    #[error("handler panicked: {msg}")]
    HandlerPanic { msg: String, trace: Vec<String> },

    #[error("timeout after {deadline_ms}ms")]
    Timeout { deadline_ms: u32 },

    #[error("worker crashed")]
    WorkerCrashed,

    #[error("backpressure saturated")]
    BackpressureSaturated,
}

#[cfg(test)]
mod tests {
    use super::*;
    use postcard::{from_bytes, to_allocvec};

    #[test]
    fn wire_error_roundtrip() {
        let cases = vec![
            WireError::HandlerNotFound { handler_id: 0xCAFE },
            WireError::DecodeFailed {
                what: "header".into(),
                detail: "bad varint".into(),
            },
            WireError::HandlerPanic {
                msg: "boom".into(),
                trace: vec!["frame_a".into(), "frame_b".into()],
            },
            WireError::Timeout { deadline_ms: 5000 },
            WireError::WorkerCrashed,
            WireError::BackpressureSaturated,
        ];
        for original in cases {
            let bytes = to_allocvec(&original).unwrap();
            let decoded: WireError = from_bytes(&bytes).unwrap();
            // round-trip via Display + variant equality (Display gives a stable string)
            assert_eq!(decoded.to_string(), original.to_string());
        }
    }
}
```

- [ ] **Step 3: Write `protocol/mod.rs` to export the modules**

Create `crates/sakura-wire/src/protocol/mod.rs`:
```rust
//! RPC protocol layer: handler IDs, error types, request/response framing.

pub mod error;
pub mod handlers;

pub use error::WireError;
pub use handlers::{
    HANDLER_CUSTOM_BASE, HANDLER_ECHO, HANDLER_EXEC_CLOUDPICKLED, HANDLER_HEARTBEAT,
    HANDLER_MODEL_CACHE_GET, HANDLER_SAVE_BLOB, HANDLER_SHUTDOWN,
};
```

- [ ] **Step 4: Add the protocol module to lib.rs**

Edit `crates/sakura-wire/src/lib.rs` — add after `pub mod codec;`:
```rust
pub mod protocol;
```

The full lib.rs is now:
```rust
//! sakura-wire: zero-copy tensor codec + RPC over QUIC.

#![deny(unsafe_op_in_unsafe_fn)]

use pyo3::prelude::*;

pub mod codec;
pub mod protocol;

#[pymodule]
fn sakura_wire(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    Ok(())
}
```

- [ ] **Step 5: Run the protocol tests — should pass**

Run:
```bash
cargo test -p sakura-wire protocol::error::tests
```
Expected:
```
running 1 test
test protocol::error::tests::wire_error_roundtrip ... ok
```

- [ ] **Step 6: Commit**

```bash
git add crates/sakura-wire/src/protocol/ crates/sakura-wire/src/lib.rs
git commit -m "feat(wire): protocol layer — handler IDs + WireError variants

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 7: Tokio runtime owned by Rust (background thread)

**Files:**
- Create: `crates/sakura-wire/src/runtime.rs`
- Modify: `crates/sakura-wire/src/lib.rs`

- [ ] **Step 1: Write the failing test (uses runtime::WireRuntime)**

Create `crates/sakura-wire/src/runtime.rs`:
```rust
//! Tokio multi-thread runtime owned by the crate, hosted in a Rust-managed
//! OS thread so Python's GIL is never blocked by async I/O.

use std::sync::{Arc, OnceLock};
use tokio::runtime::{Builder, Runtime};

/// Lazily-built shared tokio runtime. All async work in sakura-wire runs here.
pub struct WireRuntime {
    rt: Arc<Runtime>,
}

impl WireRuntime {
    pub fn shared() -> &'static WireRuntime {
        static SHARED: OnceLock<WireRuntime> = OnceLock::new();
        SHARED.get_or_init(|| WireRuntime {
            rt: Arc::new(
                Builder::new_multi_thread()
                    .worker_threads(num_cpus_capped(8))
                    .thread_name("sakura-wire")
                    .enable_all()
                    .build()
                    .expect("failed to build tokio runtime"),
            ),
        })
    }

    pub fn handle(&self) -> &tokio::runtime::Handle {
        self.rt.handle()
    }

    pub fn spawn<F>(&self, fut: F) -> tokio::task::JoinHandle<F::Output>
    where
        F: std::future::Future + Send + 'static,
        F::Output: Send + 'static,
    {
        self.rt.spawn(fut)
    }

    pub fn block_on<F: std::future::Future>(&self, fut: F) -> F::Output {
        self.rt.block_on(fut)
    }
}

fn num_cpus_capped(cap: usize) -> usize {
    std::thread::available_parallelism()
        .map(|n| n.get().min(cap).max(2))
        .unwrap_or(2)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    #[test]
    fn shared_runtime_is_singleton() {
        let a = WireRuntime::shared() as *const WireRuntime;
        let b = WireRuntime::shared() as *const WireRuntime;
        assert_eq!(a, b, "WireRuntime::shared must return the same instance");
    }

    #[test]
    fn spawn_runs_to_completion() {
        let counter = Arc::new(AtomicUsize::new(0));
        let c2 = Arc::clone(&counter);
        let handle = WireRuntime::shared().spawn(async move {
            c2.fetch_add(1, Ordering::SeqCst);
            42
        });
        let result = WireRuntime::shared().block_on(handle).expect("join");
        assert_eq!(result, 42);
        assert_eq!(counter.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn block_on_yields_value() {
        let v = WireRuntime::shared().block_on(async { 1 + 1 });
        assert_eq!(v, 2);
    }
}
```

- [ ] **Step 2: Add the runtime module to lib.rs**

Edit `crates/sakura-wire/src/lib.rs` — add `pub mod runtime;` so it becomes:
```rust
//! sakura-wire: zero-copy tensor codec + RPC over QUIC.

#![deny(unsafe_op_in_unsafe_fn)]

use pyo3::prelude::*;

pub mod codec;
pub mod protocol;
pub mod runtime;

#[pymodule]
fn sakura_wire(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    Ok(())
}
```

- [ ] **Step 3: Run the tests — should pass**

Run:
```bash
cargo test -p sakura-wire runtime::tests
```
Expected:
```
running 3 tests
test runtime::tests::block_on_yields_value ... ok
test runtime::tests::shared_runtime_is_singleton ... ok
test runtime::tests::spawn_runs_to_completion ... ok
```

- [ ] **Step 4: Commit**

```bash
git add crates/sakura-wire/src/runtime.rs crates/sakura-wire/src/lib.rs
git commit -m "feat(wire): WireRuntime — shared tokio multi-thread runtime owned by Rust

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 8: Transport — QUIC server + client (loopback, self-signed TLS)

**Files:**
- Create: `crates/sakura-wire/src/transport/mod.rs`, `crates/sakura-wire/src/transport/quic.rs`
- Modify: `crates/sakura-wire/src/lib.rs`

- [ ] **Step 1: Write `transport/quic.rs` with a server + client API**

Create `crates/sakura-wire/src/transport/quic.rs`:
```rust
//! QUIC transport via quinn. Self-signed TLS for loopback by default.

use quinn::{Connection, Endpoint, ServerConfig, TransportConfig};
use rcgen::generate_simple_self_signed;
use std::net::SocketAddr;
use std::sync::Arc;
use tokio::io::AsyncWriteExt;

use crate::protocol::WireError;

#[derive(Debug, thiserror::Error)]
pub enum TransportError {
    #[error("io: {0}")]
    Io(#[from] std::io::Error),
    #[error("quinn: {0}")]
    Quinn(String),
    #[error("rcgen: {0}")]
    Rcgen(String),
    #[error("rustls: {0}")]
    Rustls(String),
    #[error("connect: {0}")]
    Connect(String),
    #[error("read: {0}")]
    Read(String),
    #[error("write: {0}")]
    Write(String),
    #[error("wire: {0}")]
    Wire(#[from] WireError),
}

impl From<rcgen::Error> for TransportError {
    fn from(e: rcgen::Error) -> Self {
        TransportError::Rcgen(e.to_string())
    }
}

impl From<rustls::Error> for TransportError {
    fn from(e: rustls::Error) -> Self {
        TransportError::Rustls(e.to_string())
    }
}

impl From<quinn::ConnectionError> for TransportError {
    fn from(e: quinn::ConnectionError) -> Self {
        TransportError::Quinn(e.to_string())
    }
}

impl From<quinn::ConnectError> for TransportError {
    fn from(e: quinn::ConnectError) -> Self {
        TransportError::Connect(e.to_string())
    }
}

impl From<quinn::ReadExactError> for TransportError {
    fn from(e: quinn::ReadExactError) -> Self {
        TransportError::Read(e.to_string())
    }
}

impl From<quinn::WriteError> for TransportError {
    fn from(e: quinn::WriteError) -> Self {
        TransportError::Write(e.to_string())
    }
}

/// A self-signed TLS pair (DER-encoded cert + key) for loopback testing.
#[derive(Clone)]
pub struct SelfSignedPair {
    pub cert_der: Vec<u8>,
    pub key_der: Vec<u8>,
}

pub fn generate_self_signed(subject: &str) -> Result<SelfSignedPair, TransportError> {
    let cert = generate_simple_self_signed(vec![subject.into()])?;
    Ok(SelfSignedPair {
        cert_der: cert.serialize_der()?,
        key_der: cert.serialize_private_key_der(),
    })
}

fn server_config(pair: &SelfSignedPair) -> Result<ServerConfig, TransportError> {
    let cert_chain = vec![rustls::Certificate(pair.cert_der.clone())];
    let key = rustls::PrivateKey(pair.key_der.clone());
    let mut cfg = ServerConfig::with_single_cert(cert_chain, key)
        .map_err(|e| TransportError::Rustls(e.to_string()))?;
    let mut transport = TransportConfig::default();
    transport.max_concurrent_uni_streams(0u8.into());
    cfg.transport = Arc::new(transport);
    Ok(cfg)
}

fn client_config(trusted_cert: &[u8]) -> Result<quinn::ClientConfig, TransportError> {
    let mut roots = rustls::RootCertStore::empty();
    roots.add(&rustls::Certificate(trusted_cert.to_vec()))?;
    let crypto = rustls::ClientConfig::builder()
        .with_safe_defaults()
        .with_root_certificates(roots)
        .with_no_client_auth();
    let mut cfg = quinn::ClientConfig::new(Arc::new(crypto));
    let mut transport = TransportConfig::default();
    transport.max_concurrent_uni_streams(0u8.into());
    cfg.transport_config(Arc::new(transport));
    Ok(cfg)
}

/// Bind a QUIC server to `bind_addr`, returning the actual address it listens on.
pub fn bind_server(
    bind_addr: SocketAddr,
    pair: &SelfSignedPair,
) -> Result<Endpoint, TransportError> {
    let cfg = server_config(pair)?;
    let endpoint = Endpoint::server(cfg, bind_addr)?;
    Ok(endpoint)
}

/// Connect to a server with a known cert (loopback / pinned-cert use case).
pub async fn connect(
    server_addr: SocketAddr,
    server_name: &str,
    trusted_cert: &[u8],
) -> Result<Connection, TransportError> {
    let mut endpoint = Endpoint::client("0.0.0.0:0".parse().unwrap())?;
    endpoint.set_default_client_config(client_config(trusted_cert)?);
    let conn = endpoint.connect(server_addr, server_name)?.await?;
    Ok(conn)
}

/// Open a bidirectional stream, write the request bytes, signal end-of-write,
/// then read the entire response and return it.
pub async fn rpc_call(conn: &Connection, request_bytes: Vec<u8>) -> Result<Vec<u8>, TransportError> {
    let (mut send, mut recv) = conn.open_bi().await?;
    send.write_all(&request_bytes).await?;
    send.finish().await.map_err(|e| TransportError::Write(e.to_string()))?;
    let resp = recv
        .read_to_end(64 * 1024 * 1024 * 1024)
        .await
        .map_err(|e| TransportError::Read(e.to_string()))?;
    Ok(resp)
}

/// Server side: accept the next bidirectional stream and read the entire request.
pub async fn accept_request(
    conn: &Connection,
) -> Result<(quinn::SendStream, Vec<u8>), TransportError> {
    let (send, mut recv) = conn.accept_bi().await.map_err(|e| TransportError::Quinn(e.to_string()))?;
    let req = recv
        .read_to_end(64 * 1024 * 1024 * 1024)
        .await
        .map_err(|e| TransportError::Read(e.to_string()))?;
    Ok((send, req))
}

/// Server side: write a response and finish the stream.
pub async fn send_response(
    mut send: quinn::SendStream,
    bytes: Vec<u8>,
) -> Result<(), TransportError> {
    send.write_all(&bytes).await?;
    send.finish().await.map_err(|e| TransportError::Write(e.to_string()))?;
    Ok(())
}
```

- [ ] **Step 2: Create `transport/mod.rs`**

Create `crates/sakura-wire/src/transport/mod.rs`:
```rust
//! Transport layer: QUIC over UDP via quinn (default).

pub mod quic;

pub use quic::{
    accept_request, bind_server, connect, generate_self_signed, rpc_call, send_response,
    SelfSignedPair, TransportError,
};
```

- [ ] **Step 3: Add transport to lib.rs**

Edit `crates/sakura-wire/src/lib.rs` to add `pub mod transport;`:
```rust
//! sakura-wire: zero-copy tensor codec + RPC over QUIC.

#![deny(unsafe_op_in_unsafe_fn)]

use pyo3::prelude::*;

pub mod codec;
pub mod protocol;
pub mod runtime;
pub mod transport;

#[pymodule]
fn sakura_wire(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    Ok(())
}
```

- [ ] **Step 4: Verify the crate compiles**

Run:
```bash
cargo check -p sakura-wire
```
Expected: clean compile, no warnings.

- [ ] **Step 5: Commit**

```bash
git add crates/sakura-wire/src/transport/ crates/sakura-wire/src/lib.rs
git commit -m "feat(wire): QUIC transport via quinn — self-signed TLS, bind/connect/rpc helpers

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 9: Transport — loopback round-trip integration test

**Files:**
- Create: `crates/sakura-wire/tests/quic_loopback.rs`

- [ ] **Step 1: Write the failing integration test (full server+client RTT over loopback)**

Create `crates/sakura-wire/tests/quic_loopback.rs`:
```rust
//! End-to-end test: bind a QUIC server on loopback, connect a client,
//! send a request, server echoes it back, client receives. No PyO3 yet —
//! this verifies the Rust transport in isolation.

use sakura_wire::runtime::WireRuntime;
use sakura_wire::transport::{
    accept_request, bind_server, connect, generate_self_signed, rpc_call, send_response,
};
use std::net::SocketAddr;
use tokio::sync::oneshot;

#[test]
fn loopback_echo_round_trip() {
    let rt = WireRuntime::shared();
    rt.block_on(async {
        let pair = generate_self_signed("localhost").expect("cert");
        let cert_for_client = pair.cert_der.clone();

        let server_addr: SocketAddr = "127.0.0.1:0".parse().unwrap();
        let endpoint = bind_server(server_addr, &pair).expect("bind");
        let local_addr = endpoint.local_addr().unwrap();

        // Server task: accept one connection, echo the next request bytes.
        let (server_done_tx, server_done_rx) = oneshot::channel();
        tokio::spawn(async move {
            let incoming = endpoint.accept().await.expect("accept");
            let conn = incoming.await.expect("connection");
            let (send, req_bytes) = accept_request(&conn).await.expect("accept_request");
            // Echo back identical bytes.
            send_response(send, req_bytes).await.expect("send_response");
            let _ = server_done_tx.send(());
        });

        // Client.
        let conn = connect(local_addr, "localhost", &cert_for_client)
            .await
            .expect("connect");
        let payload: Vec<u8> = (0u8..200).cycle().take(4096).collect();
        let resp = rpc_call(&conn, payload.clone()).await.expect("rpc");
        assert_eq!(resp.len(), payload.len());
        assert_eq!(resp, payload);
        // Wait for server task to finish.
        let _ = tokio::time::timeout(std::time::Duration::from_secs(5), server_done_rx).await;
    });
}
```

- [ ] **Step 2: Run — should pass**

Run:
```bash
cargo test -p sakura-wire --test quic_loopback -- --test-threads=1
```
Expected:
```
running 1 test
test loopback_echo_round_trip ... ok
```

(Single-threaded because the test allocates a UDP port; run in isolation to avoid flake.)

- [ ] **Step 3: Commit**

```bash
git add crates/sakura-wire/tests/quic_loopback.rs
git commit -m "test(wire): loopback QUIC round-trip — bind server + connect client + echo

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 10: PyO3 — `Future`, `Result`, `Dispatcher`, `TlsConfig`

**Files:**
- Create: `crates/sakura-wire/src/pyo3_bindings.rs`
- Modify: `crates/sakura-wire/src/lib.rs`

- [ ] **Step 1: Implement the PyO3 bindings**

Create `crates/sakura-wire/src/pyo3_bindings.rs`:
```rust
//! PyO3 bindings: Dispatcher, Future, Result, TlsConfig, WorkerSupervisor.

use pyo3::exceptions::{PyRuntimeError, PyTimeoutError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyList};
use std::net::SocketAddr;
use std::sync::{Arc, Mutex};
use std::time::Duration;
use tokio::sync::oneshot;

use crate::codec::{pack_request, OwnedTensor, RpcRequestHeader, TensorView, WireVersion};
use crate::protocol::WireError;
use crate::runtime::WireRuntime;
use crate::transport::{connect, rpc_call, TransportError};

#[derive(Debug, thiserror::Error)]
enum PyWireError {
    #[error("transport error: {0}")]
    Transport(#[from] TransportError),
    #[error("wire error: {0}")]
    Wire(#[from] WireError),
    #[error("URI parse error: {0}")]
    UriParse(String),
    #[error("future already consumed")]
    AlreadyConsumed,
    #[error("future cancelled")]
    Cancelled,
    #[error("timeout")]
    Timeout,
    #[error("buffer protocol error: {0}")]
    Buffer(String),
}

impl From<PyWireError> for PyErr {
    fn from(e: PyWireError) -> Self {
        match &e {
            PyWireError::Timeout => PyTimeoutError::new_err(e.to_string()),
            PyWireError::UriParse(_) | PyWireError::Buffer(_) => PyValueError::new_err(e.to_string()),
            _ => PyRuntimeError::new_err(e.to_string()),
        }
    }
}

#[pyclass(name = "TlsConfig")]
#[derive(Clone)]
pub struct PyTlsConfig {
    cert_der: Vec<u8>,
    server_name: String,
}

#[pymethods]
impl PyTlsConfig {
    #[new]
    fn new(cert_der: Vec<u8>, server_name: String) -> Self {
        Self { cert_der, server_name }
    }
}

#[pyclass(name = "Result")]
pub struct PyRpcResult {
    #[pyo3(get)]
    pub elapsed_us: u64,
    aux: Vec<u8>,
    tensors: Vec<OwnedTensor>,
}

#[pymethods]
impl PyRpcResult {
    #[getter]
    fn aux<'py>(&self, py: Python<'py>) -> Bound<'py, PyBytes> {
        PyBytes::new_bound(py, &self.aux)
    }

    /// Returns a list of `bytes` objects (one per tensor). Plan 1: bytes-for-bytes
    /// fidelity is what callers need; Plan 2 layers numpy/torch unpacking on top.
    fn tensors<'py>(&self, py: Python<'py>) -> Bound<'py, PyList> {
        let items: Vec<Bound<PyBytes>> = self
            .tensors
            .iter()
            .map(|t| PyBytes::new_bound(py, &t.bytes))
            .collect();
        PyList::new_bound(py, items)
    }
}

type PendingChannel = Arc<Mutex<Option<oneshot::Receiver<Result<PyRpcResult, PyWireError>>>>>;

#[pyclass(name = "Future", unsendable)]
pub struct PyFuture {
    rx: PendingChannel,
    cancelled: Arc<Mutex<bool>>,
}

#[pymethods]
impl PyFuture {
    #[pyo3(signature = (timeout = None))]
    fn result(&self, py: Python<'_>, timeout: Option<f64>) -> PyResult<PyRpcResult> {
        let rx = self.rx.lock().unwrap().take().ok_or(PyWireError::AlreadyConsumed)?;
        py.allow_threads(move || -> PyResult<PyRpcResult> {
            let outcome = match timeout {
                None => WireRuntime::shared().block_on(async { rx.await }),
                Some(secs) => {
                    let dur = Duration::from_secs_f64(secs);
                    WireRuntime::shared().block_on(async move {
                        match tokio::time::timeout(dur, rx).await {
                            Ok(v) => v,
                            Err(_) => Ok(Err(PyWireError::Timeout)),
                        }
                    })
                }
            };
            match outcome {
                Ok(Ok(v)) => Ok(v),
                Ok(Err(e)) => Err(e.into()),
                Err(_) => Err(PyWireError::Cancelled.into()),
            }
        })
    }

    fn cancel(&self) -> bool {
        let mut c = self.cancelled.lock().unwrap();
        let was = *c;
        *c = true;
        !was
    }

    fn done(&self) -> bool {
        let g = self.rx.lock().unwrap();
        match &*g {
            Some(rx) => rx.is_terminated() || rx.is_closed(),
            None => true,
        }
    }
}

#[pyclass(name = "Dispatcher")]
pub struct PyDispatcher {
    target_uri: String,
    tls: PyTlsConfig,
}

#[pymethods]
impl PyDispatcher {
    /// Construct a dispatcher pointing at `quic://host:port`.
    #[new]
    #[pyo3(signature = (target_uri, tls))]
    fn new(target_uri: String, tls: PyTlsConfig) -> PyResult<Self> {
        if !target_uri.starts_with("quic://") {
            return Err(
                PyWireError::UriParse(format!("only quic:// is supported in v1, got: {target_uri}"))
                    .into(),
            );
        }
        Ok(Self { target_uri, tls })
    }

    /// Submit an RPC. `tensors` is a list of `(shape: list[int], dtype_id: int, device_id: int, data: bytes)` tuples;
    /// `aux_payload` is opaque bytes (cloudpickle in the calling layer).
    #[pyo3(signature = (handler_id, tensors, aux_payload, timeout_ms = None))]
    fn submit(
        &self,
        py: Python<'_>,
        handler_id: u32,
        tensors: Vec<TensorTuple>,
        aux_payload: Vec<u8>,
        timeout_ms: Option<u32>,
    ) -> PyResult<PyFuture> {
        let target = parse_quic_uri(&self.target_uri)?;
        let cert = self.tls.cert_der.clone();
        let server_name = self.tls.server_name.clone();

        let header = RpcRequestHeader {
            version: WireVersion::V1,
            request_id: next_request_id(),
            handler_id,
            n_tensors: tensors.len() as u32,
            aux_payload_bytes: aux_payload.len() as u32,
            deadline_ms: timeout_ms,
            trace_id: 0,
        };
        // Pack on the calling thread (CPU work; OK to hold the GIL briefly here).
        let tensor_views: Vec<TensorView<'_>> = tensors.iter().map(TensorTuple::as_view).collect();
        let request_bytes = pack_request(&header, &tensor_views, &aux_payload)
            .map_err(|e| PyRuntimeError::new_err(format!("pack: {e}")))?;
        // Drop the views (and thus the borrow on `tensors`) before spawning.
        drop(tensor_views);
        // Move owned tensors + aux out of the Python scope.
        let _keep_alive = py.allow_threads(|| {});

        let (tx, rx) = oneshot::channel::<Result<PyRpcResult, PyWireError>>();
        let cancelled = Arc::new(Mutex::new(false));
        let cancelled_clone = Arc::clone(&cancelled);

        WireRuntime::shared().spawn(async move {
            let result = run_rpc(&target, &server_name, &cert, request_bytes).await;
            if !*cancelled_clone.lock().unwrap() {
                let _ = tx.send(result);
            }
        });

        Ok(PyFuture {
            rx: Arc::new(Mutex::new(Some(rx))),
            cancelled,
        })
    }
}

/// Minimal Python representation of one tensor:
///   (shape: List[int], dtype_id: int, device_id: int, data: bytes)
/// dtype_id and device_id correspond to the Dtype/Device repr(u8) values.
#[derive(FromPyObject)]
struct TensorTuple {
    shape: Vec<u32>,
    dtype_id: u8,
    device_id: u8,
    data: Vec<u8>,
}

impl TensorTuple {
    fn as_view(&self) -> TensorView<'_> {
        use crate::codec::{Device, Dtype};
        let dtype = match self.dtype_id {
            0 => Dtype::F32,
            1 => Dtype::F16,
            2 => Dtype::BF16,
            3 => Dtype::F8E4M3,
            10 => Dtype::I64,
            11 => Dtype::I32,
            12 => Dtype::U8,
            13 => Dtype::Bool,
            _ => Dtype::U8,
        };
        let device = match self.device_id {
            0 => Device::Cpu,
            other => Device::Cuda(other - 1),
        };
        TensorView::new(self.shape.clone(), dtype, device, &self.data)
    }
}

fn parse_quic_uri(uri: &str) -> PyResult<SocketAddr> {
    // Parse "quic://host:port" → SocketAddr (IPv4/IPv6 supported via std parser).
    let stripped = uri
        .strip_prefix("quic://")
        .ok_or_else(|| PyWireError::UriParse(format!("not a quic:// uri: {uri}")))?;
    stripped
        .parse::<SocketAddr>()
        .map_err(|e| PyWireError::UriParse(format!("bad quic uri {uri}: {e}")).into())
}

fn next_request_id() -> u64 {
    use std::sync::atomic::{AtomicU64, Ordering};
    static NEXT: AtomicU64 = AtomicU64::new(1);
    NEXT.fetch_add(1, Ordering::Relaxed)
}

async fn run_rpc(
    target: &SocketAddr,
    server_name: &str,
    trusted_cert: &[u8],
    request_bytes: Vec<u8>,
) -> Result<PyRpcResult, PyWireError> {
    let conn = connect(*target, server_name, trusted_cert).await?;
    let resp = rpc_call(&conn, request_bytes).await?;
    parse_response(&resp)
}

fn parse_response(buf: &[u8]) -> Result<PyRpcResult, PyWireError> {
    use crate::codec::{RpcResponseHeader, TensorDesc};
    let mut cursor = 0usize;
    let read_u32 = |cur: &mut usize, buf: &[u8]| -> Result<u32, PyWireError> {
        if buf.len() < *cur + 4 {
            return Err(PyWireError::Wire(WireError::DecodeFailed {
                what: "response prefix".into(),
                detail: "truncated".into(),
            }));
        }
        let v = u32::from_le_bytes([buf[*cur], buf[*cur + 1], buf[*cur + 2], buf[*cur + 3]]);
        *cur += 4;
        Ok(v)
    };
    let h_len = read_u32(&mut cursor, buf)? as usize;
    let header: RpcResponseHeader =
        postcard::from_bytes(&buf[cursor..cursor + h_len]).map_err(|e| {
            PyWireError::Wire(WireError::DecodeFailed {
                what: "response header".into(),
                detail: e.to_string(),
            })
        })?;
    cursor += h_len;
    let descs_len = read_u32(&mut cursor, buf)? as usize;
    let descs: Vec<TensorDesc> =
        postcard::from_bytes(&buf[cursor..cursor + descs_len]).map_err(|e| {
            PyWireError::Wire(WireError::DecodeFailed {
                what: "response descriptors".into(),
                detail: e.to_string(),
            })
        })?;
    cursor += descs_len;
    let mut tensors = Vec::with_capacity(descs.len());
    for d in descs {
        let n = d.n_bytes as usize;
        if buf.len() < cursor + n {
            return Err(PyWireError::Wire(WireError::DecodeFailed {
                what: "response tensor bytes".into(),
                detail: format!("expected {n}, got {}", buf.len() - cursor),
            }));
        }
        let bytes = buf[cursor..cursor + n].to_vec();
        cursor += n;
        tensors.push(OwnedTensor::new(d, bytes));
    }
    let aux_len = header.aux_payload_bytes as usize;
    if buf.len() < cursor + aux_len {
        return Err(PyWireError::Wire(WireError::DecodeFailed {
            what: "response aux".into(),
            detail: format!("expected {aux_len}, got {}", buf.len() - cursor),
        }));
    }
    let aux = buf[cursor..cursor + aux_len].to_vec();
    Ok(PyRpcResult {
        elapsed_us: header.elapsed_us,
        aux,
        tensors,
    })
}
```

- [ ] **Step 2: Wire the bindings into the `#[pymodule]`**

Replace `crates/sakura-wire/src/lib.rs` with:
```rust
//! sakura-wire: zero-copy tensor codec + RPC over QUIC.

#![deny(unsafe_op_in_unsafe_fn)]

use pyo3::prelude::*;

pub mod codec;
pub mod protocol;
pub mod pyo3_bindings;
pub mod runtime;
pub mod transport;

#[pymodule]
fn sakura_wire(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    m.add_class::<pyo3_bindings::PyDispatcher>()?;
    m.add_class::<pyo3_bindings::PyFuture>()?;
    m.add_class::<pyo3_bindings::PyRpcResult>()?;
    m.add_class::<pyo3_bindings::PyTlsConfig>()?;
    Ok(())
}
```

- [ ] **Step 3: Build and verify the import**

Run:
```bash
maturin develop
python -c "import sakura_wire; print(sakura_wire.Dispatcher, sakura_wire.Future, sakura_wire.Result, sakura_wire.TlsConfig)"
```
Expected: prints four `<class 'builtins.…'>` lines.

- [ ] **Step 4: Commit**

```bash
git add crates/sakura-wire/src/pyo3_bindings.rs crates/sakura-wire/src/lib.rs
git commit -m "feat(wire): PyO3 bindings — Dispatcher, Future, Result, TlsConfig

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 11: `WorkerSupervisor` — spawn + manage a Python worker subprocess

**Files:**
- Create: `crates/sakura-wire/src/supervisor.rs`
- Modify: `crates/sakura-wire/src/lib.rs`, `crates/sakura-wire/src/pyo3_bindings.rs`

- [ ] **Step 1: Implement the supervisor in pure Rust**

Create `crates/sakura-wire/src/supervisor.rs`:
```rust
//! WorkerSupervisor: spawn `sakura-worker` subprocesses, harvest their listen URI
//! from stdout, and shut them down cleanly.

use std::collections::HashMap;
use std::io::{BufRead, BufReader};
use std::process::{Child, Command, Stdio};
use std::sync::Mutex;
use std::time::Duration;

#[derive(Debug, thiserror::Error)]
pub enum SupervisorError {
    #[error("spawn: {0}")]
    Spawn(#[from] std::io::Error),
    #[error("worker exited before reporting URI")]
    EarlyExit,
    #[error("did not receive listen URI within {0:?}")]
    Timeout(Duration),
    #[error("invalid listen URI: {0}")]
    BadUri(String),
}

pub struct WorkerHandle {
    pub uri: String,
    pub cert_der: Vec<u8>,
    child: Mutex<Child>,
}

impl WorkerHandle {
    pub fn shutdown(&self, timeout: Duration) {
        let _ = self.child.lock().unwrap().kill();
        let mut waited = Duration::ZERO;
        let step = Duration::from_millis(50);
        while waited < timeout {
            if self.child.lock().unwrap().try_wait().ok().flatten().is_some() {
                return;
            }
            std::thread::sleep(step);
            waited += step;
        }
        let _ = self.child.lock().unwrap().wait();
    }
}

/// Spawn the worker, read lines from its stdout until the magic line
/// `SAKURA_WORKER_LISTENING <uri> <cert_hex>` appears, then return its handle.
pub fn spawn_worker(
    cmd: &[String],
    extra_env: HashMap<String, String>,
    startup_timeout: Duration,
) -> Result<WorkerHandle, SupervisorError> {
    if cmd.is_empty() {
        return Err(SupervisorError::BadUri("empty cmd".into()));
    }
    let mut command = Command::new(&cmd[0]);
    command
        .args(&cmd[1..])
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::inherit());
    for (k, v) in extra_env {
        command.env(k, v);
    }
    let mut child = command.spawn()?;
    let stdout = child.stdout.take().ok_or_else(|| {
        SupervisorError::Spawn(std::io::Error::new(
            std::io::ErrorKind::Other,
            "no stdout",
        ))
    })?;

    let (tx, rx) = std::sync::mpsc::channel::<Result<(String, Vec<u8>), SupervisorError>>();
    std::thread::spawn(move || {
        let reader = BufReader::new(stdout);
        for line in reader.lines() {
            match line {
                Ok(line) if line.starts_with("SAKURA_WORKER_LISTENING ") => {
                    let parts: Vec<&str> = line.splitn(3, ' ').collect();
                    if parts.len() != 3 {
                        let _ = tx.send(Err(SupervisorError::BadUri(line)));
                        return;
                    }
                    let uri = parts[1].to_string();
                    let cert_hex = parts[2];
                    match decode_hex(cert_hex) {
                        Ok(cert) => {
                            let _ = tx.send(Ok((uri, cert)));
                        }
                        Err(e) => {
                            let _ = tx.send(Err(SupervisorError::BadUri(format!(
                                "bad cert hex: {e}"
                            ))));
                        }
                    }
                    return;
                }
                Ok(_) => continue,
                Err(_) => {
                    let _ = tx.send(Err(SupervisorError::EarlyExit));
                    return;
                }
            }
        }
        let _ = tx.send(Err(SupervisorError::EarlyExit));
    });

    match rx.recv_timeout(startup_timeout) {
        Ok(Ok((uri, cert_der))) => Ok(WorkerHandle {
            uri,
            cert_der,
            child: Mutex::new(child),
        }),
        Ok(Err(e)) => {
            let _ = child.kill();
            Err(e)
        }
        Err(_) => {
            let _ = child.kill();
            Err(SupervisorError::Timeout(startup_timeout))
        }
    }
}

fn decode_hex(s: &str) -> Result<Vec<u8>, String> {
    if s.len() % 2 != 0 {
        return Err("odd-length hex".into());
    }
    let mut out = Vec::with_capacity(s.len() / 2);
    for i in (0..s.len()).step_by(2) {
        let byte =
            u8::from_str_radix(&s[i..i + 2], 16).map_err(|e| format!("bad hex byte: {e}"))?;
        out.push(byte);
    }
    Ok(out)
}

pub fn encode_hex(bytes: &[u8]) -> String {
    let mut s = String::with_capacity(bytes.len() * 2);
    for b in bytes {
        s.push_str(&format!("{b:02x}"));
    }
    s
}
```

- [ ] **Step 2: Add the supervisor module + PyO3 wrapper**

Edit `crates/sakura-wire/src/lib.rs` — add `pub mod supervisor;`:
```rust
//! sakura-wire: zero-copy tensor codec + RPC over QUIC.

#![deny(unsafe_op_in_unsafe_fn)]

use pyo3::prelude::*;

pub mod codec;
pub mod protocol;
pub mod pyo3_bindings;
pub mod runtime;
pub mod supervisor;
pub mod transport;

#[pymodule]
fn sakura_wire(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    m.add_class::<pyo3_bindings::PyDispatcher>()?;
    m.add_class::<pyo3_bindings::PyFuture>()?;
    m.add_class::<pyo3_bindings::PyRpcResult>()?;
    m.add_class::<pyo3_bindings::PyTlsConfig>()?;
    m.add_class::<pyo3_bindings::PyWorkerSupervisor>()?;
    Ok(())
}
```

- [ ] **Step 3: Add `PyWorkerSupervisor` to pyo3_bindings.rs**

Append the following to `crates/sakura-wire/src/pyo3_bindings.rs`. The first three `use` lines are *new*; do NOT re-add `Duration`, `Mutex`, or `Arc` — those are already imported at the top of the file from Task 10:

```rust
use crate::supervisor::{spawn_worker, WorkerHandle};
use pyo3::types::PyDict;
use std::collections::HashMap;

#[pyclass(name = "WorkerSupervisor")]
pub struct PyWorkerSupervisor {
    handles: Mutex<Vec<Arc<WorkerHandle>>>,
    shutdown_timeout: Duration,
}

#[pymethods]
impl PyWorkerSupervisor {
    #[new]
    #[pyo3(signature = (shutdown_timeout_s = 30.0))]
    fn new(shutdown_timeout_s: f64) -> Self {
        Self {
            handles: Mutex::new(Vec::new()),
            shutdown_timeout: Duration::from_secs_f64(shutdown_timeout_s.max(0.1)),
        }
    }

    /// Spawn one worker.
    /// `cmd` is the argv list (e.g., `[sys.executable, "-m", "sakura.worker"]`).
    /// `env` is extra env vars (e.g., `CUDA_VISIBLE_DEVICES=1`).
    /// Returns a tuple `(uri, cert_der_bytes)` once the worker prints
    /// `SAKURA_WORKER_LISTENING <uri> <cert_hex>` on stdout.
    #[pyo3(signature = (cmd, env = None, startup_timeout_s = 10.0))]
    fn spawn(
        &self,
        py: Python<'_>,
        cmd: Vec<String>,
        env: Option<&Bound<'_, PyDict>>,
        startup_timeout_s: f64,
    ) -> PyResult<(String, Py<PyBytes>)> {
        let mut env_map: HashMap<String, String> = HashMap::new();
        if let Some(d) = env {
            for (k, v) in d.iter() {
                let key: String = k.extract()?;
                let val: String = v.extract()?;
                env_map.insert(key, val);
            }
        }
        let timeout = Duration::from_secs_f64(startup_timeout_s.max(0.1));
        let handle =
            spawn_worker(&cmd, env_map, timeout).map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        let uri = handle.uri.clone();
        let cert = handle.cert_der.clone();
        self.handles.lock().unwrap().push(Arc::new(handle));
        let cert_py = PyBytes::new_bound(py, &cert).unbind();
        Ok((uri, cert_py))
    }

    /// Shut down every spawned worker.
    fn shutdown(&self) {
        let handles = std::mem::take(&mut *self.handles.lock().unwrap());
        let timeout = self.shutdown_timeout;
        for h in handles {
            h.shutdown(timeout);
        }
    }

    fn __len__(&self) -> usize {
        self.handles.lock().unwrap().len()
    }
}
```

- [ ] **Step 4: Build**

Run:
```bash
maturin develop
python -c "import sakura_wire; s = sakura_wire.WorkerSupervisor(); print(s, len(s))"
```
Expected: prints supervisor repr and `0`.

- [ ] **Step 5: Commit**

```bash
git add crates/sakura-wire/src/supervisor.rs crates/sakura-wire/src/pyo3_bindings.rs crates/sakura-wire/src/lib.rs
git commit -m "feat(wire): WorkerSupervisor — spawn workers, harvest listen URI + cert from stdout

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 12: Python `sakura.worker` daemon — echo handler

**Files:**
- Create: `python/sakura/worker/__init__.py`, `python/sakura/worker/__main__.py`

(Note: We use `python/sakura/worker/` as the module location even though the existing v0.1.x code lives in `sakura/`. Per `pyproject.toml` `python-source = "."`, both paths are valid; we keep new code under a subdir that doesn't collide with v0.1.x. The console-script `sakura-worker` is wired to `sakura.worker.__main__:main` already in pyproject.toml. **Important:** because `python-source = "."`, the entry-point `sakura.worker.__main__:main` resolves against `sakura/worker/__main__.py`, not `python/sakura/worker/__main__.py`. We therefore place the worker module at `sakura/worker/` directly — *not* under a `python/` subdirectory.)

**Corrected paths:**
- Create: `sakura/worker/__init__.py`, `sakura/worker/__main__.py`

- [ ] **Step 1: Create `sakura/worker/__init__.py`**

Write `sakura/worker/__init__.py`:
```python
"""sakura.worker — daemon entry point used by the WorkerSupervisor.

In Plan 1 the worker only registers an echo handler (HANDLER_ECHO = 0xDEAD)
which bounces all input tensors back unchanged. Plan 2 adds the real
HANDLER_EXEC_CLOUDPICKLED handler that runs user-supplied callables.
"""
__all__ = ["main"]

from sakura.worker.__main__ import main
```

- [ ] **Step 2: Create the failing daemon entry**

Write `sakura/worker/__main__.py`:
```python
"""`sakura-worker` — minimal QUIC server for Plan 1.

This worker speaks the sakura-wire protocol on top of QUIC, registers a
single echo handler at HANDLER_ECHO (0xDEAD), and prints a single line to
stdout once it is listening:

    SAKURA_WORKER_LISTENING <uri> <cert_der_hex>

The supervisor parses that line to learn the dynamic port and the
self-signed cert it must trust. After printing, the worker serves until
killed.
"""
from __future__ import annotations

import argparse
import asyncio
import sys

# This module imports the maturin-built sakura_wire native module via the
# helpers exposed below. To keep Plan 1's dependency on the native side
# minimal, the actual QUIC server is implemented in Rust and exposed
# through a small Python helper at sakura.wire.serve_echo. We import it
# lazily so the package is importable even before maturin develop has run
# (e.g., during `pip show sakura-ml`).


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="sakura-worker")
    parser.add_argument(
        "--listen",
        default="quic://127.0.0.1:0",
        help="Bind address (use port :0 for an ephemeral port; default).",
    )
    parser.add_argument(
        "--print-cert-hex",
        action="store_true",
        default=True,
        help="Print the self-signed cert (hex) on the listening line. Required for the supervisor handshake.",
    )
    args = parser.parse_args(argv)

    from sakura.wire import serve_echo  # imports sakura_wire native module
    serve_echo(listen=args.listen, print_handshake=args.print_cert_hex)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 3: Create `sakura/wire/__init__.py` with the `serve_echo` helper**

Write `sakura/wire/__init__.py`:
```python
"""Thin Python wrapper around the maturin-built `sakura_wire` cdylib.

Plan 1 exposes:
    - sakura.wire.Dispatcher, Future, Result, TlsConfig (re-exports)
    - sakura.wire.WorkerSupervisor (re-export)
    - sakura.wire.serve_echo() — used by sakura-worker daemon

Plan 2 layers SakuraRuntime, Service, and the dispatcher abstractions
on top of these primitives.
"""
from __future__ import annotations

import sys

import sakura_wire as _native

Dispatcher = _native.Dispatcher
Future = _native.Future
Result = _native.Result
TlsConfig = _native.TlsConfig
WorkerSupervisor = _native.WorkerSupervisor

__version__ = _native.__version__


def serve_echo(*, listen: str, print_handshake: bool = True) -> None:
    """Run a blocking QUIC server with the echo handler registered.

    Prints a single line to stdout when ready (so the supervisor can pick up the
    URI + cert), then serves forever. Plan 1 only — Plan 2 replaces this with
    a real handler-registry server.
    """
    if not listen.startswith("quic://"):
        raise ValueError(f"--listen must be a quic:// URI, got: {listen}")
    addr = listen[len("quic://"):]
    _native.run_echo_server(addr=addr, print_handshake=print_handshake)
```

- [ ] **Step 4: Add `run_echo_server` to the native module**

Append to `crates/sakura-wire/src/pyo3_bindings.rs`:
```rust
use crate::protocol::HANDLER_ECHO;
use crate::supervisor::encode_hex;
use crate::transport::{accept_request, bind_server, generate_self_signed, send_response};

/// Bind a QUIC server, print one handshake line on stdout, then serve the
/// echo handler forever (`HANDLER_ECHO`). Used by the `sakura-worker` daemon.
#[pyfunction]
#[pyo3(signature = (addr, print_handshake = true))]
pub fn run_echo_server(py: Python<'_>, addr: String, print_handshake: bool) -> PyResult<()> {
    py.allow_threads(|| -> PyResult<()> {
        WireRuntime::shared().block_on(async move {
            let pair = generate_self_signed("localhost")
                .map_err(|e| PyRuntimeError::new_err(format!("cert: {e}")))?;
            let bind_addr: SocketAddr = addr
                .parse()
                .map_err(|e| PyRuntimeError::new_err(format!("addr {addr}: {e}")))?;
            let endpoint = bind_server(bind_addr, &pair)
                .map_err(|e| PyRuntimeError::new_err(format!("bind: {e}")))?;
            let local = endpoint
                .local_addr()
                .map_err(|e| PyRuntimeError::new_err(format!("local_addr: {e}")))?;
            if print_handshake {
                let cert_hex = encode_hex(&pair.cert_der);
                println!("SAKURA_WORKER_LISTENING quic://{local} {cert_hex}");
                let _ = std::io::Write::flush(&mut std::io::stdout().lock());
            }

            // Accept connections forever.
            while let Some(incoming) = endpoint.accept().await {
                tokio::spawn(async move {
                    match incoming.await {
                        Ok(conn) => loop {
                            match accept_request(&conn).await {
                                Ok((send, req)) => {
                                    let resp = match echo_handler(&req) {
                                        Ok(b) => b,
                                        Err(e) => {
                                            tracing::error!("echo: {e:?}");
                                            return;
                                        }
                                    };
                                    if let Err(e) = send_response(send, resp).await {
                                        tracing::error!("send_response: {e:?}");
                                        return;
                                    }
                                }
                                Err(_) => return,
                            }
                        },
                        Err(e) => tracing::error!("connection: {e:?}"),
                    }
                });
            }
            Ok(())
        })
    })
}

fn echo_handler(req_bytes: &[u8]) -> Result<Vec<u8>, PyWireError> {
    use crate::codec::{unpack_request, RpcResponseHeader, RpcStatus, TensorDesc, WireVersion};
    let (header, descs, tensors, aux) = unpack_request(req_bytes).map_err(|e| {
        PyWireError::Wire(WireError::DecodeFailed {
            what: "echo unpack".into(),
            detail: e.to_string(),
        })
    })?;
    if header.handler_id != HANDLER_ECHO {
        return Err(PyWireError::Wire(WireError::HandlerNotFound {
            handler_id: header.handler_id,
        }));
    }

    // Build the response: same descriptors + tensor bytes, identical aux.
    let resp_header = RpcResponseHeader {
        version: WireVersion::V1,
        request_id: header.request_id,
        status: RpcStatus::Ok,
        n_result_tensors: descs.len() as u32,
        aux_payload_bytes: aux.len() as u32,
        elapsed_us: 0,
    };
    let header_bytes = postcard::to_allocvec(&resp_header).map_err(|e| {
        PyWireError::Wire(WireError::DecodeFailed {
            what: "encode resp header".into(),
            detail: e.to_string(),
        })
    })?;
    let descs_bytes = postcard::to_allocvec(&descs).map_err(|e| {
        PyWireError::Wire(WireError::DecodeFailed {
            what: "encode resp descs".into(),
            detail: e.to_string(),
        })
    })?;
    let total = 4 + header_bytes.len() + 4 + descs_bytes.len()
        + tensors.iter().map(Vec::len).sum::<usize>()
        + aux.len();
    let mut out = Vec::with_capacity(total);
    out.extend_from_slice(&(header_bytes.len() as u32).to_le_bytes());
    out.extend_from_slice(&header_bytes);
    out.extend_from_slice(&(descs_bytes.len() as u32).to_le_bytes());
    out.extend_from_slice(&descs_bytes);
    for t in &tensors {
        out.extend_from_slice(t);
    }
    out.extend_from_slice(&aux);
    Ok(out)
}
```

- [ ] **Step 5: Wire `run_echo_server` into the pymodule**

Edit `crates/sakura-wire/src/lib.rs` to add `m.add_function(wrap_pyfunction!(...))`:
```rust
//! sakura-wire: zero-copy tensor codec + RPC over QUIC.

#![deny(unsafe_op_in_unsafe_fn)]

use pyo3::prelude::*;
use pyo3::wrap_pyfunction;

pub mod codec;
pub mod protocol;
pub mod pyo3_bindings;
pub mod runtime;
pub mod supervisor;
pub mod transport;

#[pymodule]
fn sakura_wire(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    m.add_class::<pyo3_bindings::PyDispatcher>()?;
    m.add_class::<pyo3_bindings::PyFuture>()?;
    m.add_class::<pyo3_bindings::PyRpcResult>()?;
    m.add_class::<pyo3_bindings::PyTlsConfig>()?;
    m.add_class::<pyo3_bindings::PyWorkerSupervisor>()?;
    m.add_function(wrap_pyfunction!(pyo3_bindings::run_echo_server, m)?)?;
    Ok(())
}
```

- [ ] **Step 6: Build + smoke test the worker (kill quickly)**

Run:
```bash
maturin develop
( sakura-worker & WPID=$!; sleep 1; kill $WPID; wait $WPID 2>/dev/null )
```
Expected: prints one line `SAKURA_WORKER_LISTENING quic://127.0.0.1:NNNNN <hex>` and exits.

- [ ] **Step 7: Commit**

```bash
git add sakura/worker/ sakura/wire/ crates/sakura-wire/src/pyo3_bindings.rs crates/sakura-wire/src/lib.rs
git commit -m "feat(wire): sakura-worker daemon with echo handler + serve_echo PyO3 hook

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 13: End-to-end Python smoke test — spawn → submit → result

**Files:**
- Create: `tests/wire/__init__.py`, `tests/wire/test_e2e_echo.py`

- [ ] **Step 1: Write the failing E2E test**

Create `tests/wire/__init__.py`:
```python
```

Create `tests/wire/test_e2e_echo.py`:
```python
"""End-to-end Plan 1 acceptance test.

Spawns a sakura-worker subprocess via WorkerSupervisor, opens a Dispatcher
against its URI with the cert it printed, submits an RPC at HANDLER_ECHO,
and verifies the response tensors are byte-identical.
"""
from __future__ import annotations

import sys

import pytest

sakura_wire = pytest.importorskip("sakura_wire")

from sakura.wire import Dispatcher, TlsConfig, WorkerSupervisor

HANDLER_ECHO = 0xDEAD
DTYPE_F32 = 0
DEVICE_CPU = 0


def _f32_bytes(values: list[float]) -> bytes:
    import struct
    return b"".join(struct.pack("<f", v) for v in values)


def test_echo_round_trip_through_worker():
    sup = WorkerSupervisor(shutdown_timeout_s=5.0)
    try:
        uri, cert = sup.spawn(
            cmd=[sys.executable, "-m", "sakura.worker", "--listen", "quic://127.0.0.1:0"],
            startup_timeout_s=10.0,
        )
        assert uri.startswith("quic://127.0.0.1:")
        assert isinstance(cert, bytes) and len(cert) > 100  # rough self-signed cert size

        tls = TlsConfig(cert, "localhost")
        d = Dispatcher(uri, tls)

        payload_a = _f32_bytes([1.0, 2.0, 3.0, 4.0])
        payload_b = _f32_bytes([10.0, 20.0])
        aux = b"hello-aux"

        fut = d.submit(
            HANDLER_ECHO,
            [
                {"shape": [4], "dtype_id": DTYPE_F32, "device_id": DEVICE_CPU, "data": payload_a},
                {"shape": [2], "dtype_id": DTYPE_F32, "device_id": DEVICE_CPU, "data": payload_b},
            ],
            aux,
            timeout_ms=5000,
        )
        result = fut.result(timeout=5.0)
        assert result.aux == aux
        tensors = result.tensors()
        assert len(tensors) == 2
        assert tensors[0] == payload_a
        assert tensors[1] == payload_b
    finally:
        sup.shutdown()


def test_dispatcher_rejects_non_quic_uri():
    tls = TlsConfig(b"\x00" * 256, "localhost")
    with pytest.raises(ValueError):
        Dispatcher("http://localhost:8080", tls)
```

- [ ] **Step 2: Adjust `Dispatcher.submit` to accept dict args**

The Rust `TensorTuple::from_pyobject` currently extracts struct fields directly from a Python object. To accept the dict shape used in the test, add `dict` extraction support. Edit `crates/sakura-wire/src/pyo3_bindings.rs` and replace `TensorTuple` with:
```rust
struct TensorTuple {
    shape: Vec<u32>,
    dtype_id: u8,
    device_id: u8,
    data: Vec<u8>,
}

impl<'py> FromPyObject<'py> for TensorTuple {
    fn extract_bound(ob: &Bound<'py, PyAny>) -> PyResult<Self> {
        // Accept either a dict {"shape", "dtype_id", "device_id", "data"} or a
        // tuple (shape, dtype_id, device_id, data).
        if let Ok(dict) = ob.downcast::<PyDict>() {
            let shape: Vec<u32> = dict
                .get_item("shape")?
                .ok_or_else(|| PyValueError::new_err("missing 'shape'"))?
                .extract()?;
            let dtype_id: u8 = dict
                .get_item("dtype_id")?
                .ok_or_else(|| PyValueError::new_err("missing 'dtype_id'"))?
                .extract()?;
            let device_id: u8 = dict
                .get_item("device_id")?
                .ok_or_else(|| PyValueError::new_err("missing 'device_id'"))?
                .extract()?;
            let data: Vec<u8> = dict
                .get_item("data")?
                .ok_or_else(|| PyValueError::new_err("missing 'data'"))?
                .extract()?;
            return Ok(Self { shape, dtype_id, device_id, data });
        }
        let tup: (Vec<u32>, u8, u8, Vec<u8>) = ob.extract()?;
        Ok(Self {
            shape: tup.0,
            dtype_id: tup.1,
            device_id: tup.2,
            data: tup.3,
        })
    }
}
```

- [ ] **Step 3: Rebuild and run the test**

Run:
```bash
maturin develop
pytest tests/wire/test_e2e_echo.py -v
```
Expected:
```
tests/wire/test_e2e_echo.py::test_echo_round_trip_through_worker PASSED
tests/wire/test_e2e_echo.py::test_dispatcher_rejects_non_quic_uri PASSED
```

- [ ] **Step 4: Commit**

```bash
git add tests/wire/ crates/sakura-wire/src/pyo3_bindings.rs
git commit -m "test(wire): end-to-end echo round-trip — spawn worker, dispatcher submit, verify

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 14: Criterion benches — codec encode/decode + loopback throughput

**Files:**
- Create: `crates/sakura-wire/benches/codec_bench.rs`, `crates/sakura-wire/benches/transport_bench.rs`

- [ ] **Step 1: Write the codec bench**

Create `crates/sakura-wire/benches/codec_bench.rs`:
```rust
use criterion::{criterion_group, criterion_main, Criterion, Throughput};
use sakura_wire::codec::{
    pack_request, unpack_request, Device, Dtype, RpcRequestHeader, TensorView, WireVersion,
};
use smallvec::smallvec;

fn make_state_dict_buf(total_mb: usize) -> Vec<u8> {
    vec![0u8; total_mb * 1024 * 1024]
}

fn bench_pack(c: &mut Criterion) {
    let mut group = c.benchmark_group("codec_pack_268mb");
    group.throughput(Throughput::Bytes(268 * 1024 * 1024));
    let buf = make_state_dict_buf(268);
    let header = RpcRequestHeader {
        version: WireVersion::V1,
        request_id: 1,
        handler_id: 0xDEAD,
        n_tensors: 1,
        aux_payload_bytes: 0,
        deadline_ms: None,
        trace_id: 0,
    };
    group.bench_function("pack", |b| {
        b.iter(|| {
            let view = TensorView::new(smallvec![268_435_456u32], Dtype::U8, Device::Cpu, &buf);
            let _ = pack_request(&header, &[view], &[]).unwrap();
        });
    });
    group.finish();
}

fn bench_unpack(c: &mut Criterion) {
    let mut group = c.benchmark_group("codec_unpack_268mb");
    group.throughput(Throughput::Bytes(268 * 1024 * 1024));
    let buf = make_state_dict_buf(268);
    let header = RpcRequestHeader {
        version: WireVersion::V1,
        request_id: 1,
        handler_id: 0xDEAD,
        n_tensors: 1,
        aux_payload_bytes: 0,
        deadline_ms: None,
        trace_id: 0,
    };
    let view = TensorView::new(smallvec![268_435_456u32], Dtype::U8, Device::Cpu, &buf);
    let packed = pack_request(&header, &[view], &[]).unwrap();
    group.bench_function("unpack", |b| {
        b.iter(|| {
            let _ = unpack_request(&packed).unwrap();
        });
    });
    group.finish();
}

criterion_group!(benches, bench_pack, bench_unpack);
criterion_main!(benches);
```

- [ ] **Step 2: Write the transport bench (RTT + small-message throughput)**

Create `crates/sakura-wire/benches/transport_bench.rs`:
```rust
use criterion::{criterion_group, criterion_main, Criterion};
use sakura_wire::runtime::WireRuntime;
use sakura_wire::transport::{
    accept_request, bind_server, connect, generate_self_signed, rpc_call, send_response,
};
use std::net::SocketAddr;
use tokio::sync::oneshot;

fn bench_rtt(c: &mut Criterion) {
    let rt = WireRuntime::shared();
    let pair = generate_self_signed("localhost").unwrap();
    let cert = pair.cert_der.clone();
    let server_addr: SocketAddr = "127.0.0.1:0".parse().unwrap();
    let endpoint = bind_server(server_addr, &pair).unwrap();
    let local = endpoint.local_addr().unwrap();

    let (stop_tx, mut stop_rx) = oneshot::channel::<()>();
    rt.spawn(async move {
        loop {
            tokio::select! {
                _ = &mut stop_rx => break,
                next = endpoint.accept() => {
                    let Some(incoming) = next else { break };
                    tokio::spawn(async move {
                        if let Ok(conn) = incoming.await {
                            loop {
                                match accept_request(&conn).await {
                                    Ok((send, req)) => {
                                        let _ = send_response(send, req).await;
                                    }
                                    Err(_) => break,
                                }
                            }
                        }
                    });
                }
            }
        }
    });

    let conn = rt.block_on(async { connect(local, "localhost", &cert).await.unwrap() });
    let payload = vec![0u8; 64]; // tiny — RTT-dominated

    c.bench_function("loopback_rtt_64b", |b| {
        b.iter(|| {
            rt.block_on(async {
                let _ = rpc_call(&conn, payload.clone()).await.unwrap();
            });
        });
    });

    let _ = stop_tx.send(());
}

criterion_group!(benches, bench_rtt);
criterion_main!(benches);
```

- [ ] **Step 3: Run the benches**

Run:
```bash
cargo bench -p sakura-wire 2>&1 | tail -30
```
Expected:
- `codec_pack_268mb/pack` reports a `time:` line with median < 30 ms.
- `codec_unpack_268mb/unpack` reports a `time:` line with median < 30 ms.
- `loopback_rtt_64b` reports a `time:` line with median < 0.5 ms.

(If any target misses on an underprovisioned machine, capture the numbers — Plan 1 acceptance is "the perf budget targets in §7.7 of the spec are within 2× on the dev box".)

- [ ] **Step 4: Commit**

```bash
git add crates/sakura-wire/benches/
git commit -m "bench(wire): criterion benches — codec MB/s + loopback RTT

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 15: CI — extend `.github/workflows/test.yml` for Rust + maturin

**Files:**
- Modify: `.github/workflows/test.yml`

- [ ] **Step 1: Read the existing workflow**

Run:
```bash
cat .github/workflows/test.yml
```
Note the current Python-only structure.

- [ ] **Step 2: Replace `.github/workflows/test.yml` with the multi-job pipeline**

Write `.github/workflows/test.yml`:
```yaml
name: test

on:
  push:
    branches: [master]
  pull_request:

env:
  CARGO_TERM_COLOR: always
  RUSTFLAGS: -D warnings

jobs:
  rust:
    name: Rust
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: dtolnay/rust-toolchain@stable
        with:
          toolchain: 1.78.0
          components: rustfmt, clippy
      - uses: Swatinem/rust-cache@v2
      - name: cargo fmt
        run: cargo fmt --all --check
      - name: cargo clippy
        run: cargo clippy --workspace --all-targets -- -D warnings
      - name: cargo test
        run: cargo test --workspace --all-features

  python:
    name: Python ${{ matrix.python }}
    runs-on: ubuntu-latest
    strategy:
      fail-fast: false
      matrix:
        python: ["3.10", "3.11", "3.12"]
    steps:
      - uses: actions/checkout@v4
      - uses: dtolnay/rust-toolchain@stable
        with:
          toolchain: 1.78.0
      - uses: Swatinem/rust-cache@v2
      - uses: actions/setup-python@v5
        with:
          python-version: ${{ matrix.python }}
      - name: Install build deps
        run: pip install --upgrade pip maturin pytest
      - name: Build + install via maturin develop
        run: maturin develop --release
      - name: Pytest (wire layer only — Plan 1)
        run: pytest tests/wire/ -v
```

- [ ] **Step 3: Validate the YAML locally**

Run:
```bash
python -c "import yaml; yaml.safe_load(open('.github/workflows/test.yml'))"
```
Expected: no exception.

- [ ] **Step 4: Commit**

```bash
git add .github/workflows/test.yml
git commit -m "ci: add Rust toolchain + cargo clippy/test + maturin develop matrix

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 16: CI — `.github/workflows/publish.yml` builds maturin wheels

**Files:**
- Modify: `.github/workflows/publish.yml`

- [ ] **Step 1: Read the existing publish workflow**

Run:
```bash
cat .github/workflows/publish.yml
```
Note the current PyPI publish step (used by v0.1.x).

- [ ] **Step 2: Replace with maturin wheel matrix**

Write `.github/workflows/publish.yml`:
```yaml
name: publish

on:
  workflow_dispatch:
  push:
    tags:
      - "v*"

jobs:
  wheels:
    name: ${{ matrix.target }}
    runs-on: ${{ matrix.os }}
    strategy:
      fail-fast: false
      matrix:
        include:
          - os: ubuntu-latest
            target: x86_64-unknown-linux-gnu
          - os: ubuntu-latest
            target: aarch64-unknown-linux-gnu
          - os: macos-14
            target: aarch64-apple-darwin
          - os: windows-latest
            target: x86_64-pc-windows-msvc
    steps:
      - uses: actions/checkout@v4
      - uses: dtolnay/rust-toolchain@stable
        with:
          toolchain: 1.78.0
          targets: ${{ matrix.target }}
      - uses: actions/setup-python@v5
        with:
          python-version: "3.10"
      - name: Build wheel
        uses: PyO3/maturin-action@v1
        with:
          target: ${{ matrix.target }}
          args: --release --out dist --strip
          manylinux: auto
      - name: Upload wheel artifact
        uses: actions/upload-artifact@v4
        with:
          name: wheel-${{ matrix.target }}
          path: dist/*.whl

  sdist:
    name: sdist
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: dtolnay/rust-toolchain@stable
        with:
          toolchain: 1.78.0
      - uses: actions/setup-python@v5
        with:
          python-version: "3.10"
      - name: Build sdist
        uses: PyO3/maturin-action@v1
        with:
          command: sdist
          args: --out dist
      - uses: actions/upload-artifact@v4
        with:
          name: sdist
          path: dist/*.tar.gz

  publish:
    if: startsWith(github.ref, 'refs/tags/v')
    needs: [wheels, sdist]
    runs-on: ubuntu-latest
    environment: pypi
    permissions:
      id-token: write
    steps:
      - uses: actions/download-artifact@v4
        with:
          path: dist
          merge-multiple: true
      - name: Publish to PyPI
        uses: PyO3/maturin-action@v1
        with:
          command: upload
          args: --non-interactive --skip-existing dist/*
```

- [ ] **Step 3: Validate the YAML**

Run:
```bash
python -c "import yaml; yaml.safe_load(open('.github/workflows/publish.yml'))"
```
Expected: no exception.

- [ ] **Step 4: Commit**

```bash
git add .github/workflows/publish.yml
git commit -m "ci: replace PyPI publish step with maturin wheel matrix (linux/mac/windows)

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 17: Acceptance — full local verification

This task runs everything Plan 1 ships and confirms the gates are green.

- [ ] **Step 1: Format + clippy clean**

Run:
```bash
cargo fmt --all --check
cargo clippy --workspace --all-targets -- -D warnings
```
Expected: both succeed, no diff, no warnings.

- [ ] **Step 2: Full Rust test suite**

Run:
```bash
cargo test --workspace --all-features -- --test-threads=1
```
Expected: all tests pass — codec (header, cast, tensor), protocol (error), runtime, codec_roundtrip, quic_loopback. No failures.

- [ ] **Step 3: Maturin build + Python tests**

Run:
```bash
maturin develop --release
pytest tests/wire/ -v
```
Expected: `test_echo_round_trip_through_worker` and `test_dispatcher_rejects_non_quic_uri` both pass.

- [ ] **Step 4: Criterion budget check**

Run:
```bash
cargo bench -p sakura-wire 2>&1 | grep -E "(time:|throughput)" | tee /tmp/sakura-bench.txt
```
Inspect:
- `codec_pack_268mb/pack` median time < 60 ms (target 30 ms; allow 2× on dev hardware).
- `codec_unpack_268mb/unpack` median time < 60 ms.
- `loopback_rtt_64b` median time < 1 ms (target 0.5 ms).

If any number exceeds 2× the §7.7 spec target on the dev box, file an issue describing the regression with the captured numbers — but do not block Plan 1; CI runs on representative hardware later.

- [ ] **Step 5: Wheel build smoke test**

Run:
```bash
maturin build --release --out /tmp/sakura-wheels
ls -la /tmp/sakura-wheels/
```
Expected: a `sakura_ml-1.0.0a0-cp310-abi3-*.whl` (or similar) is produced.

- [ ] **Step 6: Final commit (only if anything changed)**

If any of the above produced uncommitted changes, commit. If everything was already clean (likely the case), skip.

```bash
git status
```
Expected: working tree clean.

- [ ] **Step 7: Tag the milestone**

```bash
git tag -a sakura-wire-v1-foundation -m "Plan 1 complete: sakura-wire Rust foundation"
git log --oneline sakura-wire-v1-foundation~17..sakura-wire-v1-foundation
```

This tag marks the boundary between Plan 1 (transport foundation) and Plan 2 (Python runtime + services).

---

## Plan 1 — Acceptance Criteria

Plan 1 is complete when **all** of the following are true:

1. `cargo fmt --all --check` and `cargo clippy --workspace --all-targets -- -D warnings` pass.
2. `cargo test --workspace --all-features` passes (~12 unit + integration tests).
3. `maturin develop --release` succeeds; `python -c "import sakura_wire"` works.
4. `pytest tests/wire/` passes both Plan 1 acceptance tests.
5. `cargo bench -p sakura-wire` completes; recorded numbers attached to the milestone tag's commit message or an issue.
6. `maturin build --release` produces a wheel that imports cleanly in a fresh venv.
7. `.github/workflows/test.yml` and `.github/workflows/publish.yml` are valid YAML and pass linting in CI.
8. The existing `sakura/` v0.1.x package is untouched (`git diff sakura/ vs HEAD~17` shows no changes).
9. Tag `sakura-wire-v1-foundation` exists on the branch.

After Plan 1 lands, **Plan 2** (`SakuraRuntime` + `Service` ABC + `LocalDispatcher` + framework-agnostic event bus) builds on this foundation. Plan 2 will introduce `python/sakura/` for the new code (per the spec's repo layout) and start the staged migration of v0.1.x callers.

---

## Self-Review Notes

- **Spec coverage:** Plan 1 implements §4 (architecture overview down to "Dispatcher" + sakura-wire), §7 (Rust crate layout, wire format, protocol, transport, perf budget), §8 (Dispatcher implementations — only the underlying Python `Dispatcher` PyO3 class; the Python `LocalDispatcher`/`RemoteDispatcher`/`ZakuroDispatcher` wrappers are Plan 2), §9 (`WorkerSupervisor` lifecycle), §13 (packaging — maturin), §14 (testing — Rust unit + Python interop tiers; Lightning/HF E2E and perf tiers wait for Plan 2+).
- **Out-of-scope confirmed:** No `SakuraRuntime`, `Service`, framework adapters, real eval/checkpoint handlers, SHM, RDMA, mutual TLS, CUDA IPC, or removal of v0.1.x code.
- **Known follow-ups for Plan 2:** introduce `python/sakura/` source root + move new code there; add `LocalDispatcher`/`RemoteDispatcher`/`ZakuroDispatcher` Python wrappers; replace echo handler with `HANDLER_EXEC_CLOUDPICKLED`; add the `SakuraRuntime` + `Service` ABC + `Adapter` ABC.
