# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

Python package versions use [PEP 440](https://peps.python.org/pep-0440/)
pre-release spelling (`1.0.0a1`); the Rust crate uses the SemVer equivalent
(`1.0.0-alpha.1`).

## [Unreleased]

### Changed

- **ML framework dependencies are now optional extras (BREAKING for installs).**
  `pip install sakura-ml` no longer pulls PyTorch — the base install is the
  dispatch / runtime / wire surface only. Install the training stack via
  `sakura-ml[training]` (torch / torchvision / lightning), `sakura-ml[huggingface]`,
  or `sakura-ml[bench]`. Using a training feature without its extra now raises a
  `ModuleNotFoundError` with the exact `pip install` hint, via the new
  `sakura._optional.load` helper. A `bare-install` CI lane guards the core
  install against regressing to require torch. (#71)

## [1.0.0a1] — 2026-05-09

v1.0 is a clean break from v0.1.x. The library was rebuilt around an **event
bus + service registry**: a framework loop feeds typed events to a
`SakuraRuntime`, which dispatches them to composable services, optionally
handing work off to a Rust QUIC transport and an isolated worker subprocess.

See [`docs/migration-from-0.1.md`](docs/migration-from-0.1.md) for side-by-side
upgrade examples. If you cannot migrate yet, pin to the previous line:

```bash
pip install 'sakura-ml<1.0'
```

### Added

- **Event bus + service registry** (`SakuraRuntime`): dispatches typed,
  frozen-dataclass events (`OnTrainBegin`, `OnEpochEnd`, `OnOptimizerStep`,
  `OnSave`, `OnError`, …) to installed services in priority order, with
  per-service error isolation and runtime-coordinated `scale_loss()` /
  `optimizer_step()` hooks.
- **Seven v1 services**, independently composable: `Telemetry`,
  `MixedPrecision`, `ActivationCheckpoint`, `Compile`, `ZeRO1`, `AsyncEval`,
  `AsyncCheckpoint`.
- **Three framework adapters**: `LightningAdapter` (a `lightning.Callback`),
  `HFAdapter` (a `transformers.TrainerCallback`), and `DDPAdapter` (explicit
  hooks for raw DDP / any custom loop).
- **Dispatchers**: `InThreadDispatcher`, `ThreadDispatcher`, `LocalDispatcher`
  (auto-spawns a localhost worker subprocess for full GIL isolation),
  `RemoteDispatcher` (`quic://host:port`), and `ZakuroDispatcher`.
- **`sakura-wire` Rust crate**: zero-copy tensor wire codec, protocol, and a
  quinn-based QUIC transport, exposed to Python via PyO3 (`sakura_wire`) and
  built into the wheel by maturin.
- **`sakura-worker`** subprocess entry point: a QUIC server running in a
  separate interpreter so the GIL never contends between training and
  dispatched eval/checkpoint work.
- **ZeRO-1** optimizer-state sharding (`sakura.zero`), with multi-rank
  `gloo`/NCCL correctness tests.
- **`sakura-bench`** harness: runs a `Workload` through a vanilla framework loop
  or through `SakuraRuntime`, emitting `RunReport` JSON for comparison.

### Changed

- **Composition over wrappers (BREAKING).** v0.1.x shipped one thin wrapper
  class per framework that bundled the eval-dispatch pattern with framework
  glue. v1.0 replaces this with explicit composition: instantiate a
  `SakuraRuntime`, install the services you need, and attach a thin adapter to
  the framework's callback chain. Each piece is independently swappable.
- Eval now runs on the worker via a cloudpickled `eval_fn` passed to
  `AsyncEval`; `model_factory` / `val_loader_factory` are gone. Checkpointing
  moves to `AsyncCheckpoint(dir=..., every="best", …)`.
- Remote compute is addressed with `RemoteDispatcher(uri="quic://…", cert_der=…)`
  (or `Compute.at("quic://…")`) instead of `zakuro.Compute(uri="quic://…")`.

### Removed

- **All v0.1.x public APIs were removed (BREAKING):**
  - `sakura.lightning.SakuraTrainer` / `SakuraLightningCallback`
    → `SakuraRuntime` + `LightningAdapter` + `AsyncEval`.
  - `sakura.huggingface.SakuraHFCallback`
    → `SakuraRuntime` + `HFAdapter` + `AsyncEval`.
  - `sakura.ddp.DDPAsyncEvalCallback`
    → `SakuraRuntime` + `DDPAdapter` + `AsyncEval`.
  - `sakura.tensorflow.SakuraKerasCallback` — removed entirely; TensorFlow /
    Keras integration is no longer in scope.
  - `sakura.ml.async_trainer.AsyncTrainer` and
    `sakura.ml.sakura_trainer.SakuraTrainer` — removed (use `DDPAdapter`, which
    is framework-free).
  - `sakura.functional.*` — removed.

## v0.1.x and earlier

Releases prior to v1.0 were not tracked in this file. See the
[GitHub Releases](https://github.com/zakuro-ai/sakura/releases) page for the
v0.1.x history (latest: `v0.1.8`, 2026-05-04). The v0.1.x line remains
installable via `pip install 'sakura-ml<1.0'`.

[Unreleased]: https://github.com/zakuro-ai/sakura/compare/v1.0.0a1...HEAD
[1.0.0a1]: https://github.com/zakuro-ai/sakura/compare/v0.1.8...v1.0.0a1
