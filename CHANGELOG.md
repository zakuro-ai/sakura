# Changelog

## 1.0.0

v1.0 is a ground-up rewrite and a clean break from v0.1.x. Every public class
moved or was replaced; see [docs/migration-from-0.1.md](docs/migration-from-0.1.md)
for side-by-side examples. To stay on the old API, pin `sakura-ml<1.0`.

### Changed (breaking)

- Explicit composition replaces per-framework wrapper classes: instantiate a
  `SakuraRuntime`, install services on it, and attach a thin adapter to the
  framework callback chain.
- `sakura.lightning.SakuraTrainer` / `SakuraLightningCallback` are replaced by
  `SakuraRuntime` + `LightningAdapter` + `AsyncEval`.
- `sakura.huggingface.SakuraHFCallback` is replaced by `SakuraRuntime` + `HFAdapter` + `AsyncEval`.
- `sakura.ddp.DDPAsyncEvalCallback` is replaced by `SakuraRuntime` + `DDPAdapter` + `AsyncEval`.
- `zakuro.Compute(uri="quic://...")` becomes `RemoteDispatcher(uri=..., cert_der=...)`
  or `Compute.at("quic://...")`.
- `model_factory` / `val_loader_factory` are gone; `model_path` becomes
  `AsyncCheckpoint(dir=..., every="best")`; `trainer.history` becomes `async_eval.history`.
- Knobs carried over: `eval_fn`, `eval_payload`, `cache_key`, `max_pending`,
  `on_backpressure`, `drain` (now arguments of `AsyncEval`).

### Removed

- `sakura.tensorflow.SakuraKerasCallback`
- `sakura.ml.async_trainer.AsyncTrainer` (use `DDPAdapter`)
- `sakura.ml.sakura_trainer.SakuraTrainer`
- `sakura.functional.*`

### Added

- Process isolation: `LocalDispatcher` auto-spawns a worker subprocess, so
  training and evaluation never contend on the GIL.
- Independent, composable services: `AsyncEval`, `AsyncCheckpoint`,
  `MixedPrecision`, `Compile`, `ZeRO1`, `ActivationCheckpoint`.
- Three framework adapters (Lightning, HuggingFace, raw PyTorch DDP) over one runtime.
- `Telemetry(output=path)`: one JSON line per event, a single source of truth for benchmarking.
- Rust transport: `sakura-wire` over QUIC, the same wire format from localhost to LAN/WAN.
## [1.1.0]

### Added
- `sakura.dispatch.ProcessDispatcher`: a persistent worker process fed through shared memory
  (one flat buffer per message), so background work no longer contends for the training loop's GIL.
- `AsyncCheckpoint(every_seconds=..., max_pending=...)`: time-budgeted checkpoints and backpressure
  (skip instead of queueing multi-hundred-MB states behind a slow disk).
- `sakura.atelier`: `speech_recognition` task, `deepspeech2` backend (asr-deepspeech on the Sakura
  runtime), `asr_manifest` data format with tar-shard clip addressing, `speech_recognition/ctc-fast@1`.

### Fixed
- `AsyncCheckpoint(keep=N)` is now enforced; it was accepted and ignored, so checkpoints accumulated.

