# Strategy Memo: Zakuro AI Project

## 1. Executive Summary
The Zakuro project has evolved from a simple Python ML helper library (Sakura) into a sophisticated context-aware distributed-ML runtime (Zakuro) paired with an asynchronous ML training services layer (Sakura v1.0). The original commercial vision depicted a federated fog-compute marketplace, but the immediate engineering focus has successfully narrowed to delivering a robust, adaptive developer runtime that decouples ML training from blocking operations (evaluation, checkpointing) via a QUIC-backed transport and rust-based optimizations.

The project is technically sound, demonstrating rigorous benchmarking and a healthy engineering culture focused on isolating and measuring distributed failure modes. The `zakuro-poc-plugin` now establishes the first AI-agent execution boundary for Claude/Codex and has been validated end to end locally, including Docker-backed execution and `100%` package coverage. It should still be treated as an implemented POC rather than a complete production component because the native `zc` contract remains an external dependency rather than an in-repo runtime. Its intended pipeline is `Intent -> Plan -> Validation -> Consent -> Isolated Backend -> Observable Result`.

The main achievement is the concrete separation of the ML training layer (Sakura) from the execution substrate (Zakuro) and the initial abstraction of AI-agent execution through the new plugin. The largest remaining work lies in hardening the plugin boundary, maturing the Rust broker (`zc`), and bridging the current developer-runtime into the broader marketplace vision (billing, enterprise isolation).

## 2. Original Intent
- **Problem:** ML training loops block synchronously on side-tasks like evaluation, checkpointing, and logging, wasting expensive GPU cycles. Furthermore, distributed execution requires complex manual routing and rigid infrastructure.
- **Intended Outcome:** A federated compute marketplace and distributed runtime where Python functions can seamlessly execute remotely, with an intelligent mesh adaptively routing workloads based on latency, cost, and availability.
- **Success Criteria:** Measurable reductions in ML training wall-clock time via async services; verifiable multi-rank optimization correctness; secure remote execution from AI agents; and eventual sub-cent billing for a federated marketplace.
- **Intent Evolution:** The commercial "marketplace" vision (billing, ledger, providers) has been temporarily deferred in favor of a highly focused technical wedge: building a world-class, adaptive distributed ML developer runtime and asynchronous training services library.

## 3. Project Evolution
- **Phase 0 (2023):** Early ML-library experiments (Sakura v0.1).
- **Phase 1 (2025 - Early 2026):** Zakuro emerges as a distributed compute substrate. Experiments with Tailscale, mesh setups, and Docker worker nodes.
- **Phase 2 (April 2026):** Strategic pivot toward adaptive allocation (`AdaptiveCompute`) and QUIC transport over HTTP. Rejection of manual routing strategies in favor of context-aware telemetry (latency drift, dead connections).
- **Phase 3 (May 2026):** Sakura v1.0 alpha introduces `SakuraRuntime`, decoupling async eval/checkpointing using the QUIC worker transport. Strong focus on benchmark credibility (ZeRO-1 multi-rank, fp16/bf16 scaling on RTX 4090).
- **Phase 4 (Current):** Implementation of the `zakuro-poc-plugin`. Establishing a controlled, structured execution boundary enabling LLMs to safely orchestrate compute tasks within isolated Docker backends and a conservative `zc` subprocess adapter, acting as a precursor for deeper Rust broker integration.

## 4. Completed Work

### Workstream: Core Zakuro Substrate
- **Purpose:** Provide the foundational distributed execution runtime.
- **Concrete Outputs:** `@zk.fn` decorators, HTTP/QUIC workers, local fallback modes, `AdaptiveCompute` (Adam-style EMAs for latency and variance tracking).
- **Current Status:** In progress / Complete with limitations (version ~0.3-draft). Focus is on execution mechanics, not yet marketplace billing.

### Workstream: Sakura ML Training Services
- **Purpose:** Accelerate PyTorch/Lightning/HF training via decoupled services.
- **Concrete Outputs:** `SakuraRuntime`, `MixedPrecision`, `ZeRO1`, `AsyncEval`, `AsyncCheckpoint`.
- **Evidence:** Verified GPU benchmarks (e.g., 1.32x speedup on ResNet-50, 1.57x speedup on MNIST CPU via overlapped async eval).
- **Current Status:** In progress (version 1.0.0a1). Benchmarking and correctness validation heavily prioritized.

### Workstream: AI Execution Plugin (`zakuro-poc-plugin`)
- **Purpose:** Allow Claude/Codex to safely translate natural language intent into executed compute jobs.
- **Concrete Outputs:** Pydantic schemas, explicit security policies (no raw shell, no network by default), abstract `ExecutionBackend`, isolated `DockerBackend`, Typer CLI wrapper (`validate`, `plan-show`, `execute`), Claude `SKILL.md`, root GitHub Actions workflow, structured artefact handling (with concurrency safety), Docker non-root defaults, a fake backend contract suite (with failure normalisation), a `ZakuroBackend` subprocess adapter for `zc execute`, formal threat model documentation (`THREAT_MODEL.md`), and plugin-specific docs.
- **Evidence of Completion:** Local plugin suite passes with `102 passed` and `100%` coverage for `zakuro_poc` (including strict tests for `zc` JSON panics); Docker-backed integration tests pass against a reachable daemon; agent workflows enforce validation without bypasses; lint, type-check, and diff checks pass.
- **Current Status:** Implemented POC, complete with limitations. The plugin is functionally complete and secure as a local execution scaffold, but it still depends on an external `zc` runtime contract for the native Zakuro path.

## 5. Current State
- **Zakuro Runtime:** In progress.
- **Sakura Services:** In progress.
- **Zakuro POC Plugin:** Implemented POC, locally validated and security-hardened, complete with limitations.
- **Rust Broker (`zc`):** Partially implemented / External dependency.
- **Federated Marketplace / Billing:** Deferred.

## 6. Remaining Work
- **Mature the Rust Broker (`zc`):**
  - *What:* Finalize the high-performance Rust control plane.
  - *Why:* Required to scale beyond local clusters and hit the 10-50K RPS targets for the federated mesh.
- **Evolve the Plugin Backend Contract:**
  - *What:* Replace the current subprocess adapter with a deeper native integration once the Rust broker contract is stable and observable.
  - *Why:* The current plugin already executes safely, but its native Zakuro path still depends on an external executable contract.
- **Marketplace & Billing Infrastructure:**
  - *What:* Re-introduce enterprise billing, PostgreSQL ledgers, and SLA compliance models.
  - *Why:* Necessary to realize the ultimate commercial vision outlined in the investor memo.

## 7. Risks and Open Questions
- **Product Identity Drift:** The project oscillates between being a Python ML optimizer, a distributed worker runtime, and a federated compute marketplace. Marketing these simultaneously risks diluting the core developer value proposition.
- **Security Posture (Cloudpickle):** Shipping functions via `cloudpickle` inherently carries severe code execution risks. The runtime requires a hardened zero-trust isolation model before enterprise multi-tenant deployments can be certified, as formally outlined in the plugin's `THREAT_MODEL.md`.
- **Framework Overreach:** Supporting raw PyTorch DDP, Lightning, Hugging Face, Ray, Dask, and Spark simultaneously creates an enormous maintenance and compatibility surface area for an early-stage team.
- **GPU Async Validation:** AsyncEval is highly effective for CPU workloads where eval cost approximates training cost, but blocks on heavily GPU-bound workloads. Stream-based GPU dispatchers are required.

## 8. Status Matrix

| Workstream | Current Status | Evidence | Remaining Work | Risk Level | Recommended Next Action |
|------------|----------------|----------|----------------|------------|-------------------------|
| **Sakura Services** | In Progress (v1.0a1) | Reproducible bench harness, NCCL correctness tests | Stream-based GPU dispatchers, expanded multi-task workloads | Medium | Finalize v1.0.0 release. |
| **Zakuro Runtime** | In Progress (v0.3) | `AdaptiveCompute` drift detection and QUIC transport implemented | Transition from standalone clusters to Rust broker mesh | Low | Stabilize worker API and QUIC reliability. |
| **Zakuro POC Plugin** | Implemented POC, complete with limitations | Structured models, CLI, Docker backend, hardened agent guidance, root CI, robust mock tests for `zc` failure modes, `THREAT_MODEL.md`, `100%` local coverage | Replace subprocess-based native backend with deeper `zc` integration once the broker contract is stable | Medium | Stabilise the `zc` contract and decide whether a native backend API is worth the complexity. |
| **Federated Marketplace** | Deferred | Mentioned in PRD/Memos, lack of active ledger codebase | Billing, identity, SLA enforcement, Dashboard | High | Keep deferred until runtime adoption is proven. |

## 9. Recommended Next Steps
- **Immediate:** Preserve the plugin’s current `100%` coverage and Docker validation, and keep the current execution contract stable.
- **Short-Term:** Solidify `sakura`'s GPU asynchronous dispatch mechanisms to prevent blocking on heavily GPU-bound evaluation workloads.
- **Medium-Term:** Mature the `zc` Rust broker and decide whether the subprocess-based plugin backend should be replaced with a native integration or retained as the stable contract boundary.
- **Deferred:** Enterprise marketplace billing and provider liquidity generation.

## 10. Final Assessment
The project is on track and making strong technical decisions by narrowing its focus from a broad marketplace concept to a measurable distributed ML runtime. The largest gap is now strategic rather than local: deciding how far to take the `zc` / native Zakuro integration beyond the already working subprocess-backed plugin. The largest risk remains maintaining focus across too many supported ML frameworks while also securing Python serialization and container execution boundaries. The immediate next step is to preserve the current plugin contract, monitor the native broker interface, and avoid expanding scope before the `zc` path stabilises further.
