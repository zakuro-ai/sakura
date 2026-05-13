# Strategy Memo: Zakuro AI Project

## 1. Executive Summary
The Zakuro project has evolved from a simple Python ML helper library (Sakura) into a sophisticated context-aware distributed-ML runtime (Zakuro) paired with an asynchronous ML training services layer (Sakura v1.0). The original commercial vision depicted a federated fog-compute marketplace, but the immediate engineering focus has successfully narrowed to delivering a robust, adaptive developer runtime that decouples ML training from blocking operations (evaluation, checkpointing) via a QUIC-backed transport and rust-based optimizations.

The project is technically sound, demonstrating rigorous benchmarking and a healthy engineering culture focused on isolating and measuring distributed failure modes. The recent addition of the `zakuro-poc-plugin` successfully bridges AI coding agents (Claude/Codex) into this ecosystem by establishing a secure, verifiable execution pipeline (`Intent -> Plan -> Consent -> Isolated Backend -> Observable Result`).

The main achievement is the concrete separation of the ML training layer (Sakura) from the execution substrate (Zakuro) and the successful abstraction of the execution backend via the new AI plugin. The largest remaining work lies in maturing the Rust broker (`zc`) and bridging the current developer-runtime into the broader marketplace vision (billing, enterprise isolation).

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
- **Phase 4 (Current):** Implementation of the `zakuro-poc-plugin`. Establishing a controlled, structured execution boundary enabling LLMs to safely orchestrate compute tasks within isolated Docker backends, acting as a precursor for the Rust `zc` broker.

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
- **Concrete Outputs:** Pydantic schemas, explicit security policies (no raw shell, no network by default), abstract `ExecutionBackend`, isolated `DockerBackend`, Typer CLI wrapper (`validate`, `plan-show`, `execute`), and Claude `SKILL.md`.
- **Current Status:** Complete. Fully tested and CI-gated. 

## 5. Current State
- **Zakuro Runtime:** In progress.
- **Sakura Services:** In progress.
- **Zakuro POC Plugin:** Complete.
- **Rust Broker (`zc`):** Partially implemented / External dependency.
- **Federated Marketplace / Billing:** Deferred.

## 6. Remaining Work
- **Transition POC Plugin to Zakuro Backend:** 
  - *What:* Swap the `DockerBackend` abstraction in the `zakuro-poc-plugin` to utilize `zk.Compute` or `zc execute`.
  - *Why:* To fully integrate the AI agent workflow into the native Zakuro distributed mesh instead of a local Docker simulation.
- **Mature the Rust Broker (`zc`):**
  - *What:* Finalize the high-performance Rust control plane.
  - *Why:* Required to scale beyond local clusters and hit the 10-50K RPS targets for the federated mesh.
- **Marketplace & Billing Infrastructure:**
  - *What:* Re-introduce enterprise billing, PostgreSQL ledgers, and SLA compliance models.
  - *Why:* Necessary to realize the ultimate commercial vision outlined in the investor memo.

## 7. Risks and Open Questions
- **Product Identity Drift:** The project oscillates between being a Python ML optimizer, a distributed worker runtime, and a federated compute marketplace. Marketing these simultaneously risks diluting the core developer value proposition.
- **Security Posture (Cloudpickle):** Shipping functions via `cloudpickle` inherently carries severe code execution risks. The runtime requires a hardened zero-trust isolation model before enterprise multi-tenant deployments can be certified.
- **Framework Overreach:** Supporting raw PyTorch DDP, Lightning, Hugging Face, Ray, Dask, and Spark simultaneously creates an enormous maintenance and compatibility surface area for an early-stage team.
- **GPU Async Validation:** AsyncEval is highly effective for CPU workloads where eval cost approximates training cost, but blocks on heavily GPU-bound workloads. Stream-based GPU dispatchers are required.

## 8. Status Matrix

| Workstream | Current Status | Evidence | Remaining Work | Risk Level | Recommended Next Action |
|------------|----------------|----------|----------------|------------|-------------------------|
| **Sakura Services** | In Progress (v1.0a1) | Reproducible bench harness, NCCL correctness tests | Stream-based GPU dispatchers, expanded multi-task workloads | Medium | Finalize v1.0.0 release. |
| **Zakuro Runtime** | In Progress (v0.3) | `AdaptiveCompute` drift detection and QUIC transport implemented | Transition from standalone clusters to Rust broker mesh | Low | Stabilize worker API and QUIC reliability. |
| **Zakuro POC Plugin** | Complete | Passing CI, strict security policies, functional Docker Backend | Swap `DockerBackend` for `ZakuroBackend` / `zc execute` | Low | Connect POC to live Zakuro cluster. |
| **Federated Marketplace** | Deferred | Mentioned in PRD/Memos, lack of active ledger codebase | Billing, identity, SLA enforcement, Dashboard | High | Keep deferred until runtime adoption is proven. |

## 9. Recommended Next Steps
- **Immediate:** Integrate the completed `zakuro-poc-plugin` with the existing `zakuro` python runtime, allowing the AI agent to dispatch tasks via `zk.Compute` rather than raw Docker.
- **Short-Term:** Solidify `sakura`'s GPU asynchronous dispatch mechanisms to prevent blocking on heavily GPU-bound evaluation workloads.
- **Medium-Term:** Mature the `zc` Rust broker and transition the orchestration layer off Python `subprocess` into native Rust mesh management.
- **Deferred:** Enterprise marketplace billing and provider liquidity generation.

## 10. Final Assessment
The project is on track and making excellent technical decisions by narrowing its focus from a grandiose marketplace to a highly optimized, measurable distributed ML runtime. The largest gap is the missing integration between the newly minted AI execution plugin and the actual native Zakuro distributed mesh. The largest risk is maintaining focus across too many supported ML frameworks and failing to adequately secure the Python serialization boundaries for enterprise deployment. The immediate next step is to hook the `zakuro-poc-plugin` into `zk.Compute` to complete the ecosystem loop.
