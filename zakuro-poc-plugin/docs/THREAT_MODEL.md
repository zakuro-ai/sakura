# Zakuro Execution Threat Model

This document outlines the security posture, assumptions, and threat boundaries of the Zakuro plugin and execution system as of the local POC phase.

## 1. System Scope and Trust Boundaries

The current `zakuro-poc-plugin` provides an execution abstraction (`ExecutionBackend`) that allows AI agents to submit structured compute requests (`ExecutionPlan`). 

### The Trust Boundary
The primary trust boundary is the **Validation and Consent Layer**.
- AI Agents (Claude, Codex) operate *outside* the trusted execution zone. They can generate intent, but cannot execute it.
- The `zakuro-poc` CLI is the gatekeeper. It enforces structural constraints, resource ceilings, and security policies before prompting the human user for explicit consent.
- The execution backend (e.g., Docker, `zc`) operates *inside* the trusted zone but is treated with extreme defensive pessimism.

## 2. Docker Local POC (Single-Tenant)

The `DockerBackend` is designed strictly as a local, single-tenant simulator of the future Zakuro runtime. It is **not** designed for remote, multi-tenant execution.

### Mitigated Threats
- **Host Escape:** Containers are run with `--security-opt no-new-privileges`, `--cap-drop ALL`, and optionally a read-only root file system. Host networking and privileged execution are explicitly forbidden.
- **Resource Exhaustion:** Memory, CPU, timeout, and PID limits (`--pids-limit`) are enforced to prevent fork bombs or compute starvation from crippling the host machine.
- **Silent Failures:** Artifact directories are created even if validation fails or a timeout occurs, ensuring an immutable audit trail.
- **Concurrency Collisions:** Artifact directories use deterministic, non-colliding paths (`exist_ok=False`), preventing concurrent agent jobs from overwriting each other's state.

### Unmitigated Risks (Accepted for Local POC)
- **Daemon Access:** The user running the plugin must have access to the Docker socket. If an agent manages to coerce the user into running an un-sandboxed command *outside* the plugin, host compromise is possible.
- **Storage Quotas:** While RAM and CPU are capped, disk space within the `/workspace` mount is not inherently quota-limited by the current Docker backend implementation.

## 3. Future Multi-Tenant Remote Execution (Zakuro Mesh)

The ultimate goal is to replace the Docker backend with the Rust-based Zakuro broker (`zc`), connecting to a federated remote compute mesh.

### Severe Multi-Tenant Risks
- **Cloudpickle Serialisation:** Python functions and closures transmitted via `cloudpickle` are inherently capable of arbitrary code execution upon deserialisation. In a multi-tenant environment, standard OS-level user isolation is insufficient.
- **Secret Exfiltration:** If node operators in the federated marketplace are untrusted, executing sensitive workloads (e.g., fine-tuning proprietary data, passing API keys) carries a high risk of exfiltration.

### Required Future Mitigations
1. **Zero-Trust Sandboxing:** Before multi-tenant deployment, worker nodes must execute Python processes within firewalled, hardware-assisted sandboxes (e.g., Firecracker microVMs or gVisor) rather than standard Docker containers.
2. **Deterministic Deserialisation:** The reliance on `cloudpickle` must be heavily restricted or replaced with strongly typed, ahead-of-time registered computational graphs to prevent deserialisation exploits.

## 4. Conclusion

The current `zakuro-poc-plugin` is safe for local, single-tenant use by an AI agent, provided the human operator reviews the validation plan and grants explicit consent. It does not yet meet the rigorous isolation requirements needed for a public, federated compute marketplace.