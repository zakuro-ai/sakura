# Zakuro Backend Plan

## Objective

Add a `ZakuroBackend` after the Docker-backed POC is stable enough to preserve the same controlled execution contract:

```text
ExecutionPlan
  -> validation and security policy
  -> explicit user consent
  -> backend interface
  -> ZakuroBackend / zc execute
  -> ExecutionResult
  -> artifacts
```

The purpose is not to bypass Docker-specific safety checks with a new backend. The purpose is to prove that the AI-agent workflow can swap execution substrates without changing the user-facing plan, consent, and artifact model.

## Non-Negotiable Requirements

- `ZakuroBackend` must implement the existing `ExecutionBackend` interface.
- CLI commands must keep the current order: load plan, validate against config, show plan, require consent, execute.
- `validate` and `plan-show` must never call `zk.Compute`, `zc`, Docker, or any remote execution path.
- `execute` must not dispatch work unless the plan is already validated.
- `ExecutionResult` must remain the canonical machine-readable result.
- Artifacts must keep the existing deterministic layout.
- Docker-specific flags and assumptions must not leak into the `ZakuroBackend` API.
- Remote execution failures must not be reported as success.
- Network and credential policy must be explicit; no secrets may be forwarded by default.

## Proposed Workstreams

| Workstream | Purpose | Required Output |
|---|---|---|
| Backend contract audit | Confirm the current interface is sufficient for a non-Docker backend | Short design note or tests proving no Docker coupling remains |
| Backend type model | Add `zakuro` only when the backend contract is ready | `BackendType` extension and validation tests |
| Command mapping | Define how `ExecutionPlan` maps to `zc execute` or `zk.Compute` | Explicit mapping table and error cases |
| Result normalisation | Convert native Zakuro/`zc` output into `ExecutionResult` | Unit tests for success, failure, timeout, and rejection |
| Artifact preservation | Keep artifact layout identical across backends | Integration tests against a fake Zakuro backend |
| Documentation | Explain when to use Docker versus Zakuro execution | README and Codex/Claude guidance updates |

## Open Questions

- Is the first native backend `zk.Compute`, `zc execute`, or a thin adapter that can support both?
- What is the stable machine-readable output contract for `zc execute`?
- How are remote artifacts returned: copied into the local artifact directory, referenced by URI, or both?
- How are timeouts enforced across the local CLI and the remote execution substrate?
- What credentials, if any, are required for remote execution, and how are they kept out of plans and artifacts?
- What is the expected isolation boundary for untrusted agent-generated workloads?

## Recommended Sequence

1. Add contract tests that assert Docker-specific behaviour stays inside `DockerBackend`.
2. Add a fake `ZakuroBackend` in tests only, proving the runner can normalise non-Docker success and failure results.
3. Decide whether the first real backend targets `zc execute` or `zk.Compute`.
4. Add the real backend behind explicit config and tests.
5. Update Claude and Codex instructions only after the backend has passing integration tests.

## Deferral

Do not implement the real `ZakuroBackend` in the same task as broad Docker hardening. Keeping these changes separate makes safety review and regression testing more reliable.
