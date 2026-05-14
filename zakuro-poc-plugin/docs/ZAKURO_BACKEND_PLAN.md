# Zakuro Backend Plan

## Objective

Add and evolve a `ZakuroBackend` after the Docker-backed POC is stable enough to preserve the same controlled execution contract:

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
| Backend contract audit | Confirm the current interface is sufficient for a non-Docker backend | Done: fake backend contract tests cover runner and artifact invariants |
| Backend type model | Add `zakuro` only when the backend contract is ready | Done: `BackendType` accepts `zakuro` |
| Command mapping | Define how `ExecutionPlan` maps to `zc execute` or `zk.Compute` | Initial implementation: configurable `zc execute --plan <plan.json> --json` |
| Result normalisation | Convert native Zakuro/`zc` output into `ExecutionResult` | Initial implementation: stdout, stderr, exit code, timeout, and missing executable are normalised |
| Artifact preservation | Keep artifact layout identical across backends | Done for fake backend and missing-executable real backend tests |
| Documentation | Explain when to use Docker versus Zakuro execution | README and Codex/Claude guidance updates |

## Open Questions

- Should a later backend use `zk.Compute` directly, or should all native execution go through `zc execute`?
- What is the stable machine-readable output contract for `zc execute`?
- How are remote artifacts returned: copied into the local artifact directory, referenced by URI, or both?
- How are timeouts enforced across the local CLI and the remote execution substrate?
- What credentials, if any, are required for remote execution, and how are they kept out of plans and artifacts?
- What is the expected isolation boundary for untrusted agent-generated workloads?

## Recommended Sequence

1. Confirm the external `zc execute` command-line and JSON-output contract.
2. Add live `zc` integration tests behind an explicit marker once `zc` is available in CI.
3. Update Claude and Codex instructions only after live `zc` integration tests pass.
4. Decide whether `zk.Compute` should be a separate backend or remain outside this plugin.

## Current Implementation Boundary

The current `ZakuroBackend` is a conservative subprocess adapter for `zc execute`. It does not infer success from remote metadata, copy remote artifacts, or inject credentials. Those behaviours require an explicit `zc` contract before they are added.
