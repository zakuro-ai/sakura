# Zakuro POC Plugin

## What This Is
A local Codex/Claude-compatible execution plugin POC for Zakuro-style backend abstraction.

## What This Is Not
This is not the full Zakuro federated compute marketplace or Rust broker.

## Architecture
```text
Claude/Codex
  -> JSON ExecutionPlan
  -> zakuro-poc CLI
  -> plan validation
  -> backend adapter
  -> DockerBackend or ZakuroBackend
  -> ExecutionResult
  -> artifacts
```

## Installation
```bash
git clone <repo-url>
cd zakuro-poc-plugin
python3.11 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"

zakuro-poc doctor
zakuro-poc validate --plan examples/plan.echo.json
zakuro-poc plan-show --plan examples/plan.echo.json
zakuro-poc execute --plan examples/plan.echo.json --yes
```

## Docker Prerequisites
Docker must be installed and running.

## Quickstart
See installation steps.

## Example Plans
Check the `examples/` directory.

## Claude Code Setup
Run `./scripts/install-claude-skill.sh`.

Ask Claude:

1. `/zakuro run a Python hello-world job in Docker`
2. `/zakuro run a small deterministic Python computation`
3. `/zakuro clone a tiny public repository and show the files, using network only if I approve it`

## Codex Usage
See `codex/README.md`.

## Configuration
See `config/zakuro-poc.example.json`.

## Security Model
See `AGENTS.md` and security policies.

## Artifact Layout
Per-job artifacts are written to the configured artifact root.

## Development
Run `./scripts/check.sh`.

## Testing
Run `./scripts/test.sh`.

Latest local validation, run on 2026-05-14:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv pip install -e ".[dev]"
.venv/bin/pytest -m "not docker"
.venv/bin/pytest -m docker tests/integration/test_docker_backend.py
```

The Docker smoke suite requires access to the host Docker socket. In sandboxed agent sessions, Docker commands may require explicit approval even when Docker Desktop is already running.

## Troubleshooting
Check Docker daemon.

## Roadmap: Replacing Docker with `zc execute`
`ZakuroBackend` is available as a conservative adapter around a configurable `zc execute --plan <plan.json> --json` command. Live `zc` usage should remain gated until the external `zc` command contract and artifact-return behaviour are confirmed. See `docs/ZAKURO_BACKEND_PLAN.md`.
