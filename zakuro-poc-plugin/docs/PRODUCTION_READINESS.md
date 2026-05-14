# Production-Readiness Checklist

## Functional Readiness
- [x] CLI installs in editable mode.
- [x] `zakuro-poc validate` accepts safe example plans.
- [x] `zakuro-poc plan-show` is non-executing.
- [x] `zakuro-poc execute` validates before prompting for confirmation.
- [x] Rejected plans produce structured artifacts.
- [ ] Full release demo has been repeated on a clean machine.

## Security Readiness
- [x] No execution occurs without explicit confirmation or `--yes`.
- [x] Validation occurs before confirmation and backend selection.
- [x] Network disabled by default.
- [x] Shell execution rejected by default.
- [x] Secrets rejected from env.
- [x] Configured CPU, memory, and timeout maxima are enforced.
- [x] Docker image `latest` tags are rejected unless config explicitly allows them.
- [ ] Multi-tenant execution threat model is documented.

## Docker Isolation
- [x] Docker backend captures stdout.
- [x] Docker backend captures stderr.
- [x] Docker backend captures exit code.
- [x] Timeout is enforced.
- [x] Docker command includes `--security-opt no-new-privileges`.
- [x] Docker command includes `--cap-drop ALL`.
- [x] Docker command includes `--pids-limit`.
- [x] Docker root filesystem can be made read-only through config.
- [ ] Docker runs as a non-root user by default.

## Testing Readiness
- [x] Unit tests pass.
- [x] Non-Docker integration tests pass.
- [x] Docker smoke tests pass locally when Docker socket access is allowed.
- [ ] Docker smoke tests are confirmed in CI.

## CI/CD Readiness
- [x] Plugin CI is wired from the repository root workflow directory.
- [ ] Plugin CI has passed on the remote GitHub runner after this branch update.

## Observability Readiness
- [x] Artifacts written per job.
- [x] Success, failure, timeout, Docker-unavailable, and rejected outcomes write structured result artifacts.

## Documentation Readiness
- [x] README quickstart verified.
- [x] Claude skill installed and documented.
- [x] Current strategy memo classifies the plugin as an implemented POC, not complete.
- [ ] `ZakuroBackend` / `zc execute` user-facing documentation is written.

## Known Limitations
- Docker smoke tests require access to the host Docker socket; sandboxed agent sessions may need explicit approval.
- Docker containers do not yet run as a non-root user by default.
- The plugin has no implemented `ZakuroBackend` / `zc execute` backend yet.
- The current Docker backend is a local execution POC, not a substitute for a multi-tenant remote execution service.

## Release Blockers
- Do not call the plugin production-ready until Docker non-root execution and remote CI validation are complete.
- Do not implement `ZakuroBackend` until its contract preserves the current plan, validation, consent, and artifact semantics.

## Latest Local Validation
- Date: 2026-05-14.
- Editable install: `UV_CACHE_DIR=/tmp/uv-cache uv pip install -e ".[dev]"`.
- Non-Docker suite: `.venv/bin/pytest -m "not docker"` passed with 61 selected tests.
- Docker smoke suite: `.venv/bin/pytest -m docker tests/integration/test_docker_backend.py` passed with 4 selected tests after Docker socket access was approved.
