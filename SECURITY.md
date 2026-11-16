# Security Policy

## Reporting a Vulnerability

Please report security vulnerabilities **privately** to **security@zakuro-ai.com**.
Do not open a public issue for security problems.

Include where possible:
- A description of the vulnerability and its impact
- Steps to reproduce or a proof of concept
- Affected version(s) / commit

We aim to acknowledge reports within 3 business days.

## Supported Versions

Security fixes are applied to the latest released version and the default
(`main`) branch. Older versions are not maintained.

## Secrets

Never commit secrets (`.env`, private keys, credentials, database dumps) to this
repository. Such files must be listed in `.gitignore`. If a secret is committed,
treat it as compromised: rotate it immediately and purge it from git history.

## SAST

`.github/workflows/sast.yml` runs CodeQL, gitleaks and Semgrep
(`p/python`, `p/security-audit`, `p/owasp-top-ten` and the custom rules in
`.semgrep/sakura.yml`). The Semgrep lane is strict: it runs with `--error`, so
any unsuppressed finding fails the job. Its SARIF is kept as a workflow
artifact when the scan fails, and uploaded to code scanning only where GitHub
Advanced Security is on (repo variable `CODE_SCANNING_ENABLED=true`). The job
log report also lists `nosemgrep`-suppressed findings, so its count can exceed
the number that fails the job.

A finding is fixed when the fix is safe. Otherwise it is suppressed with a
`nosemgrep: <full rule id>` comment on the matched line or the line before it,
next to a comment that says why the pattern is safe there. Never use a bare
`nosemgrep` (it hides every rule), and never drop `--error` to get a green run.
To reproduce the lane locally, install `setuptools<81` and `semgrep==1.92.0`
and run the same `semgrep scan ... --error` command as the workflow.
