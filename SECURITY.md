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

## RemoteDispatcher trust model

`RemoteDispatcher` (and the `LocalDispatcher` that auto-spawns a local worker)
use `cloudpickle` to serialise callables over the QUIC wire.
**cloudpickle payloads are not cryptographically verified.** This means:

- The worker you connect to **must be operator-controlled**. Connecting to an
  untrusted or publicly-accessible endpoint lets a malicious worker return an
  arbitrary cloudpickle payload that executes code in your training process
  (CWE-502 / RCE).
- Likewise, a worker exposes a `HANDLER_EXEC_CLOUDPICKLED` endpoint: do **not**
  expose a sakura-worker's QUIC port to untrusted clients.

The safe usage pattern is:
```
LocalDispatcher()          # auto-spawned localhost worker — safe
RemoteDispatcher(uri=...)  # operator-controlled worker on a private network — safe
RemoteDispatcher(uri=...)  # public/untrusted endpoint — NOT SAFE
```

Migration to a safe postcard/serde codec that eliminates cloudpickle from the
RPC path is tracked in [zakuro#117](https://github.com/zakuro-ai/zakuro/issues/117).
Until that lands, the two call sites are marked with
`# nosemgrep: sakura.deserialization.cloudpickle-loads-untrusted` so the Semgrep
gate blocks **new** cloudpickle call sites from being merged.

## Secrets

Never commit secrets (`.env`, private keys, credentials, database dumps) to this
repository. Such files must be listed in `.gitignore`. If a secret is committed,
treat it as compromised: rotate it immediately and purge it from git history.
