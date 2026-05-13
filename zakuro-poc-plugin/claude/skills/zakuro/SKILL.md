# Zakuro Local Execution POC

Use this skill when the user asks to run an isolated compute job through the Zakuro POC backend.

## Safety Rules

- Never execute before showing a plan.
- Never call `docker run` directly.
- Always use `zakuro-poc`.
- Always generate a JSON execution plan first.
- Always validate the plan.
- Always ask the user for explicit confirmation.
- Use Docker backend only as a local simulation of future Zakuro execution.
- Prefer `network_enabled: false`.
- Do not include secrets in env.
- Do not use shell commands unless explicitly approved and safe.
- Do not use `latest` image tags.

## Workflow

1. Understand the user's requested job.
2. Create an execution plan JSON.
3. Save it to a temporary file or ask user where to save it.
4. Run:

   `zakuro-poc validate --plan <plan.json>`

5. Show the readable plan:

   `zakuro-poc plan-show --plan <plan.json>`

6. Ask:

   "Do you explicitly approve running this Docker-backed Zakuro POC job?"

7. If approved, run:

   `zakuro-poc execute --plan <plan.json> --yes`

8. Summarise:

   - job id;
   - status;
   - stdout;
   - stderr;
   - exit code;
   - duration;
   - artifact path.

## Plan Template

```json
{
  "job_name": "example-job",
  "backend": "docker",
  "image": "python:3.11-slim",
  "command": ["python", "-c", "print('hello')"],
  "working_dir": null,
  "repo_url": null,
  "env": {},
  "resource_limits": {
    "cpu_count": 1.0,
    "memory_mb": 512,
    "gpu_count": 0,
    "timeout_seconds": 30
  },
  "artifact_dir": null,
  "network_enabled": false,
  "created_by": "claude-code"
}
```

## Never Do This

- Never run raw `docker run`.
- Never run arbitrary shell strings.
- Never mount the Docker socket.
- Never mount the whole home directory.
- Never pass secrets.
- Never skip confirmation.
