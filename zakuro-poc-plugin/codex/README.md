# Using Zakuro POC with Codex

Codex should not call Docker directly.

Use this workflow:

1. Generate an execution plan JSON.
2. Save it to `tmp/plan.json`.
3. Validate:

   `zakuro-poc validate --plan tmp/plan.json`

4. Show the plan:

   `zakuro-poc plan-show --plan tmp/plan.json`

5. Ask the user for explicit confirmation.

6. Execute:

   `zakuro-poc execute --plan tmp/plan.json --yes`

7. Report:

   - job id;
   - status;
   - stdout;
   - stderr;
   - exit code;
   - duration;
   - artifact path.

Never pass secrets. Never run raw Docker. Never skip validation.
