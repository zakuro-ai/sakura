# /zakuro

Interpret the rest of the user message as a request for a Zakuro POC execution job.

Follow the Zakuro skill workflow:

1. Create JSON execution plan.
2. Validate with `zakuro-poc validate`.
3. Show plan with `zakuro-poc plan-show`.
4. Ask for explicit approval.
5. Execute only after approval with `zakuro-poc execute --yes`.
6. Return stdout, stderr, exit code, duration, and artifact path.
