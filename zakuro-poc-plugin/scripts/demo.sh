#!/usr/bin/env bash
set -euo pipefail

if [ -d ".venv" ]; then
    source .venv/bin/activate
fi

zakuro-poc doctor
zakuro-poc validate --plan examples/plan.echo.json
zakuro-poc plan-show --plan examples/plan.echo.json
zakuro-poc execute --plan examples/plan.echo.json --yes
