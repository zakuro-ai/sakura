#!/usr/bin/env bash
set -euo pipefail

if [ -d ".venv" ]; then
    source .venv/bin/activate
fi

pytest -m "not docker"
