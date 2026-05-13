#!/usr/bin/env bash
set -euo pipefail

if [ -d ".venv" ]; then
    source .venv/bin/activate
fi

ruff format --check .
ruff check .
mypy src
pytest -m "not docker" --cov=zakuro_poc --cov-report=term-missing
pip-audit || true
bandit -r src || true
