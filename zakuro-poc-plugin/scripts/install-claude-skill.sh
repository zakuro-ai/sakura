#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TARGET_DIR="${HOME}/.claude/skills/zakuro"
CONFIG_TARGET="${HOME}/.claude/zakuro-poc.json"

mkdir -p "${TARGET_DIR}"
cp "${ROOT_DIR}/claude/skills/zakuro/SKILL.md" "${TARGET_DIR}/SKILL.md"

if [ ! -f "${CONFIG_TARGET}" ]; then
  mkdir -p "${HOME}/.claude"
  cp "${ROOT_DIR}/config/zakuro-poc.example.json" "${CONFIG_TARGET}"
fi

echo "Installed Zakuro Claude skill to ${TARGET_DIR}"
echo "Config available at ${CONFIG_TARGET}"
