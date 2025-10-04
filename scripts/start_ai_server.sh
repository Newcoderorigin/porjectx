#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")"/.. && pwd)"
export PYTHONPATH="${ROOT_DIR}:${PYTHONPATH-}"

if [[ -z "${VIRTUAL_ENV-}" ]]; then
  echo "[ai-server] warning: no virtualenv detected; continuing with system interpreter" >&2
fi

python -m toptek.ai_server.app "$@"
