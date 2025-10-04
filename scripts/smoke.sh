#!/usr/bin/env bash
set -euo pipefail

echo "[smoke] compiling python modules"
python -m compileall toptek >/dev/null

echo "[smoke] running targeted tests"
pytest -q tests/test_config_schema.py tests/test_lmstudio_client.py
