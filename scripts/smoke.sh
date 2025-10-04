#!/usr/bin/env bash
set -euo pipefail

pytest -q -k 'lmstudio or futures or ui_cfg_schema' --maxfail=1
