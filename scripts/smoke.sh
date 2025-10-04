#!/usr/bin/env bash
set -euo pipefail

pytest -q --maxfail=1 \
  tests/test_lmstudio_client.py \
  tests/test_futures_yahoo_urls.py \
  tests/test_ui_cfg_schema.py
