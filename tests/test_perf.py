"""Performance harness smoke test (gated by PERF_CHECK)."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

pytest.importorskip("numpy")

from bench.run_bench import main as run_bench


@pytest.mark.skipif(os.getenv("PERF_CHECK") != "1", reason="PERF_CHECK env not set")
def test_run_bench(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    run_bench()
    output = Path("reports/baselines/latest.json")
    assert output.exists()
    data = json.loads(output.read_text(encoding="utf-8"))
    assert data["horizon"] == 32
