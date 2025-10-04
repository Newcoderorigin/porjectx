"""Synthetic performance harness producing deterministic baselines."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


def run_scenario(config: dict[str, Any]) -> dict[str, float]:
    seed = int(config.get("seed", 42))
    horizon = int(config.get("horizon", 256))
    rng = np.random.default_rng(seed)
    pnl = rng.normal(config.get("drift", 0.05), config.get("vol", 0.2), size=horizon)
    sharpe = float(np.mean(pnl) / (np.std(pnl) + 1e-9))
    hit_rate = float((pnl > 0).mean())
    return {"sharpe": sharpe, "hit_rate": hit_rate, "horizon": horizon}


def main() -> None:
    scenario_path = Path(__file__).with_name("scenario_small.yaml")
    if not scenario_path.exists():
        scenario_path = Path("bench/scenario_small.yaml")
    if not scenario_path.exists():
        raise FileNotFoundError("scenario_small.yaml not found")
    config = json.loads(scenario_path.read_text(encoding="utf-8"))
    metrics = run_scenario(config)
    out_dir = Path("reports/baselines")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "latest.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
