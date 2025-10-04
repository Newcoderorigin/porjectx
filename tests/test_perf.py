"""Performance regression harness for the moving-average crossover bench."""

import logging
import os
from math import sin, pi
from pathlib import Path
from time import perf_counter
from typing import Dict, Iterable, List, Optional

import pytest

try:  # pragma: no cover - dependency optional in some environments
    import yaml
except ImportError:  # pragma: no cover - handled at runtime in the test
    yaml = None

from toptek.core import utils as core_utils

EXPECTED_HORIZON = 512
SCENARIO_PATH = Path(__file__).resolve().parents[1] / "bench" / "scenario_small.yaml"


def _load_scenario() -> Dict[str, float | int | str]:
    """Load the small bench scenario from disk."""

    if yaml is None:  # pragma: no cover - exercised in minimal environments
        pytest.skip("PyYAML is required to load bench scenarios")
    if not SCENARIO_PATH.exists():
        raise FileNotFoundError(f"Scenario missing at {SCENARIO_PATH}")
    with SCENARIO_PATH.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _generate_prices(bars: int) -> List[float]:
    """Return a deterministic OHLC close series for the bench."""

    if bars <= 0:
        raise ValueError("bars must be positive")
    prices: List[float] = []
    for idx in range(bars):
        angle = (idx / bars) * 8 * pi
        trend = 0.006 * idx
        prices.append(100.0 + sin(angle) * 5.0 + trend)
    return prices


def _moving_average(values: Iterable[float], window: int) -> List[Optional[float]]:
    """Compute a simple moving average with ``None`` padding during warm-up."""

    if window <= 0:
        raise ValueError("window must be positive")
    values_list = list(values)
    length = len(values_list)
    result: List[Optional[float]] = [None] * length
    if length < window:
        return result
    total = sum(values_list[:window])
    result[window - 1] = total / window
    for idx in range(window, length):
        total += values_list[idx] - values_list[idx - window]
        result[idx] = total / window
    return result


def _simulate_crossover(bars: int) -> Dict[str, float]:
    """Run the deterministic crossover and return aggregated metrics."""

    prices = _generate_prices(bars)
    short_ma = _moving_average(prices, 32)
    long_ma = _moving_average(prices, 200)

    position = 0
    trades = 0
    steps = 0
    wins = 0
    profit = 0.0

    for idx in range(1, len(prices) - 1):
        short_value = short_ma[idx]
        long_value = long_ma[idx]
        if short_value is None or long_value is None:
            continue
        signal = 1 if short_value > long_value else 0
        if signal != position:
            position = signal
            trades += 1
        if position:
            delta = prices[idx + 1] - prices[idx]
            if delta > 0:
                wins += 1
            profit += delta
            steps += 1

    hit_rate = wins / steps if steps else 0.0
    expectancy = profit / trades if trades else 0.0

    return {
        "trades": trades,
        "steps": steps,
        "hit_rate": hit_rate,
        "expectancy": expectancy,
    }


@pytest.mark.perf
def test_run_bench(tmp_path: Path) -> None:
    """Ensure the CI perf harness exercises all 512 bars for indicator warm-up."""

    if os.environ.get("PERF_CHECK") != "1":
        pytest.skip("Set PERF_CHECK=1 to enable perf regression assertions")

    scenario = _load_scenario()
    horizon = int(scenario["horizon_bars"])
    assert horizon == EXPECTED_HORIZON, "Scenario horizon drifted from perf gate"

    max_runtime = float(scenario["max_runtime_seconds"])
    min_trades = int(scenario["min_trades"])
    min_steps = int(scenario["min_steps"])
    min_hit_rate = float(scenario["min_hit_rate"])
    min_expectancy = float(scenario["min_expectancy"])

    logger_name = f"bench.perf.{horizon}"
    benchmark_logger = logging.getLogger(logger_name)
    benchmark_logger.handlers.clear()
    logger = core_utils.build_logger(logger_name, level="info")
    assert logger.name == logger_name

    sample_yaml = tmp_path / "sample.yaml"
    sample_yaml.write_text("key: value\n", encoding="utf-8")
    parsed_yaml = core_utils.load_yaml(sample_yaml)
    assert parsed_yaml == {"key": "value"}
    assert core_utils.load_yaml(tmp_path / "missing.yaml") == {}

    paths = core_utils.AppPaths(
        root=tmp_path, cache=tmp_path / "cache", models=tmp_path / "models"
    )
    core_utils.ensure_directories(paths)
    assert paths.cache.exists() and paths.models.exists()

    stamp = core_utils.timestamp()
    assert stamp.tzinfo is not None
    payload = core_utils.json_dumps({"when": stamp})
    assert "when" in payload
    assert core_utils.env_or_default("UNSET_PERF_KEY", "fallback") == "fallback"
    derived_paths = core_utils.build_paths(
        tmp_path, {"cache_directory": "alt_cache", "models_directory": "alt_models"}
    )
    assert derived_paths.cache == tmp_path / "alt_cache"
    assert derived_paths.models == tmp_path / "alt_models"
    assert core_utils._version_tuple("1.2.3") == (1, 2, 3)
    assert core_utils._version_tuple("1.2.dev0") == (1, 2)
    assert core_utils._version_tuple("release-2") == (0,)
    assert core_utils._version_tuple("invalid") == (0,)
    assert core_utils._compare_versions((1, 2, 0), (1, 2)) == 0
    assert core_utils._compare_versions((1, 0), (2, 0)) == -1
    assert core_utils._compare_versions((2, 1), (1, 4)) == 1
    assert core_utils._spec_matches("1.2.3", ">=1.0,<2.0")
    assert core_utils._spec_matches("1.0.0", "==1.0.0")
    assert not core_utils._spec_matches("1.0.0", "!=1.0.0")
    assert not core_utils._spec_matches("1.0.0", ">1.0.0")
    assert core_utils._spec_matches("1.0.0", "<=1.0.0")
    assert not core_utils._spec_matches("1.0.0", "<1.0.0")
    assert core_utils._spec_matches("1.0.0", " ,>=0")
    core_utils.assert_numeric_stack({"pytest": ">=0"})
    with pytest.raises(RuntimeError):
        core_utils.assert_numeric_stack({"not-a-real-package": ">=1.0"})

    start = perf_counter()
    report = _simulate_crossover(horizon)
    elapsed = perf_counter() - start

    assert elapsed <= max_runtime
    assert report["trades"] >= min_trades
    assert report["steps"] >= min_steps
    assert report["hit_rate"] >= min_hit_rate
    assert report["expectancy"] >= min_expectancy
