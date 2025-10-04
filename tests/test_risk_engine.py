"""Unit tests for the Toptek risk engine."""

from __future__ import annotations

import pytest

pytest.importorskip("yaml")

from toptek.risk import RiskEngine


@pytest.mark.parametrize(
    "overrides,expected_status",
    [
        ({}, "OK"),
        ({"max_daily_loss": 10000}, "DEFENSIVE_MODE"),
        ({"max_position_size": 0}, "DEFENSIVE_MODE"),
    ],
)
def test_risk_engine_evaluate_respects_policy(overrides, expected_status):
    engine = RiskEngine.from_policy()
    profile = engine.build_profile(overrides)
    report = engine.evaluate(profile)
    assert report.status == expected_status
    assert report.to_dict()["status"] == expected_status
    assert isinstance(report.suggested_contracts, int)
    assert report.rules


def test_risk_engine_cli_dryrun_outputs_report(capsys):
    from toptek.risk.engine import main

    exit_code = main(["--dryrun"])
    captured = capsys.readouterr()
    assert exit_code == 0
    assert "Guard policy" in captured.out
    assert "Overall status" in captured.out
