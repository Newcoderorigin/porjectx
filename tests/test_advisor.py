from __future__ import annotations

import pytest

pytest.importorskip("numpy")
pytest.importorskip("pandas")

from toptek.advisor import AdvisorEngine


def test_advisor_engine() -> None:
    engine = AdvisorEngine()
    response = engine.advise("AAPL")
    assert response.symbol == "AAPL"
    assert len(response.bullets) == 3
    assert response.recommendation
