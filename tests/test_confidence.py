from __future__ import annotations

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("scipy")

from toptek.confidence import score_probabilities


def test_score_probabilities_fast() -> None:
    probs = np.linspace(0.55, 0.75, 20)
    result = score_probabilities(probs, method="fast")
    assert 0.6 < result.probability < 0.7
    assert result.confidence > 0
    assert result.ci_high <= 1.0
    assert result.expected_value > 0


def test_score_probabilities_beta() -> None:
    probs = [0.6] * 15
    result = score_probabilities(probs, method="beta")
    assert result.ci_low < result.probability < result.ci_high
