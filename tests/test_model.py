"""Tests for the training helpers in :mod:`toptek.core.model`."""

from pathlib import Path

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("sklearn")

from toptek.core.model import train_classifier  # noqa: E402


def test_train_classifier_records_original_feature_count(tmp_path: Path) -> None:
    """Training should succeed even when all-NaN columns are removed."""

    rng = np.random.default_rng(42)
    X = rng.normal(size=(60, 5))
    X[:, 2] = np.nan  # Simulate a column that will be dropped entirely
    X[0, :] = np.nan  # Ensure at least one row gets dropped
    y = (X[:, 0] > X[:, 1]).astype(int)

    result = train_classifier(X, y, models_dir=tmp_path)

    assert result.original_feature_count == 5
    assert result.retained_columns == (0, 1, 3, 4)
    assert result.model_path.exists()
