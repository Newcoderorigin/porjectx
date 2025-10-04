from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest

pytest.importorskip("pandas")

from toptek.databank import Bank, SyntheticBars


def test_ingest_and_read(tmp_path: Path) -> None:
    bank = Bank(tmp_path / "bank", provider=SyntheticBars(seed=1))
    target_end = datetime(2024, 1, 31, tzinfo=timezone.utc)
    bank.ingest("ES", "5m", days=2, end=target_end)
    df = bank.read("ES", "5m")
    assert not df.empty
    assert {"open", "high", "low", "close", "volume"}.issubset(df.columns)
    catalog = bank.catalog("ES", "5m")
    assert catalog["symbol"] == "ES"
    assert catalog["timeframe"] == "5m"
    partitions = catalog["partitions"]
    assert isinstance(partitions, list)
    assert len(list((tmp_path / "bank" / "ES" / "5m").glob("*.parquet"))) == len(
        partitions
    )
