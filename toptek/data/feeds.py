"""Streaming helpers for querying fresh market bars from SQLite."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Iterable, Iterator, List, Sequence

import pandas as pd
import sqlite3

_BAR_COLUMNS = ["open", "high", "low", "close", "volume"]


@dataclass(frozen=True)
class BarRecord:
    """Immutable representation of a single OHLCV bar."""

    symbol: str
    ts: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float

    @classmethod
    def from_row(cls, row: Sequence[object]) -> "BarRecord":
        return cls(
            symbol=str(row[0]),
            ts=datetime.fromisoformat(str(row[1])),
            open=float(row[2]),
            high=float(row[3]),
            low=float(row[4]),
            close=float(row[5]),
            volume=float(row[6]),
        )


class SQLiteBarFeed:
    """Utility for streaming incremental bars from the SQLite data layer."""

    def __init__(self, conn: sqlite3.Connection) -> None:
        self._conn = conn

    def insert(self, records: Iterable[BarRecord]) -> int:
        payload = [
            (
                record.symbol,
                record.ts.isoformat(),
                float(record.open),
                float(record.high),
                float(record.low),
                float(record.close),
                float(record.volume),
            )
            for record in records
        ]
        if not payload:
            return 0
        self._conn.executemany(
            (
                "INSERT OR REPLACE INTO market_bars(symbol, ts, open, high, low, close, volume) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)"
            ),
            payload,
        )
        self._conn.commit()
        return len(payload)

    def fetch(
        self,
        symbol: str,
        *,
        since: datetime | None = None,
        limit: int | None = None,
    ) -> pd.DataFrame:
        """Fetch bars newer than ``since`` ordered by timestamp."""

        conditions: List[str] = ["symbol = ?"]
        params: List[object] = [symbol]
        if since is not None:
            conditions.append("ts > ?")
            params.append(since.isoformat())
        where = " AND ".join(conditions)
        query = (
            "SELECT symbol, ts, open, high, low, close, volume "
            f"FROM market_bars WHERE {where} ORDER BY ts ASC"
        )
        if limit is not None:
            query += " LIMIT ?"
            params.append(int(limit))

        cursor = self._conn.execute(query, params)
        rows = cursor.fetchall()
        if not rows:
            return pd.DataFrame(columns=["symbol", *_BAR_COLUMNS])
        records = [BarRecord.from_row(row) for row in rows]
        frame = pd.DataFrame(
            {
                "symbol": [record.symbol for record in records],
                "open": [record.open for record in records],
                "high": [record.high for record in records],
                "low": [record.low for record in records],
                "close": [record.close for record in records],
                "volume": [record.volume for record in records],
            },
            index=pd.Index([record.ts for record in records], name="ts"),
        )
        return frame

    def stream(
        self,
        symbol: str,
        *,
        since: datetime | None = None,
    ) -> Iterator[pd.Series]:
        """Iterate through bars newer than ``since`` one-at-a-time."""

        frame = self.fetch(symbol, since=since)
        for timestamp, row in frame.iterrows():
            yield pd.Series(row, name=timestamp)


__all__ = ["BarRecord", "SQLiteBarFeed"]
