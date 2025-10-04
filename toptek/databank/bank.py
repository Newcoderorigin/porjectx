"""Parquet-backed market data bank with deterministic ingest."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path

from typing import Sequence

import pandas as pd

from .providers import BarsProvider, SyntheticBars


@dataclass(slots=True)
class Bank:
    """Manage historical bars stored as Parquet partitions."""

    root: Path
    provider: BarsProvider = field(default_factory=SyntheticBars)

    def __post_init__(self) -> None:
        if isinstance(self.root, str):  # pragma: no cover - convenience
            self.root = Path(self.root)
        self.root = self.root.expanduser().resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    def ingest(
        self,
        symbol: str,
        timeframe: str,
        *,
        days: int,
        end: datetime | None = None,
    ) -> Path:
        """Download deterministic data and persist partitions.

        Returns the path to the symbol/timeframe directory.
        """

        if days <= 0:
            raise ValueError("days must be positive")
        end = end or datetime.now(timezone.utc)
        start = end - timedelta(days=days)
        data = self.provider.fetch(symbol, timeframe, start=start, end=end)
        if data.empty:
            raise RuntimeError("Provider returned no data")
        if data.index.tzinfo is None:
            data.index = data.index.tz_localize(timezone.utc)
        path = self.root / symbol.upper() / timeframe
        path.mkdir(parents=True, exist_ok=True)
        daily_groups = data.groupby(data.index.date)
        for day, frame in daily_groups:
            day_str = datetime.combine(
                day, datetime.min.time(), tzinfo=timezone.utc
            ).strftime("%Y%m%d")
            target = path / f"{day_str}.parquet"
            if target.exists():
                continue
            frame.to_parquet(target)
        catalog_path = path / "catalog.json"
        partitions = sorted(p.name for p in path.glob("*.parquet"))
        catalog = {
            "symbol": symbol.upper(),
            "timeframe": timeframe,
            "partitions": partitions,
        }
        catalog_path.write_text(json.dumps(catalog, indent=2), encoding="utf-8")
        return path

    def read(
        self,
        symbol: str,
        timeframe: str,
        *,
        start: datetime | None = None,
        end: datetime | None = None,
    ) -> pd.DataFrame:
        """Load OHLCV bars for the given window."""

        path = self.root / symbol.upper() / timeframe
        if not path.exists():
            raise FileNotFoundError(f"No data for {symbol} {timeframe}")
        frames: list[pd.DataFrame] = []
        for parquet_file in sorted(path.glob("*.parquet")):
            frame = pd.read_parquet(parquet_file)
            frames.append(frame)
        if not frames:
            raise RuntimeError("No partitions found")
        data = pd.concat(frames).sort_index()
        if start:
            data = data[data.index >= start]
        if end:
            data = data[data.index <= end]
        return data

    def catalog(self, symbol: str, timeframe: str) -> dict[str, list[str]]:
        """Return the stored partition catalog for the instrument."""

        path = self.root / symbol.upper() / timeframe / "catalog.json"
        if not path.exists():
            raise FileNotFoundError("Catalog missing; run ingest first")
        data = json.loads(path.read_text(encoding="utf-8"))
        data["partitions"] = list(data.get("partitions", []))
        return data


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Toptek Data Bank CLI")
    parser.add_argument("command", choices=["ingest"])
    parser.add_argument("--symbol", required=True)
    parser.add_argument("--timeframe", required=True)
    parser.add_argument("--days", type=int, default=365)
    parser.add_argument("--root", default="data/bank")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    bank = Bank(Path(args.root))
    if args.command == "ingest":
        bank.ingest(args.symbol, args.timeframe, days=args.days)
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entrypoint
    raise SystemExit(main())
