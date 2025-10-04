"""Tests for Yahoo futures URL helpers."""

from __future__ import annotations

from datetime import datetime, timezone
from urllib.parse import urlparse, parse_qs

from toptek.ui import research_futures_tab


def test_build_yahoo_url_contains_expected_components() -> None:
    start = datetime(2024, 1, 1, tzinfo=timezone.utc)
    end = datetime(2024, 1, 31, tzinfo=timezone.utc)
    url = research_futures_tab.build_yahoo_csv_url("ES=F", "1d", start, end)
    parsed = urlparse(url)
    assert parsed.scheme == "https"
    assert "ES%3DF" in parsed.path
    params = parse_qs(parsed.query)
    assert params["interval"] == ["1d"]
    assert params["period1"] == [str(int(start.timestamp()))]
    assert params["period2"] == [str(int(end.timestamp()))]
