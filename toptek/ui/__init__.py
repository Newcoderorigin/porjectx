"""UI widgets for new LM Studio and Futures Research tabs."""

from __future__ import annotations

try:  # pragma: no cover - exercised via import behaviour
    from .live_tab import LiveTab  # type: ignore[F401]
except Exception:  # pragma: no cover - headless environments
    LiveTab = None  # type: ignore[assignment]

try:  # pragma: no cover - exercised via import behaviour
    from .research_futures_tab import FuturesResearchTab  # type: ignore[F401]
except Exception:  # pragma: no cover - headless environments
    FuturesResearchTab = None  # type: ignore[assignment]

__all__ = ["LiveTab", "FuturesResearchTab"]
