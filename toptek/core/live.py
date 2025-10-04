"""Live trading utilities with optional SignalR streaming stubs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

from .gateway import ProjectXGateway


@dataclass
class ExecutionContext:
    """Represents the state required for placing orders."""

    gateway: ProjectXGateway
    account_id: str


def poll_open_orders(context: ExecutionContext) -> Dict[str, object]:
    """Poll open orders using the REST API."""

    return context.gateway.search_open_orders({"accountId": context.account_id})


def poll_positions(context: ExecutionContext) -> Dict[str, object]:
    """Poll open positions."""

    return context.gateway.search_positions({"accountId": context.account_id})


def connect_market_hub(*_, **__) -> None:  # pragma: no cover - stub
    """Placeholder for SignalR market hub connection."""

    raise NotImplementedError(
        "SignalR streaming is optional; install signalrcore to enable"
    )


def subscribe_ticker(*_, **__) -> None:  # pragma: no cover - stub
    raise NotImplementedError(
        "SignalR streaming is optional; install signalrcore to enable"
    )


def subscribe_bars(*_, **__) -> None:  # pragma: no cover - stub
    raise NotImplementedError(
        "SignalR streaming is optional; install signalrcore to enable"
    )


__all__ = [
    "ExecutionContext",
    "poll_open_orders",
    "poll_positions",
    "connect_market_hub",
    "subscribe_ticker",
    "subscribe_bars",
]
