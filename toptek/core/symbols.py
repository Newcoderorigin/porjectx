"""Exchange symbol validation utilities.

Only CME Group futures (CME, CBOT, NYMEX, COMEX) are supported. Symbols must
consist of an uppercase root and a valid month code followed by a two-digit
year, for example ``ESZ5``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Set


ALLOWED_EXCHANGES: Set[str] = {"CME", "CBOT", "NYMEX", "COMEX"}
MONTH_CODES = "FGHJKMNQUVXZ"


@dataclass(frozen=True)
class ContractSymbol:
    """Canonical representation of a futures contract symbol."""

    root: str
    month: str
    year: int

    @property
    def code(self) -> str:
        return f"{self.root}{self.month}{self.year % 10}"


def validate_symbol(symbol: str, *, allowed_roots: Iterable[str] | None = None) -> ContractSymbol:
    """Validate a futures symbol, raising ``ValueError`` if invalid."""

    symbol = symbol.upper().strip()
    if len(symbol) < 3:
        raise ValueError("Symbol too short")
    root = symbol[:-2]
    month = symbol[-2:-1]
    year_char = symbol[-1]
    if month not in MONTH_CODES:
        raise ValueError(f"Invalid month code: {month}")
    if not year_char.isdigit():
        raise ValueError("Year must end with a digit")
    if allowed_roots and root not in set(r.upper() for r in allowed_roots):
        raise ValueError(f"Root {root} not permitted")
    return ContractSymbol(root=root, month=month, year=int(year_char))


__all__ = ["ContractSymbol", "validate_symbol", "ALLOWED_EXCHANGES", "MONTH_CODES"]
