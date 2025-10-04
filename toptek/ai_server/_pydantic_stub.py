"""Minimal pydantic compatibility shims for offline execution."""

from __future__ import annotations

from typing import Any, Callable, Dict


class BaseModel:
    def __init__(self, **data: Any) -> None:
        for key, value in data.items():
            setattr(self, key, value)

    def dict(self) -> Dict[str, Any]:
        return dict(self.__dict__)


def Field(default: Any = None, **_: Any) -> Any:  # pragma: no cover - trivial stub
    return default


def root_validator(*_args: Any, **_kwargs: Any) -> Callable:
    def decorator(func: Callable) -> Callable:
        return func

    return decorator


__all__ = ["BaseModel", "Field", "root_validator"]
