"""Utility helpers for configuration, logging, time conversions, and JSON handling.

This module centralises convenience helpers shared across the project. It loads
configuration files, initialises structured logging, and provides a few small
wrappers for timezone-aware timestamps and JSON serialisation.

Example:
    >>> from core import utils
    >>> config = utils.load_yaml(Path("config/app.yml"))
    >>> logger = utils.build_logger("toptek")
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

import yaml


DEFAULT_TIMEZONE = timezone.utc


@dataclass
class AppPaths:
    """Paths used throughout the application.

    Attributes:
        root: Base directory for the project.
        cache: Directory path for cached data files.
        models: Directory path for persisted models.
        user_data: Directory path for long-lived user state and history.
    """

    root: Path
    cache: Path
    models: Path
    user_data: Path


def build_logger(name: str, level: str = "INFO") -> logging.Logger:
    """Create a structured logger configured for console output.

    Args:
        name: Logger name.
        level: Log level name.

    Returns:
        A configured :class:`logging.Logger` instance.
    """

    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    logger.setLevel(level.upper())
    return logger


def load_yaml(path: Path) -> Dict[str, Any]:
    """Load a YAML document from *path*.

    Args:
        path: Path to the YAML file.

    Returns:
        Parsed data as a dictionary.
    """

    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def ensure_directories(paths: AppPaths) -> None:
    """Ensure application directories exist."""

    for directory in (paths.cache, paths.models, paths.user_data):
        directory.mkdir(parents=True, exist_ok=True)


def timestamp() -> datetime:
    """Return a timezone-aware UTC timestamp."""

    return datetime.now(tz=DEFAULT_TIMEZONE)


def json_dumps(data: Any, *, indent: int = 2) -> str:
    """Serialise *data* to JSON using sane defaults."""

    return json.dumps(data, indent=indent, default=str)


def env_or_default(key: str, default: str) -> str:
    """Return an environment variable or *default* if unset."""

    return os.environ.get(key, default)


def build_paths(root: Path, app_config: Dict[str, Any]) -> AppPaths:
    """Create :class:`AppPaths` from configuration values.

    Args:
        root: Repository root directory.
        app_config: Configuration dictionary.

    Returns:
        An :class:`AppPaths` instance.
    """

    cache_dir = root / app_config.get("cache_directory", "data/cache")
    models_dir = root / app_config.get("models_directory", "models")
    user_dir = root / app_config.get("user_data_directory", "data/user")
    return AppPaths(root=root, cache=cache_dir, models=models_dir, user_data=user_dir)
