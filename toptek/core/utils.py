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
from importlib import metadata
from importlib.metadata import PackageNotFoundError
from typing import Any, Dict, Mapping

import yaml  # type: ignore[import]


DEFAULT_TIMEZONE = timezone.utc


@dataclass
class AppPaths:
    """Paths used throughout the application.

    Attributes:
        root: Base directory for the project.
        cache: Directory path for cached data files.
        models: Directory path for persisted models.
    """

    root: Path
    cache: Path
    models: Path


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

    for directory in (paths.cache, paths.models):
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
    return AppPaths(root=root, cache=cache_dir, models=models_dir)


STACK_REQUIREMENTS: Dict[str, str | tuple[str, str]] = {
    "numpy": ">=1.21.6,<1.28",
    "scipy": ">=1.7.3,<1.12",
    "scikit-learn": "==1.3.2",
}


def _version_tuple(value: str) -> tuple[int, ...]:
    parts: list[int] = []
    for raw_part in value.replace("-", ".").split("."):
        if not raw_part:
            continue
        numeric = ""
        for char in raw_part:
            if char.isdigit():
                numeric += char
            else:
                break
        if not numeric:
            break
        parts.append(int(numeric))
    return tuple(parts) if parts else (0,)


def _compare_versions(installed: tuple[int, ...], expected: tuple[int, ...]) -> int:
    length = max(len(installed), len(expected))
    installed_padded = installed + (0,) * (length - len(installed))
    expected_padded = expected + (0,) * (length - len(expected))
    if installed_padded == expected_padded:
        return 0
    return -1 if installed_padded < expected_padded else 1


def _spec_matches(installed: str, spec: str) -> bool:
    installed_tuple = _version_tuple(installed)
    for clause in (segment.strip() for segment in spec.split(",")):
        if not clause:
            continue
        for operator in ("==", "!=", ">=", "<=", ">", "<"):
            if clause.startswith(operator):
                version_part = clause[len(operator) :].strip()
                if not version_part:
                    continue
                expected_tuple = _version_tuple(version_part)
                comparison = _compare_versions(installed_tuple, expected_tuple)
                if operator == "==" and comparison != 0:
                    return False
                if operator == "!=" and comparison == 0:
                    return False
                if operator == ">=" and comparison == -1:
                    return False
                if operator == "<=" and comparison == 1:
                    return False
                if operator == ">" and comparison != 1:
                    return False
                if operator == "<" and comparison != -1:
                    return False
                break
        else:  # pragma: no cover - defensive branch
            continue
    return True


def assert_numeric_stack(
    requirements: Mapping[str, str | tuple[str, str]] | None = None,
) -> None:
    """Validate that the numeric stack matches our supported ABI matrix.

    The guard is executed prior to importing SciPy or scikit-learn to surface
    a friendly, actionable error message whenever the environment drifts from
    the tested wheel set. This avoids confusing low-level ImportErrors during
    CLI or GUI start-up.
    """

    spec = requirements or STACK_REQUIREMENTS
    missing: list[str] = []
    mismatched: list[tuple[str, str, str]] = []

    for package_name, constraint in spec.items():
        if isinstance(constraint, tuple):
            dist_name, version_spec = constraint
        else:
            dist_name, version_spec = package_name, constraint
        try:
            installed_version = metadata.version(dist_name)
        except PackageNotFoundError:
            missing.append(f"{package_name} (requires {version_spec})")
            continue
        if version_spec and not _spec_matches(installed_version, version_spec):
            mismatched.append((package_name, installed_version, version_spec))

    if not missing and not mismatched:
        return

    summary_lines = []
    if missing:
        summary_lines.append("Missing packages: " + ", ".join(sorted(missing)))
    if mismatched:
        details = ", ".join(
            f"{name} {installed} (requires {required})"
            for name, installed, required in sorted(mismatched)
        )
        summary_lines.append("Version mismatches: " + details)
    requirement_summary = ", ".join(
        f"{name} {constraint if isinstance(constraint, str) else constraint[1]}"
        for name, constraint in spec.items()
    )
    message = (
        "Incompatible numeric stack detected before importing SciPy/sklearn.\n"
        + "\n".join(summary_lines)
        + "\nSupported versions: "
        + requirement_summary
        + "\nReinstall the vetted stack via `pip install -r toptek/requirements-lite.txt` "
        + "or align the listed packages manually."
    )
    raise RuntimeError(message)
