"""Utility helpers for configuration, logging, time conversions, and JSON handling.

This module centralises convenience helpers shared across the project. It
loads configuration files, initialises structured logging backed by rotating
file handlers, and provides a few small wrappers for timezone-aware timestamps
and JSON serialisation. It also exposes deterministic seeding helpers and
numeric stack validation utilities that fail fast when the runtime drifts away
from the supported SciPy/NumPy/sklearn matrix.

Example:
    >>> from core import utils
    >>> config = utils.load_yaml(Path("config/app.yml"))
    >>> logger = utils.build_logger("toptek")
"""

from __future__ import annotations

import importlib
import importlib.util
import json
import logging
import os
import platform
import random
from dataclasses import dataclass
from datetime import datetime, timezone
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any, Dict, Iterable

import yaml


DEFAULT_TIMEZONE = timezone.utc
REPO_ROOT = Path(__file__).resolve().parents[2]

STACK_REQUIREMENTS: Dict[str, str] = {
    "numpy": "1.26.4",
    "scipy": "1.10.1",
    "sklearn": "1.3.2",
}
STACK_OPTIONAL: Iterable[str] = ("pandas", "joblib", "threadpoolctl")


@dataclass
class AppPaths:
    """Paths used throughout the application.

    Attributes:
        root: Base directory for the project.
        cache: Directory path for cached data files.
        models: Directory path for persisted models.
        logs: Directory path for rotating log files.
        reports: Directory path for diagnostic reports.
    """

    root: Path
    cache: Path
    models: Path
    logs: Path
    reports: Path


def build_logger(name: str, level: str = "INFO") -> logging.Logger:
    """Return a logger configured against the global logging policy.

    The global logging configuration is expected to be installed via
    :func:`configure_logging`. When imported in isolation (e.g. within unit
    tests) the helper falls back to ``logging.basicConfig`` with sane defaults
    so that callers always receive a functional logger.
    """

    root_logger = logging.getLogger()
    if not root_logger.handlers:
        logging.basicConfig(
            level=level.upper(),
            format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    logger = logging.getLogger(name)
    logger.setLevel(level.upper())
    return logger


def configure_logging(log_dir: Path, level: str = "INFO") -> Path:
    """Initialise rotating file + console logging in ``log_dir``.

    Returns the path to the active log file.
    """

    log_dir.mkdir(parents=True, exist_ok=True)
    filename = f"toptek_{datetime.now(tz=DEFAULT_TIMEZONE).strftime('%Y%m%d')}.log"
    log_path = log_dir / filename

    root_logger = logging.getLogger()
    if not any(
        isinstance(handler, RotatingFileHandler) for handler in root_logger.handlers
    ):
        formatter = logging.Formatter(
            "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        file_handler = RotatingFileHandler(
            log_path,
            maxBytes=5 * 1024 * 1024,
            backupCount=5,
            encoding="utf-8",
        )
        file_handler.setFormatter(formatter)

        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)

        logging.basicConfig(
            level=level.upper(),
            handlers=[file_handler, stream_handler],
        )
    else:
        root_logger.setLevel(level.upper())
    return log_path


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

    for directory in (paths.cache, paths.models, paths.logs, paths.reports):
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
    log_dir = root / app_config.get("logs_directory", "logs")
    reports_dir = root / app_config.get("reports_directory", "reports")
    return AppPaths(
        root=root, cache=cache_dir, models=models_dir, logs=log_dir, reports=reports_dir
    )


def _collect_versions(modules: Iterable[str]) -> Dict[str, str | None]:
    versions: Dict[str, str | None] = {}
    for module_name in modules:
        spec = importlib.util.find_spec(module_name)
        if spec is None:
            versions[module_name] = None
            continue
        module = importlib.import_module(module_name)
        versions[module_name] = getattr(module, "__version__", None)
    return versions


def assert_numeric_stack(*, reports_dir: Path | None = None) -> Dict[str, str]:
    """Verify the numeric stack exactly matches the supported versions.

    Writes a diagnostic report to ``reports_dir`` (defaulting to
    ``reports/run_stack.json`` at the repository root) describing the current
    environment and raises :class:`RuntimeError` when a mismatch is detected.
    """

    reports_path = reports_dir or (REPO_ROOT / "reports")
    reports_path.mkdir(parents=True, exist_ok=True)

    required_versions = _collect_versions(STACK_REQUIREMENTS.keys())
    optional_versions = _collect_versions(STACK_OPTIONAL)

    mismatches = {
        name: version
        for name, version in required_versions.items()
        if version is not None and version != STACK_REQUIREMENTS[name]
    }
    missing = [name for name, version in required_versions.items() if version is None]

    stack_report = {
        "timestamp": datetime.now(tz=DEFAULT_TIMEZONE).isoformat(),
        "python": platform.python_version(),
        "required": required_versions,
        "expected": STACK_REQUIREMENTS,
        "optional": optional_versions,
        "status": "ok" if not (mismatches or missing) else "error",
    }

    report_path = reports_path / "run_stack.json"
    with report_path.open("w", encoding="utf-8") as handle:
        json.dump(stack_report, handle, indent=2)

    if missing or mismatches:
        problems: list[str] = []
        if missing:
            problems.append("missing: " + ", ".join(sorted(missing)))
        if mismatches:
            issues = ", ".join(
                f"{name}=={version} (expected {STACK_REQUIREMENTS[name]})"
                for name, version in mismatches.items()
            )
            problems.append(f"mismatched: {issues}")
        joined = "; ".join(problems)
        raise RuntimeError(
            "Numeric stack mismatch detected (" + joined + "). "
            "Run scripts/setup_env.ps1 to recreate the supported environment."
        )

    return {name: version or "" for name, version in required_versions.items()}


def set_seeds(seed: int) -> None:
    """Set deterministic seeds for Python and NumPy."""

    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    import numpy as np  # Imported lazily to avoid top-level dependency during import

    np.random.seed(seed)
