"""Data access package exposing IO utilities."""

from .feeds import BarRecord, SQLiteBarFeed
from .io import (
    DATA_DIR,
    VAR_DIR,
    IOPaths,
    connect,
    export_to_parquet,
    load_demo_data,
    run_migrations,
)

__all__ = [
    "DATA_DIR",
    "VAR_DIR",
    "IOPaths",
    "connect",
    "export_to_parquet",
    "load_demo_data",
    "run_migrations",
    "BarRecord",
    "SQLiteBarFeed",
]
