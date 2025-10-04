"""Tests for the SQLite schema and demo loader."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from toptek.data import io


def test_init_and_demo(tmp_path: Path, monkeypatch) -> None:
    db_path = tmp_path / "toptek.db"

    original_cls = io.IOPaths

    def _mock_paths() -> io.IOPaths:
        return original_cls(root=tmp_path, var=tmp_path, data=tmp_path, db=db_path)

    monkeypatch.setattr(io, "IOPaths", _mock_paths)
    monkeypatch.setattr(io, "DATA_DIR", tmp_path)
    monkeypatch.setattr(io, "VAR_DIR", tmp_path)
    conn = io.connect(db_path)
    try:
        io.run_migrations(conn)
        stats = io.load_demo_data(conn, rows=200)
        assert stats["trades"] == 200
        exports = io.export_to_parquet(conn, dest=tmp_path)
        for table, path in exports.items():
            assert path.exists(), table
            df = pd.read_parquet(path)
            assert not df.empty
    finally:
        conn.close()
