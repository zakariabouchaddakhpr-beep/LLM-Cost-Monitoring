"""
SQLAlchemy engine and session factory for DuckDB.
"""

import os
from pathlib import Path

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from config.settings import DUCKDB_PATH


def _ensure_dir(path: str | None) -> str:
    resolved_path = str(path or DUCKDB_PATH or "results/llm_traces.duckdb")
    if not resolved_path:
        resolved_path = "results/llm_traces.duckdb"

    path_obj = Path(resolved_path)
    if path_obj.is_absolute():
        target_path = path_obj
    else:
        target_path = Path.cwd() / path_obj

    target_path.parent.mkdir(parents=True, exist_ok=True)
    return str(target_path)


def make_engine(db_path: str | None = None):
    """Return a SQLAlchemy engine backed by DuckDB."""
    resolved_path = _ensure_dir(db_path)
    # duckdb-engine dialect: duckdb:///path/to/file.duckdb
    return create_engine(f"duckdb:///{resolved_path}")


SessionLocal = sessionmaker(autocommit=False, autoflush=False)
