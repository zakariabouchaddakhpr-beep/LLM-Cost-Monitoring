"""
SQLAlchemy engine and session factory for DuckDB.
"""

import tempfile
import logging
from pathlib import Path

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from config.settings import DUCKDB_PATH

logger = logging.getLogger(__name__)


def _is_file_accessible(path_obj: Path) -> bool:
    if not path_obj.exists():
        return True
    try:
        with path_obj.open("rb"):
            pass
        return True
    except OSError:
        return False


def _fallback_path(target_path: Path) -> str:
    fallback_dir = Path(tempfile.gettempdir()) / "llm_cost_monitor"
    fallback_dir.mkdir(parents=True, exist_ok=True)
    fallback_path = fallback_dir / target_path.name
    logger.warning(
        "DuckDB path %s is not accessible; falling back to %s",
        target_path,
        fallback_path,
    )
    return str(fallback_path)


def _ensure_dir(path: str | None) -> str:
    resolved_path = str(path or DUCKDB_PATH or "results/llm_traces.duckdb")
    if not resolved_path:
        resolved_path = "results/llm_traces.duckdb"

    path_obj = Path(resolved_path)
    if path_obj.is_absolute():
        target_path = path_obj
    else:
        target_path = Path.cwd() / path_obj

    try:
        target_path.parent.mkdir(parents=True, exist_ok=True)
    except OSError:
        return _fallback_path(target_path)

    if not _is_file_accessible(target_path):
        return _fallback_path(target_path)

    return str(target_path)


def make_engine(db_path: str | None = None):
    """Return a SQLAlchemy engine backed by DuckDB."""
    resolved_path = _ensure_dir(db_path)
    # duckdb-engine dialect: duckdb:///path/to/file.duckdb
    return create_engine(f"duckdb:///{resolved_path}")


SessionLocal = sessionmaker(autocommit=False, autoflush=False)
