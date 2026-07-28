"""Persistence sub-package — DuckDB storage via SQLAlchemy ORM."""

from core.infrastructure.persistence.repository import TraceRepository

__all__ = ["TraceRepository"]
