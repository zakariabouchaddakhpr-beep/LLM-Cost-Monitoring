"""ClickHouse storage. TraceRepository is re-exported here to keep imports short."""

from llm_observability.core.infrastructure.persistence.repository import (
    TraceRepository,
)

__all__ = ["TraceRepository"]
