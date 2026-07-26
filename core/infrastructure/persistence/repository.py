"""
Repository for LLM trace persistence.

Replaces the old hard-coded SQL TraceStore with SQLAlchemy ORM
operations. All CRUD goes through the ORM; analytics views are still
DuckDB-native (created on init) and queried via SQL.
"""

from typing import List, Optional

import pandas as pd
from sqlalchemy import text
from sqlalchemy.orm import Session, sessionmaker

from core.domain.result import LLMCallResult
from core.infrastructure.persistence.database import make_engine
from core.infrastructure.persistence.models import Base, LLMTrace
from core.infrastructure.persistence.views import VIEW_DEFINITIONS


def _result_to_orm(result: LLMCallResult) -> LLMTrace:
    """Convert domain result → ORM instance."""
    from datetime import datetime

    ts = result.timestamp
    if isinstance(ts, str):
        if ts.endswith("Z"):
            ts = ts[:-1]
        try:
            ts = datetime.fromisoformat(ts)
        except ValueError:
            ts = None

    return LLMTrace(
        provider=result.provider,
        trace_id=result.trace_id,
        timestamp=ts,
        model=result.model,
        prompt_preview=result.prompt_preview,
        response_preview=result.response_preview,
        input_tokens=result.input_tokens,
        output_tokens=result.output_tokens,
        total_tokens=result.total_tokens,
        finish_reason=result.finish_reason,
        latency_ms=result.latency_ms,
        tokens_per_second=result.tokens_per_second,
        cost_usd=result.cost_usd,
        quality_score=result.quality_score,
        quality_reasoning=result.quality_reasoning,
        cost_efficiency=result.cost_efficiency,
        request_id=result.request_id,
        attempts_used=result.attempts_used,
        error_type=result.error_type,
        error=result.error,
        span_id=result.span_id,
        trace_otel_id=result.trace_otel_id,
        tenant_id=result.tenant_id,
        use_case=result.use_case,
    )


class TraceRepository:
    """
    DuckDB-backed repository for LLM call traces.

    Usage:
        repo = TraceRepository()
        repo.save(result)
        df = repo.query_df("SELECT * FROM llm_traces WHERE provider = 'openai'")
    """

    def __init__(self, db_path: str | None = None):
        self._engine = make_engine(db_path)
        self._session_factory = sessionmaker(self._engine, autocommit=False, autoflush=False)
        self._init_schema()

    def _init_schema(self) -> None:
        # Create tables via ORM
        Base.metadata.create_all(self._engine)
        # Create views via raw DDL (DuckDB-specific)
        with self._engine.connect() as conn:
            conn.execute(text(VIEW_DEFINITIONS))
            conn.commit()

    def _session(self) -> Session:
        return self._session_factory()

    def save(self, result: LLMCallResult) -> None:
        """Persist a single LLMCallResult (upsert via merge)."""
        with self._session() as session:
            session.merge(_result_to_orm(result))
            session.commit()

    def save_many(self, results: List[LLMCallResult]) -> None:
        """Persist multiple results in one transaction."""
        if not results:
            return
        with self._session() as session:
            for result in results:
                session.merge(_result_to_orm(result))
            session.commit()

    def query(self, sql: str, params: Optional[dict] = None) -> list:
        """Raw SQL → list of tuples."""
        with self._engine.connect() as conn:
            return conn.execute(text(sql), params or {}).fetchall()

    def query_df(self, sql: str, params: Optional[dict] = None) -> pd.DataFrame:
        """Raw SQL → pandas DataFrame."""
        with self._engine.connect() as conn:
            return pd.read_sql(text(sql), conn, params=params or {})

    def get_summary(self, tenant_id: Optional[str] = None,
                    use_case: Optional[str] = None) -> dict:
        """Aggregated stats. Optional filters by tenant_id and/or use_case."""
        conditions = []
        params = {}
        if tenant_id is not None:
            conditions.append("tenant_id = :tenant_id")
            params["tenant_id"] = tenant_id
        if use_case is not None:
            conditions.append("use_case = :use_case")
            params["use_case"] = use_case

        where = "WHERE " + " AND ".join(conditions) if conditions else ""

        sql = f"""
        SELECT
            COUNT(*)           AS total_calls,
            SUM(cost_usd)      AS total_cost_usd,
            AVG(latency_ms)    AS avg_latency_ms,
            AVG(quality_score) AS avg_quality_score,
            SUM(input_tokens)  AS total_input_tokens,
            SUM(output_tokens) AS total_output_tokens
        FROM llm_traces
        {where}
        """
        with self._engine.connect() as conn:
            row = conn.execute(text(sql), params).fetchone()

        return {
            "total_calls": row[0] or 0,
            "total_cost_usd": row[1] or 0.0,
            "avg_latency_ms": row[2] or 0.0,
            "avg_quality_score": row[3] or 0.0,
            "total_input_tokens": row[4] or 0,
            "total_output_tokens": row[5] or 0,
        }

    def export_to_parquet(self, output_path: str,
                          tenant_id: Optional[str] = None,
                          use_case: Optional[str] = None) -> None:
        """Export traces to Parquet. Optional filters."""
        conditions = []
        params = {}
        if tenant_id is not None:
            conditions.append("tenant_id = :tenant_id")
            params["tenant_id"] = tenant_id
        if use_case is not None:
            conditions.append("use_case = :use_case")
            params["use_case"] = use_case

        where = "WHERE " + " AND ".join(conditions) if conditions else ""
        sql = f"COPY (SELECT * FROM llm_traces {where}) TO '{output_path}' (FORMAT PARQUET)"
        with self._engine.connect() as conn:
            conn.execute(text(sql), params)
            conn.commit()

    def close(self) -> None:
        self._engine.dispose()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
