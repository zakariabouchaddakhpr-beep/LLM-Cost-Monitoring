"""
DuckDB persistence layer for LLM call traces.
Zero server setup — just pip install duckdb and go.

Schema:
- llm_traces       : one row per instrumented LLM call
- v_cost_by_tenant : aggregated cost per tenant
- v_cost_by_use_case : aggregated cost per use_case
- v_latency_percentiles : p50/p90/p99 latency per provider+model
"""

import os
from typing import List, Optional, Any

import duckdb
import pandas as pd

from config.settings import DUCKDB_PATH
from core.models import LLMCallResult


_CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS llm_traces (
    provider        TEXT NOT NULL,
    trace_id        TEXT PRIMARY KEY,
    timestamp       TIMESTAMP NOT NULL,
    model           TEXT NOT NULL,
    prompt_preview  TEXT,
    response_preview TEXT,
    input_tokens    INTEGER,
    output_tokens   INTEGER,
    total_tokens    INTEGER,
    finish_reason   TEXT,
    latency_ms      DOUBLE,
    tokens_per_second DOUBLE,
    cost_usd        DOUBLE,
    quality_score   INTEGER,
    quality_reasoning TEXT,
    cost_efficiency DOUBLE,
    request_id      TEXT,
    attempts_used   INTEGER,
    error_type      TEXT,
    error           TEXT,
    span_id         TEXT,
    trace_otel_id   TEXT,
    tenant_id       TEXT,
    use_case        TEXT
);
"""

_CREATE_VIEWS_SQL = """
CREATE OR REPLACE VIEW v_cost_by_tenant AS
SELECT
    tenant_id,
    COUNT(*)          AS call_count,
    SUM(cost_usd)     AS total_cost_usd,
    AVG(cost_usd)     AS avg_cost_usd,
    SUM(input_tokens) AS total_input_tokens,
    SUM(output_tokens) AS total_output_tokens
FROM llm_traces
WHERE tenant_id IS NOT NULL
GROUP BY tenant_id;

CREATE OR REPLACE VIEW v_cost_by_use_case AS
SELECT
    use_case,
    COUNT(*)          AS call_count,
    SUM(cost_usd)     AS total_cost_usd,
    AVG(cost_usd)     AS avg_cost_usd,
    AVG(quality_score) AS avg_quality_score
FROM llm_traces
WHERE use_case IS NOT NULL
GROUP BY use_case;

CREATE OR REPLACE VIEW v_latency_percentiles AS
SELECT
    provider,
    model,
    COUNT(*)                              AS call_count,
    quantile_cont(latency_ms, 0.50)       AS p50_ms,
    quantile_cont(latency_ms, 0.90)       AS p90_ms,
    quantile_cont(latency_ms, 0.99)       AS p99_ms,
    AVG(latency_ms)                       AS avg_ms
FROM llm_traces
WHERE latency_ms IS NOT NULL
GROUP BY provider, model;
"""

_INSERT_SQL = """
INSERT OR REPLACE INTO llm_traces
VALUES (
    $provider, $trace_id, $timestamp, $model, $prompt_preview,
    $response_preview, $input_tokens, $output_tokens, $total_tokens,
    $finish_reason, $latency_ms, $tokens_per_second, $cost_usd,
    $quality_score, $quality_reasoning, $cost_efficiency, $request_id,
    $attempts_used, $error_type, $error, $span_id, $trace_otel_id,
    $tenant_id, $use_case
)
"""


def _ensure_dir(path: str) -> None:
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)


def _result_to_row(result: LLMCallResult) -> dict:
    """Convert an LLMCallResult to a dict of DuckDB-friendly values."""
    from datetime import datetime

    ts = result.timestamp
    # DuckDB prefers a native timestamp; attempt to parse ISO strings.
    if isinstance(ts, str):
        # Some timestamps may end with Z (UTC); strip it since fromisoformat
        # handles +00:00 but not the Z suffix.
        if ts.endswith("Z"):
            ts = ts[:-1]
        try:
            ts = datetime.fromisoformat(ts)
        except ValueError:
            ts = None
    # DuckDB prefers a native timestamp; attempt to parse ISO strings.
    if isinstance(ts, str):
        # Handle trailing Z correctly: "...Z" -> "...+00:00"
        if ts.endswith("Z"):
            ts = ts[:-1] + "+00:00"
        try:
            ts = datetime.fromisoformat(ts)
        except ValueError:
            ts = None
    # DuckDB prefers a native timestamp; attempt to parse ISO strings.
    if isinstance(ts, str):
        # Remove the trailing 'Z' if present and parse
        ts_clean = ts.replace("Z", "+00:00")
        try:
            ts = datetime.fromisoformat(ts_clean)
        except ValueError:
            ts = None

    return {
        "provider": result.provider,
        "trace_id": result.trace_id,
        "timestamp": ts,
        "model": result.model,
        "prompt_preview": result.prompt_preview,
        "response_preview": result.response_preview,
        "input_tokens": result.input_tokens,
        "output_tokens": result.output_tokens,
        "total_tokens": result.total_tokens,
        "finish_reason": result.finish_reason,
        "latency_ms": result.latency_ms,
        "tokens_per_second": result.tokens_per_second,
        "cost_usd": result.cost_usd,
        "quality_score": result.quality_score,
        "quality_reasoning": result.quality_reasoning,
        "cost_efficiency": result.cost_efficiency,
        "request_id": result.request_id,
        "attempts_used": result.attempts_used,
        "error_type": result.error_type,
        "error": result.error,
        "span_id": result.span_id,
        "trace_otel_id": result.trace_otel_id,
        "tenant_id": result.tenant_id,
        "use_case": result.use_case,
    }


class TraceStore:
    """
    DuckDB-backed store for LLM call traces.

    Usage:
        store = TraceStore()
        store.save(result)
        df = store.query_df("SELECT * FROM llm_traces WHERE provider = 'openai'")
    """

    def __init__(self, db_path: str = DUCKDB_PATH):
        _ensure_dir(db_path)
        self._conn = duckdb.connect(db_path)
        self._init_schema()

    def _init_schema(self) -> None:
        self._conn.execute(_CREATE_TABLE_SQL)
        self._conn.execute(_CREATE_VIEWS_SQL)

    def save(self, result: LLMCallResult) -> None:
        """Persist a single LLMCallResult."""
        self._conn.execute(_INSERT_SQL, _result_to_row(result))

    def save_many(self, results: List[LLMCallResult]) -> None:
        """Persist multiple results in one transaction."""
        if not results:
            return
        rows = [_result_to_row(r) for r in results]
        # Use executemany for bulk insert
        self._conn.executemany(_INSERT_SQL, rows)

    def query(self, sql: str, params: Optional[dict] = None) -> List[Any]:
        """
        Execute raw SQL and return a list of tuples.
        params is an optional dict for named parameter binding.
        """
        if params:
            return self._conn.execute(sql, params).fetchall()
        return self._conn.execute(sql).fetchall()

    def query_df(self, sql: str, params: Optional[dict] = None) -> pd.DataFrame:
        """
        Execute raw SQL and return a pandas DataFrame.
        params is an optional dict for named parameter binding.
        """
        if params:
            return self._conn.execute(sql, params).fetchdf()
        return self._conn.execute(sql).fetchdf()

    def get_summary(self, tenant_id: Optional[str] = None,
                    use_case: Optional[str] = None) -> dict:
        """
        Return a summary dict with total calls, total cost, avg latency,
        and avg quality score. Optional filters by tenant_id and/or use_case.
        """
        where_clauses = []
        params = {}
        if tenant_id is not None:
            where_clauses.append("tenant_id = $tenant_id")
            params["tenant_id"] = tenant_id
        if use_case is not None:
            where_clauses.append("use_case = $use_case")
            params["use_case"] = use_case

        where = "WHERE " + " AND ".join(where_clauses) if where_clauses else ""

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
        row = self._conn.execute(sql, params).fetchone()
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
        """
        Export traces to a Parquet file. Optional filters by tenant_id
        and/or use_case.
        """
        where_clauses = []
        params = {}
        if tenant_id is not None:
            where_clauses.append("tenant_id = $tenant_id")
            params["tenant_id"] = tenant_id
        if use_case is not None:
            where_clauses.append("use_case = $use_case")
            params["use_case"] = use_case

        where = "WHERE " + " AND ".join(where_clauses) if where_clauses else ""
        sql = f"COPY (SELECT * FROM llm_traces {where}) TO '{output_path}' (FORMAT PARQUET)"
        self._conn.execute(sql, params)

    def close(self) -> None:
        self._conn.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
