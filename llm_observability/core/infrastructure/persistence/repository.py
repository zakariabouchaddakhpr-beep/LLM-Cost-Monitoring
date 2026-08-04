"""
Repository for LLM trace persistence, backed by ClickHouse.

Append-only: traces are immutable events, so there is no update path and no
upsert. The previous version used session.merge() keyed on trace_id, but
trace_id is a fresh uuid4() per call and nothing ever re-saves a trace, so that
merge was always an insert in practice.

There is also no `read_only` flag and no lock-retry loop. Both existed only to
negotiate DuckDB's single-writer file lock, which a server does not have.
"""

import logging
import re
from dataclasses import astuple
from datetime import datetime, timezone
from typing import List, Optional

import pandas as pd

from llm_observability.config.settings import ClickHouseConfig, get_clickhouse_config
from llm_observability.core.domain.result import LLMCallResult
from llm_observability.core.infrastructure.persistence import schema
from llm_observability.core.infrastructure.persistence.database import get_client

logger = logging.getLogger(__name__)

_OFFSET_RE = re.compile(r"[+-]\d{2}:?\d{2}$")

# astuple() follows dataclass declaration order.
_FIELD_ORDER = [name for name in LLMCallResult.__dataclass_fields__]


def to_utc(ts) -> datetime:
    """
    Coerce a timestamp to timezone-aware UTC.

    Naive datetimes are treated as UTC rather than local time. This is not
    cosmetic: clickhouse-connect interprets a naive datetime as client-local and
    converts it, so on a machine at +01:00 every trace would be stored an hour
    early. Measured, not assumed.

    Also tolerates the malformed '...+00:00Z' that monitor.py used to emit by
    appending "Z" to an already-offset isoformat().
    """
    if isinstance(ts, str):
        s = ts.strip()
        if s.endswith("Z"):
            s = s[:-1]
            if not _OFFSET_RE.search(s):
                s += "+00:00"
        ts = datetime.fromisoformat(s)
    if ts.tzinfo is None:
        return ts.replace(tzinfo=timezone.utc)
    return ts.astimezone(timezone.utc)


def _to_row(result: LLMCallResult) -> list:
    """LLMCallResult -> a row ordered to match schema.COLUMNS."""
    values = dict(zip(_FIELD_ORDER, astuple(result)))
    values["timestamp"] = to_utc(values["timestamp"])
    return [values[name] for name in schema.COLUMNS]


class TraceRepository:
    """
    ClickHouse-backed repository for LLM call traces.

        repo = TraceRepository()
        repo.save(result)
        df = repo.query_df("SELECT * FROM llm_traces WHERE provider = 'openai'")

    Readers and writers coexist freely — there is no lock to negotiate, which is
    the whole reason the storage layer moved off DuckDB.
    """

    def __init__(self, database: str | None = None, *,
                 config: ClickHouseConfig | None = None,
                 create_schema: bool = True):
        self._config = config or get_clickhouse_config(database)
        self._client = get_client(self._config, create_database=create_schema)
        if create_schema:
            self._init_schema()

    @property
    def database(self) -> str:
        return self._config.database

    def _init_schema(self) -> None:
        """Idempotent. The old version cached this per path in a module-level
        set because re-running DDL meant re-taking the write lock; with no lock
        it is just a few cheap round trips. One statement per call — the HTTP
        interface rejects multi-statement blobs."""
        self._client.command(schema.table_ddl(self.database))
        # Migrations run after the table exists and before the views, because a
        # view may reference a column a migration is about to add.
        for statement in schema.migrations(self.database):
            self._client.command(statement)
        for statement in schema.view_ddl(self.database):
            self._client.command(statement)

    def save(self, result: LLMCallResult) -> None:
        """Persist one trace. Never raises — see save_many."""
        self.save_many([result])

    def save_many(self, results: List[LLMCallResult]) -> None:
        """
        Persist several traces in one insert.

        A failure is logged and swallowed rather than raised. Observability must
        never be load-bearing: the same principle already governs the call path,
        where a failed LLM call returns a result with .error set instead of
        taking down the caller. Against a local file, raising was tolerable;
        against a network service, transient failures are normal.
        """
        if not results:
            return
        try:
            self._client.insert(
                "llm_traces",
                [_to_row(r) for r in results],
                column_names=schema.COLUMNS,
                database=self.database,
            )
        except Exception as exc:
            logger.warning("Could not save %d trace(s) to ClickHouse: %s",
                           len(results), exc)

    def query(self, sql: str, params: Optional[dict] = None) -> list:
        """Raw SQL -> list of tuples."""
        return self._client.query(sql, parameters=params or {}).result_rows

    def query_df(self, sql: str, params: Optional[dict] = None) -> pd.DataFrame:
        """Raw SQL -> pandas DataFrame."""
        return self._client.query_df(sql, parameters=params or {})

    def _filters(self, tenant_id, use_case):
        conditions, params = [], {}
        if tenant_id is not None:
            conditions.append("tenant_id = %(tenant_id)s")
            params["tenant_id"] = tenant_id
        if use_case is not None:
            conditions.append("use_case = %(use_case)s")
            params["use_case"] = use_case
        where = "WHERE " + " AND ".join(conditions) if conditions else ""
        return where, params

    def get_summary(self, tenant_id: Optional[str] = None,
                    use_case: Optional[str] = None) -> dict:
        """Aggregated stats, optionally filtered by tenant_id and/or use_case.

        Mirrors v_summary when called with no filters; the view exists for HTTP
        callers that cannot parameterise a query."""
        where, params = self._filters(tenant_id, use_case)
        sql = f"""
        SELECT count(*), sum(cost_usd), avg(latency_ms), avg(quality_score),
               sum(input_tokens), sum(output_tokens)
        FROM llm_traces {where}
        """
        row = self.query(sql, params)[0]
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
        """Export traces to Parquet. Optional filters.

        Written client-side. ClickHouse's INTO OUTFILE writes on the SERVER,
        which is not what this API promises — and the old DuckDB version
        interpolated output_path into SQL unescaped."""
        where, params = self._filters(tenant_id, use_case)
        self.query_df(f"SELECT * FROM llm_traces {where}",
                      params).to_parquet(output_path)

    def close(self) -> None:
        self._client.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
