import logging
import re
from dataclasses import asdict
from datetime import datetime, timezone
from typing import List, Optional

import pandas as pd

from llm_observability.config.settings import (
    ClickHouseConfig,
    get_clickhouse_config,
)
from llm_observability.core.domain.result import LLMCallResult
from llm_observability.core.infrastructure.persistence import schema
from llm_observability.core.infrastructure.persistence.database import get_client

logger = logging.getLogger(__name__)

_OFFSET_RE = re.compile(r"[+-]\d{2}:?\d{2}$")


def to_utc(ts) -> datetime:
    if isinstance(ts, str):
        text = ts.strip()
        if text.endswith("Z"):
            text = text[:-1]
            if not _OFFSET_RE.search(text):
                text += "+00:00"

        ts = datetime.fromisoformat(text)
    if ts.tzinfo is None:
        return ts.replace(tzinfo=timezone.utc)
    return ts.astimezone(timezone.utc)


def _to_row(result: LLMCallResult) -> list:
    values = asdict(result)
    values["timestamp"] = to_utc(values["timestamp"])
    return [values[column] for column in schema.COLUMNS]


class TraceRepository:

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
        self._client.command(schema.table_ddl(self.database))
        for statement in schema.view_ddl(self.database):
            self._client.command(statement)

    def save(self, result: LLMCallResult) -> None:
        self.save_many([result])

    def save_many(self, results: List[LLMCallResult]) -> None:
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
        return self._client.query(sql, parameters=params or {}).result_rows

    def query_df(self, sql: str, params: Optional[dict] = None) -> pd.DataFrame:
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
        where, params = self._filters(tenant_id, use_case)
        frame = self.query_df(f"SELECT * FROM llm_traces {where}", params)
        frame.to_parquet(output_path)

    def close(self) -> None:
        self._client.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
