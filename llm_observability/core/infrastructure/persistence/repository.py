"""
TraceRepository: saving records to the database and reading them back out.
"""

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

# A pattern for spotting a timezone marker at the end of a timestamp, like
# "+00:00" or "-0500". Reading it left to right:
#   [+-]      a plus or a minus sign
#   \d{2}     exactly two digits
#   :?        an optional colon
#   \d{2}     exactly two more digits
#   $         and this must be at the very END of the text
# Compiled once here, rather than rebuilt every time to_utc() runs.
_OFFSET_RE = re.compile(r"[+-]\d{2}:?\d{2}$")


def to_utc(ts) -> datetime:
    """
    Turn any timestamp into a UTC time that knows it is UTC.
    """
    if isinstance(ts, str):
        text = ts.strip()
        if text.endswith("Z"):
            text = text[:-1]   # everything except the last character
            if not _OFFSET_RE.search(text):
                text += "+00:00"

        ts = datetime.fromisoformat(text)
    if ts.tzinfo is None:
        return ts.replace(tzinfo=timezone.utc)
    return ts.astimezone(timezone.utc)


def _to_row(result: LLMCallResult) -> list:
    """
    Turn one LLMCallResult into a list of values, ready to be inserted.
    """
    values = asdict(result)
    # The timestamp is text at this point. Convert it to a proper UTC time
    values["timestamp"] = to_utc(values["timestamp"])
    # Build the list in column order. Fields not in COLUMNS (the embeddings)
    return [values[column] for column in schema.COLUMNS]


class TraceRepository:
    """
    Saves records to the database and reads them back.
        repo = TraceRepository("my_project")
        repo.save(result)
        df = repo.query_df("SELECT * FROM llm_traces WHERE provider = 'groq'")
        repo.close()
    """

    def __init__(self, database: str | None = None, *,
                 config: ClickHouseConfig | None = None,
                 create_schema: bool = True):
        """
        Connect to this project's database.

        """
        self._config = config or get_clickhouse_config(database)
        self._client = get_client(self._config, create_database=create_schema)
        if create_schema:
            self._init_schema()

    @property
    def database(self) -> str:
        """The name of the database we're connected to."""
        return self._config.database

    def _init_schema(self) -> None:
        """
        Make sure the table and views exist and are up to date.
        """
        self._client.command(schema.table_ddl(self.database))
        for statement in schema.migrations(self.database):
            self._client.command(statement)
        for statement in schema.view_ddl(self.database):
            self._client.command(statement)

    def save(self, result: LLMCallResult) -> None:
        """Save one record. Never crashes - see save_many for the reason."""
        # Wrap it in a list and hand it to the method that does the real work,
        # rather than writing the insert logic out twice.
        self.save_many([result])

    def save_many(self, results: List[LLMCallResult]) -> None:
        """
        Save several records in one go, in a single trip to the database.

        If saving fails, we write a warning to the log and carry on. We do NOT
        crash.

        That is a deliberate decision. Monitoring must never be the thing that
        breaks your app. The same rule already applies further up: a failed AI
        call comes back as a result with an error on it, rather than crashing.
        It would be absurd if your app stayed up through a failed AI call, then
        fell over while trying to write a note about it.
        """
        # Nothing to save. Leave now, rather than sending an empty insert.
        if not results:
            return

        try:
            self._client.insert(
                "llm_traces",
                # Convert every result into a row. This is a list
                # comprehension: it does the same as a for-loop that appends
                # to a list, written on one line.
                [_to_row(r) for r in results],
                # Sending the names alongside the rows is what lets the driver
                # match values to columns by name rather than by position.
                column_names=schema.COLUMNS,
                database=self.database,
            )
        except Exception as exc:
            logger.warning("Could not save %d trace(s) to ClickHouse: %s",
                           len(results), exc)

    def query(self, sql: str, params: Optional[dict] = None) -> list:
        """Run a query and get back a list of rows, each one a tuple."""
        # `params or {}` means: use the values we were given, or an empty
        # dictionary if we were given None.
        return self._client.query(sql, parameters=params or {}).result_rows

    def query_df(self, sql: str, params: Optional[dict] = None) -> pd.DataFrame:
        """
        Run a query and get back a pandas DataFrame.

        A DataFrame is a table in memory, like a spreadsheet you can do maths
        on. The dashboard uses this version because the charting library
        expects data in that shape.
        """
        return self._client.query_df(sql, parameters=params or {})

    def _filters(self, tenant_id, use_case):
        """
        Build the optional "only show me these" part of a query.

        Hands back two things - the text, and the values to go with it:
            ("WHERE tenant_id = %(tenant_id)s", {"tenant_id": "acme"})

        With no filters at all it returns ("", {}), and an empty string simply
        disappears when it's dropped into the query, leaving a valid one.

        The values travel SEPARATELY from the query text, as placeholders that
        the driver fills in. That is what makes it impossible for a value like
        `'; DROP TABLE llm_traces; --` to be run as a command.
        """
        conditions, params = [], {}

        # `is not None` rather than a plain `if`, so that a genuinely empty
        # string still counts as a filter the caller asked for.
        if tenant_id is not None:
            conditions.append("tenant_id = %(tenant_id)s")
            params["tenant_id"] = tenant_id
        if use_case is not None:
            conditions.append("use_case = %(use_case)s")
            params["use_case"] = use_case

        # Join with " AND " so two filters become "a = 1 AND b = 2".
        where = "WHERE " + " AND ".join(conditions) if conditions else ""
        return where, params

    def get_summary(self, tenant_id: Optional[str] = None,
                    use_case: Optional[str] = None) -> dict:
        """
        Totals across all records, optionally narrowed to one customer or
        one feature.

        With no filters this gives exactly the same numbers as the v_summary
        view in schema.py. Both exist for a reason: the view is for simple
        tools that can't send filter values, and this method is for when you
        want filters, which a saved view can't do. Keep the two in step.
        """
        where, params = self._filters(tenant_id, use_case)
        # An f-string, so {where} gets replaced by the filter text (or by
        # nothing). Note that only the FILTER TEXT is inserted here - the
        # actual values still travel separately in `params`.
        sql = f"""
        SELECT count(*), sum(cost_usd), avg(latency_ms), avg(quality_score),
               sum(input_tokens), sum(output_tokens)
        FROM llm_traces {where}
        """

        # This query always gives back exactly one row, with the six values in
        # the same order we asked for them. [0] takes that single row.
        row = self.query(sql, params)[0]

        # `or 0` turns an empty value into a number. Summing or averaging a
        # table with no matching rows gives back nothing, and this saves every
        # caller from having to check for that.
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
        Save the records to a Parquet file.

        Parquet is a compact file format for tables - much smaller and faster
        to read than CSV, and it remembers which column is a number and which
        is text.

        The file is written HERE, on the machine running this code. ClickHouse
        does have its own way of writing files, but that writes them on the
        SERVER, which is not what you'd expect when you hand this a path from
        your own laptop.
        """
        where, params = self._filters(tenant_id, use_case)
        frame = self.query_df(f"SELECT * FROM llm_traces {where}", params)
        frame.to_parquet(output_path)

    def close(self) -> None:
        """Close the connection. Always do this when you're finished."""
        self._client.close()

    # These two methods are what make `with TraceRepository(...) as repo:`
    # work. Python calls __enter__ when the block starts and __exit__ when it
    # ends - including when it ends because of an error. So the connection
    # always gets closed, and you can never forget to do it.
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # The three arguments describe the error, if there was one. We don't
        # need them - we just want to close the connection either way.
        self.close()
