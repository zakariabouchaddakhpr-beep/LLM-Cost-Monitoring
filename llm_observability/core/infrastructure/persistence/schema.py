"""
All schema text for the trace store: database, table and views.

DDL is written by hand rather than generated. ClickHouse needs ENGINE,
PARTITION BY and ORDER BY clauses that no ORM expresses, and the table is
append-only, so there is nothing for an ORM's change tracking to do. Removing
the ORM also removed a 38-line hand-written converter between the domain object
and a near-identical mirror of it.

Nullability follows one mechanical rule: a column is Nullable exactly when
LLMCallResult declares the field Optional. That is not a style choice —
clickhouse-connect rejects None into a non-nullable column outright, and the
project's real traces carry NULL error and error_type on every row.
"""

# Fields on LLMCallResult that are never written to the database. The vectors
# an embedding call returns are far too large to store and the caller needs
# them in hand, not in a table.
NON_PERSISTED = {"embeddings"}

# Order matters only for readability: inserts pass column_names alongside the
# rows, so clickhouse-connect matches by name.
COLUMNS = [
    "provider", "trace_id", "timestamp", "model", "prompt",
    "response", "call_type", "input_tokens", "output_tokens", "total_tokens",
    "cached_tokens", "reasoning_tokens", "finish_reason", "latency_ms",
    "tokens_per_second", "attempts_used", "cost_usd", "quality_score",
    "quality_reasoning", "cost_efficiency", "request_id", "error_type",
    "error", "span_id", "trace_otel_id", "tenant_id", "use_case",
    "query_type", "vector_dimensions", "input_count",
]

# DateTime64(3) is milliseconds, which truncates the microseconds DuckDB kept.
# Deliberate: the whole tool measures in milliseconds and sub-millisecond call
# start times carry no meaning. DateTime64(6) is a one-word change if it ever
# does.
TABLE_DDL = """
CREATE TABLE IF NOT EXISTS {database}.llm_traces (
    provider          LowCardinality(String),
    trace_id          String,
    timestamp         DateTime64(3, 'UTC'),
    model             LowCardinality(String),
    prompt            String TTL timestamp + INTERVAL 90 DAY,
    response          Nullable(String) TTL timestamp + INTERVAL 90 DAY,
    call_type         LowCardinality(String) DEFAULT 'chat',
    input_tokens      Nullable(UInt32),
    output_tokens     Nullable(UInt32),
    total_tokens      Nullable(UInt32),
    cached_tokens     Nullable(UInt32),
    reasoning_tokens  Nullable(UInt32),
    finish_reason     LowCardinality(Nullable(String)),
    latency_ms        Nullable(Float64),
    tokens_per_second Nullable(Float64),
    attempts_used     Nullable(UInt8),
    cost_usd          Nullable(Float64),
    quality_score     Nullable(UInt8),
    quality_reasoning Nullable(String),
    cost_efficiency   Nullable(Float64),
    request_id        Nullable(String),
    error_type        LowCardinality(Nullable(String)),
    error             Nullable(String),
    span_id           Nullable(String),
    trace_otel_id     Nullable(String),
    tenant_id         LowCardinality(Nullable(String)),
    use_case          LowCardinality(Nullable(String)),
    query_type        LowCardinality(Nullable(String)),
    vector_dimensions Nullable(UInt32),
    input_count       Nullable(UInt32)
) ENGINE = MergeTree
PARTITION BY toYYYYMM(timestamp)
ORDER BY (timestamp, provider, model)
"""

# quantile(), not quantileExact(). Both DuckDB's quantile_cont and ClickHouse's
# quantile interpolate linearly, so the figures match to 0.0003 ms on the
# project's own traces; quantileExact returns an actual member of the dataset
# and came out 527 ms higher at p99 over 15 rows. Interpolated and discrete
# quantiles diverge MOST on small samples, not least — there are fewer points to
# land on. quantile uses reservoir sampling and is therefore approximate on very
# large datasets; revisit if percentiles ever look unstable.
VIEW_DDL = [
    """
CREATE OR REPLACE VIEW {database}.v_cost_by_tenant AS
SELECT tenant_id,
       count(*)           AS call_count,
       sum(cost_usd)      AS total_cost_usd,
       avg(cost_usd)      AS avg_cost_usd,
       sum(input_tokens)  AS total_input_tokens,
       sum(output_tokens) AS total_output_tokens
FROM {database}.llm_traces
WHERE tenant_id IS NOT NULL
GROUP BY tenant_id
""",
    """
CREATE OR REPLACE VIEW {database}.v_cost_by_use_case AS
SELECT use_case,
       count(*)           AS call_count,
       sum(cost_usd)      AS total_cost_usd,
       avg(cost_usd)      AS avg_cost_usd,
       avg(quality_score) AS avg_quality_score
FROM {database}.llm_traces
WHERE use_case IS NOT NULL
GROUP BY use_case
""",
    """
CREATE OR REPLACE VIEW {database}.v_latency_percentiles AS
SELECT provider,
       model,
       count(*)                   AS call_count,
       quantile(0.50)(latency_ms) AS p50_ms,
       quantile(0.90)(latency_ms) AS p90_ms,
       quantile(0.99)(latency_ms) AS p99_ms,
       avg(latency_ms)            AS avg_ms
FROM {database}.llm_traces
WHERE latency_ms IS NOT NULL
GROUP BY provider, model
""",
    # Replaces the deleted metrics service. Unfiltered by design: it exists for
    # callers reaching the server over HTTP with no way to parameterise a query.
    # TraceRepository.get_summary() keeps its own SQL because it takes filters,
    # which a static view cannot express. The two must agree when unfiltered.
    """
CREATE OR REPLACE VIEW {database}.v_summary AS
SELECT count(*)           AS total_calls,
       sum(cost_usd)      AS total_cost_usd,
       avg(latency_ms)    AS avg_latency_ms,
       avg(quality_score) AS avg_quality_score,
       sum(input_tokens)  AS total_input_tokens,
       sum(output_tokens) AS total_output_tokens
FROM {database}.llm_traces
""",
    # The question the supervisor actually asked: which models are better for
    # which kind of request. This is the argument the routing recommender will
    # eventually automate, available before that recommender exists.
    #
    # nullIf guards the divide - a zero or missing average cost would otherwise
    # produce infinity and poison any sort on this column.
    #
    # Embeddings are excluded: they are never judged, so they have no quality
    # score to divide by and no query_type to group on.
    """
CREATE OR REPLACE VIEW {database}.v_model_by_query_type AS
SELECT query_type,
       provider,
       model,
       count(*)                   AS call_count,
       avg(cost_usd)              AS avg_cost_usd,
       avg(quality_score)         AS avg_quality,
       quantile(0.90)(latency_ms) AS p90_ms,
       avg(quality_score) / nullIf(avg(cost_usd), 0) AS quality_per_dollar
FROM {database}.llm_traces
WHERE query_type IS NOT NULL AND call_type = 'chat'
GROUP BY query_type, provider, model
""",
]


# Applied on every connection, so every statement must be idempotent.
# ADD COLUMN IF NOT EXISTS and RENAME COLUMN IF EXISTS were both measured
# idempotent on 26.7.1 - running each twice succeeds. Adding a nullable column
# is metadata-only, so this is fast even on a populated table.
#
# These exist because CREATE TABLE IF NOT EXISTS does nothing to a table that
# already exists: without them an established database would silently keep its
# old columns and every insert would fail on the new ones.
_MIGRATIONS = [
    "ALTER TABLE {database}.llm_traces RENAME COLUMN IF EXISTS prompt_preview TO prompt",
    "ALTER TABLE {database}.llm_traces RENAME COLUMN IF EXISTS response_preview TO response",
    "ALTER TABLE {database}.llm_traces ADD COLUMN IF NOT EXISTS call_type LowCardinality(String) DEFAULT 'chat'",
    "ALTER TABLE {database}.llm_traces ADD COLUMN IF NOT EXISTS cached_tokens Nullable(UInt32)",
    "ALTER TABLE {database}.llm_traces ADD COLUMN IF NOT EXISTS reasoning_tokens Nullable(UInt32)",
    "ALTER TABLE {database}.llm_traces ADD COLUMN IF NOT EXISTS query_type LowCardinality(Nullable(String))",
    "ALTER TABLE {database}.llm_traces ADD COLUMN IF NOT EXISTS vector_dimensions Nullable(UInt32)",
    "ALTER TABLE {database}.llm_traces ADD COLUMN IF NOT EXISTS input_count Nullable(UInt32)",
    "ALTER TABLE {database}.llm_traces MODIFY COLUMN prompt String TTL timestamp + INTERVAL 90 DAY",
    "ALTER TABLE {database}.llm_traces MODIFY COLUMN response Nullable(String) TTL timestamp + INTERVAL 90 DAY",
]


def database_ddl(database: str) -> str:
    return f"CREATE DATABASE IF NOT EXISTS {database}"


def migrations(database: str) -> list[str]:
    """Bring an existing table up to the current schema."""
    return [m.format(database=database) for m in _MIGRATIONS]


def table_ddl(database: str) -> str:
    return TABLE_DDL.format(database=database).strip()


def view_ddl(database: str) -> list[str]:
    """One statement per element: the HTTP interface takes them one at a time,
    unlike DuckDB which accepted the whole blob in a single execute()."""
    return [v.format(database=database).strip() for v in VIEW_DDL]
