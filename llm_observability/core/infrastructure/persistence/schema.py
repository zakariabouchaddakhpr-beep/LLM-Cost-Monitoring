NON_PERSISTED = {"embeddings"}

COLUMNS = [
    "provider", "trace_id", "timestamp", "model", "prompt",
    "response", "call_type", "input_tokens", "output_tokens", "total_tokens",
    "cached_tokens", "reasoning_tokens", "finish_reason", "latency_ms",
    "tokens_per_second", "attempts_used", "cost_usd", "quality_score",
    "quality_reasoning", "cost_efficiency", "request_id", "error_type",
    "error", "span_id", "trace_otel_id", "tenant_id", "use_case",
    "query_type", "vector_dimensions", "input_count",
    "task_type", "complexity", "tier",
]

ADDED_COLUMNS = [
    ("task_type", "LowCardinality(Nullable(String))"),
    ("complexity", "Nullable(Float64)"),
    ("tier", "LowCardinality(Nullable(String))"),
]

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
    input_count       Nullable(UInt32),
    task_type         LowCardinality(Nullable(String)),
    complexity        Nullable(Float64),
    tier              LowCardinality(Nullable(String))
) ENGINE = MergeTree
PARTITION BY toYYYYMM(timestamp)
ORDER BY (timestamp, provider, model)
"""

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


def database_ddl(database: str) -> str:
    return f"CREATE DATABASE IF NOT EXISTS {database}"


def table_ddl(database: str) -> str:
    return TABLE_DDL.format(database=database).strip()


def migration_ddl(database: str) -> list[str]:
    return [
        f"ALTER TABLE {database}.llm_traces "
        f"ADD COLUMN IF NOT EXISTS {column} {column_type}"
        for column, column_type in ADDED_COLUMNS
    ]


def view_ddl(database: str) -> list[str]:
    return [view.format(database=database).strip() for view in VIEW_DDL]
