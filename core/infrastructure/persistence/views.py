"""
Pre-built analytics views for the DuckDB persistence layer.

Views are created via raw DDL because SQLAlchemy has no native
first-class view support without external libraries.
"""

VIEW_DEFINITIONS = """
CREATE OR REPLACE VIEW v_cost_by_tenant AS
SELECT
    tenant_id,
    COUNT(*)           AS call_count,
    SUM(cost_usd)      AS total_cost_usd,
    AVG(cost_usd)      AS avg_cost_usd,
    SUM(input_tokens)  AS total_input_tokens,
    SUM(output_tokens) AS total_output_tokens
FROM llm_traces
WHERE tenant_id IS NOT NULL
GROUP BY tenant_id;

CREATE OR REPLACE VIEW v_cost_by_use_case AS
SELECT
    use_case,
    COUNT(*)           AS call_count,
    SUM(cost_usd)      AS total_cost_usd,
    AVG(cost_usd)      AS avg_cost_usd,
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
