# LLM Monitor — Grafana Dashboard Spec

A lightweight text-based dashboard reference for the LLM monitoring layer.
Each panel maps directly to data your `TraceRepository` and OpenTelemetry
collectors already emit.

---

## Dashboard: "LLM Cost & Quality"

| Field | Value |
|-------|-------|
| **Name** | LLM Cost & Quality |
| **UID** | `llm-monitor` |
| **Data sources** | DuckDB (via Grafana SQL data source) + OTLP metrics |
| **Refresh** | 30s |

---

## Row 1: Cost Attribution

### Panel 1.1 — Total Cost by Tenant

| Property | Value |
|----------|-------|
| **Title** | Total Cost by Tenant |
| **Type** | Bar chart (or Table) |
| **Query** | `SELECT tenant_id, total_cost_usd FROM v_cost_by_tenant` |
| **Unit** | Currency USD |

```
┌─────────────┬────────────────┐
│ tenant_id   │ total_cost_usd │
├─────────────┼────────────────┤
│ acme-corp   │        $12.45  │
│ docs        │         $8.20  │
│ engineering │         $3.10  │
└─────────────┴────────────────┘
```

---

### Panel 1.2 — Total Cost by Use Case

| Property | Value |
|----------|-------|
| **Title** | Cost vs. Quality by Use Case |
| **Type** | Table |
| **Query** | `SELECT use_case, total_cost_usd, avg_quality_score FROM v_cost_by_use_case` |

```
┌───────────────┬────────────────┬───────────────────┐
│ use_case      │ total_cost_usd │ avg_quality_score │
├───────────────┼────────────────┼───────────────────┤
│ code-gen      │         $5.50  │               8.5 │
│ summarization │         $3.20  │               9.1 │
│ tutorial      │         $2.10  │               8.0 │
│ support       │         $1.20  │               7.2 │
└───────────────┴────────────────┴───────────────────┘
```

---

## Row 2: Latency

### Panel 2.1 — Latency Percentiles by Provider + Model

| Property | Value |
|----------|-------|
| **Title** | Latency Percentiles |
| **Type** | Table (or Time series) |
| **Query** | `SELECT provider, model, p50_ms, p90_ms, p99_ms FROM v_latency_percentiles` |
| **Unit** | milliseconds |

```
┌──────────┬──────────────────────────────┬────────┬────────┬────────┐
│ provider │ model                        │ p50_ms │ p90_ms │ p99_ms │
├──────────┼──────────────────────────────┼────────┼────────┼────────┤
│ gemini   │ gemini/gemini-3.5-flash      │  210.0 │  420.0 │  890.0 │
│ openai   │ openai/gpt-4o                │  340.0 │  580.0 │ 1100.0 │
│ anthropic│ anthropic/claude-3-5-sonnet  │  280.0 │  510.0 │  950.0 │
└──────────┴──────────────────────────────┴────────┴────────┴────────┘
```

---

## Row 3: Token Usage (OTel Metrics)

### Panel 3.1 — Token Usage Over Time

| Property | Value |
|----------|-------|
| **Title** | Input vs. Output Tokens / minute |
| **Type** | Time series (stacked area) |
| **Metric** | `gen_ai.client.token.usage` |
| **Dimensions** | `gen_ai.token.type = input` / `output` |
| **Aggregation** | Rate per minute |

```
tokens/min
    │
300 │      ╭─╮        ╭──╮
200 │   ╭──╯ ╰──╮  ╭──╯  ╰──╮      ← output
100 │╭──╯        ╰──╯         ╰──╮  ← input
  0 └───────────────────────────────→ time
```

---

## Row 4: Operation Health

### Panel 4.1 — Call Duration Distribution

| Property | Value |
|----------|-------|
| **Title** | LLM Call Duration Distribution |
| **Type** | Histogram / Heatmap |
| **Metric** | `gen_ai.client.operation.duration` |
| **Unit** | milliseconds |

```
count
    │
 20 │    ██
 15 │   ████
 10 │  ██████
  5 │ ████████
  0 └─────────────────→ duration (ms)
         100   500   1s   2s   5s
```

---

### Panel 4.2 — Cost per Call (USD)

| Property | Value |
|----------|-------|
| **Title** | Cost per Call |
| **Type** | Time series |
| **Metric** | `llm.cost.usd` |
| **Dimensions** | `gen_ai.system`, `gen_ai.request.model` |

```
cost ($)
0.05│
0.04│        ╭──╮
0.03│   ╭────╯  ╰────╮
0.02│╭──╯            ╰──╮
0.01│                    ╰────╮
  0 └──────────────────────────→ time
```

---

### Panel 4.3 — Error Rate

| Property | Value |
|----------|-------|
| **Title** | Error Rate by Provider |
| **Type** | Stat panel (percentage) |
| **Source** | `llm_traces` table — count where `error IS NOT NULL` |

```
┌────────────────────────────────────────┐
│                                        │
│           2.4% ERROR RATE              │
│                                        │
│   gemini: 1.1%  │  openai: 3.8%        │
│                                        │
└────────────────────────────────────────┘
```

---

### Panel 4.4 — Quality Score Trend

| Property | Value |
|----------|-------|
| **Title** | Avg Quality Score Over Time |
| **Type** | Time series |
| **Query** | `SELECT timestamp, quality_score FROM llm_traces` |
| **Unit** | score (1–10) |

```
score
 10 │
  8 │  ╭──╮  ╭────╮
  6 │ ╭╯  ╰──╯    ╰──╮
  4 │╭╯              ╰──╮
  2 │
  0 └──────────────────────→ time
```

---

## Row 5: Summary (single stat row)

### Panel 5.1 — Overall Summary

| Property | Value |
|----------|-------|
| **Title** | Last 24h Summary |
| **Type** | Stat row (4 stats) |
| **Query** | `SELECT * FROM v_cost_by_tenant` + `SELECT COUNT(*) FROM llm_traces` |

```
┌─────────────┬────────────┬─────────────┬──────────────┐
│ Total Calls │ Total Cost │ Avg Latency │ Avg Quality  │
├─────────────┼────────────┼─────────────┼──────────────┤
│     1,247   │   $23.75   │   285 ms    │    8.4 / 10  │
└─────────────┴────────────┴─────────────┴──────────────┘
```

---

## How to wire this up (for reference)

### Option A: DuckDB + Grafana (local file)

Grafana has a **DuckDB data source plugin**. Point it at:
```
results/llm_traces.duckdb
```
Then create panels using the SQL queries above.

### Option B: OTLP → Prometheus/Grafana

If you set `OTEL_EXPORTER_OTLP_ENDPOINT` to a collector that writes
to Prometheus, the OTel metrics (`gen_ai.client.token.usage`,
`gen_ai.client.operation.duration`, `llm.cost.usd`) appear automatically
as Prometheus metrics and Grafana can graph them directly.

### Option C: DuckDB → Parquet → S3 → Grafana

```python
repo.export_to_parquet("s3://bucket/traces/")
```
Grafana can query Parquet from S3 via DuckDB or Athena.

---

## Variable (dropdown filter)

| Variable | Query | Multi-select |
|----------|-------|--------------|
| `tenant_id` | `SELECT DISTINCT tenant_id FROM llm_traces` | ✅ |
| `use_case` | `SELECT DISTINCT use_case FROM llm_traces` | ✅ |
| `provider` | `SELECT DISTINCT provider FROM llm_traces` | ✅ |
| `model` | `SELECT DISTINCT model FROM llm_traces` | ✅ |

All panels should include `WHERE tenant_id IN ($tenant_id)` etc. when
variables are used.
