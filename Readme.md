# LLM Monitor

A middleware library that sits between your application and whichever
LLM provider you use (Claude, GPT, Gemini, Groq, Cerebras), instrumenting
every call with consistent metrics: latency, tokens, cost, throughput,
retry behavior, and an optional quality score — regardless of which
provider answered.

Built on [LiteLLM](https://github.com/BerriAI/litellm) for the actual
provider calls. LiteLLM handles all provider-specific routing via the
model-string prefix (e.g. `openai/gpt-4o`, `anthropic/claude-3-5-sonnet`),
so the library needs only a single thin adapter.

**New in this version:**
- **OpenTelemetry GenAI Semantic Conventions** — every call emits a span
  with standard attributes and metrics. Console exporter by default; OTLP
  when `OTEL_EXPORTER_OTLP_ENDPOINT` is set.
- **DuckDB persistence via SQLAlchemy ORM** — all traces are saved through
  a proper ORM layer (`core/infrastructure/persistence/`) with pre-built
  analytics views for cost attribution by tenant and use case.

## Architecture (layered)

```
core/
├── domain/              # Pure business logic — zero external deps
│   ├── result.py        # LLMCallResult dataclass
│   └── metrics.py       # throughput / cost-efficiency math
├── application/         # Orchestration layer
│   ├── monitor.py       # LLMMonitor — the middleman
│   └── quality_scoring.py
└── infrastructure/      # Adapters for external systems
    ├── provider.py      # LiteLLM adapter
    ├── otel.py          # OpenTelemetry tracer & meter
    └── persistence/     # SQLAlchemy ORM + DuckDB
        ├── database.py
        ├── models.py
        ├── repository.py
        └── views.py
```

**Design principle:** `domain` knows nothing about HTTP, databases, or OTel.
`application` wires domain + infrastructure together. `infrastructure`
adapts external APIs to the domain's interfaces.

## Project structure

```
llm_monitor/
├── quickstart.py              usage example — OTel + DuckDB showcase
├── config/
│   └── settings.py            judge model, retry settings, OTel config
├── core/
│   ├── domain/
│   │   ├── result.py          LLMCallResult dataclass
│   │   └── metrics.py         pure math functions
│   ├── application/
│   │   ├── monitor.py         LLMMonitor orchestrator
│   │   └── quality_scoring.py
│   ├── infrastructure/
│   │   ├── provider.py        single LiteLLM adapter
│   │   ├── otel.py            OpenTelemetry setup
│   │   └── persistence/
│   │       ├── database.py    SQLAlchemy engine factory
│   │       ├── models.py      ORM models (declarative)
│   │       ├── repository.py  TraceRepository (CRUD + analytics)
│   │       └── views.py       DuckDB analytics views
│   ├── logging_config.py
│   └── __init__.py            backward-compat re-exports
├── tests/
│   ├── domain/
│   │   └── test_metrics.py
│   ├── application/
│   │   └── test_monitor.py
│   └── infrastructure/
│       └── test_persistence.py
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
└── results/                    generated at runtime, not committed
```

## Usage

```python
from core.application.monitor import LLMMonitor
from core.infrastructure.persistence import TraceRepository

monitor = LLMMonitor(provider_name="anthropic", model="anthropic/claude-3-5-sonnet-20241022")
repo    = TraceRepository()

result = monitor.call(
    "Explain the Factory design pattern in 3 sentences.",
    tenant_id="acme-corp",
    use_case="docs"
)

print(result.total_tokens, result.latency_ms, result.cost_usd)
repo.save(result)

# Query aggregated analytics
print(repo.query_df("SELECT * FROM v_cost_by_tenant"))
```

### Cost attribution

Pass `tenant_id` and `use_case` to `monitor.call()` to slice costs in
DuckDB:

```python
repo.get_summary(tenant_id="acme-corp")
repo.get_summary(use_case="support")
repo.query_df("SELECT * FROM v_cost_by_use_case")
```

## Setup (Docker — fully self-contained)

The Docker image bundles Python, DuckDB, SQLAlchemy, OpenTelemetry, and all
other dependencies. You don't need Python, pip, or anything else installed
locally.

```bash
cp .env.example .env        # fill in your real API keys
docker compose up --build
```

That's it. The container will:
1. Install all Python dependencies inside the image
2. Run `quickstart.py` which calls the LLM and saves traces
3. Persist results to `./results/` on your host (JSON + DuckDB file)
4. Print OTel spans and metrics to the console

### What's inside the image

| Component | Version | Purpose |
|-----------|---------|---------|
| Python | 3.12 | Runtime |
| LiteLLM | >=1.40 | Unified LLM provider API |
| DuckDB | >=1.0 | Embedded analytics database |
| SQLAlchemy | >=2.0 | ORM layer |
| OpenTelemetry | >=1.25 | Traces and metrics |
| pandas | >=2.0 | DataFrame support |

### Accessing the DuckDB file

After running, the database is at `./results/llm_traces.duckdb` on your host:

```bash
# Query from host using DuckDB CLI (if installed)
duckdb ./results/llm_traces.duckdb "SELECT * FROM v_cost_by_tenant"

# Or attach to the running container
docker exec -it llm-monitor python -c "
from core.infrastructure.persistence import TraceRepository
r = TraceRepository('/app/results/llm_traces.duckdb')
print(r.query_df('SELECT * FROM v_cost_by_tenant'))
"
```

### Environment variables in Docker

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `OPENAI_API_KEY` | * | — | OpenAI API key |
| `ANTHROPIC_API_KEY` | * | — | Anthropic API key |
| `GEMINI_API_KEY` | * | — | Google Gemini API key |
| `GROQ_API_KEY` | * | — | Groq API key |
| `CEREBRAS_API_KEY` | * | — | Cerebras API key |
| `OTEL_EXPORTER_OTLP_ENDPOINT` | — | — | OTLP collector URL (omit for console) |
| `OTEL_SERVICE_NAME` | — | `llm-monitor` | Service name in traces |
| `DUCKDB_PATH` | — | `/app/results/llm_traces.duckdb` | DB file location |

\* At least one provider key matching the `model=` string in `quickstart.py`.

### Re-running after code changes

```bash
docker compose up --build
```

The `--build` flag rebuilds the image when you change source code.

## Setup (local, no Docker)

```bash
git clone <this-repo>
cd llm_monitor
cp .env.example .env        # then fill in your real API keys
pip install -r requirements.txt
python quickstart.py
```

Edit the `model=` argument in `quickstart.py` to change which LLM gets
called. Results are written to both `results/call_log_*.json` and
`results/llm_traces.duckdb`.

## Setup (Docker)

```bash
cp .env.example .env        # then fill in your real API keys
docker compose up --build
```

Results still land in `./results/` on your host machine via the volume
mount in `docker-compose.yml`.

## Grafana

A local Grafana instance is available through Docker at http://localhost:3000 with the default login `admin / admin`.

The compose file now provisions a starter dashboard and datasource config from the [grafana/provisioning](grafana/provisioning) folder.

## OpenTelemetry

Every `monitor.call()` emits:

- **Span** named `chat {provider}` (e.g. `chat openai`)
  - Attributes: `gen_ai.operation.name`, `gen_ai.system`,
    `gen_ai.request.model`, `gen_ai.usage.input_tokens`,
    `gen_ai.usage.output_tokens`
  - Error spans get `error.type` and `error.message` on failure
- **Metrics**
  - `gen_ai.client.token.usage` (counter) — tagged `gen_ai.token.type: input|output`
  - `gen_ai.client.operation.duration` (histogram) — in milliseconds
  - `llm.cost.usd` (histogram)

By default spans and metrics go to **stdout** (ConsoleExporter). To send
to an OTLP collector instead, set the environment variable:

```bash
export OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4317
```

## DuckDB persistence (SQLAlchemy ORM)

`TraceRepository` creates a local DuckDB file via SQLAlchemy with the
following schema:

- **Table `llm_traces`** — mapped by the `LLMTrace` ORM model. One row
  per call with all `LLMCallResult` fields plus `tenant_id`, `use_case`,
  `span_id`, and `trace_otel_id`.
- **View `v_cost_by_tenant`** — aggregated cost and token usage per tenant.
- **View `v_cost_by_use_case`** — aggregated cost and average quality per use case.
- **View `v_latency_percentiles`** — p50 / p90 / p99 latency per provider+model.

API:

```python
repo = TraceRepository()
repo.save(result)                          # ORM merge (upsert)
repo.save_many([r1, r2, r3])               # bulk merge
repo.query("SELECT * FROM llm_traces")     # raw SQL → list of tuples
repo.query_df("SELECT * FROM v_cost_by_tenant")  # raw SQL → pandas DataFrame
repo.get_summary(tenant_id="acme")         # dict with totals & averages
repo.export_to_parquet("traces.parquet")   # export to Parquet
```

## Switching providers

LiteLLM routes to the correct provider based on the model-string prefix:

| Provider | Example model string |
|----------|---------------------|
| OpenAI | `openai/gpt-4o` |
| Anthropic | `anthropic/claude-3-5-sonnet-20241022` |
| Gemini | `gemini/gemini-3.5-flash` |
| Groq | `groq/llama-3.3-70b-versatile` |
| Cerebras | `cerebras/gpt-oss-120b` |

Just change the `model=` argument when creating `LLMMonitor`:

```python
monitor = LLMMonitor(provider_name="anthropic", model="anthropic/claude-3-5-sonnet-20241022")
```

`provider_name` is used for logging, OTel spans, and DuckDB analytics.
`model` is passed straight to LiteLLM, which handles the actual API call.

## Metrics tracked per call

- `input_tokens` / `output_tokens` / `total_tokens`
- `latency_ms`
- `tokens_per_second` — output tokens generated per second
- `finish_reason` — whether the response completed naturally or was cut off
- `cost_usd` — via LiteLLM's built-in pricing table
- `attempts_used` / `error_type` — retry and failure diagnostics
- `request_id` — provider-assigned ID, if exposed, for cross-referencing later
- `quality_score` — optional 1–10 rating from a judge LLM
- `cost_efficiency` — quality per dollar spent
- `span_id` / `trace_otel_id` — correlation with OpenTelemetry traces
- `tenant_id` / `use_case` — cost attribution fields

## Logging

The library uses Python's `logging` module, not `print()`. Call
`configure_logging()` once at your app's entry point (already done in
`quickstart.py`); every module then logs through `get_logger(__name__)`
with timestamps and severity levels, which can be filtered or redirected
in one place without touching library code.

## Testing

```bash
python -m pytest tests/ -v
```

## Notes

- Never commit `.env` — it's git-ignored. Use `.env.example` as the
template for which variables need to be set.
- `results/` is git-ignored since it's generated output, not source code.
- Model strings passed to `LLMMonitor`/provider constructors should be
checked against LiteLLM's current supported model list — some provider
model IDs are version-dated and get deprecated over time.
