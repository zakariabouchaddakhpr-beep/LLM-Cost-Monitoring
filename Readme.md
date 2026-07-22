# LLM Monitor

A middleware library that sits between your application and whichever
LLM provider you use (Claude, GPT, Gemini, Groq, Cerebras), instrumenting
every call with consistent metrics: latency, tokens, cost, throughput,
retry behavior, and an optional quality score — regardless of which
provider answered.

**New in this version:**
- **OpenTelemetry GenAI Semantic Conventions** — every call emits a span
  with standard attributes (`gen_ai.system`, `gen_ai.usage.*`, etc.) and
  metrics (`gen_ai.client.token.usage`, `gen_ai.client.operation.duration`,
  `llm.cost.usd`). Console exporter by default; OTLP when
  `OTEL_EXPORTER_OTLP_ENDPOINT` is set.
- **DuckDB persistence** — all traces are saved to a local DuckDB file
  with pre-built analytics views for cost attribution by tenant and use case.

Built on [LiteLLM](https://github.com/BerriAI/litellm) for the actual
provider calls, and a Factory pattern for provider selection: swap
providers by changing which class the Factory looks up, no other code
changes needed elsewhere in the library.

## Project structure

```
llm_monitor/
├── quickstart.py              usage example — OTel + DuckDB showcase
├── config/
│   └── settings.py            judge model, retry settings, OTel config
├── core/
│   ├── base_provider.py        the interface every provider adapter implements
│   ├── provider_factory.py     Factory: provider name -> provider instance
│   ├── monitor.py              LLMMonitor - the middleman, wraps any provider with metrics & OTel
│   ├── models.py                LLMCallResult - the structured result object
│   ├── trace_store.py           DuckDB persistence layer with analytics views
│   ├── otel_setup.py            OpenTelemetry tracer & meter initialization
│   ├── logging_config.py        shared logging setup (used instead of print())
│   ├── metrics.py               throughput / cost-efficiency math
│   ├── quality_scoring.py       optional LLM-as-judge quality scoring
│   └── providers/
│       ├── openai_provider.py
│       ├── anthropic_provider.py
│       ├── gemini_provider.py
│       ├── groq_provider.py
│       └── cerebras_provider.py
├── tests/
│   ├── test_metrics.py
│   ├── test_monitor.py
│   └── test_trace_store.py
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
└── results/                    generated at runtime (JSON logs + DuckDB), not committed
```

## Usage

```python
from core.monitor import LLMMonitor
from core.trace_store import TraceStore

monitor = LLMMonitor(provider_name="anthropic", model="anthropic/claude-3-5-sonnet-20241022")
store   = TraceStore()

result = monitor.call(
    "Explain the Factory design pattern in 3 sentences.",
    tenant_id="acme-corp",
    use_case="docs"
)

print(result.total_tokens, result.latency_ms, result.cost_usd)
store.save(result)

# Query aggregated analytics
print(store.query_df("SELECT * FROM v_cost_by_tenant"))
```

`result` is an `LLMCallResult` instance (see `core/models.py`) with the
same fields no matter which provider was called — see "Metrics tracked"
below. `provider_name` and `model` are required and explicit — there's
no default model lookup; you always state exactly which model you want.

### Cost attribution

Pass `tenant_id` and `use_case` to `monitor.call()` to slice costs in
DuckDB:

```python
store.get_summary(tenant_id="acme-corp")
store.get_summary(use_case="support")
store.query_df("SELECT * FROM v_cost_by_use_case")
```

## Setup (local, no Docker)

```bash
git clone <this-repo>
cd llm_monitor
cp .env.example .env        # then fill in your real API keys
pip install -r requirements.txt
python quickstart.py
```

Edit `PROVIDER_NAME` and `MODEL` at the top of `quickstart.py` to change
which LLM gets called. Results are written to both `results/call_log_*.json`
and `results/llm_traces.duckdb`.

## Setup (Docker)

```bash
cp .env.example .env        # then fill in your real API keys
docker compose up --build
```

Results still land in `./results/` on your host machine via the volume
mount in `docker-compose.yml`.

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

## DuckDB persistence

`TraceStore` creates a local DuckDB file (default: `results/llm_traces.duckdb`)
with the following schema:

- **Table `llm_traces`** — one row per call with all `LLMCallResult` fields
  plus `tenant_id`, `use_case`, `span_id`, and `trace_otel_id`.
- **View `v_cost_by_tenant`** — aggregated cost and token usage per tenant.
- **View `v_cost_by_use_case`** — aggregated cost and average quality per use case.
- **View `v_latency_percentiles`** — p50 / p90 / p99 latency per provider+model.

API:

```python
store = TraceStore()
store.save(result)                          # single insert
store.save_many([r1, r2, r3])               # bulk insert
store.query("SELECT * FROM llm_traces")     # raw SQL -> list of tuples
store.query_df("SELECT * FROM v_cost_by_tenant")  # raw SQL -> pandas DataFrame
store.get_summary(tenant_id="acme")         # dict with totals & averages
store.export_to_parquet("traces.parquet")   # export to Parquet
```

## Adding a new provider

1. Create `core/providers/my_provider.py`, implementing `call()`,
   `get_provider_name()`, and `get_model_string()`, with `model: str`
   as a required constructor argument (see any existing provider for
   the pattern — there is no default-model fallback).
2. Register it in `core/provider_factory.py`:
   `LLMProviderFactory.register_provider("my_provider", MyProvider)`

Nothing in `monitor.py` needs to change.

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
