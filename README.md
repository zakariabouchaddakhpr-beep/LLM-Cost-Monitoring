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
- **ClickHouse persistence** — traces are appended to a ClickHouse server, one
  database per project, with pre-built analytics views for cost attribution by
  tenant and use case.

## Architecture (layered)

```
llm_observability/
├── api.py                   the public API: init / call / dashboard
├── config/
│   └── settings.py          connection config, retries, OTel config
├── core/
│   ├── domain/              pure business logic — zero external deps
│   │   ├── result.py        LLMCallResult dataclass
│   │   └── metrics.py       throughput / cost-efficiency math
│   ├── application/         orchestration
│   │   ├── monitor.py       LLMMonitor — the middleman
│   │   └── quality_scoring.py
│   └── infrastructure/      adapters for external systems
│       ├── provider.py      LiteLLM adapter
│       ├── otel.py          OpenTelemetry tracer & meter
│       └── persistence/     ClickHouse via clickhouse-connect
│           ├── database.py  client factory
│           ├── schema.py    all DDL: database, table, four views
│           └── repository.py TraceRepository (writes + analytics)
└── dashboard/
    ├── app.py               the Streamlit dashboard
    └── __main__.py          python -m llm_observability.dashboard
```

There is no ORM. ClickHouse needs `ENGINE` and `ORDER BY` clauses that no ORM
expresses, and the trace table is append-only, so change tracking has nothing to
do. Removing it also removed a hand-written converter between the domain object
and a near-identical mirror of it.

**Design principle:** `domain` knows nothing about HTTP, databases, or OTel.
`application` wires domain + infrastructure together. `infrastructure`
adapts external APIs to the domain's interfaces. `api.py` is a thin facade over
all of it, so the three public calls stay small.

Everything lives under **one** top-level package on purpose: `config` and `core`
at the top level would collide with a consuming project's own modules of those
names.

## Project structure

```
├── llm_observability/       the library (this is what you import)
├── main.py                  example app — OTel + ClickHouse showcase
├── tests/                   87 tests, mirroring the package layout
├── pyproject.toml           packaging
├── requirements.txt
├── Dockerfile
├── docker-compose.yaml
└── results/                 generated at runtime, not committed
```

## Usage

```python
from llm_observability.core.application.monitor import LLMMonitor
from llm_observability.core.infrastructure.persistence import TraceRepository

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

Pass `tenant_id` and `use_case` to `monitor.call()` to slice costs:

```python
repo.get_summary(tenant_id="acme-corp")
repo.get_summary(use_case="support")
repo.query_df("SELECT * FROM v_cost_by_use_case")
```

## Configuration

All configuration is read from the environment. The library looks for a `.env`
file by searching upward from the **working directory**, so the file it finds is
the one belonging to the application that imported it — not one sitting next to
the library's own source.

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `CLICKHOUSE_DATABASE` | — | none | Trace database. Wins over `name`; if neither is given, an error |
| `CLICKHOUSE_HOST` | — | `localhost` | Server address. docker-compose overrides this to `clickhouse` |
| `CLICKHOUSE_PORT` | — | `8123` | HTTP interface — **not** the native protocol on 9000 |
| `CLICKHOUSE_USER` | yes | `default` | Must be set for docker-compose; the image refuses network access otherwise |
| `CLICKHOUSE_PASSWORD` | yes | empty | Must be non-empty for docker-compose, same reason |
| `PROJECT_NAME` | — | — | Dashboard title, docker-compose container names |
| `DASHBOARD_PORT` | — | `8501` | Host port for the dashboard (docker-compose) |
| `OPENAI_API_KEY` | * | — | OpenAI API key |
| `ANTHROPIC_API_KEY` | * | — | Anthropic API key |
| `GEMINI_API_KEY` | * | — | Google Gemini API key |
| `GROQ_API_KEY` | * | — | Groq API key |
| `CEREBRAS_API_KEY` | * | — | Cerebras API key |
| `OTEL_EXPORTER_OTLP_ENDPOINT` | — | — | OTLP collector URL (omit for console) |
| `OTEL_SERVICE_NAME` | — | `llm-monitor` | Service name in traces |

\* At least one provider key matching the `model=` string you pass to `LLMMonitor`.

There is **no default database on purpose.** A built-in fallback is exactly what
causes unrelated applications to pile their traces into one shared store, so a
project with neither `CLICKHOUSE_DATABASE` nor `name` fails fast with an
explanatory error instead of silently writing somewhere generic. `name` is
per-project by construction, so it does not reintroduce that problem. Host, port,
user and password all default, because they describe infrastructure rather than
identity.

**Database names take letters, digits and underscores.** A hyphen is not legal in
a ClickHouse identifier without backtick quoting, and neither is a leading digit,
so `slug()` converts `My App!` to `my_app` and prefixes a leading digit with `p_`.

**Leave `CLICKHOUSE_HOST` as `localhost` in `.env`.** The same file is read by
Compose and by a local run, but the server is `localhost:<published port>` from
the host and `clickhouse:8123` inside the Docker network. `docker-compose.yaml`
sets both variables for the application containers, so one `.env` covers both.

Reading happens at call time, not import time, so a host application may also set
these in code before its first LLM call. Importing the library never requires
configuration, and never requires a reachable server.

## Using it in your own project

Three calls. Each project gets its own database; nothing is shared.

**1. Install it**

```bash
pip install /path/to/this/repo
```

**2. Use it**

```python
import os
import llm_observability as obs

obs.init(name="ticket-triage",
         model="cerebras/gpt-oss-120b",
         api_key=os.getenv("CEREBRAS_API_KEY"))

result = obs.call("Summarise this ticket…", tenant_id="acme", use_case="support")
print(result.cost_usd, result.latency_ms)      # the trace is already saved

obs.dashboard()                                 # → http://localhost:8501
```

That's it — beyond a reachable ClickHouse, no connection string and no `.env`
required.

| Argument | |
|---|---|
| `name` | picks the database: `slug(name)` on the configured server |
| `model` | the LiteLLM model string; the provider is read from its prefix |
| `api_key` | passed straight to LiteLLM. Omit it and LiteLLM finds its own from the environment |

An existing database is **reused, never replaced** — call `init()` as often as you
like.

### The rest of the surface

```python
obs.summary()          # {'total_calls': 15, 'total_cost_usd': 0.0045, ...}
obs.database()         # which ClickHouse database this project's traces are in
obs.dashboard(port=8502, block=False)   # a second project, non-blocking

client = obs.init(...)  # or hold the object instead of using the module default
client.call("…")
```

### Embeddings

Instrumented the same way, and landing in the same table behind a `call_type`
column so cost queries still span everything:

```python
result = obs.embed("some text", tenant_id="acme", use_case="search")
result = obs.embed(["doc one", "doc two"], tenant_id="acme", use_case="indexing")

vectors = result.embeddings     # the vectors themselves — NOT persisted
result.vector_dimensions        # 3072
result.input_count              # 2
```

**One row per API call, not per input.** A batch of 100 documents is one billed
request, so it is one trace with `input_count = 100`. Counting them individually
would multiply the call count and misstate the cost.

Embeddings are never judged — there is no response to score, and classifying the
input is a different question. That is most of why they stay cheap.

### Quality scoring and query classification

Both are off by default and both are powered by the same judge call, so turning
on the second one is free:

| Configuration | API calls per `obs.call()` |
|---|---|
| Neither | 1 |
| `score_quality=True` | 2 |
| `classify_query=True` | 2 |
| **Both** | **2** |

```python
obs.init(name="my-app", model="…", score_quality=True, classify_query=True)
```

`classify_query` fills in `query_type` from a closed vocabulary —
`summarization`, `question-answering`, `code-generation`, `extraction`,
`classification`, `translation`, `reasoning`, `creative-writing`,
`conversation`, `other` — and anything outside it becomes `other`. The fixed
list is deliberate: a judge answering in free text would produce many spellings
of the same idea and `GROUP BY query_type` would stop meaning anything.

That column is what makes `v_model_by_query_type` answerable.

Two more options worth knowing:

- **`score_quality=False` is the default here**, unlike `LLMMonitor.call`. Judging
  makes a *second* API call to a different provider (a Groq model), so it would be
  a surprise on both the bill and the latency. Opt in with
  `obs.init(..., score_quality=True, judge_api_key=...)`.
- **`CLICKHOUSE_DATABASE` still wins** if it is set, so an existing `.env` or the
  Docker services keep behaving exactly as before. `name` only fills in when it is
  unset. You can also pass `database=` to override both.

### One store or many

Isolation and sharing are the same mechanism, not two features — it comes down to
whether two projects name the same database.

**Separate stores.** Each project keeps its own traces; nothing is shared:

```bash
# ticket-triage/.env
CLICKHOUSE_DATABASE=ticket_triage
# doc-search/.env
CLICKHOUSE_DATABASE=doc_search
```

**One company-wide store.** Several applications, run by different people, all
visible in one dashboard — which is what you want when a team is spending on
models and somebody has to account for it:

```bash
# BOTH projects
CLICKHOUSE_DATABASE=elyadata_shared
```

```python
# project A — one person's support bot
obs.init(name="support-bot", model="cerebras/gpt-oss-120b")
obs.call(prompt, tenant_id="zakaria", use_case="support")

# project B — a colleague's summariser, a completely separate program
obs.init(name="doc-summariser", model="gemini/gemini-2.0-flash")
obs.call(prompt, tenant_id="fadoua", use_case="summarisation")
```

One dashboard then shows both, filterable by tenant and use case:

```
fadoua     doc-summariser   1 call    $0.0100
zakaria    support-bot      2 calls   $0.0070
```

Two applications may write **at the same time** — that is what a database server
buys you, and it is the reason this project moved off DuckDB, whose single-writer
lock made the arrangement impossible.

> **`name=` is ignored when `CLICKHOUSE_DATABASE` is set.** It is only used to
> *derive* a database name when the variable is absent. That is the documented
> precedence, and it is exactly what makes the shared setup above work — but it
> does mean `obs.init(name="support-bot")` will not put the traces in a
> `support_bot` database if the environment says otherwise.

### Without installing

The lower-level API is still there if you want it, and works over `PYTHONPATH`
with no install:

```python
from llm_observability.core.application.monitor import LLMMonitor
from llm_observability.core.infrastructure.persistence import TraceRepository

monitor = LLMMonitor(provider_name="cerebras", model="cerebras/gpt-oss-120b")
with TraceRepository("my_project") as repo:
    repo.save(monitor.call("Summarise this ticket…", tenant_id="acme"))
```

And the dashboard can be launched without `init()`:

```bash
python -m llm_observability.dashboard --name ticket-triage
CLICKHOUSE_DATABASE=other_project python -m llm_observability.dashboard --port 8502
```

### One thing that will surprise you

`import litellm` calls `load_dotenv()` itself, walking up from wherever it is
installed. If your virtualenv sits inside a project that has a `.env` with
`CLICKHOUSE_DATABASE` in it, that value lands in your environment without you
setting it — and since it wins, it would override `name`. Pass `database=`
explicitly if you ever need to be certain.

This matters more than it did under DuckDB: a leaked value now points a project
at another project's *database* on a shared server, not merely at another file.

## Setup (local, no Docker)

```bash
git clone <this-repo>
cd llm_monitor
cp .env.example .env        # fill in API keys and CLICKHOUSE_*
pip install -r requirements.txt
docker compose up -d clickhouse    # the one piece of infrastructure needed
python main.py              # generates traces
python -m llm_observability.dashboard
```

Edit the `model=` argument in `main.py` to change which LLM gets called. The
traces go to the ClickHouse database named by `CLICKHOUSE_DATABASE`; the example
also writes `call_log_*.json` into `results/`.

## Setup (Docker)

The image bundles Python, clickhouse-connect, OpenTelemetry and everything else;
nothing needs to be installed on the host.

```bash
cp .env.example .env        # fill in API keys and CLICKHOUSE_*
docker compose up --build
```

Three services are defined:

| Service | Role | Endpoint |
|---------|------|----------|
| `clickhouse` | The trace store | http://localhost:8123 |
| `monitor` | Runs `main.py` once and exits | — |
| `dashboard` | Streamlit UI | http://localhost:8501 |

Both application services declare `depends_on: clickhouse` with
`condition: service_healthy`, so they wait for the database rather than for each
other. Query it directly from the host at any time:

```bash
curl 'http://localhost:8123/?query=SELECT * FROM my_project.v_summary FORMAT JSON' \
     --user llmobs:your-password
```

That view is what replaced the old standalone metrics container — ClickHouse
already serves JSON over HTTP, so a fourth service to expose six numbers was
redundant.

### Readers and writers run at the same time

This is the reason the storage layer is ClickHouse rather than DuckDB.

DuckDB is embedded and permits **many readers or one writer, never both**, so
the dashboard could not display data while the monitored application was still
writing. The readers had to wait for the writer to *exit*:

```yaml
depends_on:
  monitor:
    condition: service_completed_successfully   # no longer needed
```

An observability tool that only works after the observed process has finished is
a report generator. With a server there is no lock to negotiate: start the
dashboard, run your application, and watch rows arrive. `read_only=`, the
lock-retry loop and the startup ordering are all gone.

`docker compose up` still runs the `monitor` service, which makes **real, paid
API calls**. To bring up only the database and the dashboard:

```bash
docker compose up -d clickhouse dashboard   # no LLM calls
```

To run several projects at once, give each its own directory and `.env`, and set
a distinct `PROJECT_NAME` and `DASHBOARD_PORT` in each so container names and
published ports don't collide.

`--build` rebuilds the image after source changes.

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

## ClickHouse persistence

`TraceRepository` creates this project's database if it is absent, then keeps one
client for its lifetime. All DDL lives in `schema.py`:

- **Table `llm_traces`** — `MergeTree`, partitioned by month, ordered by
  `(timestamp, provider, model)`. One row per call with every `LLMCallResult`
  field. Append-only: traces are immutable events, so there is no update path
  and no upsert.
- **View `v_cost_by_tenant`** — aggregated cost and token usage per tenant.
- **View `v_cost_by_use_case`** — aggregated cost and average quality per use case.
- **View `v_latency_percentiles`** — p50 / p90 / p99 latency per provider+model.
- **View `v_summary`** — the six headline totals, for HTTP callers.
- **View `v_model_by_query_type`** — for each kind of request, what each model
  costs, how it scores, and its p90 latency. This is the routing argument.

`prompt` and `response` are stored whole and carry a **90-day column TTL**: the
text blanks itself while the row survives, so the cost and latency history stays
permanent and only the sensitive content ages out. A table-level TTL would have
dropped the whole row and taken the spend record with it.

Adding a column to a live database is handled by `schema.migrations()`, applied
on every connection. `CREATE TABLE IF NOT EXISTS` does nothing to a table that
already exists, so without it an established database would keep its old columns
and fail every insert. Every statement uses `IF NOT EXISTS` / `IF EXISTS` and is
safe to re-run.

A column is `Nullable` exactly when `LLMCallResult` declares the field
`Optional`. That is not stylistic: the driver rejects `None` into a non-nullable
column, and real traces carry `NULL` error fields on every successful row.

Percentiles use `quantile`, not `quantileExact`. Both it and DuckDB's
`quantile_cont` interpolate linearly, so the figures match to 0.0003 ms;
`quantileExact` returns a real member of the dataset and read 527 ms higher at
p99 over 15 traces. Interpolated and discrete quantiles diverge *most* on small
samples, not least.

API:

```python
repo = TraceRepository()                   # database from CLICKHOUSE_DATABASE
repo.save(result)                          # append; logs on failure, never raises
repo.save_many([r1, r2, r3])               # one insert
repo.query("SELECT * FROM llm_traces")     # raw SQL → list of tuples
repo.query_df("SELECT * FROM v_cost_by_tenant")  # raw SQL → pandas DataFrame
repo.get_summary(tenant_id="acme")         # dict with totals & averages
repo.export_to_parquet("traces.parquet")   # export to Parquet, client-side
```

`save()` deliberately does not raise. Against a network service transient
failures are normal, and instrumentation must never be the reason the calling
application goes down — the same principle that makes a failed LLM call return a
result with `.error` set instead of propagating.

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

`provider_name` is used for logging, OTel spans, and ClickHouse analytics.
`model` is passed straight to LiteLLM, which handles the actual API call.

## Metrics tracked per call

- `prompt` / `response` — stored **whole**, never truncated
- `call_type` — `chat` or `embedding`
- `input_tokens` / `output_tokens` / `total_tokens`
- `cached_tokens` — prompt-cache hits, billed at a fraction of the normal rate.
  `None` means the provider did not report it; `0` means it reported zero
- `reasoning_tokens` — billed on thinking models but absent from the
  input/output split
- `latency_ms`
- `tokens_per_second` — output tokens generated per second
- `finish_reason` — whether the response completed naturally or was cut off
- `cost_usd` — via LiteLLM's built-in pricing table, which covers embedding
  models too
- `attempts_used` / `error_type` — retry and failure diagnostics
- `request_id` — provider-assigned ID, if exposed, for cross-referencing later
- `quality_score` — optional 1–10 rating from a judge LLM
- `query_type` — optional classification from that same judge call
- `cost_efficiency` — quality per dollar spent
- `span_id` / `trace_otel_id` — correlation with OpenTelemetry traces
- `tenant_id` / `use_case` — cost attribution fields
- `vector_dimensions` / `input_count` — embedding calls only

## Logging

The library uses Python's `logging` module, not `print()`. Call
`configure_logging()` once at your app's entry point (already done in
`main.py`); every module then logs through `get_logger(__name__)`
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
