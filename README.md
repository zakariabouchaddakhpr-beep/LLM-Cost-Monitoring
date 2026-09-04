# LLM Monitor

A middleware library that sits between your application and whichever
LLM provider you use (Claude, GPT, Gemini, Groq, Cerebras), instrumenting
every call with consistent metrics: latency, tokens, cost, throughput,
retry behavior, and an optional quality score — regardless of which
provider answered. Embedding calls are instrumented the same way and land in
the same table, so a cost query spans everything the application spent.

Built on [LiteLLM](https://github.com/BerriAI/litellm) for the actual
provider calls. LiteLLM handles all provider-specific routing via the
model-string prefix (e.g. `openai/gpt-4o`, `anthropic/claude-3-5-sonnet`),
so the library needs only a single thin adapter.

Six pieces:

- **A three-call API** — `obs.init()`, `obs.call()`, `obs.dashboard()`. Adding
  observability to an existing project is three lines and no connection string.
- **OpenTelemetry GenAI Semantic Conventions** — every call emits a span
  with standard attributes and metrics. Console exporter by default; OTLP
  when `OTEL_EXPORTER_OTLP_ENDPOINT` is set.
- **ClickHouse persistence** — traces are appended to a ClickHouse server, one
  database per project, with five pre-built analytics views for cost attribution
  by tenant and use case.
- **A Streamlit dashboard** — cost, latency, quality, tokens and activity over
  whatever the filters select, reading live while the monitored application is
  still writing.
- **Complexity routing** — pass a pool of models instead of one and each prompt
  is classified locally, then sent to the cheapest tier that should be able to
  answer it. The decision is recorded on the trace.
- **A routing recommender** — every routed call is judged after the fact against
  a public benchmark, and the cost of a wrong tier is reported in dollars or in
  quality points.

## Architecture (layered)

```
llm_observability/
├── api.py                   the public API: init / call / embed / dashboard
├── .streamlit/config.toml   dashboard theme, shipped as package data
├── config/
│   └── settings.py          connection config, retries, OTel config
├── core/
│   ├── domain/              pure business logic — zero external deps
│   │   ├── result.py        LLMCallResult dataclass
│   │   ├── metrics.py       throughput / cost-efficiency math
│   │   └── routing.py       the tier policy: thresholds, choose, resolve
│   ├── application/         orchestration
│   │   ├── monitor.py       LLMMonitor — the middleman
│   │   └── quality_scoring.py
│   ├── infrastructure/      adapters for external systems
│   │   ├── provider.py      LiteLLM adapter
│   │   ├── otel.py          OpenTelemetry tracer & meter
│   │   ├── classifier.py    prompt task type + complexity, run locally
│   │   └── persistence/     ClickHouse via clickhouse-connect
│   │       ├── database.py  client factory
│   │       ├── schema.py    all DDL: database, table, five views
│   │       └── repository.py TraceRepository (writes + analytics)
│   └── logging_config.py    configure_logging / get_logger
├── dashboard/
│   ├── app.py               the Streamlit dashboard
│   └── __main__.py          python -m llm_observability.dashboard
└── recommender/             offline analysis — never in the request path
    ├── sprout.py            reads the SPROUT benchmark files
    ├── probes.py            picks the probe prompts
    ├── grading.py           marks an answer without a judge model
    ├── tiering.py           matches your models to reference stand-ins
    ├── neighbours.py        finds benchmark prompts resembling yours
    ├── verdict.py           one verdict for one routing decision
    ├── review.py            aggregates verdicts into a report
    ├── thresholds.py        fits the routing thresholds from measured data
    └── bulk_classify.py     batch prompt classification, resumable
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
├── sproutdb/                the SPROUT benchmark, used by the recommender
├── recommender_output/      cached probe set, scores and classifications
├── threshold_study/         evidence behind the shipped routing thresholds
├── pyproject.toml           packaging
├── requirements.txt
├── Dockerfile
├── docker-compose.yaml
└── results/                 generated at runtime, not committed
```

`main.py` is deliberately outside the package and is not shipped by a
`pip install`: only `llm_observability/` is. The dataset and study directories
are inputs and evidence, not code, and are likewise not packaged.

**Requirements:** Python 3.10 or newer — the code uses the `str | None` union
syntax — plus a reachable ClickHouse server. Everything else is in
`requirements.txt`, which `pyproject.toml` reads directly so a Docker build and a
`pip install` of the package can never disagree about versions.

## Usage

Two ways in. This is the lower-level one, where you hold the monitor and the
repository yourself; skip to
[Using it in your own project](#using-it-in-your-own-project) for the
three-call API that wires them together for you.

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

### Not environment variables

Three settings are constants in [settings.py](llm_observability/config/settings.py)
rather than environment variables, because they describe the library's behaviour
rather than the deployment:

| Constant | Value | |
|---|---|---|
| `MAX_RETRIES` | `2` | retries *after* the first attempt, so 3 calls at most. The count that was actually used lands in `attempts_used` |
| `RETRY_BACKOFF_SECONDS` | `1.5` | base delay, doubled on each subsequent retry |
| `JUDGE_MODEL` | `groq/llama-3.3-70b-versatile` | the model that scores quality and classifies queries, when either is enabled |

`tenant_id` and `use_case` are per-call arguments, not configuration — the same
process serves many tenants, so they cannot be fixed at startup.

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
| `model` | a LiteLLM model string, or a `{tier: model}` dict to turn routing on |
| `api_key` | passed straight to LiteLLM. Omit it and LiteLLM finds its own from the environment |

An existing database is **reused, never replaced** — call `init()` as often as you
like.

### The rest of the surface

```python
obs.summary()          # {'total_calls': 15, 'total_cost_usd': 0.0045, ...}
obs.database()         # which ClickHouse database this project's traces are in
obs.embed("some text") # embeddings, instrumented the same way (see below)
obs.review(days=30)    # judge the routing decisions of the last 30 days
obs.dashboard(port=8502, block=False)   # a second project, non-blocking

client = obs.init(...)  # or hold the object instead of using the module default
client.call("…")
client.close()          # release the ClickHouse connection
```

`init()` returns an `Observer` and also stores it as the module default, which is
what lets the bare `obs.call(...)` form work. Calling any of these before
`init()` raises with an explanation rather than a `NoneType` error.

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

### Routing

Pass a **pool** of models instead of one and every prompt is classified before it
is sent, then routed to the cheapest tier that should be able to answer it:

```python
obs.init(name="my-app", model={
    "weak":   "groq/llama-3.1-8b-instant",
    "mid":    "groq/llama-3.3-70b-versatile",
    "strong": "openai/gpt-4o",
})
```

The classifier is NVIDIA's `prompt-task-and-complexity-classifier`, which runs
**locally** — no network call and no per-request cost. It returns a task type and
a complexity score between 0 and 1, and the table in
[routing.py](llm_observability/core/domain/routing.py) maps that score to a tier,
with two cut points per task type.

Three fields land on the trace — `task_type`, `complexity` and `tier` — so a
routing decision is auditable rather than invisible, and the dashboard's Routing
tab is built from them.

**Two rules keep it safe.** An unknown task, or a classifier that failed
outright, routes to `strong`. And if `choose()` asks for a tier that isn't in
your pool, `resolve()` rounds **up**, never down. Both lean the same way on
purpose: overpaying is visible on a cost chart and recoverable, while answering
too weakly damages output quietly.

> **Four of the eleven task types now carry measured thresholds; the other seven
> are still the original hand-picked values.** Where they were fitted, the signal
> underneath turned out to be weak. Against 10,000 benchmark prompts, complexity predicts
> whether the weak tier will succeed with an AUC of **0.531**, where 0.500 is a
> coin flip. Only `Code Generation` shows real signal (0.585). The current values
> still cut spend 30–86% on the four best-supported task types at unchanged
> quality — they work by no longer overpaying, not by predicting difficulty well.
> The full evidence is in `threshold_study/`, and
> `thresholds.band_table()` regenerates the per-task working.

**Routing needs `torch` and `transformers`, which are not in
`requirements.txt`.** A base install is deliberately light; without them,
`obs.init()` with a dict raises `ModuleNotFoundError`.

### The routing recommender

Once calls have been routed, `obs.review()` judges those decisions after the fact
and reports what the wrong ones cost:

```python
obs.review(days=30)
```

You cannot re-run the other tiers retrospectively, so the answer comes
indirectly. The recommender uses **SPROUT** — a public dataset of 44,241 prompts
already answered and graded for 13 reference models — in five stages:

| Stage | What it does |
|---|---|
| `probes` | Picks 96 prompts whose answers can be checked mechanically, spread over six difficulty bands |
| `grading` | Marks an answer without a judge model: the marker, then "the answer is…", then the last number; compared as text, as a number, then symbolically |
| `tiering` | Scores each of your models on the probes and matches it to the closest of the 13 references |
| `neighbours` | For a real trace, finds SPROUT prompts of the same task type and near complexity |
| `verdict` | Reads each tier's score and cost off its stand-in and compares against what actually ran |

Every call gets one of four verdicts — `OPTIMAL`, `OVERPAID`, `UNDERSERVED`,
`NO_VERDICT` — and the regret is reported in **two separate units**: dollars
wasted for overpaying, quality points lost for under-serving. They are never
summed, because they call for opposite fixes.

Only `tiering` makes paid calls, roughly 17 minutes per model, and the result is
cached per model in `recommender_output/`, so a second review costs nothing.
`sproutdb/` must be present — about 555 MB of parquet.

> The prices in [verdict.py](llm_observability/recommender/verdict.py) are
> placeholders. Measurement showed the fitted thresholds don't move across a 16×
> price swing, because the policy is quality-constrained rather than
> cost-constrained — but any **dollar total** the recommender reports is only as
> good as those numbers.

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
buys you.

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

| Flag | |
|---|---|
| `--name` | derives the database, used only when `CLICKHOUSE_DATABASE` is unset |
| `--port` | default `8501` |
| `--address` | bind address. Containers need `0.0.0.0`, or the published port looks dead from the host |

A missing database or an unreachable server exits `2` with a one-line message
rather than a driver traceback.

### One thing that will surprise you

`import litellm` calls `load_dotenv()` itself, walking up from wherever it is
installed. If your virtualenv sits inside a project that has a `.env` with
`CLICKHOUSE_DATABASE` in it, that value lands in your environment without you
setting it — and since it wins, it would override `name`. Pass `database=`
explicitly if you ever need to be certain.

This matters: a leaked value points a project at another project's *database* on
a shared server.

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

This is the reason the storage layer is a database server.

There is no lock to negotiate: start the dashboard, run your application, and
watch rows arrive. The dashboard reads while the monitored application is still
writing, so it never has to wait for that application to exit — an observability
tool that only works after the observed process has finished is a report
generator.

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

A column is `Nullable` exactly when `LLMCallResult` declares the field
`Optional`. That is not stylistic: the driver rejects `None` into a non-nullable
column, and real traces carry `NULL` error fields on every successful row.

Percentiles use `quantile`, not `quantileExact`. `quantile` interpolates
linearly; `quantileExact` returns a real member of the dataset and read 527 ms
higher at p99 over 15 traces. Interpolated and discrete quantiles diverge *most* on small
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

## The dashboard

`obs.dashboard()`, or `python -m llm_observability.dashboard`, opens a Streamlit
UI over one database. A row of filters — date range, provider, tenant, use case,
and query type — scopes everything below it, above a KPI row (calls, cost,
latency, quality, tokens, throughput, failures, retries, truncations) and seven
tabs:

| Tab | What it answers |
|---|---|
| **Cost** | Spend per day, per tenant, per use case, per model, and the ten most expensive calls |
| **Latency** | p50/p90/p99 per model, the distribution behind them, throughput, and the ten slowest calls |
| **Quality** | Judge-score distribution, cost against quality, score per use case, models by query type, finish reasons |
| **Routing** | Complexity distribution, which tasks get escalated, tiers and models actually used |
| **Recommender** | Verdicts on routing decisions, wasted spend and quality lost |
| **Tokens** | Input vs output split per use case and per model |
| **Activity** | Calls per day and per hour of the day |
| **Calls** | The raw traces, newest first, capped at 200 rows |
| **SQL** | An explorer for arbitrary queries against the table and views |

**The filters run on the server, not on a loaded DataFrame.** They are compiled
into one query, so narrowing a selection sends less over the wire rather than
more — the opposite of fetching everything and filtering in pandas, which gets
slower exactly as the trace history grows.

The **Query type** row of pills only appears once something has been classified.
With `classify_query=False` — the default — an always-visible empty filter would
be noise.

The theme lives in `llm_observability/.streamlit/config.toml` and ships as
package data, which is why both `obs.dashboard()` and the Docker service run
Streamlit with the working directory set to the package: Streamlit reads that
file from the cwd, so launching from the caller's folder would silently drop the
light/dark palette.

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

Every call returns an `LLMCallResult` carrying all of the following, and that
same object is what `TraceRepository.save()` writes as one row.

- `trace_id` / `timestamp` — identity of the call
- `provider` / `model`
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
- `quality_score` / `quality_reasoning` — optional 1–10 rating from a judge LLM,
  and the sentence it gave for that rating
- `query_type` — optional classification from that same judge call
- `cost_efficiency` — quality per dollar spent
- `span_id` / `trace_otel_id` — correlation with OpenTelemetry traces
- `tenant_id` / `use_case` — cost attribution fields
- `task_type` / `complexity` / `tier` — the routing decision, when a model pool
  is configured. `NULL` on unrouted calls, which is what the recommender filters on
- `vector_dimensions` / `input_count` — embedding calls only

Plus a few conveniences on the object itself:

```python
result.succeeded            # False when .error is set
result.to_dict()            # plain dict
result.to_json(indent=2)
result.save("call_log.json")
```

`result.embeddings` holds the vectors on an embedding call and is the one field
that is **not** persisted — the caller needs them in hand, and a database is the
wrong place for them.

## Logging

The library uses Python's `logging` module, not `print()`. Call
`configure_logging()` once at your app's entry point (already done in
`main.py`); every module then logs through `get_logger(__name__)`
with timestamps and severity levels, which can be filtered or redirected
in one place without touching library code.

## Testing

There is currently **no test suite in the repository.** An earlier `tests/`
directory covering the recommender's grading, verdict and review logic has been
removed, so `pytest` collects nothing.

When it is restored, two constraints are worth keeping. **No test should make a
paid API call** — the provider is mocked, and the recommender's pure functions
(`grading`, `verdict`, `review`, `thresholds`) all take plain DataFrames, so they
need neither a network nor a database. Anything that does want a real ClickHouse
should `pytest.skip` with an explanatory message when there isn't one rather than
failing, and should point at `TEST_CLICKHOUSE_HOST`/`TEST_CLICKHOUSE_PORT` when
set, so a test run never touches the database holding real traces.

## Notes

- Never commit `.env` — it's git-ignored. Use `.env.example` as the
template for which variables need to be set.
- `results/` is git-ignored since it's generated output, not source code.
- Model strings passed to `LLMMonitor`/provider constructors should be
checked against LiteLLM's current supported model list — some provider
model IDs are version-dated and get deprecated over time.
- **litellm is pinned exactly** in `requirements.txt`, unlike everything else,
which only has a floor. This code reads its response shape and its pricing
table in detail, so a minor release that reshapes either would silently change
what `cost_usd` means. That pin is also what fixes which model strings are
priced: bumping it is how you pick up newly released models.
