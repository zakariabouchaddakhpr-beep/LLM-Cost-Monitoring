# LLM Monitor

Measures what every LLM call costs, how long it took, and how good the answer
was — whichever provider answered. Optionally picks a cheaper model when it can,
and tells you afterwards whether that was the right call.

Built on [LiteLLM](https://github.com/BerriAI/litellm), so it works with OpenAI,
Anthropic, Gemini, Groq and Cerebras through one adapter.

## What you get

- **Every call measured** — cost, latency, tokens, retries, failures, and an
  optional quality score, tagged by customer and feature
- **A dashboard** that reads live while your app is still running
- **Automatic model choice** — send each prompt to the cheapest model that can
  handle it
- **An audit** of those choices, priced in dollars wasted or quality lost

## Requirements

Python 3.10+, and a reachable ClickHouse server:

```bash
docker compose up -d clickhouse
```

## Install

```bash
pip install /path/to/this/repo
```

## Use it

```python
import llm_observability as obs

obs.init(name="my-app", model="cerebras/gpt-oss-120b")

result = obs.call("Summarise this ticket…", tenant_id="acme", use_case="support")
print(result.cost_usd, result.latency_ms)   # the trace is already saved

obs.dashboard()                             # → http://localhost:8501
```

That's the whole API. `init()` picks a database from `name`, and the trace is
written before `call()` returns.

The rest of the surface:

```python
obs.embed("some text")   # embeddings, measured the same way
obs.summary()            # {'total_calls': 15, 'total_cost_usd': 0.0045, ...}
obs.review(days=30)      # judge the last 30 days of routing decisions
obs.database()           # which database this project writes to
```

## Configuration

All from the environment, or a `.env` file. Copy `.env.example` and fill it in.

| Variable | Default | |
|---|---|---|
| `CLICKHOUSE_DATABASE` | — | wins over `name=`; without either you get an error |
| `CLICKHOUSE_HOST` | `localhost` | leave as localhost — Compose overrides it |
| `CLICKHOUSE_PORT` | `8123` | the HTTP port, not 9000 |
| `CLICKHOUSE_USER` | `default` | must be set for Docker |
| `CLICKHOUSE_PASSWORD` | empty | must be non-empty for Docker |
| `OPENAI_API_KEY` etc. | — | one key for whichever provider you call |

Database names take letters, digits and underscores only.

## Quality scoring

Off by default, because it makes a second, billable call:

```python
obs.init(name="my-app", model="…", score_quality=True, classify_query=True)
```

`score_quality` adds a 1–10 rating from a judge model. `classify_query` labels
the request type. Both come from the same judge call, so turning on the second
one is free.

## Routing

Pass a pool of models instead of one:

```python
obs.init(name="my-app", model={
    "weak":   "groq/llama-3.1-8b-instant",
    "mid":    "groq/llama-3.3-70b-versatile",
    "strong": "openai/gpt-4o",
})
```

A small classifier runs locally — no API call, no cost — and scores how hard the
prompt is. Thresholds in
[routing.py](llm_observability/core/domain/routing.py) map that score to a tier.
If anything is unclear, the prompt goes to `strong`; if a tier isn't configured,
it rounds up, never down.

`task_type`, `complexity` and `tier` land on every routed trace.

> Routing needs `torch` and `transformers`, which are **not** in
> `requirements.txt` — a base install stays light. Without them, passing a dict
> to `init()` raises `ModuleNotFoundError`.

## The recommender

```python
obs.review(days=30)
```

Judges past routing decisions and reports what the wrong ones cost. It compares
your prompts against SPROUT, a public dataset of 44,241 questions already graded
for 13 models, so it needs no extra API calls.

Four verdicts per call — `OPTIMAL`, `OVERPAID`, `UNDERSERVED`, `NO_VERDICT` —
with money wasted and quality lost reported separately.

The first run scores each of your models on 96 probe prompts (~17 minutes per
model, paid, cached afterwards). `sproutdb/` must be present, about 555 MB.

## The dashboard

```bash
python -m llm_observability.dashboard --name my-app
```

Nine tabs — Cost, Latency, Quality, Routing, Recommender, Tokens, Activity,
Calls, SQL — over one filter row. Filters run in the database, not in the page.

## Docker

```bash
cp .env.example .env
docker compose up -d clickhouse dashboard    # no LLM calls
docker compose up --build                    # also runs main.py, which costs money
```

Three services: `clickhouse` (the store), `monitor` (`main.py`, an example
workload), `dashboard` (the UI on port 8501).

## Layout

```
llm_observability/
├── api.py            init / call / embed / review / dashboard
├── config/           settings, read from the environment
├── core/
│   ├── domain/       pure logic: the result record, metrics, routing policy
│   ├── application/  the monitor, and the quality judge
│   └── infrastructure/  LiteLLM, OpenTelemetry, the classifier, ClickHouse
├── dashboard/        the Streamlit app
└── recommender/      offline analysis, never in the request path
```

Traces go to one append-only ClickHouse table plus five views. Prompt and
response text expires after 90 days; the cost history is kept permanently.

Every call also emits an OpenTelemetry span and three metrics, using the GenAI
semantic conventions. Console by default, OTLP if you set
`OTEL_EXPORTER_OTLP_ENDPOINT`.

## More detail

- **[CODE_NOTES.md](CODE_NOTES.md)** — why the code is the way it is, line by line
- **[README_full.md](README_full.md)** — the long-form version of this file
- **`threshold_study/`** — the measurements behind the routing thresholds

## Notes

- Never commit `.env`.
- There is currently no test suite in the repository.
- `litellm` is pinned exactly, because the cost figures depend on its pricing
  table. Bumping it is how you pick up newly released models.
