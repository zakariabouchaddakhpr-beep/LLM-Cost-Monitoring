# LLM Monitor

A middleware library that sits between your application and whichever
LLM provider you use (Claude, GPT, Gemini, Groq, Cerebras), instrumenting
every call with consistent metrics: latency, tokens, cost, throughput,
retry behavior, and an optional quality score — regardless of which
provider answered.

Built on [LiteLLM](https://github.com/BerriAI/litellm) for the actual
provider calls, and a Factory pattern for provider selection: swap
providers by changing which class the Factory looks up, no other code
changes needed elsewhere in the library.

## Project structure

```
llm_monitor/
├── quickstart.py              usage example — set PROVIDER_NAME/MODEL here
├── config/
│   └── settings.py            judge model, retry settings (no provider/model defaults)
├── core/
│   ├── base_provider.py        the interface every provider adapter implements
│   ├── provider_factory.py     Factory: provider name -> provider instance
│   ├── monitor.py              LLMMonitor - the middleman, wraps any provider with metrics
│   ├── models.py                LLMCallResult - the structured result object
│   ├── logging_config.py        shared logging setup (used instead of print())
│   ├── metrics.py               throughput / cost-efficiency math
│   ├── quality_scoring.py       optional LLM-as-judge quality scoring
│   └── providers/
│       ├── openai_provider.py
│       ├── anthropic_provider.py
│       ├── gemini_provider.py
│       ├── groq_provider.py
│       └── cerebras_provider.py
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
└── results/                    generated at runtime, not committed
```

## Usage

```python
from core.monitor import LLMMonitor

monitor = LLMMonitor(provider_name="anthropic", model="anthropic/claude-3-5-sonnet-20241022")
result = monitor.call("Explain the Factory design pattern in 3 sentences.")

print(result.total_tokens, result.latency_ms, result.cost_usd)
result.save("results/call_log.json")   # or result.to_json() / result.to_dict()
```

`result` is an `LLMCallResult` instance (see `core/models.py`) with the
same fields no matter which provider was called — see "Metrics tracked"
below. `provider_name` and `model` are required and explicit — there's
no default model lookup; you always state exactly which model you want.

## Setup (local, no Docker)

```bash
git clone <this-repo>
cd llm_monitor
cp .env.example .env        # then fill in your real API keys
pip install -r requirements.txt
python quickstart.py
```

Edit `PROVIDER_NAME` and `MODEL` at the top of `quickstart.py` to change
which LLM gets called. Results are written to `results/call_log.json`.

## Setup (Docker)

```bash
cp .env.example .env        # then fill in your real API keys
docker compose up --build
```

Results still land in `./results/` on your host machine via the volume
mount in `docker-compose.yml`.

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

## Logging

The library uses Python's `logging` module, not `print()`. Call
`configure_logging()` once at your app's entry point (already done in
`quickstart.py`); every module then logs through `get_logger(__name__)`
with timestamps and severity levels, which can be filtered or redirected
in one place without touching library code.

## Notes

- Never commit `.env` — it's git-ignored. Use `.env.example` as the
  template for which variables need to be set.
- `results/` is git-ignored since it's generated output, not source code.
- Model strings passed to `LLMMonitor`/provider constructors should be
  checked against LiteLLM's current supported model list — some
  provider model IDs are version-dated and get deprecated over time.