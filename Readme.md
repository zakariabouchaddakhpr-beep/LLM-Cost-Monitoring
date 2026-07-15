# LLM Monitor

A middleware library that sits between your application and whichever
LLM provider you use (Claude, GPT, Gemini, Groq, Cerebras), instrumenting
every call with consistent metrics: latency, tokens, cost, throughput,
retry behavior, and an optional quality score — regardless of which
provider answered.

Built on [LiteLLM](https://github.com/BerriAI/litellm) for the actual
provider calls, and a Factory pattern for provider selection: swap
providers by changing one string, no other code changes.

## Project structure

```
llm_monitor/
├── quickstart.py          how to use the library (not the library's entry point)
├── config/
│   └── settings.py           judge model, retry settings, per-provider default models
├── core/
│   ├── base_provider.py       the interface every provider adapter implements
│   ├── provider_factory.py    Factory: provider name -> provider instance
│   ├── monitor.py             LLMMonitor - the middleman, wraps any provider with metrics
│   ├── metrics.py             throughput / cost-efficiency math
│   ├── quality_scoring.py     optional LLM-as-judge quality scoring
│   └── providers/
│       ├── openai_provider.py
│       ├── anthropic_provider.py
│       ├── gemini_provider.py
│       ├── groq_provider.py
│       └── cerebras_provider.py
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
└── results/                   generated at runtime, not committed
```

## Usage

```python
from core.monitor import LLMMonitor

monitor = LLMMonitor(provider_name="anthropic")   # or "openai", "gemini", "groq", "cerebras"
result = monitor.call("Explain the Factory design pattern in 3 sentences.")

print(result["total_tokens"], result["latency_ms"], result["cost_usd"])
```

`result` is a dictionary with the same shape no matter which provider
was called — see "Metrics tracked" below.

## Setup (local, no Docker)

```bash
git clone <this-repo>
cd llm_monitor
cp .env.example .env        # then fill in your real API keys
pip install -r requirements.txt
python quickstart.py
```

Results are written to `results/call_log.json`.

## Setup (Docker)

```bash
cp .env.example .env        # then fill in your real API keys
docker compose up --build
```

Results still land in `./results/` on your host machine via the volume
mount in `docker-compose.yml`.

## Adding a new provider

1. Create `core/providers/my_provider.py`, implementing `call()`,
   `get_provider_name()`, and `get_model_string()` (see any existing
   provider for the pattern).
2. Register it in `core/provider_factory.py`:
   `LLMProviderFactory.register_provider("my_provider", MyProvider)`
3. Add a default model string to `DEFAULT_MODELS` in `config/settings.py`.

Nothing in `monitor.py` or `quickstart.py` needs to change.

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

## Notes

- Never commit `.env` — it's git-ignored. Use `.env.example` as the
  template for which variables need to be set.
- `results/` is git-ignored since it's generated output, not source code.
- Model strings in `DEFAULT_MODELS` should be double-checked against
  LiteLLM's current supported model list before production use — some
  provider model IDs are version-dated and get deprecated over time.