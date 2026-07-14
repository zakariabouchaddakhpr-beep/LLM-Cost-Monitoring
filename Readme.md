# LLM Benchmark

Benchmarks Groq, Gemini, and Cerebras on the same prompt through
[LiteLLM](https://github.com/BerriAI/litellm), logging latency, tokens,
cost, throughput, quality score, and cost-efficiency for each provider.

## Project structure

```
llm_benchmark/
├── main.py                 entry point
├── config/
│   └── settings.py         judge model + provider list
├── core/
│   ├── metrics.py           throughput / cost-efficiency math
│   ├── quality_scoring.py   LLM-as-judge quality scoring
│   └── providers.py         the actual provider call + log assembly
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
└── results/                 generated at runtime, not committed
```

## Setup (local, no Docker)

```bash
git clone <this-repo>
cd llm_benchmark
cp .env.example .env        # then fill in your real API keys
pip install -r requirements.txt
python main.py
```

Results are written to `results/output_litellm.json`.

## Setup (Docker)

```bash
cp .env.example .env        # then fill in your real API keys
docker compose up --build
```

Results still land in `./results/output_litellm.json` on your host
machine via the volume mount in `docker-compose.yml`.

## Metrics tracked per call

- `input_tokens` / `output_tokens` / `total_tokens`
- `latency_ms`
- `tokens_per_second` — output tokens generated per second
- `finish_reason` — whether the response completed naturally or was cut off
- `cost_usd` — via LiteLLM's built-in pricing table
- `quality_score` — 1–10 rating from a judge LLM
- `cost_efficiency` — quality per dollar spent

## Notes

- Never commit `.env` — it's git-ignored. Use `.env.example` as the
  template for which variables need to be set.
- `results/` is git-ignored since it's generated output, not source code.