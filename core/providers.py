"""
Handles calling each LLM provider through litellm and assembling the
result log for each call (tokens, latency, cost, quality, etc).
"""

import time
import uuid
from datetime import datetime, timezone

import litellm

from core.metrics import compute_throughput, compute_cost_efficiency
from core.quality_scoring import score_quality


def build_error_log(provider, model_name, prompt, latency_ms, error):
    return {
        "provider": provider,
        "trace_id": str(uuid.uuid4()),
        "timestamp": datetime.now(timezone.utc).isoformat() + "Z",
        "model": model_name,
        "prompt_preview": prompt[:100] + ("..." if len(prompt) > 100 else ""),
        "response_preview": None,
        "input_tokens": None,
        "output_tokens": None,
        "total_tokens": None,
        "finish_reason": None,
        "latency_ms": latency_ms,
        "tokens_per_second": None,
        "cost_usd": None,
        "quality_score": None,
        "quality_reasoning": None,
        "cost_efficiency": None,
        "error": error
    }


def call_provider(provider_name: str, model_name: str, prompt: str) -> dict:
    """
    Generic call used for every provider. Having one shared function
    (instead of one copy-pasted block per provider) means every new
    metric only needs to be added in one place.
    """
    start_time = time.perf_counter()
    error = None
    response = None

    try:
        response = litellm.completion(
            model=model_name,
            messages=[{"role": "user", "content": prompt}]
        )
    except Exception as e:
        error = str(e)

    end_time = time.perf_counter()
    latency_ms = (end_time - start_time) * 1000

    if error:
        print(f" ERREUR {provider_name} : {error}")
        return build_error_log(provider_name, model_name, prompt, latency_ms, error)

    usage = response.usage
    input_tokens = usage.prompt_tokens
    output_tokens = usage.completion_tokens
    total_tokens = usage.total_tokens

    finish_reason = response.choices[0].finish_reason
    response_text = response.choices[0].message.content

    try:
        total_cost_usd = litellm.completion_cost(response)
    except Exception:
        total_cost_usd = None

    tokens_per_second = compute_throughput(output_tokens, latency_ms)
    quality = score_quality(prompt, response_text)
    cost_efficiency = compute_cost_efficiency(quality["quality_score"], total_cost_usd)

    return {
        "provider": provider_name,
        "trace_id": str(uuid.uuid4()),
        "timestamp": datetime.now(timezone.utc).isoformat() + "Z",
        "model": model_name,
        "prompt_preview": prompt[:100] + ("..." if len(prompt) > 100 else ""),
        "response_preview": response_text,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
        "finish_reason": finish_reason,
        "latency_ms": latency_ms,
        "tokens_per_second": tokens_per_second,
        "cost_usd": round(total_cost_usd, 8) if total_cost_usd is not None else None,
        "quality_score": quality["quality_score"],
        "quality_reasoning": quality["quality_reasoning"],
        "cost_efficiency": cost_efficiency,
        "error": None
    }