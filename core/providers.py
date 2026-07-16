"""
Handles calling each LLM provider through litellm and assembling the
result log for each call (tokens, latency, cost, quality....).
"""

import time
import uuid
from datetime import datetime, timezone

import litellm

from config.settings import MAX_RETRIES, RETRY_BACKOFF_SECONDS
from core.metrics import compute_throughput, compute_cost_efficiency
from core.quality_scoring import score_quality


def build_error_log(provider, model_name, prompt, latency_ms, error, error_type, attempts_used):
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
        "request_id": None,
        "attempts_used": attempts_used,
        "error_type": error_type,
        "error": error
    }


def _call_with_retries(model_name: str, prompt: str):
    """
    Calls litellm.completion with a simple retry-with-backoff loop.
    Returns a tuple: (response, error_message, error_type, attempts_used).
    On success, error_message and error_type are None.
    On final failure (all retries exhausted), response is None.
    This lives in its own function because retrying is an instrumentation
    concern, separate from building the result log.
    """

    last_error = None
    last_error_type = None

    for attempt in range(1, MAX_RETRIES + 2):  # +2: first attempt + MAX_RETRIES retries
        try:
            response = litellm.completion(
                model=model_name,
                messages=[{"role": "user", "content": prompt}]
            )
            return response, None, None, attempt
        except Exception as e:
            last_error = str(e)
            last_error_type = type(e).__name__
            if attempt <= MAX_RETRIES:
                backoff = RETRY_BACKOFF_SECONDS * (2 ** (attempt - 1))
                print(f"   ...retry {attempt}/{MAX_RETRIES} after {last_error_type}, "
                      f"waiting {backoff}s")
                time.sleep(backoff)

    return None, last_error, last_error_type, MAX_RETRIES + 1


def call_provider(provider_name: str, model_name: str, prompt: str) -> dict:
    """
    Generic call used for every provider. Having one shared function
    means every new metric only needs to be added in one place.
    """
    start_time = time.perf_counter()

    response, error, error_type, attempts_used = _call_with_retries(model_name, prompt)

    end_time = time.perf_counter()
    latency_ms = (end_time - start_time) * 1000

    if error:
        print(f" ERREUR {provider_name} : {error}")
        return build_error_log(provider_name, model_name, prompt, latency_ms,
                                error, error_type, attempts_used)

    usage = response.usage
    input_tokens = usage.prompt_tokens
    output_tokens = usage.completion_tokens
    total_tokens = usage.total_tokens

    finish_reason = response.choices[0].finish_reason
    response_text = response.choices[0].message.content

    # Provider-assigned request ID, if exposed on the response object.
    request_id = getattr(response, "id", None)

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
        "request_id": request_id,
        "attempts_used": attempts_used,
        "error_type": None,
        "error": None
    }