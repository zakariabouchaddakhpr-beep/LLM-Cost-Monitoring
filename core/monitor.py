"""
LLMMonitor is the middleman between a developer's app and whichever
LLM provider they've chosen. It uses the Factory to get the right
provider adapter, then wraps every call with uniform instrumentation:
retries, latency, tokens, cost, throughput, and (optionally) a quality
score — the exact same metrics regardless of whether the underlying
call went to Claude, GPT, Gemini, Groq, or Cerebras.
"""

import time
import uuid
from datetime import datetime, timezone

import litellm

from config.settings import MAX_RETRIES, RETRY_BACKOFF_SECONDS
from core.metrics import compute_throughput, compute_cost_efficiency
from core.quality_scoring import score_quality
from core.provider_factory import LLMProviderFactory


def build_error_log(provider_name, model_name, prompt, latency_ms, error, error_type, attempts_used):
    return {
        "provider": provider_name,
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


def _call_with_retries(provider, prompt: str):
    """
    provider is an LLMProvider instance (any concrete adapter from the
    Factory). Because every adapter implements .call(prompt) the same
    way, this retry loop works identically no matter which provider
    was actually chosen.
    """
    last_error = None
    last_error_type = None

    for attempt in range(1, MAX_RETRIES + 2):
        try:
            response = provider.call(prompt)
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


class LLMMonitor:
    """
    Usage:
        monitor = LLMMonitor(provider_name="anthropic")
        result = monitor.call("Explain the Factory pattern in 3 sentences.")

    Swap provider_name to "openai", "gemini", "groq", or "cerebras" and
    every downstream metric (latency, cost, throughput, ...) is computed
    identically — the app code calling .call() never needs to change.
    """

    def __init__(self, provider_name: str, **provider_kwargs):
        self.provider_name = provider_name
        self.provider = LLMProviderFactory.get_provider(provider_name, **provider_kwargs)

    def call(self, prompt: str, score_response_quality: bool = True) -> dict:
        model_name = self.provider.get_model_string()

        start_time = time.perf_counter()
        response, error, error_type, attempts_used = _call_with_retries(self.provider, prompt)
        end_time = time.perf_counter()
        latency_ms = (end_time - start_time) * 1000

        if error:
            print(f" ERREUR {self.provider_name} : {error}")
            return build_error_log(self.provider_name, model_name, prompt,
                                    latency_ms, error, error_type, attempts_used)

        usage = response.usage
        input_tokens = usage.prompt_tokens
        output_tokens = usage.completion_tokens
        total_tokens = usage.total_tokens

        finish_reason = response.choices[0].finish_reason
        response_text = response.choices[0].message.content
        request_id = getattr(response, "id", None)

        try:
            total_cost_usd = litellm.completion_cost(response)
        except Exception:
            total_cost_usd = None

        tokens_per_second = compute_throughput(output_tokens, latency_ms)

        if score_response_quality:
            quality = score_quality(prompt, response_text)
        else:
            quality = {"quality_score": None, "quality_reasoning": None}

        cost_efficiency = compute_cost_efficiency(quality["quality_score"], total_cost_usd)

        return {
            "provider": self.provider_name,
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