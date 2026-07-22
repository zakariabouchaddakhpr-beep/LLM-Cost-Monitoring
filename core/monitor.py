"""
LLMMonitor is the middleman between a developer's app and whichever
LLM provider they've chosen. It uses the Factory to get the right
provider adapter, then wraps every call with uniform instrumentation:
retries, latency, tokens, cost, throughput, and a quality score — the exact same metrics
regardless of whether the underlying call went to the providers.

OpenTelemetry GenAI Semantic Conventions (2026):
- Span name: "chat {provider}" (e.g. "chat openai")
- Attributes: gen_ai.system, gen_ai.request.model, gen_ai.usage.input_tokens,
  gen_ai.usage.output_tokens, gen_ai.operation.name
- Metrics: gen_ai.client.token.usage (counter), gen_ai.client.operation.duration (histogram),
  llm.cost.usd (histogram)

The result is built ONCE, as an LLMCallResult, right at the start of
call(). Every step below just fills in more of its fields as data
becomes available. There's a single return at the end.
"""

import time
import uuid
from datetime import datetime, timezone

import litellm

from config.settings import MAX_RETRIES, RETRY_BACKOFF_SECONDS
from core.metrics import compute_throughput, compute_cost_efficiency
from core.quality_scoring import score_quality
from core.provider_factory import LLMProviderFactory
from core.models import LLMCallResult
from core.logging_config import get_logger
from core.otel_setup import get_tracer, get_meter

logger = get_logger(__name__)


def _get_otel_metrics():
    """Lazily create OTel metrics so tests can swap the meter first."""
    meter = get_meter()
    return {
        "token_usage": meter.create_counter(
            "gen_ai.client.token.usage",
            description="Number of tokens used in GenAI operations",
            unit="token",
        ),
        "op_duration": meter.create_histogram(
            "gen_ai.client.operation.duration",
            description="Duration of GenAI client operations",
            unit="ms",
        ),
        "cost_usd": meter.create_histogram(
            "llm.cost.usd",
            description="Cost of LLM call in USD",
            unit="USD",
        ),
    }


def _preview(prompt: str) -> str:
    return prompt[:100] + ("..." if len(prompt) > 100 else "")


def _call_with_retries(provider, prompt: str):
    """
    provider is an LLMProvider instance.
    Because every adapter implements .call(prompt) the same way,
    this retry loop works identically no matter which provider
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
                logger.warning(
                    "Retry %d/%d after %s, waiting %.2fs",
                    attempt, MAX_RETRIES, last_error_type, backoff
                )
                time.sleep(backoff)

    return None, last_error, last_error_type, MAX_RETRIES + 1


class LLMMonitor:
    """
    monitor = LLMMonitor(provider_name="provider", model="model")
    result = monitor.call(prompt)
    Swap provider_name/model to point at a different LLM entirely —
    every downstream metric (latency, cost, throughput, ...) is computed
    identically — the app code calling .call() never needs to change.
    """

    def __init__(self, provider_name: str, model: str, **provider_kwargs):
        self.provider_name = provider_name
        self.provider = LLMProviderFactory.get_provider(provider_name, model=model, **provider_kwargs)
        logger.info("LLMMonitor initialized for provider=%s model=%s",
                    provider_name, self.provider.get_model_string())

    def call(self, prompt: str, score_response_quality: bool = True,
             tenant_id: str = None, use_case: str = None) -> LLMCallResult:

        tracer = get_tracer()
        with tracer.start_as_current_span(f"chat {self.provider_name}") as span:
            from opentelemetry.trace import Status, StatusCode

            # GenAI semantic convention attributes (always set)
            span.set_attribute("gen_ai.operation.name", "chat")
            span.set_attribute("gen_ai.system", self.provider_name)
            span.set_attribute("gen_ai.request.model", self.provider.get_model_string())

            # Built once, right away, with everything we already know.
            # Every step below fills in more fields on this SAME object.
            result = LLMCallResult(
                provider=self.provider_name,
                trace_id=str(uuid.uuid4()),
                timestamp=datetime.now(timezone.utc).isoformat() + "Z",
                model=self.provider.get_model_string(),
                prompt_preview=_preview(prompt),
                tenant_id=tenant_id,
                use_case=use_case,
            )

            # Capture OTel correlation IDs from the current span context
            ctx = span.get_span_context()
            if ctx.is_valid:
                result.span_id = format(ctx.span_id, "016x")
                result.trace_otel_id = format(ctx.trace_id, "032x")

            start_time = time.perf_counter()
            response, error, error_type, attempts_used = _call_with_retries(self.provider, prompt)
            result.latency_ms = (time.perf_counter() - start_time) * 1000
            result.attempts_used = attempts_used

            if error:
                result.error = error
                result.error_type = error_type
                span.set_attribute("error.type", error_type)
                span.set_attribute("error.message", error)
                span.set_status(Status(StatusCode.ERROR, description=error))
                logger.error("Call to %s failed: %s", self.provider_name, error)
                return result

            usage = response.usage
            result.input_tokens = usage.prompt_tokens
            result.output_tokens = usage.completion_tokens
            result.total_tokens = usage.total_tokens
            result.finish_reason = response.choices[0].finish_reason
            result.response_preview = response.choices[0].message.content
            result.request_id = getattr(response, "id", None)

            # GenAI semantic convention: token usage on span
            span.set_attribute("gen_ai.usage.input_tokens", result.input_tokens)
            span.set_attribute("gen_ai.usage.output_tokens", result.output_tokens)

            try:
                cost = litellm.completion_cost(response)
                result.cost_usd = round(cost, 8) if cost is not None else None
            except Exception:
                result.cost_usd = None
                logger.warning("Could not compute cost for model=%s", result.model)

            result.tokens_per_second = compute_throughput(result.output_tokens, result.latency_ms)

            if score_response_quality:
                quality = score_quality(prompt, result.response_preview)
                result.quality_score = quality["quality_score"]
                result.quality_reasoning = quality["quality_reasoning"]

            result.cost_efficiency = compute_cost_efficiency(result.quality_score, result.cost_usd)

            # Emit OTel metrics
            otel = _get_otel_metrics()
            otel["op_duration"].record(result.latency_ms, {
                "gen_ai.system": self.provider_name,
                "gen_ai.request.model": result.model,
                "gen_ai.operation.name": "chat",
            })

            if result.input_tokens is not None:
                otel["token_usage"].add(result.input_tokens, {
                    "gen_ai.system": self.provider_name,
                    "gen_ai.token.type": "input",
                })
            if result.output_tokens is not None:
                otel["token_usage"].add(result.output_tokens, {
                    "gen_ai.system": self.provider_name,
                    "gen_ai.token.type": "output",
                })
            if result.cost_usd is not None:
                otel["cost_usd"].record(result.cost_usd, {
                    "gen_ai.system": self.provider_name,
                    "gen_ai.request.model": result.model,
                })

            logger.info(
                "Call succeeded provider=%s tokens=%d latency_ms=%.1f cost_usd=%s attempts=%d",
                self.provider_name, result.total_tokens, result.latency_ms,
                result.cost_usd, result.attempts_used
            )

            return result
