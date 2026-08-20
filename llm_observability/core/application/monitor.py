import os
import time
import uuid
from datetime import datetime, timezone
from typing import NamedTuple

if "LITELLM_MODE" not in os.environ:
    os.environ["LITELLM_MODE"] = "PRODUCTION"

import litellm

from llm_observability.config.settings import MAX_RETRIES, RETRY_BACKOFF_SECONDS
from llm_observability.core.domain.metrics import (
    compute_throughput,
    compute_cost_efficiency,
)
from llm_observability.core.domain.result import LLMCallResult
from llm_observability.core.application.quality_scoring import evaluate
from llm_observability.core.infrastructure.provider import LiteLLMProvider
from llm_observability.core.infrastructure.otel import get_tracer, get_meter
from llm_observability.core.logging_config import get_logger

logger = get_logger(__name__)


def _get_otel_metrics():
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


def _token_details(usage):
    prompt_details = getattr(usage, "prompt_tokens_details", None)
    completion_details = getattr(usage, "completion_tokens_details", None)

    cached = getattr(prompt_details, "cached_tokens", None) if prompt_details else None
    reasoning = (getattr(completion_details, "reasoning_tokens", None)
                 if completion_details else None)
    return cached, reasoning


class RetryOutcome(NamedTuple):
    response: object
    error: str | None
    error_type: str | None
    attempts_used: int
    last_attempt_ms: float


def _call_with_retries(provider, prompt: str) -> RetryOutcome:
    last_error = None
    last_error_type = None
    last_attempt_ms = 0.0
    total_attempts = MAX_RETRIES + 1

    for attempt in range(1, total_attempts + 1):
        attempt_start = time.perf_counter()
        try:
            response = provider.call(prompt)
            last_attempt_ms = (time.perf_counter() - attempt_start) * 1000
            return RetryOutcome(response, None, None, attempt, last_attempt_ms)
        except Exception as e:
            last_attempt_ms = (time.perf_counter() - attempt_start) * 1000
            last_error = str(e)
            last_error_type = type(e).__name__
            if attempt <= MAX_RETRIES:

                backoff = RETRY_BACKOFF_SECONDS * (2 ** (attempt - 1))
                logger.warning("Retry %d/%d after %s, waiting %.2fs",
                               attempt, MAX_RETRIES, last_error_type, backoff)
                time.sleep(backoff)
    return RetryOutcome(None, last_error, last_error_type, total_attempts,
                        last_attempt_ms)


def _compute_cost(response, model: str):
    try:
        cost = litellm.completion_cost(completion_response=response, model=model)
        return round(cost, 8) if cost is not None else None
    except Exception:
        logger.warning("Could not compute cost for model=%s", model)
        return None


def _attach_otel_ids(result: LLMCallResult, span) -> None:

    context = span.get_span_context()
    if context.is_valid:
        result.span_id = format(context.span_id, "016x")
        result.trace_otel_id = format(context.trace_id, "032x")


def _mark_span_failed(span, error: str, error_type: str) -> None:
    from opentelemetry.trace import Status, StatusCode

    span.set_attribute("error.type", error_type)
    span.set_status(Status(StatusCode.ERROR, description=error))


def _record_otel_metrics(result: LLMCallResult, provider_name: str) -> None:

    otel = _get_otel_metrics()

    otel["op_duration"].record(result.latency_ms, {
        "gen_ai.system": provider_name,
        "gen_ai.request.model": result.model,
        "gen_ai.operation.name": "chat",
    })
    if result.input_tokens is not None:
        otel["token_usage"].add(result.input_tokens, {
            "gen_ai.system": provider_name,
            "gen_ai.token.type": "input",
        })
    if result.output_tokens is not None:
        otel["token_usage"].add(result.output_tokens, {
            "gen_ai.system": provider_name,
            "gen_ai.token.type": "output",
        })
    if result.cost_usd is not None:
        otel["cost_usd"].record(result.cost_usd, {
            "gen_ai.system": provider_name,
            "gen_ai.request.model": result.model,
        })


class LLMMonitor:
    def __init__(self, provider_name: str, model: str,
                 api_key: str | None = None):
        self.provider_name = provider_name
        self.provider = LiteLLMProvider(model, api_key=api_key)
        logger.info("LLMMonitor initialized for provider=%s model=%s",
                    provider_name, self.provider.get_model_string())

    def _new_result(self, prompt: str, tenant_id, use_case,
                    call_type: str = "chat", input_count=None) -> LLMCallResult:
        return LLMCallResult(
            provider=self.provider_name,
            trace_id=str(uuid.uuid4()),
            timestamp=datetime.now(timezone.utc).isoformat(),
            model=self.provider.get_model_string(),
            prompt=prompt,
            call_type=call_type,
            input_count=input_count,
            tenant_id=tenant_id,
            use_case=use_case,
        )

    def _set_request_attributes(self, span, operation: str) -> None:
        span.set_attribute("gen_ai.operation.name", operation)
        span.set_attribute("gen_ai.system", self.provider_name)
        span.set_attribute("gen_ai.request.model", self.provider.get_model_string())

    def call(self, prompt: str, score_response_quality: bool = True,
             tenant_id: str = None, use_case: str = None,
             judge_api_key: str = None,
             classify_query: bool = False) -> LLMCallResult:
        tracer = get_tracer()
        with tracer.start_as_current_span(f"chat {self.provider_name}") as span:
            self._set_request_attributes(span, "chat")

            result = self._new_result(prompt, tenant_id, use_case)
            _attach_otel_ids(result, span)
            start_time = time.perf_counter()
            outcome = _call_with_retries(self.provider, prompt)
            result.latency_ms = (time.perf_counter() - start_time) * 1000
            result.attempts_used = outcome.attempts_used
            if outcome.error:
                result.error = outcome.error
                result.error_type = outcome.error_type
                span.set_attribute("error.message", outcome.error)
                _mark_span_failed(span, outcome.error, outcome.error_type)
                logger.error("Call to %s failed: %s",
                             self.provider_name, outcome.error)
                return result

            self._record_usage(result, outcome.response, span)
            result.cost_usd = _compute_cost(outcome.response, result.model)

            result.tokens_per_second = compute_throughput(
                result.output_tokens, outcome.last_attempt_ms)

            self._add_judge_verdict(result, prompt, score_response_quality,
                                    classify_query, judge_api_key)

            result.cost_efficiency = compute_cost_efficiency(result.quality_score,
                                                             result.cost_usd)

            _record_otel_metrics(result, self.provider_name)

            logger.info(
                "Call succeeded provider=%s tokens=%d latency_ms=%.1f "
                "cost_usd=%s attempts=%d",
                self.provider_name, result.total_tokens, result.latency_ms,
                result.cost_usd, result.attempts_used,
            )
            return result

    def _record_usage(self, result: LLMCallResult, response, span) -> None:
        usage = response.usage
        result.input_tokens = usage.prompt_tokens
        result.output_tokens = usage.completion_tokens
        result.total_tokens = usage.total_tokens
        result.cached_tokens, result.reasoning_tokens = _token_details(usage)
        result.finish_reason = response.choices[0].finish_reason
        result.response = response.choices[0].message.content
        result.request_id = getattr(response, "id", None)

        span.set_attribute("gen_ai.usage.input_tokens", result.input_tokens)
        span.set_attribute("gen_ai.usage.output_tokens", result.output_tokens)

    def _add_judge_verdict(self, result: LLMCallResult, prompt: str,
                           want_score: bool, want_query_type: bool,
                           judge_api_key) -> None:

        if not (want_score or want_query_type):
            return
        verdict = evaluate(
            prompt, result.response,
            want_score=want_score,
            want_query_type=want_query_type,
            api_key=judge_api_key,
        )
        result.quality_score = verdict["quality_score"]
        result.quality_reasoning = verdict["quality_reasoning"]
        result.query_type = verdict["query_type"]

    def embed(self, inputs, *, tenant_id: str = None,
              use_case: str = None) -> LLMCallResult:
        texts = [inputs] if isinstance(inputs, str) else list(inputs)

        tracer = get_tracer()
        with tracer.start_as_current_span(
                f"embeddings {self.provider_name}") as span:
            self._set_request_attributes(span, "embeddings")

            result = self._new_result(
                "\n".join(texts), tenant_id, use_case,
                call_type="embedding", input_count=len(texts),
            )
            _attach_otel_ids(result, span)

            start_time = time.perf_counter()
            try:
                response = self.provider.embed(texts)
            except Exception as exc:
                result.latency_ms = (time.perf_counter() - start_time) * 1000
                result.error = str(exc)
                result.error_type = type(exc).__name__
                _mark_span_failed(span, result.error, result.error_type)
                logger.error("Embedding call to %s failed: %s",
                             self.provider_name, exc)
                return result

            result.latency_ms = (time.perf_counter() - start_time) * 1000
            result.attempts_used = 1

            vectors = [row["embedding"] for row in response.data]
            result.embeddings = vectors

            result.vector_dimensions = len(vectors[0]) if vectors else None
            result.request_id = getattr(response, "id", None)

            usage = getattr(response, "usage", None)
            if usage is not None:
                result.input_tokens = getattr(usage, "prompt_tokens", None)
                result.total_tokens = getattr(usage, "total_tokens", None)
                span.set_attribute("gen_ai.usage.input_tokens",
                                   result.input_tokens or 0)

            result.cost_usd = _compute_cost(response, result.model)

            logger.info("Embedding succeeded provider=%s inputs=%d dims=%s "
                        "tokens=%s latency_ms=%.1f cost_usd=%s",
                        self.provider_name, result.input_count,
                        result.vector_dimensions, result.input_tokens,
                        result.latency_ms, result.cost_usd)
            return result
