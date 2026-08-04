"""
LLMMonitor is the middleman between a developer's app and whichever
LLM provider they've chosen. It wraps every call with uniform
instrumentation: retries, latency, tokens, cost, throughput, and a quality
score — the exact same metrics regardless of which provider answered.

OpenTelemetry GenAI Semantic Conventions (2026):
- Span name: "chat {provider}" (e.g. "chat openai")
- Attributes: gen_ai.system, gen_ai.request.model, gen_ai.usage.input_tokens,
  gen_ai.usage.output_tokens, gen_ai.operation.name
- Metrics: gen_ai.client.token.usage (counter), gen_ai.client.operation.duration (histogram),
  llm.cost.usd (histogram)
"""

import os
import time
import uuid
from datetime import datetime, timezone

if "LITELLM_MODE" not in os.environ:
    os.environ["LITELLM_MODE"] = "PRODUCTION"

import litellm

from llm_observability.config.settings import MAX_RETRIES, RETRY_BACKOFF_SECONDS
from llm_observability.core.domain.metrics import compute_throughput, compute_cost_efficiency
from llm_observability.core.domain.result import LLMCallResult
from llm_observability.core.application.quality_scoring import evaluate
from llm_observability.core.infrastructure.provider import LiteLLMProvider
from llm_observability.core.infrastructure.otel import get_tracer, get_meter
from llm_observability.core.logging_config import get_logger

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


def _token_details(usage):
    """
    Pull cached and reasoning token counts out of a usage object.

    Both are absent on providers that do not report them, and the nesting
    differs between providers, so this walks the chain with getattr rather than
    indexing. None means "not reported"; 0 means the provider reported zero, and
    the difference matters when judging whether prompt caching is working.

    Cached tokens are billed at a fraction of the normal rate, so a workload
    that caches heavily is much cheaper than its raw token count suggests.
    Reasoning tokens are billed but do not appear in the input/output split.
    """
    prompt_details = getattr(usage, "prompt_tokens_details", None)
    completion_details = getattr(usage, "completion_tokens_details", None)
    cached = getattr(prompt_details, "cached_tokens", None) if prompt_details else None
    reasoning = getattr(completion_details, "reasoning_tokens", None) if completion_details else None
    return cached, reasoning


def _call_with_retries(provider, prompt: str):
    """Retry loop with exponential backoff."""
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
    monitor = LLMMonitor(provider_name="openai", model="openai/gpt-4o")
    result = monitor.call(prompt)
    """

    def __init__(self, provider_name: str, model: str, api_key: str | None = None):
        self.provider_name = provider_name
        self.provider = LiteLLMProvider(model, api_key=api_key)
        logger.info("LLMMonitor initialized for provider=%s model=%s",
                    provider_name, self.provider.get_model_string())

    def call(self, prompt: str, score_response_quality: bool = True,
             tenant_id: str = None, use_case: str = None,
             judge_api_key: str = None,
             classify_query: bool = False) -> LLMCallResult:

        tracer = get_tracer()
        with tracer.start_as_current_span(f"chat {self.provider_name}") as span:
            from opentelemetry.trace import Status, StatusCode

            span.set_attribute("gen_ai.operation.name", "chat")
            span.set_attribute("gen_ai.system", self.provider_name)
            span.set_attribute("gen_ai.request.model", self.provider.get_model_string())

            result = LLMCallResult(
                provider=self.provider_name,
                trace_id=str(uuid.uuid4()),
                # isoformat() on an aware datetime already ends "+00:00".
                # Appending "Z" produced "...+00:00Z", which is not valid
                # ISO 8601 - it was masked in the database because the ORM
                # converter stripped a trailing Z, but to_json() and the
                # call_log_*.json dumps carried the malformed value.
                timestamp=datetime.now(timezone.utc).isoformat(),
                model=self.provider.get_model_string(),
                # Stored whole. This used to be truncated at 100 characters.
                prompt=prompt,
                tenant_id=tenant_id,
                use_case=use_case,
            )

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
            result.cached_tokens, result.reasoning_tokens = _token_details(usage)
            result.finish_reason = response.choices[0].finish_reason
            result.response = response.choices[0].message.content
            result.request_id = getattr(response, "id", None)

            span.set_attribute("gen_ai.usage.input_tokens", result.input_tokens)
            span.set_attribute("gen_ai.usage.output_tokens", result.output_tokens)

            try:
                # model= is passed explicitly rather than letting litellm infer
                # it from the response. Providers report the BARE model name
                # ("llama-3.3-70b-versatile"), and litellm's pricing table is
                # keyed on the prefixed one ("groq/llama-3.3-70b-versatile"), so
                # inference silently yields no price - measured against Groq,
                # where every call came back with cost_usd NULL.
                cost = litellm.completion_cost(
                    completion_response=response,
                    model=self.provider.get_model_string(),
                )
                result.cost_usd = round(cost, 8) if cost is not None else None
            except Exception:
                result.cost_usd = None
                logger.warning("Could not compute cost for model=%s", result.model)

            result.tokens_per_second = compute_throughput(result.output_tokens, result.latency_ms)

            # ONE judge call serves both features. Guarded here as well as
            # inside evaluate() so that with neither feature on there is no
            # call at all, not even a function call that returns early.
            # Enabling classification alongside scoring costs no extra API call.
            if score_response_quality or classify_query:
                enrichment = evaluate(
                    prompt, result.response,
                    want_score=score_response_quality,
                    want_query_type=classify_query,
                    api_key=judge_api_key,
                )
                result.quality_score = enrichment["quality_score"]
                result.quality_reasoning = enrichment["quality_reasoning"]
                result.query_type = enrichment["query_type"]

            result.cost_efficiency = compute_cost_efficiency(result.quality_score, result.cost_usd)

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

    def embed(self, inputs, *, tenant_id: str = None,
              use_case: str = None) -> LLMCallResult:
        """
        Make an instrumented embedding call and return the trace.

        One row per API call, not per input: a batch of 100 documents is one
        billed request, and every other row in this table means one call.
        Counting them individually would multiply the call count and misstate
        the cost.

        The vectors ride back on result.embeddings, which is deliberately not
        persisted - far too large, and the caller needs them in hand.

        No judge call: there is no response to score, and classifying the input
        text is a different question from classifying a request. Skipping it is
        most of why embeddings stay cheap.
        """
        texts = [inputs] if isinstance(inputs, str) else list(inputs)

        tracer = get_tracer()
        with tracer.start_as_current_span(f"embeddings {self.provider_name}") as span:
            from opentelemetry.trace import Status, StatusCode

            span.set_attribute("gen_ai.operation.name", "embeddings")
            span.set_attribute("gen_ai.system", self.provider_name)
            span.set_attribute("gen_ai.request.model", self.provider.get_model_string())

            result = LLMCallResult(
                provider=self.provider_name,
                trace_id=str(uuid.uuid4()),
                timestamp=datetime.now(timezone.utc).isoformat(),
                model=self.provider.get_model_string(),
                # Stored whole, like a prompt. A list is joined so the text is
                # preserved rather than summarised away.
                prompt="\n".join(texts),
                call_type="embedding",
                input_count=len(texts),
                tenant_id=tenant_id,
                use_case=use_case,
            )

            ctx = span.get_span_context()
            if ctx.is_valid:
                result.span_id = format(ctx.span_id, "016x")
                result.trace_otel_id = format(ctx.trace_id, "032x")

            start_time = time.perf_counter()
            try:
                response = self.provider.embed(texts)
            except Exception as exc:
                result.latency_ms = (time.perf_counter() - start_time) * 1000
                result.error = str(exc)
                result.error_type = type(exc).__name__
                span.set_attribute("error.type", result.error_type)
                span.set_status(Status(StatusCode.ERROR, description=result.error))
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

            try:
                # model= passed explicitly, for the same reason as the chat
                # path: providers report the bare model name and litellm's
                # pricing table is keyed on the prefixed one.
                cost = litellm.completion_cost(
                    completion_response=response,
                    model=self.provider.get_model_string(),
                )
                result.cost_usd = round(cost, 8) if cost is not None else None
            except Exception:
                # Unpriced is not failed. Logged loudly because a silently NULL
                # cost makes the whole tool undercount spend.
                result.cost_usd = None
                logger.warning("Could not compute embedding cost for model=%s",
                               result.model)

            logger.info("Embedding succeeded provider=%s inputs=%d dims=%s "
                        "tokens=%s latency_ms=%.1f cost_usd=%s",
                        self.provider_name, result.input_count,
                        result.vector_dimensions, result.input_tokens,
                        result.latency_ms, result.cost_usd)
            return result
