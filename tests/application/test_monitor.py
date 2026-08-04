"""
Unit tests for core/application/monitor.py with mocked providers.
Also verifies OpenTelemetry span emission.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from llm_observability.core.application.monitor import LLMMonitor, _token_details
from llm_observability.core.domain.result import LLMCallResult


class FakeUsage:
    prompt_tokens = 10
    completion_tokens = 20
    total_tokens = 30


class FakeChoice:
    finish_reason = "stop"
    message = MagicMock(content="Hello world")


class FakeResponse:
    usage = FakeUsage()
    choices = [FakeChoice()]
    id = "resp-123"


def _make_monitor():
    """Return a monitor with a mocked LiteLLMProvider.call()."""
    with patch("llm_observability.core.application.monitor.LiteLLMProvider") as MockProvider:
        instance = MockProvider.return_value
        instance.get_model_string.return_value = "fake/my-model"
        instance.call.return_value = FakeResponse()
        return LLMMonitor(provider_name="fake", model="fake/my-model")


class TestLLMMonitor:
    @patch("llm_observability.core.application.monitor.litellm.completion_cost", return_value=0.001)
    @patch("llm_observability.core.application.monitor.evaluate", return_value={"quality_score": 8, "quality_reasoning": "OK", "query_type": None})
    def test_call_success(self, mock_score, mock_cost):
        monitor = _make_monitor()
        result = monitor.call("Say hello", score_response_quality=True)

        assert isinstance(result, LLMCallResult)
        assert result.provider == "fake"
        assert result.model == "fake/my-model"
        assert result.input_tokens == 10
        assert result.output_tokens == 20
        assert result.total_tokens == 30
        assert result.finish_reason == "stop"
        assert result.cost_usd == 0.001
        assert result.quality_score == 8
        assert result.succeeded is True
        assert result.error is None
        assert result.attempts_used == 1

    @patch("llm_observability.core.application.monitor.litellm.completion_cost", return_value=None)
    @patch("llm_observability.core.application.monitor.evaluate", return_value={"quality_score": None, "quality_reasoning": None, "query_type": None})
    def test_call_no_quality_scoring(self, mock_score, mock_cost):
        monitor = _make_monitor()
        result = monitor.call("Say hello", score_response_quality=False)
        assert result.quality_score is None
        mock_score.assert_not_called()

    @patch("llm_observability.core.application.monitor.litellm.completion_cost", side_effect=Exception("boom"))
    @patch("llm_observability.core.application.monitor.evaluate", return_value={"quality_score": 7, "quality_reasoning": "Fine", "query_type": None})
    def test_call_cost_computation_failure(self, mock_score, mock_cost):
        monitor = _make_monitor()
        result = monitor.call("Say hello")
        assert result.cost_usd is None
        assert result.succeeded is True

    @patch("llm_observability.core.application.monitor._call_with_retries", return_value=(None, "Rate limited", "RateLimitError", 3))
    def test_call_failure(self, mock_retries):
        monitor = _make_monitor()
        result = monitor.call("Say hello")
        assert result.succeeded is False
        assert result.error == "Rate limited"
        assert result.error_type == "RateLimitError"
        assert result.attempts_used == 3

    def test_call_with_tenant_and_use_case(self):
        monitor = _make_monitor()
        with patch("llm_observability.core.application.monitor.litellm.completion_cost", return_value=0.001):
            with patch("llm_observability.core.application.monitor.evaluate", return_value={"quality_score": 5, "quality_reasoning": "Meh", "query_type": None}):
                result = monitor.call("Hi", tenant_id="acme", use_case="support")
        assert result.tenant_id == "acme"
        assert result.use_case == "support"


class TestOTelSpanAttributes:
    """Verify OTel spans with expected GenAI attributes."""

    @pytest.fixture
    def tracer_fixture(self):
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import SimpleSpanProcessor
        from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
        from opentelemetry import trace

        exporter = InMemorySpanExporter()
        provider = TracerProvider()
        provider.add_span_processor(SimpleSpanProcessor(exporter))
        tracer = trace.get_tracer("test-tracer", tracer_provider=provider)
        return tracer, exporter

    @patch("llm_observability.core.application.monitor.litellm.completion_cost", return_value=0.001)
    @patch("llm_observability.core.application.monitor.evaluate", return_value={"quality_score": 8, "quality_reasoning": "OK", "query_type": None})
    def test_span_name_and_attributes(self, mock_score, mock_cost, tracer_fixture):
        tracer, exporter = tracer_fixture
        with patch("llm_observability.core.application.monitor.get_tracer", return_value=tracer):
            monitor = _make_monitor()
            result = monitor.call("Say hello")

        spans = exporter.get_finished_spans()
        assert len(spans) == 1
        span = spans[0]

        assert span.name == "chat fake"
        assert span.attributes.get("gen_ai.operation.name") == "chat"
        assert span.attributes.get("gen_ai.system") == "fake"
        assert span.attributes.get("gen_ai.request.model") == "fake/my-model"
        assert span.attributes.get("gen_ai.usage.input_tokens") == 10
        assert span.attributes.get("gen_ai.usage.output_tokens") == 20

        assert result.span_id is not None
        assert result.trace_otel_id is not None

    @patch("llm_observability.core.application.monitor._call_with_retries", return_value=(None, "Boom", "Exception", 1))
    def test_error_span(self, mock_retries, tracer_fixture):
        tracer, exporter = tracer_fixture
        with patch("llm_observability.core.application.monitor.get_tracer", return_value=tracer):
            monitor = _make_monitor()
            monitor.call("Say hello")

        spans = exporter.get_finished_spans()
        assert len(spans) == 1
        span = spans[0]
        assert span.attributes.get("error.type") == "Exception"
        assert span.status.status_code.name == "ERROR"


class TestTokenDetails:
    """Cached and reasoning tokens are billed but were previously invisible.

    Both are absent on providers that do not report them, and the nesting
    differs between providers, so the reader must tolerate absence rather
    than index into it.
    """

    def test_reads_both_when_present(self):
        usage = SimpleNamespace(
            prompt_tokens_details=SimpleNamespace(cached_tokens=128),
            completion_tokens_details=SimpleNamespace(reasoning_tokens=64))
        assert _token_details(usage) == (128, 64)

    def test_absent_details_are_none_not_zero(self):
        """None means 'not reported'; 0 would claim the provider said there
        were none. The difference matters when judging whether caching works."""
        assert _token_details(SimpleNamespace()) == (None, None)

    def test_present_but_empty_details(self):
        usage = SimpleNamespace(prompt_tokens_details=None,
                                completion_tokens_details=None)
        assert _token_details(usage) == (None, None)

    def test_partial_support(self):
        usage = SimpleNamespace(
            prompt_tokens_details=SimpleNamespace(cached_tokens=32))
        assert _token_details(usage) == (32, None)

    def test_zero_cached_is_preserved_not_coerced_to_none(self):
        usage = SimpleNamespace(
            prompt_tokens_details=SimpleNamespace(cached_tokens=0))
        assert _token_details(usage)[0] == 0
