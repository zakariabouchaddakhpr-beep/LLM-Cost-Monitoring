"""litellm.embedding is mocked throughout — a real call costs money."""
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from llm_observability.core.application.monitor import LLMMonitor


def _embedding_response(n_vectors=1, dims=8, prompt_tokens=5):
    return SimpleNamespace(
        data=[{"embedding": [0.1] * dims} for _ in range(n_vectors)],
        usage=SimpleNamespace(prompt_tokens=prompt_tokens, total_tokens=prompt_tokens),
        id="emb-1",
    )


@pytest.fixture
def monitor():
    with patch("llm_observability.core.application.monitor.LiteLLMProvider") as P:
        m = LLMMonitor(provider_name="openai", model="openai/text-embedding-3-small")
        m.provider = P.return_value
        m.provider.get_model_string.return_value = "openai/text-embedding-3-small"
        yield m


class TestEmbedShape:
    def test_single_string(self, monitor):
        monitor.provider.embed.return_value = _embedding_response(1, 8)
        with patch("llm_observability.core.application.monitor.litellm.completion_cost",
                   return_value=0.000002):
            r = monitor.embed("hello")
        assert r.call_type == "embedding"
        assert r.input_count == 1
        assert r.vector_dimensions == 8
        assert r.prompt == "hello"
        assert r.embeddings is not None and len(r.embeddings) == 1

    def test_batch_is_one_row_not_many(self, monitor):
        """A batch is one billed request, so it is one trace. One row per input
        would multiply the call count and misstate cost."""
        monitor.provider.embed.return_value = _embedding_response(3, 8)
        with patch("llm_observability.core.application.monitor.litellm.completion_cost",
                   return_value=0.000006):
            r = monitor.embed(["a", "b", "c"])
        assert r.input_count == 3
        assert r.prompt == "a\nb\nc"          # stored whole, joined
        assert len(r.embeddings) == 3

    def test_chat_only_fields_stay_none(self, monitor):
        monitor.provider.embed.return_value = _embedding_response()
        with patch("llm_observability.core.application.monitor.litellm.completion_cost",
                   return_value=0.0):
            r = monitor.embed("hello")
        assert r.output_tokens is None
        assert r.finish_reason is None
        assert r.tokens_per_second is None
        assert r.response is None

    def test_never_judged(self, monitor):
        """No response to score, and classifying an embedding input is a
        different question. Skipping the judge is most of why embeddings
        are cheap."""
        monitor.provider.embed.return_value = _embedding_response()
        with patch("llm_observability.core.application.monitor.evaluate") as judge, \
             patch("llm_observability.core.application.monitor.litellm.completion_cost",
                   return_value=0.0):
            r = monitor.embed("hello")
        judge.assert_not_called()
        assert r.quality_score is None
        assert r.query_type is None

    def test_input_tokens_recorded(self, monitor):
        monitor.provider.embed.return_value = _embedding_response(prompt_tokens=42)
        with patch("llm_observability.core.application.monitor.litellm.completion_cost",
                   return_value=0.0):
            r = monitor.embed("hello")
        assert r.input_tokens == 42

    def test_attribution_passes_through(self, monitor):
        monitor.provider.embed.return_value = _embedding_response()
        with patch("llm_observability.core.application.monitor.litellm.completion_cost",
                   return_value=0.0):
            r = monitor.embed("hi", tenant_id="acme", use_case="search")
        assert (r.tenant_id, r.use_case) == ("acme", "search")


class TestEmbedFailure:
    def test_failure_returns_an_errored_result_rather_than_raising(self, monitor):
        monitor.provider.embed.side_effect = RuntimeError("provider down")
        r = monitor.embed("hello")
        assert not r.succeeded
        assert r.error_type == "RuntimeError"
        assert r.call_type == "embedding"

    def test_unpriceable_embedding_leaves_cost_none(self, monitor):
        monitor.provider.embed.return_value = _embedding_response()
        with patch("llm_observability.core.application.monitor.litellm.completion_cost",
                   side_effect=Exception("no pricing")):
            r = monitor.embed("hello")
        assert r.cost_usd is None
        assert r.succeeded          # unpriced is not failed
