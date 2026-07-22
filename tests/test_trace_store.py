"""Unit tests for core/trace_store.py — uses a temporary DuckDB file."""

import os
import tempfile
import pytest
from datetime import datetime, timezone

from core.trace_store import TraceStore
from core.models import LLMCallResult


def _make_result(**overrides) -> LLMCallResult:
    defaults = dict(
        provider="openai",
        trace_id="t-001",
        timestamp=datetime.now(timezone.utc).isoformat() + "Z",
        model="gpt-4o",
        prompt_preview="Hello",
        response_preview="Hi there",
        input_tokens=10,
        output_tokens=20,
        total_tokens=30,
        finish_reason="stop",
        latency_ms=150.0,
        tokens_per_second=133.33,
        cost_usd=0.0005,
        quality_score=9,
        quality_reasoning="Good",
        cost_efficiency=18000.0,
        request_id="req-1",
        attempts_used=1,
        error_type=None,
        error=None,
        span_id="span-1",
        trace_otel_id="trace-1",
        tenant_id="tenant-a",
        use_case="summarization",
    )
    defaults.update(overrides)
    return LLMCallResult(**defaults)


@pytest.fixture
def store():
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test.duckdb")
        s = TraceStore(db_path=db_path)
        yield s
        s.close()


class TestSave:
    def test_save_single(self, store):
        r = _make_result()
        store.save(r)
        rows = store.query("SELECT * FROM llm_traces")
        assert len(rows) == 1
        assert rows[0][0] == "openai"  # provider column

    def test_save_many(self, store):
        store.save_many([_make_result(trace_id="t-1"), _make_result(trace_id="t-2")])
        rows = store.query("SELECT trace_id FROM llm_traces")
        assert len(rows) == 2

    def test_save_empty_list(self, store):
        store.save_many([])
        rows = store.query("SELECT * FROM llm_traces")
        assert len(rows) == 0


class TestQueryDf:
    def test_basic(self, store):
        store.save(_make_result())
        df = store.query_df("SELECT provider, cost_usd FROM llm_traces")
        assert len(df) == 1
        assert df["provider"][0] == "openai"


class TestViews:
    def test_v_cost_by_tenant(self, store):
        store.save_many([
            _make_result(trace_id="a", tenant_id="t1", cost_usd=1.0),
            _make_result(trace_id="b", tenant_id="t1", cost_usd=2.0),
            _make_result(trace_id="c", tenant_id="t2", cost_usd=3.0),
        ])
        df = store.query_df("SELECT * FROM v_cost_by_tenant")
        assert len(df) == 2
        t1 = df[df["tenant_id"] == "t1"]
        assert t1["total_cost_usd"].iloc[0] == 3.0

    def test_v_cost_by_use_case(self, store):
        store.save_many([
            _make_result(trace_id="a", use_case="chat", cost_usd=1.0, quality_score=8),
            _make_result(trace_id="b", use_case="chat", cost_usd=2.0, quality_score=10),
        ])
        df = store.query_df("SELECT * FROM v_cost_by_use_case")
        assert len(df) == 1
        assert df["avg_quality_score"].iloc[0] == 9.0

    def test_v_latency_percentiles(self, store):
        store.save_many([
            _make_result(trace_id="a", latency_ms=100.0),
            _make_result(trace_id="b", latency_ms=200.0),
            _make_result(trace_id="c", latency_ms=300.0),
        ])
        df = store.query_df("SELECT * FROM v_latency_percentiles")
        assert len(df) == 1
        assert df["p50_ms"].iloc[0] == 200.0


class TestGetSummary:
    def test_overall(self, store):
        store.save_many([
            _make_result(trace_id="a", cost_usd=1.0, latency_ms=100.0, quality_score=8),
            _make_result(trace_id="b", cost_usd=2.0, latency_ms=200.0, quality_score=10),
        ])
        summary = store.get_summary()
        assert summary["total_calls"] == 2
        assert summary["total_cost_usd"] == 3.0
        assert summary["avg_latency_ms"] == 150.0
        assert summary["avg_quality_score"] == 9.0

    def test_filtered_by_tenant(self, store):
        store.save_many([
            _make_result(trace_id="a", tenant_id="t1", cost_usd=1.0),
            _make_result(trace_id="b", tenant_id="t2", cost_usd=2.0),
        ])
        summary = store.get_summary(tenant_id="t1")
        assert summary["total_calls"] == 1
        assert summary["total_cost_usd"] == 1.0


class TestExportToParquet:
    def test_export(self, store):
        store.save(_make_result())
        with tempfile.TemporaryDirectory() as tmpdir:
            out = os.path.join(tmpdir, "out.parquet")
            store.export_to_parquet(out)
            assert os.path.exists(out)
            # Verify readable by duckdb
            df = store.query_df(f"SELECT * FROM '{out}'")
            assert len(df) == 1
