"""
Repository tests against a real ClickHouse server.

Skipped rather than failed when no server is reachable: mocking the client would
test nothing about the SQL dialect, which is the part most likely to be wrong.
"""
import os
import uuid
from datetime import datetime, timedelta, timezone

import pytest

from llm_observability.config.settings import ClickHouseConfig
from llm_observability.core.domain.result import LLMCallResult
from llm_observability.core.infrastructure.persistence.database import ClickHouseUnavailable
from llm_observability.core.infrastructure.persistence.repository import (
    TraceRepository,
    to_utc,
)

# Falls back to the project's own CLICKHOUSE_* settings, which settings.py has
# already loaded from .env by the time this module imports. That means the tests
# follow the running compose stack and keep working if its password changes.
# TEST_CLICKHOUSE_* overrides them when you want a separate throwaway server.
CH_HOST = os.getenv("TEST_CLICKHOUSE_HOST") or os.getenv("CLICKHOUSE_HOST", "localhost")
CH_PORT = int(os.getenv("TEST_CLICKHOUSE_PORT") or os.getenv("CLICKHOUSE_PORT", "8123"))
CH_USER = os.getenv("TEST_CLICKHOUSE_USER") or os.getenv("CLICKHOUSE_USER", "default")
CH_PASS = os.getenv("TEST_CLICKHOUSE_PASSWORD") or os.getenv("CLICKHOUSE_PASSWORD", "")


def _config(database: str) -> ClickHouseConfig:
    return ClickHouseConfig(host=CH_HOST, port=CH_PORT, database=database,
                            username=CH_USER, password=CH_PASS)


@pytest.fixture(scope="session")
def server_available():
    import clickhouse_connect
    try:
        c = clickhouse_connect.get_client(host=CH_HOST, port=CH_PORT,
                                          username=CH_USER, password=CH_PASS)
        c.command("SELECT 1")
        c.close()
    except Exception as exc:
        pytest.skip(f"No ClickHouse at {CH_HOST}:{CH_PORT} ({exc}). "
                    "Start one with: docker compose up -d clickhouse")


@pytest.fixture
def repo(server_available):
    name = "test_" + uuid.uuid4().hex[:12]
    r = TraceRepository(config=_config(name))
    yield r
    # command(), not query(): DDL returns no result set.
    r._client.command(f"DROP DATABASE IF EXISTS {name}")
    r.close()


def _result(**over) -> LLMCallResult:
    defaults = dict(
        provider="openai", trace_id="t-001",
        timestamp=datetime.now(timezone.utc).isoformat(),
        model="gpt-4o", prompt="Hello", response="Hi there",
        input_tokens=10, output_tokens=20, total_tokens=30, finish_reason="stop",
        latency_ms=150.0, tokens_per_second=133.33, cost_usd=0.0005,
        quality_score=9, quality_reasoning="Good", cost_efficiency=18000.0,
        request_id="req-1", attempts_used=1, error_type=None, error=None,
        span_id="span-1", trace_otel_id="trace-1", tenant_id="tenant-a",
        use_case="summarization",
    )
    defaults.update(over)
    return LLMCallResult(**defaults)


class TestToUtc:
    def test_naive_is_assumed_utc_not_local(self):
        """A naive datetime would otherwise be shifted by the client's offset —
        measured at +01:00 on the development machine."""
        out = to_utc(datetime(2026, 7, 29, 23, 49, 55))
        assert out.tzinfo is not None
        assert out.hour == 23

    def test_aware_is_converted_not_relabelled(self):
        plus_two = timezone(timedelta(hours=2))
        out = to_utc(datetime(2026, 7, 29, 23, 0, 0, tzinfo=plus_two))
        assert out.hour == 21

    def test_iso_string(self):
        assert to_utc("2026-07-29T23:49:55+00:00").hour == 23

    def test_legacy_malformed_suffix(self):
        """monitor.py used to emit isoformat() + 'Z', giving '+00:00Z'."""
        assert to_utc("2026-07-29T23:49:55.999+00:00Z").hour == 23

    def test_plain_z_suffix(self):
        assert to_utc("2026-07-29T23:49:55Z").hour == 23


class TestSave:
    def test_save_single(self, repo):
        repo.save(_result())
        assert repo.query("SELECT count() FROM llm_traces")[0][0] == 1

    def test_save_many(self, repo):
        repo.save_many([_result(trace_id="t-1"), _result(trace_id="t-2")])
        assert repo.query("SELECT count() FROM llm_traces")[0][0] == 2

    def test_save_empty_list(self, repo):
        repo.save_many([])
        assert repo.query("SELECT count() FROM llm_traces")[0][0] == 0

    def test_failed_call_shape_with_every_optional_none(self, repo):
        """monitor.py returns exactly this shape when a call errors, and the
        real trace data carries NULL error/error_type on every row."""
        repo.save(_result(
            trace_id="failed", response=None, input_tokens=None,
            output_tokens=None, total_tokens=None, finish_reason=None,
            latency_ms=None, tokens_per_second=None, cost_usd=None,
            quality_score=None, quality_reasoning=None, cost_efficiency=None,
            request_id=None, attempts_used=None, error_type="TimeoutError",
            error="timed out", span_id=None, trace_otel_id=None,
            tenant_id=None, use_case=None))
        assert repo.query("SELECT count() FROM llm_traces")[0][0] == 1

    def test_append_only_duplicate_trace_id_is_not_an_upsert(self, repo):
        repo.save(_result(trace_id="same"))
        repo.save(_result(trace_id="same"))
        assert repo.query("SELECT count() FROM llm_traces")[0][0] == 2

    def test_timestamp_round_trips_without_shifting(self, repo):
        repo.save(_result(trace_id="ts",
                          timestamp=datetime(2026, 7, 29, 23, 49, 55,
                                             tzinfo=timezone.utc)))
        stored = repo.query("SELECT toHour(timestamp) FROM llm_traces")[0][0]
        assert stored == 23

    def test_full_prompt_is_stored_untruncated(self, repo):
        """The supervisor's request: prompts saved as-is, not compressed or
        reduced. The old _preview() cut them at 100 characters."""
        long_prompt = "word " * 500          # 2500 chars, far past the old limit
        repo.save(_result(trace_id="long", prompt=long_prompt))
        stored = repo.query("SELECT prompt FROM llm_traces")[0][0]
        assert stored == long_prompt
        assert "..." not in stored

    def test_call_type_defaults_to_chat(self, repo):
        repo.save(_result(trace_id="ct"))
        assert repo.query("SELECT call_type FROM llm_traces")[0][0] == "chat"

    def test_save_failure_does_not_raise(self, repo, monkeypatch):
        """Instrumentation must never take down the calling application."""
        def boom(*a, **k):
            raise ClickHouseUnavailable("server gone")
        monkeypatch.setattr(repo._client, "insert", boom)
        repo.save(_result(trace_id="swallowed"))  # must not raise


class TestConcurrency:
    def test_reader_and_writer_coexist(self, repo):
        """The whole reason for the migration: no single-writer lock."""
        writer = TraceRepository(config=_config(repo.database))
        reader = TraceRepository(config=_config(repo.database), create_schema=False)
        try:
            writer.save(_result(trace_id="live"))
            assert reader.query("SELECT count() FROM llm_traces")[0][0] >= 1
            # The writer is still open — under DuckDB this second write would
            # have collided with the reader holding the file.
            writer.save(_result(trace_id="live-2"))
            assert reader.query("SELECT count() FROM llm_traces")[0][0] >= 2
        finally:
            writer.close()
            reader.close()


class TestViews:
    def test_v_cost_by_tenant(self, repo):
        repo.save_many([
            _result(trace_id="a", tenant_id="t1", cost_usd=1.0),
            _result(trace_id="b", tenant_id="t1", cost_usd=2.0),
            _result(trace_id="c", tenant_id="t2", cost_usd=3.0)])
        df = repo.query_df("SELECT * FROM v_cost_by_tenant ORDER BY tenant_id")
        assert len(df) == 2
        assert df[df["tenant_id"] == "t1"]["total_cost_usd"].iloc[0] == 3.0

    def test_v_cost_by_use_case(self, repo):
        repo.save_many([
            _result(trace_id="a", use_case="chat", cost_usd=1.0, quality_score=8),
            _result(trace_id="b", use_case="chat", cost_usd=2.0, quality_score=10)])
        df = repo.query_df("SELECT * FROM v_cost_by_use_case")
        assert df["avg_quality_score"].iloc[0] == 9.0

    def test_v_latency_percentiles(self, repo):
        repo.save_many([_result(trace_id=str(i), latency_ms=v)
                        for i, v in enumerate([100.0, 200.0, 300.0])])
        df = repo.query_df("SELECT * FROM v_latency_percentiles")
        assert df["p50_ms"].iloc[0] == 200.0

    def test_v_summary_matches_get_summary_when_unfiltered(self, repo):
        repo.save_many([
            _result(trace_id="a", cost_usd=1.0, latency_ms=100.0, quality_score=8),
            _result(trace_id="b", cost_usd=2.0, latency_ms=200.0, quality_score=10)])
        view = repo.query_df("SELECT * FROM v_summary").iloc[0]
        summary = repo.get_summary()
        assert view["total_calls"] == summary["total_calls"]
        assert abs(view["total_cost_usd"] - summary["total_cost_usd"]) < 1e-9


class TestModelByQueryType:
    """The view that answers 'which models are better for which requests' —
    the argument the routing recommender will eventually automate."""

    def test_compares_models_within_a_query_type(self, repo):
        repo.save_many([
            _result(trace_id="a", model="cheap", query_type="summarization",
                    cost_usd=0.001, quality_score=8, latency_ms=100.0),
            _result(trace_id="b", model="cheap", query_type="summarization",
                    cost_usd=0.001, quality_score=8, latency_ms=120.0),
            _result(trace_id="c", model="pricey", query_type="summarization",
                    cost_usd=0.010, quality_score=9, latency_ms=300.0),
        ])
        df = repo.query_df("SELECT * FROM v_model_by_query_type ORDER BY model")
        assert len(df) == 2
        cheap = df[df["model"] == "cheap"].iloc[0]
        pricey = df[df["model"] == "pricey"].iloc[0]
        # Nearly the same quality for a tenth of the money.
        assert cheap["quality_per_dollar"] > pricey["quality_per_dollar"]

    def test_embeddings_are_excluded(self, repo):
        repo.save_many([
            _result(trace_id="chat", query_type="summarization", cost_usd=0.001,
                    quality_score=8),
            _result(trace_id="emb", call_type="embedding", query_type=None,
                    cost_usd=0.001),
        ])
        df = repo.query_df("SELECT * FROM v_model_by_query_type")
        assert df["call_count"].sum() == 1

    def test_zero_cost_does_not_produce_infinity(self, repo):
        repo.save(_result(trace_id="free", query_type="reasoning",
                          cost_usd=0.0, quality_score=9))
        df = repo.query_df("SELECT quality_per_dollar FROM v_model_by_query_type")
        assert df["quality_per_dollar"].isna().all()


class TestGetSummary:
    def test_overall(self, repo):
        repo.save_many([
            _result(trace_id="a", cost_usd=1.0, latency_ms=100.0, quality_score=8),
            _result(trace_id="b", cost_usd=2.0, latency_ms=200.0, quality_score=10)])
        s = repo.get_summary()
        assert s["total_calls"] == 2
        assert s["total_cost_usd"] == 3.0
        assert s["avg_latency_ms"] == 150.0
        assert s["avg_quality_score"] == 9.0

    def test_filtered_by_tenant(self, repo):
        repo.save_many([
            _result(trace_id="a", tenant_id="t1", cost_usd=1.0),
            _result(trace_id="b", tenant_id="t2", cost_usd=2.0)])
        s = repo.get_summary(tenant_id="t1")
        assert s["total_calls"] == 1
        assert s["total_cost_usd"] == 1.0

    def test_empty_database_returns_zeros_not_none(self, repo):
        s = repo.get_summary()
        assert s["total_calls"] == 0
        assert s["total_cost_usd"] == 0.0


class TestSharedStore:
    """Several applications deliberately sharing one database.

    The mirror image of TestIsolation: same mechanism, opposite configuration.
    Two projects naming the SAME database get a company-wide view; two projects
    naming DIFFERENT ones stay separate.
    """

    def test_two_applications_write_to_one_store(self, repo):
        """Under DuckDB this was impossible - the second writer would have hit
        the single-writer lock. It is the behaviour the migration was for, so a
        change that reintroduced per-writer exclusivity must fail here."""
        app_a = TraceRepository(config=_config(repo.database))
        app_b = TraceRepository(config=_config(repo.database))
        reader = TraceRepository(config=_config(repo.database), create_schema=False)
        try:
            # Both applications are open at the same time, interleaved.
            app_a.save(_result(trace_id="a1", tenant_id="zakaria",
                               use_case="support-bot", cost_usd=0.004))
            app_b.save(_result(trace_id="b1", tenant_id="fadoua",
                               use_case="doc-summariser", cost_usd=0.010))
            app_a.save(_result(trace_id="a2", tenant_id="zakaria",
                               use_case="support-bot", cost_usd=0.003))

            rows = reader.query(
                "SELECT tenant_id, count(), sum(cost_usd) FROM llm_traces "
                "GROUP BY tenant_id ORDER BY tenant_id")
            assert [(r[0], r[1]) for r in rows] == [("fadoua", 1), ("zakaria", 2)]
            # The company-wide figure spans both applications.
            assert reader.get_summary()["total_calls"] == 3
            assert abs(reader.get_summary()["total_cost_usd"] - 0.017) < 1e-9
        finally:
            app_a.close()
            app_b.close()
            reader.close()

    def test_per_tenant_view_separates_the_two(self, repo):
        repo.save_many([
            _result(trace_id="a1", tenant_id="zakaria", cost_usd=0.004),
            _result(trace_id="b1", tenant_id="fadoua", cost_usd=0.010),
        ])
        df = repo.query_df("SELECT * FROM v_cost_by_tenant ORDER BY tenant_id")
        assert list(df["tenant_id"]) == ["fadoua", "zakaria"]


class TestIsolation:
    def test_two_projects_do_not_share_traces(self, server_available):
        a = "test_" + uuid.uuid4().hex[:12]
        b = "test_" + uuid.uuid4().hex[:12]
        ra, rb = TraceRepository(config=_config(a)), TraceRepository(config=_config(b))
        try:
            ra.save(_result(trace_id="a-1"))
            rb.save(_result(trace_id="b-1"))
            assert [r[0] for r in rb.query("SELECT trace_id FROM llm_traces")] == ["b-1"]
        finally:
            ra._client.command(f"DROP DATABASE IF EXISTS {a}")
            ra.close()
            rb._client.command(f"DROP DATABASE IF EXISTS {b}")
            rb.close()


class TestColumnTTL:
    def test_expiring_text_leaves_the_row_and_its_metrics(self, repo):
        """Column TTL must blank the text and keep the row.

        A table-level TTL would drop the row and take the cost history with it,
        which is the opposite of what is wanted: personal data should age out
        while the spend record stays. Uses a one-second interval and a forced
        merge, because a 90-day one cannot be waited out.
        """
        from datetime import datetime, timedelta, timezone

        db = repo.database
        repo._client.command(f"""
            CREATE TABLE {db}.ttl_probe (
                id String,
                timestamp DateTime64(3, 'UTC'),
                prompt String TTL timestamp + INTERVAL 1 SECOND,
                cost_usd Float64
            ) ENGINE = MergeTree ORDER BY id
        """)
        past = datetime.now(timezone.utc) - timedelta(minutes=5)
        repo._client.insert(f"{db}.ttl_probe", [["r1", past, "secret text", 0.5]],
                            column_names=["id", "timestamp", "prompt", "cost_usd"])
        # Column TTL is applied during merges, so force one.
        repo._client.command(f"OPTIMIZE TABLE {db}.ttl_probe FINAL")

        row = repo.query(f"SELECT id, prompt, cost_usd FROM {db}.ttl_probe")[0]
        assert row[0] == "r1"          # the row survives
        assert row[2] == 0.5           # and so does the metric
        assert row[1] == ""            # only the text is blanked


class TestQueryDf:
    def test_basic(self, repo):
        repo.save(_result())
        df = repo.query_df("SELECT provider, cost_usd FROM llm_traces")
        assert len(df) == 1
        assert df["provider"][0] == "openai"


class TestExportToParquet:
    def test_export(self, repo, tmp_path):
        repo.save(_result())
        out = tmp_path / "out.parquet"
        repo.export_to_parquet(str(out))
        assert out.exists()
        import pandas as pd
        assert len(pd.read_parquet(out)) == 1
