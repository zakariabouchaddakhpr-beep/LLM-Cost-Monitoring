"""The schema module is pure text — these checks need no server."""
from dataclasses import fields
from typing import get_args, get_type_hints

from llm_observability.core.domain.result import LLMCallResult
from llm_observability.core.infrastructure.persistence import schema


def _optional_fields():
    """Fields whose annotation admits None.

    NOT 'fields with a default' — call_type has a default and is never None,
    which is why the column is non-nullable. Reading the annotations means this
    cannot drift from the dataclass.
    """
    hints = get_type_hints(LLMCallResult)
    return {name for name, t in hints.items() if type(None) in get_args(t)}


def test_columns_are_the_persisted_subset_of_the_domain_object():
    declared = {f.name for f in fields(LLMCallResult)}
    assert set(schema.COLUMNS) == declared - schema.NON_PERSISTED
    assert schema.NON_PERSISTED == {"embeddings"}


def test_nullable_exactly_where_the_annotation_admits_none():
    """clickhouse-connect rejects None into a non-nullable column, and the real
    trace data carries NULL error and error_type on every successful row."""
    ddl = schema.table_ddl("proj")
    optional = _optional_fields() - schema.NON_PERSISTED
    for name in optional:
        line = next(l for l in ddl.splitlines() if l.strip().startswith(name + " "))
        assert "Nullable(" in line, f"{name} must be Nullable"
    for name in set(schema.COLUMNS) - optional:
        line = next(l for l in ddl.splitlines() if l.strip().startswith(name + " "))
        assert "Nullable(" not in line, f"{name} must NOT be Nullable"


def test_call_type_defaults_to_chat_so_old_rows_read_correctly():
    line = next(l for l in schema.table_ddl("p").splitlines()
                if l.strip().startswith("call_type "))
    assert "DEFAULT 'chat'" in line
    assert "Nullable" not in line


def test_text_columns_expire_but_the_row_does_not():
    ddl = schema.table_ddl("proj")
    for name in ("prompt", "response"):
        line = next(l for l in ddl.splitlines() if l.strip().startswith(name + " "))
        assert "TTL timestamp + INTERVAL 90 DAY" in line
    # A table-level TTL would drop whole rows, losing the cost history with them.
    assert "\nTTL " not in ddl


def test_migrations_are_idempotent_in_form():
    stmts = schema.migrations("proj")
    adds = [s for s in stmts if "ADD COLUMN" in s]
    assert adds and all("IF NOT EXISTS" in s for s in adds)
    renames = [s for s in stmts if "RENAME COLUMN" in s]
    assert renames and all("IF EXISTS" in s for s in renames)
    for s in stmts:
        # One statement each: the HTTP interface rejects multi-statement blobs.
        assert s.strip().rstrip(";").count(";") == 0
        assert "proj." in s


def test_every_new_column_has_a_migration():
    stmts = " ".join(schema.migrations("proj"))
    for name in ("call_type", "query_type", "cached_tokens", "reasoning_tokens",
                 "vector_dimensions", "input_count"):
        assert name in stmts


def test_table_is_append_only_mergetree():
    ddl = schema.table_ddl("proj")
    assert "ENGINE = MergeTree" in ddl
    assert "ReplacingMergeTree" not in ddl
    assert "PARTITION BY toYYYYMM(timestamp)" in ddl
    assert "ORDER BY (timestamp, provider, model)" in ddl


def test_percentiles_use_the_interpolating_quantile():
    """quantileExact diverges from the existing dashboard figures by 527ms at p99."""
    joined = " ".join(schema.view_ddl("proj"))
    assert "quantile(0.5" in joined
    assert "quantileExact" not in joined


def test_views_are_each_a_single_statement():
    views = schema.view_ddl("proj")
    assert len(views) == 5
    for statement in views:
        assert statement.strip().rstrip(";").count(";") == 0
    assert any("v_summary" in v for v in views)
    assert any("v_cost_by_tenant" in v for v in views)
    assert any("v_cost_by_use_case" in v for v in views)
    assert any("v_latency_percentiles" in v for v in views)
    assert any("v_model_by_query_type" in v for v in views)


def test_model_comparison_guards_the_divide_and_skips_embeddings():
    view = next(v for v in schema.view_ddl("proj") if "v_model_by_query_type" in v)
    # Without nullIf a zero average cost gives infinity, poisoning any sort.
    assert "nullIf(avg(cost_usd), 0)" in view
    # Embeddings are never judged, so they have no quality to divide.
    assert "call_type = 'chat'" in view


def test_database_name_is_interpolated():
    assert "myproj" in schema.table_ddl("myproj")
    assert "{database}" not in schema.table_ddl("myproj")
    assert all("{database}" not in v for v in schema.view_ddl("myproj"))
    assert all("{database}" not in m for m in schema.migrations("myproj"))


def test_database_ddl_is_idempotent():
    assert "IF NOT EXISTS" in schema.database_ddl("myproj")
